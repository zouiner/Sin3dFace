import os, sys, math, random
from torchvision.utils import save_image
import cv2
import numpy as np
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf

from data.utils import util_net
from data.utils import util_image
from data.utils import util_common

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from data.datapipe_SinSR.datasets_new import create_dataset 
from data.utils.util_image import ImageSpliterTh

import trimesh

from tqdm import tqdm

class BaseSampler:
    def __init__(
            self,
            configs,
            sf=None,
            use_fp16=False,
            chop_size=128,
            chop_stride=128,
            chop_bs=1,
            desired_min_size=64,
            seed=10000,
            ddim=False
            ):
        '''
        Input:
            configs: config, see the yaml file in folder ./configs/
            sf: int, super-resolution scale
            seed: int, random seed
        '''
        self.configs = configs
        self.chop_size = chop_size
        self.chop_stride = chop_stride
        self.chop_bs = chop_bs
        self.seed = seed
        self.use_fp16 = use_fp16
        self.desired_min_size = desired_min_size
        self.ddim=ddim
        if sf is None:
            sf = configs.diffusion.params.sf
        self.sf = sf

        self.setup_dist()  # setup distributed training: self.num_gpus, self.rank

        self.setup_seed()

        self.build_model()

    def setup_seed(self, seed=None):
        seed = self.seed if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_dist(self, gpu_id=None):
        num_gpus = torch.cuda.device_count()
        assert num_gpus == 1, 'Please assign one available GPU using CUDA_VISIBLE_DEVICES!'

        self.num_gpus = num_gpus
        self.rank = int(os.environ['LOCAL_RANK']) if num_gpus > 1 else 0

    def write_log(self, log_str):
        if self.rank == 0:
            print(log_str)

    def build_model(self):
        # diffusion model
        log_str = f'Building the diffusion model with length: {self.configs.diffusion.params.steps}...'
        self.write_log(log_str)
        self.base_diffusion = util_common.instantiate_from_config(self.configs.diffusion)
        model = util_common.instantiate_from_config(self.configs.model).cuda()
        ckpt_path =self.configs.model.ckpt_path
        assert ckpt_path is not None
        self.write_log(f'Loading Diffusion model from {ckpt_path}...')
        
        # MICA model
        from models.mica.mica import MICA
        self.mica_model = MICA(config=self.configs.MICA, device=f"cuda:0").cuda()
        self.mica_model.eval()

        self.load_model(model, ckpt_path)
        if self.use_fp16:
            model.dtype = torch.float16
            model.convert_to_fp16()
        self.model = model.eval()

        # autoencoder model
        if self.configs.autoencoder is not None:
            ckpt_path = self.configs.autoencoder.ckpt_path
            assert ckpt_path is not None
            self.write_log(f'Loading AutoEncoder model from {ckpt_path}...')
            autoencoder = util_common.instantiate_from_config(self.configs.autoencoder).cuda()
            self.load_model(autoencoder, ckpt_path)
            autoencoder.eval()
            if self.configs.autoencoder.use_fp16:
                self.autoencoder = autoencoder.half()
            else:
                self.autoencoder = autoencoder
        else:
            self.autoencoder = None

    def load_model(self, model, ckpt_path=None):
        state = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in state:
            state_SR = state['state_dict']
            util_net.reload_model(model, state_SR)
            self.write_log(f'Loading SR model from checkpoint. No MICA model in checkpoint.')
        elif 'state_dict_SinSR' in state:
            state_SR = state['state_dict_SinSR']
            util_net.reload_model(model, state_SR)
            self.mica_model.flameModel.load_state_dict(state['flameModel'])
            self.mica_model.arcface.load_state_dict(state['arcface'])
            self.write_log(f'Loading SR and MICA model from checkpoint.')

        else:
            util_net.reload_model(model, state)
        if 'iters_start' in state:
            self.current_iter = state['iters_start']
        
        

class Sampler(BaseSampler):    
    def sample_func(self, y0, noise_repeat=False, one_step=False, apply_decoder=True):
        '''
        Input:
            y0: n x c x h x w torch tensor, low-quality image, [-1, 1], RGB
        Output:
            sample: n x c x h x w, torch tensor, [-1, 1], RGB
        '''
        if noise_repeat:
            self.setup_seed()

        desired_min_size = self.desired_min_size
        ori_h, ori_w = y0.shape[2:]
        if not (ori_h % desired_min_size == 0 and ori_w % desired_min_size == 0):
            flag_pad = True
            pad_h = (math.ceil(ori_h / desired_min_size)) * desired_min_size - ori_h
            pad_w = (math.ceil(ori_w / desired_min_size)) * desired_min_size - ori_w
            y0 = F.pad(y0, pad=(0, pad_w, 0, pad_h), mode='reflect')
        else:
            flag_pad = False


        model_kwargs={'lq':y0,} if self.configs.model.params.cond_lq else None
        
        if not self.ddim:        
            results = self.base_diffusion.p_sample_loop(
                    y=y0,
                    model=self.model,
                    first_stage_model=self.autoencoder,
                    noise=None,
                    noise_repeat=noise_repeat,
                    clip_denoised=(self.autoencoder is None),
                    denoised_fn=None,
                    model_kwargs=model_kwargs,
                    progress=False,
                    one_step=one_step,
                    apply_decoder=apply_decoder
                    )    # This has included the decoding for latent space
        else:
            results = self.base_diffusion.ddim_sample_loop(
                    y=y0,
                    model=self.model,
                    first_stage_model=self.autoencoder,
                    noise=None,
                    clip_denoised=(self.autoencoder is None),
                    denoised_fn=None,
                    model_kwargs=model_kwargs,
                    progress=True,
                    one_step=one_step,
                    apply_decoder=apply_decoder
                    )    # This has included the decoding for latent space
        if flag_pad and apply_decoder:
            results = results[:, :, :ori_h*self.sf, :ori_w*self.sf]
            
        if not apply_decoder:
            return results["pred_xstart"]
        return results.clamp_(-1.0, 1.0)

    
    def inference(self, in_path, out_path, bs=1, noise_repeat=False, one_step=False, return_tensor=False, apply_decoder=True):
        '''
        Inference demo.
        Input:
            in_path: str, folder or image path for LQ image
            out_path: str, folder save the results
            bs: int, default bs=1, bs % num_gpus == 0
        '''
        def _process_per_image(im_lq_tensor):
            '''
            Input:
                im_lq_tensor: b x c x h x w, torch tensor, [0,1], RGB
            Output:
                im_sr: h x w x c, numpy array, [0,1], RGB
            '''

            if im_lq_tensor.shape[2] > self.chop_size or im_lq_tensor.shape[3] > self.chop_size:
                im_spliter = ImageSpliterTh(
                        im_lq_tensor,
                        self.chop_size,
                        stride=self.chop_stride,
                        sf=self.sf,
                        extra_bs=self.chop_bs,  
                        )
                for im_lq_pch, index_infos in im_spliter:
                    # print(im_lq_pch.shape)
                    im_sr_pch = self.sample_func(
                            (im_lq_pch - 0.5) / 0.5,
                            noise_repeat=noise_repeat, one_step=one_step, apply_decoder=apply_decoder
                            )     # 1 x c x h x w, [-1, 1]
                    im_spliter.update(im_sr_pch.detach(), index_infos)
                    
                im_sr_tensor = im_spliter.gather()
            else:
                im_sr_tensor = self.sample_func(
                        (im_lq_tensor - 0.5) / 0.5,
                        noise_repeat=noise_repeat, one_step=one_step, apply_decoder=apply_decoder
                        )     # 1 x c x h x w, [-1, 1]

            if apply_decoder:
                im_sr_tensor = im_sr_tensor * 0.5 + 0.5
            return im_sr_tensor

        in_path = Path(in_path) if not isinstance(in_path, Path) else in_path
        out_path = Path(out_path) if not isinstance(out_path, Path) else out_path
        if not out_path.exists():
            out_path.mkdir(parents=True)
        
        return_res = {}
        if bs > 1:
            assert in_path.is_dir(), "Input path must be folder when batch size is larger than 1."

            data_config = {'type': 'folder',
                           'params': {'dir_path': str(in_path),
                                      'transform_type': 'default',
                                      'transform_kwargs': {
                                          'mean': 0.0,
                                          'std': 1.0,
                                          },
                                      'need_path': True,
                                      'recursive': True,
                                      'length': None,
                                      }
                           }
            dataset = create_dataset(data_config)
            self.write_log(f'Find {len(dataset)} images in {in_path}')
            dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=bs,
                    shuffle=False,
                    drop_last=False,
                    )
            for micro_data in dataloader:
                results = _process_per_image(micro_data['lq'].cuda())    # b x h x w x c, [0, 1], RGB
                for jj in range(results.shape[0]):
                    im_sr = util_image.tensor2img(results[jj], rgb2bgr=True, min_max=(0.0, 1.0))
                    im_name = Path(micro_data['path'][jj]).stem
                    im_path = out_path / f"{im_name}.png"
                    util_image.imwrite(im_sr, im_path, chn='bgr', dtype_in='uint8')
                    if return_tensor:
                        return_res[im_path.stem]=results[jj]
                        
        else:
            if not in_path.is_dir():
                im_lq = util_image.imread(in_path, chn='rgb', dtype='float32')  # h x w x c
                im_lq_tensor = util_image.img2tensor(im_lq).cuda()              # 1 x c x h x w
                im_sr_tensor = _process_per_image(im_lq_tensor)
                im_sr = util_image.tensor2img(im_sr_tensor, rgb2bgr=True, min_max=(0.0, 1.0))

                im_path = out_path / f"{in_path.stem}.png"
                util_image.imwrite(im_sr, im_path, chn='bgr', dtype_in='uint8')
                if return_tensor:
                    return_res[im_path.stem]=im_sr_tensor
            else:
                im_path_list = [x for x in in_path.glob("*.[jpJP][pnPN]*[gG]")]
                self.write_log(f'Find {len(im_path_list)} images in {in_path}')

                faces = self.mica_model.flameModel.generator.faces_tensor.cpu()
                self.mica_model.testing = True 

                ex_path = out_path / str(self.current_iter)

                for im_path in tqdm(im_path_list, desc="Processing Images", unit="img"):
                    im_lq = util_image.imread(im_path, chn='rgb', dtype='float32')  # h x w x c
                    im_lq_tensor = util_image.img2tensor(im_lq).cuda()              # 1 x c x h x w
                    if im_lq.shape[0] <= 64:
                        im_lq_tensor = F.interpolate(im_lq_tensor, size=64, mode='bilinear', align_corners=False) # -> 64 x 64
                    for k in range(self.configs.sample):
                        if self.configs.randseed and self.configs.sample == 1:
                            self.setup_seed()
                        elif self.configs.randseed:
                            self.setup_seed(seed=self.seed + k)
                        else:
                            self.setup_seed(seed=self.seed)
                        im_sr_tensor = _process_per_image(im_lq_tensor)
                        
                        # make it to be the same size with the real
                        if im_lq.shape[0] <= 256:
                            im_sr_tensor = F.interpolate(im_sr_tensor, size=(int(im_lq.shape[0]), int(im_lq.shape[0])))
                        
                        im_sr = util_image.tensor2img(im_sr_tensor, rgb2bgr=True, min_max=(0.0, 1.0))

                        im_path_img = ex_path / 'sr_img' 
                        im_path_img.mkdir(parents=True, exist_ok=True)
                        

                        if self.configs.sample == 1:
                            name = f"{im_path.stem}.png"
                            save_path = im_path_img / name
                        else:
                            name = f"{im_path.stem}" + '_' + str(k).zfill(len(str(self.configs.sample))) + '.png'
                            save_path = im_path_img / name
                        util_image.imwrite(im_sr, save_path, chn='bgr', dtype_in='uint8')

                        # MICA
                        
                        image_mica = self.mica_model.tensor2tensor_img(im_sr_tensor, size = 224) * 255.0
                        temp = self.mica_model.create_tensor_blob(image_mica)
                        arcface = temp.detach().cuda().unsqueeze(0) # (3, 112, 112) -> (1,3,112,112)
                        arcface = 2*arcface - 1 # (0,1) -> (-1,1)
                        image = image_mica.detach().cuda().unsqueeze(0) / 255.0 # (3, 224, 224) -> (1,3,224,224)
                        
                        codedict = self.mica_model.encode(image, arcface)
                        opdict = self.mica_model.decode(codedict)
                        meshes = opdict['pred_canonical_shape_vertices']
                        code = opdict['pred_shape_code']
                        lmk = self.mica_model.flame.compute_landmarks(meshes)

                        mesh = meshes[0].detach()
                        landmark_51 = lmk[0, 17:]
                        landmark_7 = landmark_51[[19, 22, 25, 28, 16, 31, 37]]

                        im_path_img = ex_path / '3d_obj' 
                        save_path = im_path_img / name[:-4]
                        save_path.mkdir(parents=True, exist_ok=True)

                        trimesh.Trimesh(vertices=mesh.cpu() * 1000.0, faces=faces, process=False).export(f'{save_path}/mesh.ply')  # save in millimeters
                        trimesh.Trimesh(vertices=mesh.cpu() * 1000.0, faces=faces, process=False).export(f'{save_path}/mesh.obj')
                        np.save(f'{save_path}/identity', code[0].cpu().detach().numpy())
                        np.save(f'{save_path}/kpt7', landmark_7.cpu().detach().numpy() * 1000.0)
                        np.save(f'{save_path}/kpt68', lmk.cpu().detach().numpy() * 1000.0)


                    if return_tensor:
                        return_res[im_path.stem]=im_sr_tensor

        self.write_log(f"Processing done, enjoy the results in {str(out_path)}")
        return return_res
    
if __name__ == '__main__':
    pass

