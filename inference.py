import os, sys
import argparse
from pathlib import Path

from omegaconf import OmegaConf
from sampler import Sampler

from data.utils.util_opts import str2bool
from basicsr.utils.download_util import load_file_from_url

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, default="", help="Input path.")
    parser.add_argument("-o", "--out_path", type=str, default="./results", help="Output path.")
    parser.add_argument("-r", "--ref_path", type=str, default=None, help="reference image")
    parser.add_argument("-s", "--steps", type=int, default=15, help="Diffusion length. (The number of steps that the model trained on.)")
    parser.add_argument("-c", "--config", type=str, default=None)
    parser.add_argument("-is", "--infer_steps", type=int, default=None, help="Diffusion length for inference")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor for SR.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument("--one_step", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--sample", type=int, default=1)
    parser.add_argument("--randseed", type=bool, default=True, help="Random seed.")
    parser.add_argument(
            "--chop_size",
            type=int,
            default=512,
            choices=[512, 256],
            help="Chopping forward.",
            )
    parser.add_argument(
            "--task",
            type=str,
            default="SinSR",
            choices=["SinSR",'realsrx4', 'bicsrx4_opencv', 'bicsrx4_matlab'],
            help="Chopping forward.",
            )
    parser.add_argument("--ddim", action="store_true")
    
    
    args = parser.parse_args()
    if args.infer_steps is None:
        args.infer_steps = args.steps
    print(f"[INFO] Using the inference step: {args.steps}")
    return args

def get_configs(args):
    if args.config is None:
        if args.task == "SinSR":
            configs = OmegaConf.load('./configs/SinSR.yaml')
        elif args.task == 'realsrx4':
            configs = OmegaConf.load('./configs/realsr_swinunet_realesrgan256.yaml')
    else:
        configs = OmegaConf.load(args.config)
    # prepare the checkpoint
    ckpt_dir = Path('./data/weights')
    if args.ckpt is None:
        if not ckpt_dir.exists():
            ckpt_dir.mkdir()
        if args.task == "SinSR":
            ckpt_path = ckpt_dir / f'SinSR_v1.pth'
        elif args.task == 'realsrx4':
            ckpt_path = ckpt_dir / f'resshift_{args.task}_s{args.steps}_v1.pth'
    else:
        ckpt_path = Path(args.ckpt)
    print(f"[INFO] Using the checkpoint {ckpt_path}")
    
    if not ckpt_path.exists():
        if args.task == "SinSR":
            load_file_from_url(
                url=f"https://github.com/wyf0912/SinSR/releases/download/v1.0/{ckpt_path.name}",
                model_dir=ckpt_dir,
                progress=True,
                file_name=ckpt_path.name,
                )
        else:
            load_file_from_url(
                url=f"https://github.com/zsyOAOA/ResShift/releases/download/v2.0/{ckpt_path.name}",
                model_dir=ckpt_dir,
                progress=True,
                file_name=ckpt_path.name,
                )
    vqgan_path = ckpt_dir / f'autoencoder_vq_f4.pth'
    if not vqgan_path.exists():
         load_file_from_url(
            url="https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth",
            model_dir=ckpt_dir,
            progress=True,
            file_name=vqgan_path.name,
            )

    configs.model.ckpt_path = str(ckpt_path)
    configs.diffusion.params.timestep_respacing = args.infer_steps
    configs.diffusion.params.sf = args.scale
    configs.autoencoder.ckpt_path = str(vqgan_path)
    configs.sample = args.sample
    configs.randseed = args.randseed

    # save folder
    if not Path(args.out_path).exists():
        Path(args.out_path).mkdir(parents=True)

    if args.chop_size == 512:
        chop_stride = 448
    elif args.chop_size == 256:
        chop_stride = 224
    else:
        raise ValueError("Chop size must be in [512, 384, 256]")

    return configs, chop_stride

def main():
    args = get_parser()

    configs, chop_stride = get_configs(args)

    resshift_sampler = Sampler(
            configs,
            chop_size=args.chop_size,
            chop_stride=chop_stride,
            chop_bs=1,
            use_fp16=True,
            seed=args.seed,
            ddim=args.ddim
            )

    resshift_sampler.inference(args.in_path, args.out_path, bs=1, noise_repeat=False, one_step=args.one_step)
    # import evaluate
    # evaluate.evaluate(args.out_path, args.ref_path, None)
    
    
if __name__ == '__main__':
    main()


# python3 -m debugpy --listen 0.0.0.0:5678 --wait-for-client inference.py -i /users/ps1510/scratch/Programs/3d-super-resolution-Face-reconstruction/contents/NoW_MICA_16_64/sr_16_64 -o results/NoW_16_64_test --scale 4 --ckpt /users/ps1510/scratch/Programs/SinSR/logs/SinSR_train_16_64/2025-03-08-02-59/ckpts/model_50000.pth --one_step --config configs/Sin3d_vggface2_16_64_256_64.yaml

# python3 inference.py -i /users/ps1510/scratch/Programs/3d-super-resolution-Face-reconstruction/contents/NoW_MICA_8_32/sr_8_32 -o results/NoW_8_32_model3 --scale 4 --ckpt /users/ps1510/scratch/Programs/Sin3dFace/logs/SinSR_vggface2_8_32_256_32/2025-03-16-14-38/ckpts/model_55000.pth --one_step --config configs/Sin3d_vggface2_8_32_256_32.yaml # --sample 15
# python3 inference.py -i /users/ps1510/scratch/Programs/3d-super-resolution-Face-reconstruction/contents/NoW_MICA_16_64/sr_16_64 -o results/NoW_16_64_model3 --scale 4 --ckpt /users/ps1510/scratch/Programs/Sin3dFace/logs/Sin3d_vggface2_16_64_256_64/2025-03-21-02-50/best_model.pth --one_step --config configs/Sin3d_vggface2_16_64_256_64.yaml --sample 15
# python3 inference.py -i /users/ps1510/scratch/Programs/3d-super-resolution-Face-reconstruction/contents/NoW_MICA_32_128/sr_32_128 -o results/NoW_32_128_model3 --scale 4 --ckpt /users/ps1510/scratch/Programs/Sin3dFace/logs/Sin3d_vggface2_32_128_256_128/2025-03-22-03-55/best_model.pth --one_step --config configs/Sin3d_vggface2_32_128_256_128.yaml --sample 15
# python3 inference.py -i /users/ps1510/scratch/Programs/3d-super-resolution-Face-reconstruction/contents/NoW_MICA_64_256/sr_64_256 -o results/NoW_64_256_model3 --scale 4 --ckpt /users/ps1510/scratch/Programs/Sin3dFace/logs/Sin3d_vggface2_64_256_256_256/2025-03-24-22-43/best_model.pth --one_step --config configs/Sin3d_vggface2_64_256_256_256.yaml --sample 15

# python3 inference.py -i /users/ps1510/scratch/Programs/3d-super-resolution-Face-reconstruction/contents/NoW_MICA_8_32/sr_8_32 -o results/NoW_8_32_model2 --scale 4 --ckpt /users/ps1510/scratch/Programs/Sin3dFace/logs/Sin3d_vggface2_8_32_256_32_model2/2025-04-01-10-17/ckpts/model_55000.pth --one_step --config configs/Sin3d_vggface2_8_32_256_32.yaml # --sample 15
# python3 inference.py -i /users/ps1510/scratch/Programs/3d-super-resolution-Face-reconstruction/contents/NoW_MICA_16_64/sr_16_64 -o results/NoW_16_64_model2 --scale 4 --ckpt /users/ps1510/scratch/Programs/Sin3dFace/logs/Sin3d_vggface2_16_64_256_64_model2/2025-03-31-01-32/best_model.pth --one_step --config configs/Sin3d_vggface2_16_64_256_64.yaml --sample 15
# python3 inference.py -i /users/ps1510/scratch/Programs/3d-super-resolution-Face-reconstruction/contents/NoW_MICA_32_128/sr_32_128 -o results/NoW_32_128_model3_001 --scale 4 --ckpt /users/ps1510/scratch/Programs/Sin3dFace/logs/Sin3d_vggface2_32_128_256_128_001/2025-04-18-15-59/ckpts/model_147000.pth --one_step --config configs/Sin3d_vggface2_32_128_256_128.yaml --sample 15
# python3 inference.py -i /users/ps1510/scratch/Programs/3d-super-resolution-Face-reconstruction/contents/NoW_MICA_64_256/sr_64_256 -o results/NoW_64_256_model3_001 --scale 4 --ckpt /users/ps1510/scratch/Programs/Sin3dFace/logs/Sin3d_vggface2_64_256_256_256_001/2025-04-20-01-28/best_model.pth --one_step --config configs/Sin3d_vggface2_64_256_256_256.yaml # --sample 15
# python3 inference.py -i /users/ps1510/scratch/Programs/3d-super-resolution-Face-reconstruction/contents/NoW_MICA_64_256/sr_64_256 -o results/NoW_64_256_model3_001 --scale 4 --ckpt /users/ps1510/scratch/Programs/Sin3dFace/logs/Sin3d_vggface2_64_256_256_256_001/2025-04-20-01-28/ckpts/model_53000.pth --one_step --config configs/Sin3d_vggface2_64_256_256_256.yaml # --sample 15


# python3 inference.py -i /users/ps1510/scratch/Programs/3d-super-resolution-Face-reconstruction/contents/NoW_MICA_8_32/sr_8_32 -o results/NoW_8_32_model3 --scale 4 --ckpt /users/ps1510/scratch/Programs/Sin3dFace/logs/SinSR_vggface2_8_32_256_32/2025-03-16-14-38/ckpts/model_55000.pth --one_step --config configs/Sin3d_vggface2_8_32_256_32.yaml 
# python3 inference.py -i /users/ps1510/scratch/Programs/3d-super-resolution-Face-reconstruction/contents/NoW_MICA_8_32/sr_8_32 -o results/NoW_8_32_model3 --scale 4 --ckpt /users/ps1510/scratch/Programs/Sin3dFace/logs/SinSR_vggface2_8_32_256_32/2025-03-16-14-38/ckpts/model_80000.pth --one_step --config configs/Sin3d_vggface2_8_32_256_32.yaml 
# python3 inference.py -i /users/ps1510/scratch/Programs/3d-super-resolution-Face-reconstruction/contents/NoW_MICA_8_32/sr_8_32 -o results/NoW_8_32_model2 --scale 4 --ckpt /users/ps1510/scratch/Programs/Sin3dFace/logs/Sin3d_vggface2_8_32_256_32_model2/2025-04-01-10-17/ckpts/model_55000.pth --one_step --config configs/Sin3d_vggface2_8_32_256_32.yaml 










# # model 1
# python3 inference.py -i /users/ps1510/scratch/Programs/3d-super-resolution-Face-reconstruction/contents/NoW_MICA_8_32/sr_8_32 -o results/NoW_8_32_model1_s1 --scale 4 --ckpt /users/ps1510/scratch/Programs/SinSR/logs/SinSR_train_8_32/2025-03-07-19-30/ckpts/model_50000.pth --one_step --config configs/Sin3d_vggface2_8_32_256_32.yaml # --sample 15
# python3 inference.py -i /users/ps1510/scratch/Programs/3d-super-resolution-Face-reconstruction/contents/NoW_MICA_16_64/sr_16_64 -o results/NoW_16_64_model1_s1 --scale 4 --ckpt /users/ps1510/scratch/Programs/SinSR/logs/SinSR_train_16_64/2025-03-08-02-59/ckpts/model_50000.pth --one_step --config configs/Sin3d_vggface2_16_64_256_64.yaml # --sample 15
# python3 inference.py -i /users/ps1510/scratch/Programs/3d-super-resolution-Face-reconstruction/contents/NoW_MICA_32_128/sr_32_128 -o results/NoW_32_128_model1_s1 --scale 4 --ckpt /users/ps1510/scratch/Programs/SinSR/logs/SinSR_train_32_128/2025-03-09-10-09/ckpts/model_50000.pth --one_step --config configs/Sin3d_vggface2_32_128_256_128.yaml # --sample 15
# python3 inference.py -i /users/ps1510/scratch/Dataset/all_unknown_img_64_256/sr_64_256 -o results/NoW_64_256_model1_s1_unkown_img --scale 4 --ckpt /users/ps1510/scratch/Programs/SinSR/logs/SinSR_train_64_256/2025-03-09-02-23/ckpts/model_50000.pth --one_step --config configs/Sin3d_vggface2_64_256_256_256.yaml --sample 50 -- andseed False

# # model2
# python3 inference.py -i /users/ps1510/scratch/Programs/3d-super-resolution-Face-reconstruction/contents/NoW_MICA_8_32/sr_8_32 -o results/NoW_8_32_model2_s1 --scale 4 --ckpt /users/ps1510/scratch/Programs/Sin3dFace/logs/Sin3d_vggface2_8_32_256_32_model2/2025-04-01-10-17/ckpts/model_55000.pth --one_step --config configs/Sin3d_vggface2_8_32_256_32.yaml # --sample 15
# python3 inference.py -i /users/ps1510/scratch/Programs/3d-super-resolution-Face-reconstruction/contents/NoW_MICA_16_64/sr_16_64 -o results/NoW_16_64_model2_s1 --scale 4 --ckpt /users/ps1510/scratch/Programs/Sin3dFace/logs/Sin3d_vggface2_16_64_256_64_model2/2025-03-31-01-32/ckpts/model_55000.pth --one_step --config configs/Sin3d_vggface2_16_64_256_64.yaml # --sample 15
# python3 inference.py -i /users/ps1510/scratch/Programs/3d-super-resolution-Face-reconstruction/contents/NoW_MICA_32_128/sr_32_128 -o results/NoW_32_128_model2_s1 --scale 4 --ckpt /users/ps1510/scratch/Programs/Sin3dFace/logs/Sin3d_vggface2_32_128_256_128_model2/2025-04-16-10-09/ckpts/model_54000.pth --one_step --config configs/Sin3d_vggface2_32_128_256_128.yaml # --sample 15
# python3 inference.py -i /users/ps1510/scratch/Programs/3d-super-resolution-Face-reconstruction/contents/NoW_MICA_64_256/sr_64_256 -o results/NoW_64_256_model2_s1 --scale 4 --ckpt /users/ps1510/scratch/Programs/Sin3dFace/logs/Sin3d_vggface2_64_256_256_256_model2/2025-03-28-00-38/ckpts/model_53000.pth --one_step --config configs/Sin3d_vggface2_64_256_256_256.yaml # --sample 15

# # model3
# python3 inference.py -i /users/ps1510/scratch/Programs/3d-super-resolution-Face-reconstruction/contents/NoW_MICA_8_32/sr_8_32 -o results/NoW_8_32_model3_001_s1 --scale 4 --ckpt /users/ps1510/scratch/Programs/Sin3dFace/logs/Sin3d_vggface2_8_32_256_32_001/2025-04-23-01-25/ckpts/model_106000.pth --one_step --config configs/Sin3d_vggface2_8_32_256_32.yaml # --sample 15
# python3 inference.py -i /users/ps1510/scratch/Programs/3d-super-resolution-Face-reconstruction/contents/NoW_MICA_16_64/sr_16_64 -o results/NoW_16_64_model3_001_s1 --scale 4 --ckpt /users/ps1510/scratch/Programs/Sin3dFace/logs/Sin3d_vggface2_16_64_256_64_001/2025-04-22-01-39/ckpts/model_157000.pth --one_step --config configs/Sin3d_vggface2_16_64_256_64.yaml # --sample 15
# python3 inference.py -i /users/ps1510/scratch/Programs/3d-super-resolution-Face-reconstruction/contents/NoW_MICA_32_128/sr_32_128 -o results/NoW_32_128_model3_001_s1 --scale 4 --ckpt /users/ps1510/scratch/Programs/Sin3dFace/logs/Sin3d_vggface2_32_128_256_128_001/2025-04-18-15-59/ckpts/model_159000.pth --one_step --config configs/Sin3d_vggface2_32_128_256_128.yaml # --sample 15
# python3 inference.py -i /users/ps1510/scratch/Programs/3d-super-resolution-Face-reconstruction/contents/NoW_MICA_64_256/sr_64_256 -o results/NoW_64_256_model3_001_s1 --scale 4 --ckpt /users/ps1510/scratch/Programs/Sin3dFace/logs/Sin3d_vggface2_64_256_256_256_001/2025-04-20-01-28/ckpts/model_119000.pth --one_step --config configs/Sin3d_vggface2_64_256_256_256.yaml #--sample 15

# python3 inference.py -i /users/ps1510/scratch/Programs/3d-super-resolution-Face-reconstruction/contents/NoW_MICA_8_32/sr_8_32 -o results/NoW_8_32_model3_001_s1 --scale 4 --ckpt /users/ps1510/scratch/Programs/Sin3dFace/logs/Sin3d_vggface2_8_32_256_32_001/2025-04-23-01-25/ckpts/model_84000.pth --one_step --config configs/Sin3d_vggface2_8_32_256_32.yaml # --sample 15
# python3 inference.py -i /users/ps1510/scratch/Programs/3d-super-resolution-Face-reconstruction/contents/NoW_MICA_16_64/sr_16_64 -o results/NoW_16_64_model3_001_s1 --scale 4 --ckpt /users/ps1510/scratch/Programs/Sin3dFace/logs/Sin3d_vggface2_16_64_256_64_001/2025-04-22-01-39/best_model.pth --one_step --config configs/Sin3d_vggface2_16_64_256_64.yaml # --sample 15
# python3 inference.py -i /users/ps1510/scratch/Programs/3d-super-resolution-Face-reconstruction/contents/NoW_MICA_32_128/sr_32_128 -o results/NoW_32_128_model3_001_s1 --scale 4 --ckpt /users/ps1510/scratch/Programs/Sin3dFace/logs/Sin3d_vggface2_32_128_256_128_001/2025-04-18-15-59/best_model.pth --one_step --config configs/Sin3d_vggface2_32_128_256_128.yaml # --sample 15
# python3 inference.py -i /users/ps1510/scratch/Programs/3d-super-resolution-Face-reconstruction/contents/NoW_MICA_64_256/sr_64_256 -o results/NoW_64_256_model3_001_s1 --scale 4 --ckpt /users/ps1510/scratch/Programs/Sin3dFace/logs/Sin3d_vggface2_64_256_256_256_001/2025-04-20-01-28/best_model.pth --one_step --config configs/Sin3d_vggface2_64_256_256_256.yaml # --sample 15
