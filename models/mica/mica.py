# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2023 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: mica@tue.mpg.de


import os
import sys

import torch
import torch.nn.functional as F

from models.mica.arcface import Arcface
from models.mica.generator import Generator
from models.mica.lib.micalib.base_model import BaseModel

from loguru import logger

from models.mica.lib.micalib.renderer import MeshShapeRenderer


class MICA(BaseModel):
    def __init__(self, config=None, device=None, tag='MICA'):
        super(MICA, self).__init__(config, device, tag)

        self.initialize()
        
        # add by patipol
        self.render = MeshShapeRenderer(obj_filename=config.model.topology_path)

    def create_model(self, model_cfg):
        mapping_layers = model_cfg.mapping_layers
        pretrained_path = None
        if not model_cfg.use_pretrained:
            pretrained_path = model_cfg.arcface_pretrained_model
        self.arcface = Arcface(pretrained_path=pretrained_path).to(self.device)
        self.flameModel = Generator(512, 300, self.cfg.model.n_shape, mapping_layers, model_cfg, self.device)

    def load_model(self):
        # model_path = os.path.join(self.cfg.output_dir, 'model_mica.tar')
        # if os.path.exists(self.cfg.ckpt_path) and self.cfg.model.use_pretrained:
        #     model_path = self.cfg.ckpt_path
        # if os.path.exists(model_path):
        #     logger.info(f'[{self.tag}] Trained model found. Path: {model_path} | GPU: {self.device}')
        #     checkpoint = torch.load(model_path)
        #     if 'arcface' in checkpoint:
        #         self.arcface.load_state_dict(checkpoint['arcface'])
        #     if 'flameModel' in checkpoint:
        #         self.flameModel.load_state_dict(checkpoint['flameModel'])
        # else:
        #     logger.info(f'[{self.tag}] Checkpoint not available starting from scratch!')
        pass

    def model_dict(self):
        return {
            'flameModel': self.flameModel.state_dict(),
            'arcface': self.arcface.state_dict()
        }

    def parameters_to_optimize(self):
        return [
            {'params': self.flameModel.parameters(), 'lr': self.cfg.train.lr},
            {'params': self.arcface.parameters(), 'lr': self.cfg.train.arcface_lr},
        ]

    def encode(self, images, arcface_imgs):
        codedict = {}

        codedict['arcface'] = F.normalize(self.arcface(arcface_imgs))
        codedict['images'] = images

        return codedict

    def decode(self, codedict):

        flame_verts_shape = None
        shapecode = None

        if not self.testing:
            flame = codedict['flame']
            shapecode = flame['shape_params'].view(-1, flame['shape_params'].shape[2])
            shapecode = shapecode.to(self.device)[:, :self.cfg.model.n_shape]
            with torch.no_grad():
                flame_verts_shape, _, _ = self.flame(shape_params=shapecode)

        identity_code = codedict['arcface']
        pred_canonical_vertices, pred_shape_code = self.flameModel(identity_code)
        

        output = {
            'flame_verts_shape': flame_verts_shape,
            'flame_shape_code': shapecode,
            'pred_canonical_shape_vertices': pred_canonical_vertices,
            'pred_shape_code': pred_shape_code,
            'faceid': codedict['arcface']
        }

        return output

    def compute_losses(self, input, encoder_output, decoder_output):
        losses = {}

        pred_verts = decoder_output['pred_canonical_shape_vertices']
        gt_verts = decoder_output['flame_verts_shape'].detach()

        pred_verts_shape_canonical_diff = (pred_verts - gt_verts).abs()

        if self.use_mask:
            pred_verts_shape_canonical_diff *= self.vertices_mask

        losses['pred_verts_shape_canonical_diff'] = torch.mean(pred_verts_shape_canonical_diff) * 1000.0

        return losses
    
    def create_tensor_blob(self, images, input_mean=127.5, input_std=127.5, size=(112, 112), swapRB=True):
        """
        images: tensor of shape (3, H, W), assumed to be in range [0, 1]
        input_mean: mean value for normalization
        input_std: standard deviation for normalization
        size: target size for resizing (width, height)
        swapRB: swap the Red and Blue channels (if True, swap channels)
        """

        # Normalize: (image - mean) / std
        images = (images - input_mean) / input_std
        
        # Resize the image using interpolation
        resized_images = F.interpolate(images.unsqueeze(0), size=size, mode='bilinear', align_corners=False).squeeze(0)

        # Swap the channels from RGB to BGR if necessary
        if swapRB:
            resized_images = resized_images[[2, 1, 0], :, :]  # Swap channels if needed
        
        return resized_images
    
    def tensor2tensor_img(self, tensor, min_max=(-1, 1), size = False):
        if size:
            if tensor.ndim == 3:  # (C, H, W)
                tensor = F.interpolate(tensor.unsqueeze(0), size=(size, size), mode='bilinear', align_corners=False)
                tensor = tensor.squeeze(0)  # Back to (C, H, W)

            elif tensor.ndim == 4:  # (N, C, H, W)
                tensor = F.interpolate(tensor, size=(size, size), mode='bilinear', align_corners=False)

        tensor = tensor.float().cpu().clamp_(*min_max)  # clamp
        tensor = (tensor - min_max[0]) / \
            (min_max[1] - min_max[0])  # to range [0,1]
        return tensor.squeeze(0)


    def training_MICA(self, batch):

        images = batch['image']
        images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        flame = batch['flame']
        arcface = batch['arcface'].cuda()
        arcface = arcface.view(-1, arcface.shape[-3], arcface.shape[-2], arcface.shape[-1])

        inputs = {
            'images': images,
            'dataset': batch['dataset'][0]
        }
        encoder_output = self.encode(images, arcface)
        encoder_output['flame'] = flame
        decoder_output = self.decode(encoder_output)

        opdict = \
            {
                'images': images,
                'flame_verts_shape': decoder_output['flame_verts_shape'],
                'pred_canonical_shape_vertices': decoder_output['pred_canonical_shape_vertices'],
            }

        if 'deca' in decoder_output:
            opdict['deca'] = decoder_output['deca']

        return inputs, opdict, encoder_output, decoder_output