import os
import json
import sys
from typing import *
import importlib

import click
import torch
import utils3d

from moge.test.baseline import MGEBaselineInterface
from test_time.data import generate_jittered_batch
from test_time.loss import compute_moge2_ttt_loss, compute_moge2_ttt_loss_from_orig, moge2_ttt_loss
from moge.utils.geometry_torch import normalized_view_plane_uv, recover_focal_shift

class Baseline(MGEBaselineInterface):

    def __init__(self, num_tokens: int, resolution_level: int, pretrained_model_name_or_path: str, use_fp16: bool, device: str = 'cuda:0', version: str = 'v1'):
        super().__init__()
        from moge.model import import_model_class_by_version
        MoGeModel = import_model_class_by_version(version)
        self.version = version

        self._pretrained_model_name_or_path = pretrained_model_name_or_path
        self._original_model = MoGeModel.from_pretrained(pretrained_model_name_or_path).to(device).eval()
        self._original_model.requires_grad_(False)
        
        self.device = torch.device(device)
        self.num_tokens = num_tokens
        self.resolution_level = resolution_level
        self.use_fp16 = False

    @click.command()
    @click.option('--num_tokens', type=int, default=None)
    @click.option('--resolution_level', type=int, default=9)
    @click.option('--pretrained', 'pretrained_model_name_or_path', type=str, default='Ruicheng/moge-vitl')
    @click.option('--fp16', 'use_fp16', is_flag=True)
    @click.option('--device', type=str, default='cuda:0')
    @click.option('--version', type=str, default='v1')
    @staticmethod
    def load(num_tokens: int, resolution_level: int, pretrained_model_name_or_path: str, use_fp16: bool, device: str = 'cuda:0', version: str = 'v1'):
        return Baseline(num_tokens, resolution_level, pretrained_model_name_or_path, use_fp16, device, version)

    def infer(self, image: torch.FloatTensor, intrinsics: Optional[torch.FloatTensor] = None):
        if intrinsics is not None:
            fov_x, _ = utils3d.torch.intrinsics_to_fov(intrinsics)
            fov_x = torch.rad2deg(fov_x)
        else:
            fov_x = None
        output = self._original_model.infer(image, fov_x=fov_x,
                                            apply_mask=True,
                                            num_tokens=self.num_tokens)
        
        if self.version == 'v1':
            return {
                'points_scale_invariant': output['points'],
                'depth_scale_invariant': output['depth'],
                'intrinsics': output['intrinsics'],
            }
        else:
            return {
                'points_metric': output['points'],
                'depth_metric': output['depth'],
                'intrinsics': output['intrinsics'],
            }
        
    def initial_inference(self, image, intrinsics=None, fov_x=None):
        with torch.inference_mode():
            output = self._original_model(image.unsqueeze(0),
                                          num_tokens=self.num_tokens)
            
            _, img_h, img_w = image.shape
            aspect_ratio = img_w / img_h
            if fov_x is None:
                # Recover focal and shift from predicted point map
                focal, shift = recover_focal_shift(output["points"], output['mask'])
            else:
                # Focal is known, recover shift only
                focal = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5 / torch.tan(torch.deg2rad(torch.as_tensor(fov_x, device=points.device, dtype=points.dtype) / 2))
                if focal.ndim == 0:
                    focal = focal[None].expand(output["points"].shape[0])
                _, shift = recover_focal_shift(output["points"], output["mask"], focal=focal)
            fx, fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio, focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 
            intrinsics = utils3d.torch.intrinsics_from_focal_center(fx, fy, 0.5, 0.5)
            output['intrinsics'] = intrinsics

            output = {k: v.squeeze(0) for k, v in output.items()}

            if "mask" in output:
                output["mask"] = output["mask"] > 0.5
                output["mask"] &= output["points"][..., 2] > 0
        
        output['points'] = output['points'] * output['metric_scale']

        # import ipdb; ipdb.set_trace()
        return output

    def infer_for_evaluation(self, image: torch.FloatTensor,
                             config: dict,
                             intrinsics: torch.FloatTensor = None,
                             ):
        if intrinsics is not None:
            fov_x, _ = utils3d.torch.intrinsics_to_fov(intrinsics)
            fov_x = torch.rad2deg(fov_x)
        else:
            fov_x = None

        output = self.initial_inference(image, intrinsics, fov_x)

        from moge.model import import_model_class_by_version
        MoGeModel = import_model_class_by_version(self.version)
        ttt_model = MoGeModel.from_pretrained(self._pretrained_model_name_or_path).to(self.device)

        if self.version == 'v1':
            ttt_model.backbone.requires_grad_(False)
        elif self.version == 'v2':
            ttt_model.encoder.requires_grad_(False)
        ttt_model.train()

        # ------------------- START: Test-Time Training Loop ------------------- #
        
        # Setup optimizer to update only the trainable parameters (unfrozen layers)
        optimizer_config = config['optim']
        optimizer = torch.optim.SGD(
            [p for p in ttt_model.parameters() if p.requires_grad],
            **optimizer_config
        )
        
        # Setup mixed-precision training scaler
        scaler = torch.amp.GradScaler(enabled=self.use_fp16)
        
        num_ttt_steps = config["num_ttt_steps"]
        grad_accum_steps = config["grad_accum_steps"]

        losses_logs = list()

        for step in range(num_ttt_steps):
            optimizer.zero_grad()
            
            # Generate a batch of jittered/augmented views from the single input image
            batch = generate_jittered_batch(image, output, config['batch_size'],
                                            **config['augs'])

            # Use autocast for mixed-precision forward pass
            with torch.amp.autocast(enabled=self.use_fp16, device_type="cuda"):
                # Forward pass through the model being adapted
                ttt_output = ttt_model(batch['images'], num_tokens=self.num_tokens)
                
                ttt_output["points"] = ttt_output["points"] * ttt_output["metric_scale"][:, None, None, None]

                # Combine original data and model output for loss calculation
                data = batch | ttt_output
                
                # Compute the self-supervised loss (total, across the batch)
                
                # losses = compute_moge2_ttt_loss(data, use_transforms_inv=True,
                #                                 config=config, device='cuda')
                # losses = moge2_ttt_loss(data)

                losses = compute_moge2_ttt_loss_from_orig(data, config,
                                                          device='cuda', use_transforms_inv=True)
                loss = losses['loss']
                losses_logs.append({loss_name: loss_value.item() for loss_name, loss_value in losses.items()})

                print('---------------------')
                print(loss)
                print('---------------------')

            # Backward pass: scale loss and compute gradients
            scaler.scale(loss).backward()

            # Update model weights after accumulating gradients for grad_accum_steps
            if (step + 1) % grad_accum_steps == 0:
                scaler.step(optimizer) # Unscale gradients and update weights
                scaler.update()        # Update the scaler for the next iteration
                optimizer.zero_grad()  # Reset gradients for the next accumulation cycle
        
        # -------------------- END: Test-Time Training Loop -------------------- #
        with torch.inference_mode():
            output_ttt = ttt_model.infer(image, fov_x=fov_x, apply_mask=False,
                                         num_tokens=self.num_tokens, use_fp16=self.use_fp16)


        if self.version == 'v1':
            return {
                'points_scale_invariant': output_ttt['points'],
                'depth_scale_invariant': output_ttt['depth'],
                'intrinsics': output_ttt['intrinsics'],
                "losses_logs": losses_logs,
            }
        else:
            return {
                'points_metric': output_ttt['points'],
                'depth_metric': output_ttt['depth'],
                'intrinsics': output_ttt['intrinsics'],
                "losses_logs": losses_logs
            }
