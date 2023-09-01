import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from smplx import SMPL
from tqdm import tqdm

from predict.predict_hrnet import predict_hrnet

from utils.renderers.pytorch3d_textured_renderer import TexturedIUVRenderer

from utils.image_utils import batch_crop_pytorch_affine
from utils.label_conversions import convert_2Djoints_to_gaussian_heatmaps_torch
from utils.rigid_transform_utils import aa_rotate_translate_points_pytorch3d
from utils.sampling_utils import compute_vertex_variance_from_samples, joints2D_error_sorted_verts_sampling
from utils.predict_utils import save_pred_output
from utils.visualise_utils import (render_point_est_visualisation,
                                   render_samples_visualisation,
                                   uncrop_point_est_visualisation,
                                   plot_xyz_vertex_variance)

def predict_humaniflow(humaniflow_model,
                       humaniflow_cfg,
                       hrnet_model,
                       hrnet_cfg,
                       edge_detect_model,
                       device,
                       image,
                       object_detect_model=None,
                       num_pred_samples=50,
                       joints2Dvisib_threshold=0.75):

    with torch.no_grad():
        # ------------------------- INPUT LOADING AND PROXY REPRESENTATION GENERATION -------------------------

        # Predict Person Bounding Box + 2D Joints
        hrnet_output = predict_hrnet(hrnet_model=hrnet_model,
                                     hrnet_config=hrnet_cfg,
                                     object_detect_model=object_detect_model,
                                     image=image,
                                     object_detect_threshold=humaniflow_cfg.DATA.BBOX_THRESHOLD,
                                     bbox_scale_factor=humaniflow_cfg.DATA.BBOX_SCALE_FACTOR)

        # Transform predicted 2D joints and image from HRNet input size to input proxy representation size
        hrnet_input_centre = torch.tensor([[hrnet_output['cropped_image'].shape[1],
                                            hrnet_output['cropped_image'].shape[2]]],
                                          dtype=torch.float32,
                                          device=device) * 0.5
        hrnet_input_height = torch.tensor([hrnet_output['cropped_image'].shape[1]],
                                          dtype=torch.float32,
                                          device=device)
        cropped_for_proxy = batch_crop_pytorch_affine(
            input_wh=(hrnet_cfg.MODEL.IMAGE_SIZE[0], hrnet_cfg.MODEL.IMAGE_SIZE[1]),
            output_wh=(humaniflow_cfg.DATA.PROXY_REP_SIZE, humaniflow_cfg.DATA.PROXY_REP_SIZE),
            num_to_crop=1,
            device=device,
            joints2D=hrnet_output['joints2D'][None, :, :],
            rgb=hrnet_output['cropped_image'][None, :, :, :],
            bbox_centres=hrnet_input_centre,
            bbox_heights=hrnet_input_height,
            bbox_widths=hrnet_input_height,
            orig_scale_factor=1.0)

        # Create proxy representation with 1) Edge detection and 2) 2D joints heatmaps generation
        edge_detector_output = edge_detect_model(cropped_for_proxy['rgb'])
        proxy_rep_img = edge_detector_output['thresholded_thin_edges'] if humaniflow_cfg.DATA.EDGE_NMS else \
        edge_detector_output['thresholded_grad_magnitude']
        proxy_rep_heatmaps = convert_2Djoints_to_gaussian_heatmaps_torch(joints2D=cropped_for_proxy['joints2D'],
                                                                         img_wh=humaniflow_cfg.DATA.PROXY_REP_SIZE,
                                                                         std=humaniflow_cfg.DATA.HEATMAP_GAUSSIAN_STD)
        hrnet_joints2Dvisib = hrnet_output['joints2Dconfs'] > joints2Dvisib_threshold
        hrnet_joints2Dvisib[[0, 1, 2, 3, 4, 5, 6]] = True  # Only removing joints [7, 8, 9, 10, 11, 12, 13, 14, 15, 16] if occluded
        proxy_rep_heatmaps = proxy_rep_heatmaps * hrnet_joints2Dvisib[None, :, None, None]
        proxy_rep_input = torch.cat([proxy_rep_img, proxy_rep_heatmaps], dim=1).float()  # (1, 18, img_wh, img_wh)

        # ------------------------------- POSE AND SHAPE DISTRIBUTION PREDICTION -------------------------------
        pred = humaniflow_model(proxy_rep_input,
                                num_samples=num_pred_samples,
                                use_shape_mode_for_samples=False,
                                return_input_feats=True)

        out = pred["shape_mode"].tolist()
    return out
