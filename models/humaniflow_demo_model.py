from collections import defaultdict
import torch
from torch import nn as nn
from torch.distributions import Normal

from smplx.lbs import batch_rodrigues

from models.resnet import resnet18, resnet50
from models.norm_flows import ConditionalLocalDiffeoTransformedDistribution
from models.norm_flows.pyro_conditional_norm_flow import create_conditional_norm_flow, forward_trans_conditional_norm_flow
from models.norm_flows.transforms import ToTransform, SO3ExpCompactTransform

from utils.rigid_transform_utils import rotmat_to_rot6d, rot6d_to_rotmat


def immediate_parent_to_all_ancestors(immediate_parents):
    """

    :param immediate_parents: list with len = num joints, contains index of each joint's parent.
            - includes root joint, but its parent index is -1.
    :return: ancestors_dict: dict of lists, dict[joint] is ordered list of parent joints.
            - DOES NOT INCLUDE ROOT JOINT! Joint 0 here is actually joint 1 in SMPL.
    """
    ancestors_dict = defaultdict(list)
    for i in range(1, len(immediate_parents)):  # Excluding root joint
        joint = i - 1
        immediate_parent = immediate_parents[i] - 1
        if immediate_parent >= 0:
            ancestors_dict[joint] += [immediate_parent] + ancestors_dict[immediate_parent]
    return ancestors_dict


class HumaniflowModel(nn.Module):
    def __init__(self,
                 device,
                 model_cfg,
                 smpl_parents):
        """
        """
        super(HumaniflowModel, self).__init__()

        # Num pose parameters + Kinematic tree pre-processing
        self.parents = smpl_parents
        self.ancestors_dict = immediate_parent_to_all_ancestors(smpl_parents)
        self.num_bodyparts = len(self.ancestors_dict)

        # Number of shape, glob and cam parameters
        self.num_shape_params = model_cfg.NUM_SMPL_BETAS

        self.num_glob_params = 6  # 6D rotation representation for glob
        init_glob = rotmat_to_rot6d(torch.eye(3)[None, :].float())
        self.register_buffer('init_glob', init_glob)

        self.num_cam_params = 3
        init_cam = torch.tensor([0.9, 0.0, 0.0]).float()  # Initialise orthographic camera scale at 0.9
        self.register_buffer('init_cam', init_cam)

        # ResNet Image Encoder
        if model_cfg.NUM_RESNET_LAYERS == 18:
            self.image_encoder = resnet18(in_channels=model_cfg.NUM_IN_CHANNELS,
                                          pretrained=False)
            input_feats_dim = 512
            fc1_dim = 512
        elif model_cfg.NUM_RESNET_LAYERS == 50:
            self.image_encoder = resnet50(in_channels=model_cfg.NUM_IN_CHANNELS,
                                          pretrained=False)
            input_feats_dim = 2048
            fc1_dim = 1024

        # FC Shape/Glob/Cam networks
        self.activation = nn.ELU()
        self.fc1 = nn.Linear(input_feats_dim, fc1_dim)
        self.fc_shape = nn.Linear(fc1_dim, self.num_shape_params * 2)  # Means and variances for SMPL betas and/or measurements
        self.fc_glob = nn.Linear(fc1_dim, self.num_glob_params)
        self.fc_cam = nn.Linear(fc1_dim, self.num_cam_params)

        # Pose Normalising Flow networks for each bodypart
        self.fc_input_shape_glob_cam_feats = nn.Linear(input_feats_dim + self.num_shape_params + 9 + self.num_cam_params,
                                                       model_cfg.INPUT_SHAPE_GLOB_CAM_FEATS_DIM)
        self.fc_flow_context = nn.ModuleList()
        self.pose_so3flow_transform_modules = nn.ModuleList()
        self.pose_so3flow_transforms = []
        self.pose_so3flow_dists = []
        self.pose_SO3flow_dists = []

        for bodypart in range(self.num_bodyparts):
            num_ancestors = len(self.ancestors_dict[bodypart])
            # Input to fc_flow_context is input/shape/glob/cam features and 3x3 rotmats for all ancestors
            self.fc_flow_context.append(nn.Linear(model_cfg.INPUT_SHAPE_GLOB_CAM_FEATS_DIM + num_ancestors * 9,
                                                  model_cfg.NORM_FLOW.CONTEXT_DIM))
            # Set up normalising flow distribution on Lie algebra so(3) for each bodypart.
            so3flow_dist, so3flow_transform_modules, so3flow_transforms = create_conditional_norm_flow(
                device=device,
                event_dim=3,
                context_dim=model_cfg.NORM_FLOW.CONTEXT_DIM,
                num_transforms=model_cfg.NORM_FLOW.NUM_TRANSFORMS,
                transform_type=model_cfg.NORM_FLOW.TRANSFORM_TYPE,
                transform_hidden_dims=model_cfg.NORM_FLOW.TRANSFORM_NN_HIDDEN_DIMS,
                permute_type=model_cfg.NORM_FLOW.PERMUTE_TYPE,
                permute_hidden_dims=model_cfg.NORM_FLOW.PERMUTE_NN_HIDDEN_DIMS,
                bound=model_cfg.NORM_FLOW.COMPACT_SUPPORT_RADIUS,
                count_bins=model_cfg.NORM_FLOW.NUM_SPLINE_SEGMENTS,
                radial_tanh_radius=model_cfg.NORM_FLOW.COMPACT_SUPPORT_RADIUS,
                base_dist_std=model_cfg.NORM_FLOW.BASE_DIST_STD)

            # Pushforward distribution on Lie group SO(3) for each bodypart.
            SO3flow_dist = ConditionalLocalDiffeoTransformedDistribution(base_dist=so3flow_dist,
                                                                         transforms=[ToTransform(dict(dtype=torch.float32), dict(dtype=torch.float64)),
                                                                                     SO3ExpCompactTransform(support_radius=model_cfg.NORM_FLOW.COMPACT_SUPPORT_RADIUS)])

            self.pose_so3flow_transform_modules.extend(so3flow_transform_modules)
            self.pose_so3flow_transforms.append(so3flow_transforms)
            self.pose_so3flow_dists.append(so3flow_dist)
            self.pose_SO3flow_dists.append(SO3flow_dist)

    def forward(self,
                input,
                num_samples=0,
                use_shape_mode_for_samples=False,
                input_feats=None,
                return_input_feats=False,
                return_input_feats_only=False):
        """
        :param input: (batch_size, num_channels, D, D) tensor.
        :param num_samples: int, number of hierarchical samples to draw from predicted shape and pose distribution.
        :param use_shape_mode_for_samples: bool, only use the shape distribution mode for hierarchical sampling.
        :param compute_for_loglik: bool, trying to compute log-likelihood of given target shape and pose rotmats.
        :param shape_for_loglik: (B, num shape params) tensor of target shapes.
        :param pose_R_for_loglik: (B, num joints, 3, 3) tensor of target pose rotation matrices.
        :param glob_R_for_loglik: (B, 3, 3) tensor of target global body rotation matrices.

        Need to input shape_for_loglik, pose_R_for_loglik and glob_R_for_loglik to compute the log-likelihood of
        target shape and pose parameters w.r.t predicted distributions.
        Since this is an auto-regressive model, predicted distribution parameters depend on samples from up the kinematic tree.
        Need to set target shape and pose as "samples" to compute distribution parameters down the kinematic tree.
        """
        if input_feats is None:
            input_feats = self.image_encoder(input)  # (bsize, num_image_features)

        if return_input_feats_only:
            return {'input_feats': input_feats}

        x = self.activation(self.fc1(input_feats))
        #######################################################################################################
        # ----------------------------------------------- Shape ----------------------------------------------
        #######################################################################################################
        shape_params = self.fc_shape(x)  # (bsize, num_shape_params * 2)
        #print("Shape Params : ", shape_params)
        shape_mode = shape_params[:, :self.num_shape_params]
        #print("Shape mode : ", shape_mode)
        shape_log_std = shape_params[:, self.num_shape_params:]
        #print("Shape log std : ", shape_log_std)
        shape_dist = Normal(loc=shape_mode, scale=torch.exp(shape_log_std), validate_args=False)
        #print("Shape Dist : ", shape_dist)
        if num_samples > 0:
            if use_shape_mode_for_samples:
                shape_samples = shape_mode[:, None, :].expand(-1, num_samples, -1)  # (bsize, num_samples, num_shape_params)
            else:
                shape_samples = shape_dist.rsample([num_samples]).transpose(0, 1)  # (bsize, num_samples, num_shape_params)

        return_dict = {'shape_mode': shape_mode,
                       'shape_log_std': shape_log_std,
                       'shape_dist_for_loglik': shape_dist}
        if num_samples > 0:
            return_dict['shape_samples'] = shape_samples

        return return_dict
