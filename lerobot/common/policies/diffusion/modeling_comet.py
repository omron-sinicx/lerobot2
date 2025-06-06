#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

TODO(alexander-soare):
  - Remove reliance on diffusers for DDPMScheduler and LR scheduler.
"""

import math
from collections import deque
from typing import Callable, Optional, Union

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor, nn

from lerobot.common.policies.diffusion.dm.utils import (
    at_least_ndim, to_tensor,
    SUPPORTED_NOISE_SCHEDULES, SUPPORTED_DISCRETIZATIONS, SUPPORTED_SAMPLING_STEP_SCHEDULE)

from lerobot.common.policies.diffusion.model.unet import DiffusionConditionalUnet1d
from lerobot.common.policies.diffusion.model.rgb_encoder import DiffusionRgbEncoder
from lerobot.common.policies.diffusion.model.plain_transformer import PlainTransformer
from lerobot.common.constants import OBS_ENV, OBS_ROBOT
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.configuration_comet import CometConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)

SUPPORTED_SOLVERS = [
    "ddpm", "ddim",
    "ode_dpmsolver_1", "ode_dpmsolver++_1", "ode_dpmsolver++_2M",
    "sde_dpmsolver_1", "sde_dpmsolver++_1", "sde_dpmsolver++_2M",]


def epstheta_to_xtheta(x, alpha, sigma, eps_theta):
    """
    x_theta = (x - sigma * eps_theta) / alpha
    """
    return (x - sigma * eps_theta) / alpha


def xtheta_to_epstheta(x, alpha, sigma, x_theta):
    """
    eps_theta = (x - alpha * x_theta) / sigma
    """
    return (x - alpha * x_theta) / sigma


class CometPolicy(PreTrainedPolicy):

    config_class = CometConfig
    name = "sde"

    def __init__(
        self,
        config: CometConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None

        self.diffusion = SdeModel(config)
        
        '''
        whats missing
        list of output sizes 
        max horizon prediction ? 
        tempoeral ensemble coeff :(
        '''
        # TODO(malek) remove dependancy on these keys
        # self.input_keys = config.keys_order['state'] 
        # self.output_keys = config.keys_order['action']
        # Check that all the keys are defined for normalization
        # assert set(config.input_shapes).issuperset({*self.input_keys})
        # assert set(config.output_shapes).issuperset({*self.output_keys})
        ###########################
        
        # TODO(malek) make this depend on the feature
        self.output_sizes = config.get_output_sizes

        self.reset()

    def get_optim_params(self) -> dict:
        return self.diffusion.parameters() #different from the og 

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method handles caching a history of observations and an action trajectory generated by the
        underlying diffusion model. Here's how it works:
          - `n_obs_steps` steps worth of observations are cached (for the first steps, the observation is
            copied `n_obs_steps` times to fill the cache).
          - The diffusion model generates `horizon` steps worth of actions.
          - `n_action_steps` worth of actions are actually kept for execution, starting from the current step.
        Schematically this looks like:
            ----------------------------------------------------------------------------------------------
            (legend: o = n_obs_steps, h = horizon, a = n_action_steps)
            |timestep            | n-o+1 | n-o+2 | ..... | n     | ..... | n+a-1 | n+a   | ..... | n-o+h |
            |observation is used | YES   | YES   | YES   | YES   | NO    | NO    | NO    | NO    | NO    |
            |action is generated | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   |
            |action is used      | NO    | NO    | NO    | YES   | YES   | YES   | NO    | NO    | NO    |
            ----------------------------------------------------------------------------------------------
        Note that this means we require: `n_action_steps <= horizon - n_obs_steps + 1`. Also, note that
        "horizon" may not the best name to describe what the variable actually means, because this period is
        actually measured from the first observation which (if `n_obs_steps` > 1) happened in the past.
        """
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        
        ###check the logic here
        if self.config.robot_state_features:
            batch = dict(batch)
            batch["observation.state"] = torch.cat(
                [batch[key] for key in self.config.robot_state_features], dim=-1
            )
        
        # Note: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues["action"]) == 0:
            # stack n latest observations from the queue
            batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
            actions = self.diffusion.generate_actions(batch)
            
            # here temporal ensemble was used 

            # TODO(rcadene): make above methods return output dictionary?
            # TODO(malek) check its doing the seperation correctly
            if self.config.action_features:
                action_list = torch.split(actions, split_size_or_sections=self.output_sizes, dim=-1)
                actions = self.unnormalize_outputs(dict(zip(self.config.action_features, action_list)))
                actions = torch.cat([actions[key] for key in self.config.action_features], dim=-1)
            else:
                actions = self.unnormalize_outputs({"action": actions})["action"]
            
            self._queues["action"].extend(actions.transpose(0, 1))

        action = self._queues["action"].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        
        ###check the logic here
        if self.config.robot_state_features:
            batch = dict(batch)
            batch["observation.state"] = torch.cat(
                [batch[key] for key in self.config.robot_state_features], dim=-1
            )   
        
        batch = self.normalize_targets(batch)
        
        if self.config.action_features:
            batch["action"] = torch.cat([batch[key] for key in self.config.action_features], dim=-1)
        
        
        loss = self.diffusion.compute_loss(batch)
        # no output_dict so returning None
        return loss, None


class SdeModel(nn.Module):
    def __init__(self, config: CometConfig):
        super().__init__()
        self.config = config

        # Build observation encoders (depending on which observations are provided).
        global_cond_dim = sum(v.shape[0] for k, v in self.config.robot_state_features.items())
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        # add the option to train using transformer
        if config.model == "FILM":
            self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)
        elif config.model == "PLAIN_TRANSFORMER": 
            self.unet = PlainTransformer(
                input_dim=sum(config.get_output_sizes),
                output_dim=sum(config.get_output_sizes),  # replace with output sum
                horizon=config.horizon,
                n_obs_steps=config.n_obs_steps,
                cond_dim=global_cond_dim,  # image + obs take from resnet global dim
                n_layer=config.n_layer,
                n_head=config.n_head,
                n_emb=config.n_emb,
                p_drop_emb=0.0,
                p_drop_attn=0.01,
                causal_attn=config.casual_attn,
                time_as_cond=config.obs_as_cond,
                obs_as_cond=config.obs_as_cond,
                n_cond_layers=config.n_cond_layers
            )
        else:
            raise NotImplementedError("the model you specify is not implemented")
            
        device = get_device_from_parameters(self)
        
        self.classifier = None # TODO (malek) look into the original code and change it to make it appropriate
        # self.sample_step_schedule = self.config.sample_step_schedule
        
        # ----------------- Masks ----------------- #
        # Fix some portion of the input data, and only allow the diffusion model to complete the rest part.
        fix_mask: Union[list, np.ndarray, torch.Tensor] = None  # be in the shape of `x_shape`
        # Add loss weight
        loss_weight: Union[list, np.ndarray, torch.Tensor] = None  # be in the shape of `x_shape`
        
        self.fix_mask = to_tensor(fix_mask, device)[None,] if fix_mask is not None else 0.
        self.loss_weight = to_tensor(loss_weight, device)[None,] if loss_weight is not None else 1.
            
        # ==================== Continuous Time-step Range ====================
        if config.noise_schedule == "cosine":
            self.t_diffusion = [config.epsilon, 0.9946]
        else:
            self.t_diffusion = [config.epsilon, 1.] 
            
        # ===================== Noise Schedule ======================
        if isinstance(config.noise_schedule, str):
            if config.noise_schedule in SUPPORTED_NOISE_SCHEDULES.keys():
                self.noise_schedule_funcs = SUPPORTED_NOISE_SCHEDULES[config.noise_schedule]
                self.noise_schedule_params = config.noise_schedule_params
            else:
                raise ValueError(f"Noise schedule {config.noise_schedule} is not supported.")
        elif isinstance(config.noise_schedule, dict):
            self.noise_schedule_funcs = config.noise_schedule
            self.noise_schedule_params = config.noise_schedule_params
        else:
            raise ValueError("noise_schedule must be a callable or a string")
        
    @property
    def supported_solvers(self):
        return SUPPORTED_SOLVERS

    @property
    def clip_pred(self):
        return (self.config.x_max is not None) or (self.config.x_min is not None)
    # ==================== Training: Score Matching ======================
    def add_noise(self, x0, t=None, eps=None):
        
        device = get_device_from_parameters(self)

        t = (torch.rand((x0.shape[0],), device=device) *
             (self.t_diffusion[1] - self.t_diffusion[0]) + self.t_diffusion[0]) if t is None else t

        eps = torch.randn_like(x0) if eps is None else eps

        alpha, sigma = self.noise_schedule_funcs["forward"](t, **(self.noise_schedule_params or {}))
        alpha = at_least_ndim(alpha, x0.dim())
        sigma = at_least_ndim(sigma, x0.dim())

        xt = alpha * x0 + sigma * eps
        xt = (1. - self.fix_mask) * xt + self.fix_mask * x0

        return xt, t, eps
    
    # ==================== Sampling: Solving SDE/ODE ======================
    def classifier_guidance(
            self, xt, t, alpha, sigma,
            model, condition=None, w: float = 1.0,
            pred=None):
        """
        Guided Sampling CG:
        bar_eps = eps - w * sigma * grad
        bar_x0  = x0 + w * (sigma ** 2) * alpha * grad
        """
        if pred is None:
            pred = model["diffusion"](xt, t, None)
        if self.classifier is None or w == 0.0:
            return pred, None
        else:
            log_p, grad = self.classifier.gradients(xt.clone(), t, condition)
            if self.config.predict_noise:
                pred = pred - w * sigma * grad
            else:
                pred = pred + w * ((sigma ** 2) / alpha) * grad

        return pred, log_p

    def classifier_free_guidance(
            self, xt, t,
            model, condition=None, w: float = 1.0,
            pred=None, pred_uncond=None,
            requires_grad: bool = False):
        """
        Guided Sampling CFG:
        bar_eps = w * pred + (1 - w) * pred_uncond
        bar_x0  = w * pred + (1 - w) * pred_uncond
        """
        with torch.set_grad_enabled(requires_grad):
            if w != 0.0 and w != 1.0:
                if pred is None or pred_uncond is None:
                    b = xt.shape[0]
                    repeat_dim = [2 if i == 0 else 1 for i in range(xt.dim())]
                    condition = torch.cat([condition, torch.zeros_like(condition)], 0)
                    pred_all = model["diffusion"](
                        xt.repeat(*repeat_dim), t.repeat(2), condition)
                    pred, pred_uncond = pred_all[:b], pred_all[b:]
            elif w == 0.0:
                pred = 0.
                pred_uncond = model["diffusion"](xt, t, None)
            else:
                # pred = model["diffusion"](xt, t, condition)
                pred = model(xt, t, condition)
                pred_uncond = 0.

        if self.config.predict_noise or not self.config.predict_noise:
            bar_pred = w * pred + (1 - w) * pred_uncond
        else:
            bar_pred = pred

        return bar_pred

    def clip_prediction(self, pred, xt, alpha, sigma):
        """
        Clip the prediction at each sampling step to stablize the generation.
        (xt - alpha * x_max) / sigma <= eps <= (xt - alpha * x_min) / sigma
                               x_min <= x0  <= x_max
        """
        if self.config.predict_noise:
            if self.clip_pred:
                upper_bound = (xt - alpha * self.config.x_min) / sigma if self.config.x_min is not None else None
                lower_bound = (xt - alpha * self.config.x_max) / sigma if self.config.x_max is not None else None
                pred = pred.clip(lower_bound, upper_bound)
        else:
            if self.clip_pred:
                pred = pred.clip(self.config.x_min, self.config.x_max)

        return pred

    def guided_sampling(
            self, xt, t, alpha, sigma,
            model,
            condition_cfg=None, w_cfg: float = 0.0,
            condition_cg=None, w_cg: float = 0.0,
            requires_grad: bool = False):
        """
        One-step epsilon/x0 prediction with guidance.
        """

        pred = self.classifier_free_guidance(
            xt, t, model, condition_cfg, w_cfg, None, None, requires_grad)

        pred, logp = self.classifier_guidance(
            xt, t, alpha, sigma, model, condition_cg, w_cg, pred)

        return pred, logp
    
    # ==================== Sampling: Solving SDE/ODE ======================
    def conditional_sample(
            self,
            # ---------- the known fixed portion ---------- #
            batch: int,
            # prior: torch.Tensor,
            # ----------------- sampling ----------------- #
            # solver: str = "ddpm",
            # n_samples: int = 1,
            # sample_steps: int = 5,
            # sample_step_schedule: Union[str, Callable] = "uniform_continuous",
            # use_ema: bool = True,
            # temperature: float = 1.0,
            # ------------------ guidance ------------------ #
            condition_cfg: Tensor | None = None,
            # mask_cfg=None,
            # w_cfg: float = 0.0,
            # condition_cg=None,
            # w_cg: float = 0.0,
            # ----------- Diffusion-X sampling ----------
            # diffusion_x_sampling_steps: int = 0,
            # ----------- Warm-Starting -----------
            # warm_start_reference: Optional[torch.Tensor] = None,
            # warm_start_forward_level: float = 0.3,
            # ------------------ others ------------------ #
            # requires_grad: bool = False,
            # preserve_history: bool = False,
            # **kwargs,
    ):
        """Sampling.
        
        Inputs:
        - prior: torch.Tensor
            The known fixed portion of the input data. Should be in the shape of generated data.
            Use `torch.zeros((n_samples, *x_shape))` for non-prior sampling.
        
        - solver: str
            The solver for the reverse process. Check `supported_solvers` property for available solvers.
        - n_samples: int
            The number of samples to generate.
        - sample_steps: int
            The number of sampling steps. Should be greater than 1.
        - sample_step_schedule: Union[str, Callable]
            The schedule for the sampling steps.
        - use_ema: bool
            Whether to use the exponential moving average model.
        - temperature: float
            The temperature for sampling.
        
        - condition_cfg: Optional
            Condition for Classifier-free-guidance.
        - mask_cfg: Optional
            Mask for Classifier-guidance.
        - w_cfg: float
            Weight for Classifier-free-guidance.
        - condition_cg: Optional
            Condition for Classifier-guidance.
        - w_cg: float
            Weight for Classifier-guidance.
            
        - diffusion_x_sampling_steps: int
            The number of diffusion steps for diffusion-x sampling.
        
        - warm_start_reference: Optional[torch.Tensor]
            Reference data for warm-starting sampling. `None` indicates no warm-starting.
        - warm_start_forward_level: float
            The forward noise level to perturb the reference data. Should be in the range of `[0., 1.]`, where `1` indicates pure noise.
        
        - requires_grad: bool
            Whether to preserve gradients.
        - preserve_history: bool
            Whether to preserve the sampling history.
            
        Outputs:
        - x0: torch.Tensor
            Generated samples. Be in the shape of `(n_samples, *x_shape)`.
        - log: dict
            The log dictionary.
        """
        assert self.config.solver in SUPPORTED_SOLVERS, f"Solver {self.config.solver} is not supported."
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)
        self.sample_step_schedule = self.config.sample_step_schedule

        # ===================== Initialization =====================
        log = {
            "sample_history": np.empty((self.config.n_samples, self.config.sample_steps + 1, *prior.shape)) if self.config.preserve_history else None, }

        model = self.unet

        prior = torch.zeros((batch, self.config.horizon, sum(self.config.get_output_sizes)))
        prior = prior.to(device)
        if isinstance(self.config.warm_start_reference, torch.Tensor) and self.config.warm_start_forward_level > 0.:
            self.config.warm_start_forward_level = self.config.epsilon + self.config.warm_start_forward_level * (1. - self.config.epsilon)
            fwd_alpha, fwd_sigma = self.noise_schedule_funcs["forward"](
                torch.ones((1,), device=device) * self.config.warm_start_forward_level, **(self.noise_schedule_params or {}))
            xt = self.config.warm_start_reference * fwd_alpha + fwd_sigma * torch.randn_like(self.config.warm_start_reference)
        else:
            xt = torch.randn_like(prior) * self.config.temperature
        # xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
        if self.config.preserve_history:
            log["sample_history"][:, 0] = xt.cpu().numpy()

        with torch.set_grad_enabled(self.config.requires_grad):
            # condition_vec_cfg = model["condition"](condition_cfg, self.config.mask_cfg) if condition_cfg is not None else None
            condition_vec_cfg = condition_cfg if condition_cfg is not None else None
            condition_vec_cg = self.config.condition_cg

        # ===================== Sampling Schedule ====================
        if isinstance(self.config.warm_start_reference, torch.Tensor) and self.config.warm_start_forward_level > 0.:
            t_diffusion = [self.t_diffusion[0], self.config.warm_start_forward_level]
        else:
            t_diffusion = self.t_diffusion
        if isinstance(self.sample_step_schedule, str):
            if self.sample_step_schedule in SUPPORTED_SAMPLING_STEP_SCHEDULE.keys():
                self.sample_step_schedule = SUPPORTED_SAMPLING_STEP_SCHEDULE[self.sample_step_schedule](
                    t_diffusion, self.config.sample_steps)
            else:
                raise ValueError(f"Sampling step schedule {self.sample_step_schedule} is not supported.")
        elif callable(self.sample_step_schedule):
            self.sample_step_schedule = self.sample_step_schedule(t_diffusion, self.config.sample_steps)
        else:
            raise ValueError("sample_step_schedule must be a callable or a string")

        alphas, sigmas = self.noise_schedule_funcs["forward"](
            self.sample_step_schedule, **(self.noise_schedule_params or {}))
        logSNRs = torch.log(alphas / sigmas)
        hs = torch.zeros_like(logSNRs)
        hs[1:] = logSNRs[:-1] - logSNRs[1:]  # hs[0] is not correctly calculated, but it will not be used.
        stds = torch.zeros((self.config.sample_steps + 1,), device=device)
        stds[1:] = sigmas[:-1] / sigmas[1:] * (1 - (alphas[1:] / alphas[:-1]) ** 2).sqrt()

        buffer = []

        # ===================== Denoising Loop ========================
        loop_steps = [1] * self.config.diffusion_x_sampling_steps + list(range(1, self.config.sample_steps + 1))
        for i in reversed(loop_steps):

            t = torch.full((self.config.n_samples,), self.sample_step_schedule[i], dtype=torch.float32, device=device)

            # guided sampling
            pred, logp = self.guided_sampling(
                xt, t, alphas[i], sigmas[i],
                model, condition_vec_cfg, self.config.w_cfg, condition_vec_cg, self.config.w_cg, self.config.requires_grad)

            # clip the prediction
            pred = self.clip_prediction(pred, xt, alphas[i], sigmas[i])

            # transform to eps_theta
            eps_theta = pred if self.config.predict_noise else xtheta_to_epstheta(xt, alphas[i], sigmas[i], pred)
            x_theta = pred if not self.config.predict_noise else epstheta_to_xtheta(xt, alphas[i], sigmas[i], pred)

            # one-step update
            if self.config.solver == "ddpm":
                xt = (
                        (alphas[i - 1] / alphas[i]) * (xt - sigmas[i] * eps_theta) +
                        (sigmas[i - 1] ** 2 - stds[i] ** 2 + 1e-8).sqrt() * eps_theta)
                if i > 1:
                    xt += (stds[i] * torch.randn_like(xt))

            elif self.config.solver == "ddim":
                xt = (alphas[i - 1] * ((xt - sigmas[i] * eps_theta) / alphas[i]) + sigmas[i - 1] * eps_theta)

            elif self.config.solver == "ode_dpmsolver_1":
                xt = (alphas[i - 1] / alphas[i]) * xt - sigmas[i - 1] * torch.expm1(hs[i]) * eps_theta

            elif self.config.solver == "ode_dpmsolver++_1":
                xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * x_theta

            elif self.config.solver == "ode_dpmsolver++_2M":
                buffer.append(x_theta)
                if i < self.config.sample_steps:
                    r = hs[i + 1] / hs[i]
                    D = (1 + 0.5 / r) * buffer[-1] - 0.5 / r * buffer[-2]
                    xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * D
                else:
                    xt = (sigmas[i - 1] / sigmas[i]) * xt - alphas[i - 1] * torch.expm1(-hs[i]) * x_theta

            elif self.config.solver == "sde_dpmsolver_1":
                xt = ((alphas[i - 1] / alphas[i]) * xt -
                      2 * sigmas[i - 1] * torch.expm1(hs[i]) * eps_theta +
                      sigmas[i - 1] * torch.expm1(2 * hs[i]).sqrt() * torch.randn_like(xt))

            elif self.config.solver == "sde_dpmsolver++_1":
                xt = ((sigmas[i - 1] / sigmas[i]) * (-hs[i]).exp() * xt -
                      alphas[i - 1] * torch.expm1(-2 * hs[i]) * x_theta +
                      sigmas[i - 1] * (-torch.expm1(-2 * hs[i])).sqrt() * torch.randn_like(xt))

            elif self.config.solver == "sde_dpmsolver++_2M":
                buffer.append(x_theta)
                if i < self.config.sample_steps:
                    r = hs[i + 1] / hs[i]
                    D = (1 + 0.5 / r) * buffer[-1] - 0.5 / r * buffer[-2]
                    xt = ((sigmas[i - 1] / sigmas[i]) * (-hs[i]).exp() * xt -
                          alphas[i - 1] * torch.expm1(-2 * hs[i]) * D +
                          sigmas[i - 1] * (-torch.expm1(-2 * hs[i])).sqrt() * torch.randn_like(xt))
                else:
                    xt = ((sigmas[i - 1] / sigmas[i]) * (-hs[i]).exp() * xt -
                          alphas[i - 1] * torch.expm1(-2 * hs[i]) * x_theta +
                          sigmas[i - 1] * (-torch.expm1(-2 * hs[i])).sqrt() * torch.randn_like(xt))

            # fix the known portion, and preserve the sampling history
            xt = xt * (1. - self.fix_mask) + prior * self.fix_mask
            if self.config.preserve_history:
                log["sample_history"][:, self.config.sample_steps - i + 1] = xt.cpu().numpy()

        # ================= Post-processing =================
        if self.config.classifier is not None and self.config.w_cg != 0.:
            with torch.no_grad():
                t = torch.zeros((self.config.n_samples,), dtype=torch.long, device=self.device)
                logp = self.classifier.logp(xt, t, condition_vec_cg)
            log["log_p"] = logp

        if self.clip_pred:
            xt = xt.clip(self.x_min, self.x_max)

        # return xt, log
        return xt

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode image features and concatenate them all together along with the state vector."""
        batch_size, n_obs_steps = batch[OBS_ROBOT].shape[:2]
        global_cond_feats = [batch[OBS_ROBOT]]
        # Extract image features.
        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                # Combine batch and sequence dims while rearranging to make the camera index dimension first.
                images_per_camera = einops.rearrange(batch["observation.images"], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                    ]
                )
                # Separate batch and sequence dims back out. The camera index dim gets absorbed into the
                # feature dim (effectively concatenating the camera features).
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                # Combine batch, sequence, and "which camera" dims before passing to shared encoder.
                img_features = self.rgb_encoder(
                    einops.rearrange(batch["observation.images"], "b s n ... -> (b s n) ...")
                )
                # Separate batch dim and sequence dim back out. The camera index dim gets absorbed into the
                # feature dim (effectively concatenating the camera features).
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV])

        # Concatenate features then flatten to (B, global_cond_dim). incase of transformer it would be (B, T, global_cond_dim)
        if self.config.model == "FILM":
            output = torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)
        elif self.config.model == "PLAIN_TRANSFORMER":
            output = torch.cat([batch["observation.state"], img_features], dim=-1)  # u add the state to the end of the image feature [B, To, sp*num_cam + statedim]
            
        return output

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)
        }
        """
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # run sampling
        actions = self.conditional_sample(batch=batch_size, condition_cfg=global_cond)

        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions
    
    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        assert set(batch).issuperset({"observation.state", "action", "action_is_pad"})
        assert "observation.images" in batch or "observation.environment_state" in batch
        n_obs_steps = batch["observation.state"].shape[1]
        horizon = batch["action"].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # Forward diffusion.
        trajectory = batch["action"]
        # get back a noised trajectory, the time step it was noised to and the eps used 
        xt, t, eps = self.add_noise(trajectory)
        
        # Run the denoising network (that might denoise the trajectory, or attempt to predict the noise).
        pred = self.unet(xt, t, global_cond=global_cond)
        
        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = batch["action"]
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")
        
        loss = F.mse_loss(pred, target, reduction="none")

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()