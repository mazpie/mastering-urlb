import copy

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from agent.dreamer import DreamerAgent, stop_gradient
import agent.dreamer_utils as common

class RND(nn.Module):
    def __init__(self,
                 obs_dim,
                 hidden_dim,
                 rnd_rep_dim,
                 encoder,
                 aug,
                 obs_shape,
                 obs_type,
                 clip_val=5.):
        super().__init__()
        self.clip_val = clip_val
        self.aug = aug

        if obs_type == "pixels":
            self.normalize_obs = nn.BatchNorm2d(obs_shape[0], affine=False)
        else:
            self.normalize_obs = nn.BatchNorm1d(obs_shape[0], affine=False)

        self.predictor = nn.Sequential(encoder, nn.Linear(obs_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim),
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim, rnd_rep_dim))
        self.target = nn.Sequential(copy.deepcopy(encoder),
                                    nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, rnd_rep_dim))

        for param in self.target.parameters():
            param.requires_grad = False

        self.apply(utils.weight_init)

    def forward(self, obs):
        if type(obs) == dict:
            img = obs['observation']
            img = self.aug(img)
            img = self.normalize_obs(img)
            img = torch.clamp(img, -self.clip_val, self.clip_val)
            obs['observation'] = img
        else:
            obs = self.aug(obs)
            obs = self.normalize_obs(obs)
            obs = torch.clamp(obs, -self.clip_val, self.clip_val)
        prediction, target = self.predictor(obs), self.target(obs)
        prediction_error = torch.square(target.detach() - prediction).mean(
            dim=-1, keepdim=True)
        return prediction_error

class RNDDreamerAgent(DreamerAgent):
    def __init__(self, rnd_rep_dim, rnd_scale, **kwargs):
        super().__init__(**kwargs)

        self.reward_free = True
        self.rnd_scale = rnd_scale

        self.obs_dim = self.wm.embed_dim
        self.hidden_dim = self.wm.embed_dim
        self.aug = nn.Identity()
        self.obs_shape = (3,64,64) 
        self.obs_type = self.cfg.obs_type

        encoder = copy.deepcopy(self.wm.encoder)

        self.rnd = RND(self.obs_dim, self.hidden_dim, rnd_rep_dim,
                       encoder, self.aug, self.obs_shape,
                       self.obs_type).to(self.device)
        self.intrinsic_reward_rms = utils.RMS(device=self.device)

        # optimizers
        self.rnd_opt = common.Optimizer('rnd', self.rnd.parameters(), **self.cfg.model_opt, use_amp=self._use_amp)

        self.rnd.train()
        self.requires_grad_(requires_grad=False)

    def update_rnd(self, obs, step):
        metrics = dict()

        prediction_error = self.rnd(obs)

        loss = prediction_error.mean()

        metrics.update(self.rnd_opt(loss, self.rnd.parameters()))

        metrics['rnd_loss'] = loss.item()

        return metrics

    def compute_intr_reward(self, obs):
        prediction_error = self.rnd(obs)
        _, intr_reward_var = self.intrinsic_reward_rms(prediction_error)
        reward = self.rnd_scale * prediction_error / (
            torch.sqrt(intr_reward_var) + 1e-8)
        return reward

    def update(self, data, step):
        metrics = {}
        B, T, _ = data['action'].shape
        obs_shape = data['observation'].shape[2:]

        if self.reward_free:
            temp_data = self.wm.preprocess(data)
            temp_data['observation'] = temp_data['observation'].reshape(B*T, *obs_shape)
            with common.RequiresGrad(self.rnd):
                with torch.cuda.amp.autocast(enabled=self._use_amp):
                    metrics.update(self.update_rnd(temp_data, step))

            with torch.no_grad():
                intr_reward = self.compute_intr_reward(temp_data).reshape(B, T, 1)

            data['reward'] = intr_reward 

        state, outputs, mets = self.wm.update(data, state=None)
        metrics.update(mets)
        start = outputs['post']
        start = {k: stop_gradient(v) for k,v in start.items()}
        reward_fn = lambda seq: self.wm.heads['reward'](seq['feat']).mean #.mode()
        metrics.update(self._task_behavior.update(
            self.wm, start, data['is_terminal'], reward_fn))
        return state, metrics