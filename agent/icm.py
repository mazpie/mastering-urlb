import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from agent.dreamer import DreamerAgent, stop_gradient
import agent.dreamer_utils as common


class ICM(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.forward_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim))

        self.backward_net = nn.Sequential(nn.Linear(2 * obs_dim, hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, action_dim),
                                          nn.Tanh())

        self.apply(utils.weight_init)

    def forward(self, obs, action, next_obs):
        assert obs.shape[0] == next_obs.shape[0]
        assert obs.shape[0] == action.shape[0]

        next_obs_hat = self.forward_net(torch.cat([obs, action], dim=-1))
        action_hat = self.backward_net(torch.cat([obs, next_obs], dim=-1))

        forward_error = torch.norm(next_obs - next_obs_hat,
                                   dim=-1,
                                   p=2,
                                   keepdim=True)
        backward_error = torch.norm(action - action_hat,
                                    dim=-1,
                                    p=2,
                                    keepdim=True)

        return forward_error, backward_error


class ICMDreamerAgent(DreamerAgent):
    def __init__(self, icm_scale, **kwargs):
        super().__init__(**kwargs)
        in_dim = self.wm.inp_size
        pred_dim = self.wm.embed_dim
        self.hidden_dim = pred_dim
        self.reward_free = True
        self.icm_scale = icm_scale

        self.icm = ICM(pred_dim, self.act_dim,
                       self.hidden_dim).to(self.device)

        # optimizers
        self.icm_opt = common.Optimizer('icm', self.icm.parameters(), **self.cfg.model_opt, use_amp=self._use_amp)

        self.icm.train()
        self.requires_grad_(requires_grad=False)

    def update_icm(self, obs, action, next_obs, step):
        metrics = dict()

        forward_error, backward_error = self.icm(obs, action, next_obs)

        loss = forward_error.mean() # + backward_error.mean()

        metrics.update(self.icm_opt(loss, self.icm.parameters()))

        metrics['icm_loss'] = loss.item()

        return metrics

    def compute_intr_reward(self, obs, action, next_obs, step):
        forward_error, _ = self.icm(obs, action, next_obs)

        reward = forward_error * self.icm_scale
        reward = torch.log(reward + 1.0)
        return reward

    def update(self, data, step):
        metrics = {}
        B, T, _ = data['action'].shape

        if self.reward_free:
            T = T-1
            temp_data = self.wm.preprocess(data)
            embed = self.wm.encoder(temp_data)
            inp = stop_gradient(embed[:, :-1]).reshape(B*T, -1)
            action = data['action'][:, 1:].reshape(B*T, -1)
            out = stop_gradient(embed[:,1:]).reshape(B*T,-1)
            with common.RequiresGrad(self.icm):
                with torch.cuda.amp.autocast(enabled=self._use_amp):
                    metrics.update(
                        self.update_icm(inp, action, out, step))

            with torch.no_grad():
                intr_reward = self.compute_intr_reward(inp, action, out, step).reshape(B, T, 1)
            
            data['reward'][:, 0] = 1
            data['reward'][:, 1:] = intr_reward 

        state, outputs, mets = self.wm.update(data, state=None)
        metrics.update(mets)
        start = outputs['post']
        start = {k: stop_gradient(v) for k,v in start.items()}
        reward_fn = lambda seq: self.wm.heads['reward'](seq['feat']).mean #.mode()
        metrics.update(self._task_behavior.update(
            self.wm, start, data['is_terminal'], reward_fn))
        return state, metrics