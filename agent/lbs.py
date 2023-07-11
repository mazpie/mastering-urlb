import copy

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

import utils
from agent.dreamer import DreamerAgent, WorldModel, stop_gradient
import agent.dreamer_utils as common

class LBSDreamerAgent(DreamerAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.reward_free = True
        
        # LBS
        self.lbs = common.MLP(self.wm.inp_size, (1,), **self.cfg.reward_head).to(self.device)
        self.lbs_opt = common.Optimizer('lbs', self.lbs.parameters(), **self.cfg.model_opt, use_amp=self._use_amp)
        self.lbs.train()

        self.requires_grad_(requires_grad=False)

    def update_lbs(self, outs):
        metrics = dict()
        B, T, _ = outs['feat'].shape
        feat, kl = outs['feat'].detach(), outs['kl'].detach()
        feat = feat.reshape(B*T, -1)
        kl = kl.reshape(B*T, -1)

        loss = -self.lbs(feat).log_prob(kl).mean()
        metrics.update(self.lbs_opt(loss, self.lbs.parameters()))
        metrics['lbs_loss'] = loss.item()
        return metrics

    def update(self, data, step):
        metrics = {}
        state, outputs, mets = self.wm.update(data, state=None)
        metrics.update(mets)
        start = outputs['post']
        start = {k: stop_gradient(v) for k,v in start.items()}

        if self.reward_free:
            with common.RequiresGrad(self.lbs):
                with torch.cuda.amp.autocast(enabled=self._use_amp):
                    metrics.update(
                        self.update_lbs(outputs))
            reward_fn = lambda seq: self.lbs(seq['feat']).mean
        else:
            reward_fn = lambda seq: self.wm.heads['reward'](seq['feat']).mean #.mode()
                
        metrics.update(self._task_behavior.update(
            self.wm, start, data['is_terminal'], reward_fn))
        return state, metrics