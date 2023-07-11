import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from agent.dreamer import DreamerAgent, stop_gradient
import agent.dreamer_utils as common

class APTDreamerAgent(DreamerAgent):
    def __init__(self, knn_rms, knn_k, knn_avg, knn_clip, **kwargs):
        super().__init__(**kwargs)
        self.reward_free = True

        # particle-based entropy
        rms = utils.RMS(self.device)
        self.pbe = utils.PBE(rms, knn_clip, knn_k, knn_avg, knn_rms,
                             self.device)

        self.requires_grad_(requires_grad=False)

    def compute_intr_reward(self, seq):
        rep = stop_gradient(seq['deter'])
        B, T, _ = rep.shape
        rep = rep.reshape(B*T, -1)
        reward = self.pbe(rep, cdist=True)
        reward = reward.reshape(B, T, 1)
        return reward

    def update(self, data, step):
        metrics = {}
        B, T, _ = data['action'].shape

        state, outputs, mets = self.wm.update(data, state=None)
        metrics.update(mets)
        start = outputs['post']
        start = {k: stop_gradient(v) for k,v in start.items()}
        if self.reward_free:
            reward_fn = lambda seq: self.compute_intr_reward(seq)
        else:
            reward_fn = lambda seq: self.wm.heads['reward'](seq['feat']).mean #.mode()
        metrics.update(self._task_behavior.update(
            self.wm, start, data['is_terminal'], reward_fn))
        return state, metrics