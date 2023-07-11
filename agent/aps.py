import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs
import math
from collections import OrderedDict

import utils

from agent.dreamer import DreamerAgent, stop_gradient
import agent.dreamer_utils as common
from agent.skill_utils import *

class APS(nn.Module):
    def __init__(self, obs_dim, sf_dim, hidden_dim):
        super().__init__()
        self.state_feat_net = nn.Sequential(nn.Linear(obs_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, sf_dim))

        self.apply(utils.weight_init)

    def forward(self, obs, norm=True):
        state_feat = self.state_feat_net(obs)
        state_feat = F.normalize(state_feat, dim=-1) if norm else state_feat
        return state_feat

class APSDreamerAgent(DreamerAgent):
    def __init__(self, update_skill_every_step, skill_dim, knn_rms, knn_k, knn_avg,
                 knn_clip, num_init_frames, lstsq_batch_size, **kwargs):
        self.update_skill_every_step = update_skill_every_step
        self.skill_dim = skill_dim
        self.num_init_frames = num_init_frames
        self.lstsq_batch_size = lstsq_batch_size

        super().__init__(**kwargs)
        in_dim = self.wm.inp_size
        self._task_behavior = SFActorCritic(self.cfg, self.act_spec, self.tfstep, self.skill_dim, discrete_skills=False).to(self.device)

        self.hidden_dim = in_dim
        self.reward_free = True
        self.solved_meta = None

        self.aps = APS(in_dim, self.skill_dim,
                       self.hidden_dim).to(self.device)

        # particle-based entropy
        rms = utils.RMS(self.device)
        self.aps_pbe = utils.PBE(rms, knn_clip, knn_k, knn_avg, knn_rms, self.device)

        # optimizers
        self.aps_opt = common.Optimizer('aps', self.aps.parameters(), **self.cfg.model_opt, use_amp=self._use_amp)
        self.aps.train()
        self.requires_grad_(requires_grad=False)
    
    def finetune_mode(self):
        self.is_ft = True
        self.reward_free = False
        self._task_behavior.rewnorm = common.StreamNorm(**{"momentum": 1.00, "scale": 1.0, "eps": 1e-8}, device=self.device)
        self.cfg.actor_ent = 1e-4
        self.cfg.sf_actor_ent = 1e-4

    def act(self, obs, meta, step, eval_mode, state):
        obs = {k : torch.as_tensor(np.copy(v), device=self.device).unsqueeze(0) for k, v in obs.items()}
        meta = {k : torch.as_tensor(np.copy(v), device=self.device).unsqueeze(0) for k, v in meta.items()}

        if state is None:
            latent = self.wm.rssm.initial(len(obs['reward']))
            action = torch.zeros((len(obs['reward']),) + self.act_spec.shape, device=self.device)
        else:
            latent, action = state
        embed = self.wm.encoder(self.wm.preprocess(obs))
        should_sample = (not eval_mode) or (not self.cfg.eval_state_mean)
        latent, _ = self.wm.rssm.obs_step(latent, action, embed, obs['is_first'], should_sample)
        feat = self.wm.rssm.get_feat(latent)
        
        skill = meta['skill']
        inp = torch.cat([feat, skill], dim=-1)
        if eval_mode:
            actor = self._task_behavior.actor(inp)
            action = actor.mean
        else:
            actor = self._task_behavior.actor(inp)
            action = actor.sample()
        new_state = (latent, action)
        return action.cpu().numpy()[0], new_state

    def get_meta_specs(self):
        return (specs.Array((self.skill_dim,), np.float32, 'skill'),)

    def init_meta(self):
        if self.solved_meta is not None:
            return self.solved_meta
        skill = torch.randn(self.skill_dim)
        skill = skill / torch.norm(skill)
        skill = skill.cpu().numpy()
        meta = OrderedDict()
        meta['skill'] = skill
        return meta

    def update_meta(self, meta, global_step, time_step):
        if global_step % self.update_skill_every_step == 0:
            return self.init_meta()
        return meta

    @torch.no_grad()
    def regress_meta(self, replay_iter, step):
        if self.solved_meta is not None:
            return self.solved_meta
        feats, reward = [], []
        batch_size = 0
        while batch_size < self.lstsq_batch_size:
            data = next(replay_iter)
            data = self.wm.preprocess(data)
            embed = self.wm.encoder(data)
            post, prior = self.wm.rssm.observe(
                embed, data['action'], data['is_first'])
            feat = self.wm.rssm.get_feat(post)
            B, T, _ = feat.shape

            feats.append(feat.reshape(B*T,-1))
            reward.append(data['reward'].reshape(B*T,-1))
            batch_size += B*T
        feats, reward = torch.cat(feats, 0), torch.cat(reward, 0)

        rep = self.aps(feats)
        skill = torch.linalg.lstsq(reward, rep)[0][:rep.size(1), :][0]
        skill = skill / torch.norm(skill)
        skill = skill.cpu().numpy()
        meta = OrderedDict()
        meta['skill'] = skill

        print("skill selected : ", meta['skill'])

        # save for evaluation
        self.solved_meta = meta
        self._task_behavior.solved_meta = meta 
        return meta

    def update_aps(self, skill, next_obs, step):
        metrics = dict()

        B, T, _ = skill.shape
        skill = skill.reshape(B*T, -1)
        next_obs = next_obs.reshape(B*T, -1)

        loss = self.compute_aps_loss(next_obs, skill)
        metrics.update(self.aps_opt(loss, self.aps.parameters()))

        metrics['aps_loss'] = loss.item()
        return metrics

    def compute_aps_loss(self, next_obs, skill):
        """MLE loss"""
        loss = -torch.einsum("bi,bi->b", skill, self.aps(next_obs)).mean()
        return loss

    def compute_intr_reward(self, seq, step):
        skill, next_obs = seq['skill'], seq['feat']
        
        B, T, _ = skill.shape
        skill = skill.reshape(B*T, -1)
        next_obs = next_obs.reshape(B*T, -1)

        # maxent reward
        with torch.no_grad():
            rep = self.aps(next_obs, norm=False)
            reward = self.aps_pbe(rep, cdist=True)
            intr_ent_reward = reward.reshape(-1, 1)

        # successor feature reward
        rep = rep / torch.norm(rep, dim=1, keepdim=True)
        intr_sf_reward = torch.einsum("bi,bi->b", skill, rep).reshape(-1, 1)

        reward = intr_ent_reward + intr_sf_reward

        return reward.reshape(B, T, 1).detach()

    def update(self, data, step):
        metrics = {}
        state, outputs, mets = self.wm.update(data, state=None)
        metrics.update(mets)
        start = outputs['post']
        start = {k: stop_gradient(v) for k,v in start.items()}

        if self.reward_free:
            skill = data['skill']
            feat = self.wm.rssm.get_feat(start)
            with common.RequiresGrad(self.aps):
                with torch.cuda.amp.autocast(enabled=self._use_amp):
                    metrics.update(self.update_aps(skill, feat, step))
            reward_fn = lambda seq: self.compute_intr_reward(seq, step)
        else:
            reward_fn = lambda seq: self.wm.heads['reward'](seq['feat']).mean #.mode()
            if self.solved_meta is None:
                return state, metrics

        metrics.update(self._task_behavior.update(
            self.wm, start, data['is_terminal'], reward_fn))
        return state, metrics
  
    @torch.no_grad()
    def estimate_value(self, start, actions, horizon):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in start.items()}
        start['feat'] = self.wm.rssm.get_feat(start)
        start['action'] = torch.zeros_like(actions[0], device=self.device) 
        seq = {k: [v] for k, v in start.items()}
        for t in range(horizon):
            action = actions[t] 
            state = self.wm.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
            feat = self.wm.rssm.get_feat(state)
            for key, value in {**state, 'action': action, 'feat': feat}.items():
                seq[key].append(value)

        seq = {k: torch.stack(v, 0) for k, v in seq.items()}
        reward = self.wm.heads['reward'](seq['feat']).mean
        if self.cfg.mpc_opt.use_value:
            B, T, _ = seq['feat'].shape
            seq['skill'] = torch.from_numpy(self.solved_meta['skill']).repeat(B, T, 1).to(self.device)
            value = self._task_behavior._target_critic(get_feat_ac(seq)).mean
            value = torch.einsum('tbk,tbk->tb', value, seq['skill']).unsqueeze(-1)
        else:
            value = torch.zeros_like(reward, device=self.device)
        disc = self.cfg.discount * torch.ones(list(seq['feat'].shape[:-1]) + [1], device=self.device)

        lambda_ret = common.lambda_return(
            reward[:-1], value[:-1], disc[:-1],
            bootstrap=value[-1],
            lambda_=self.cfg.discount_lambda, 
            axis=0)

        # First step is lost because the reward is from the start state
        return lambda_ret[1]

    @torch.no_grad()
    def plan(self, obs, meta, step, eval_mode, state, t0=True):
        """
        Plan next action using Dyna-MPC inference.
        """
        if self.solved_meta is None:  
            return self.act(obs, meta, step, eval_mode, state)

        # Get Dreamer's state and features
        obs = {k : torch.as_tensor(np.copy(v), device=self.device).unsqueeze(0) for k, v in obs.items()}
        if state is None:
            latent = self.wm.rssm.initial(len(obs['reward']))
            action = torch.zeros((len(obs['reward']),) + self.act_spec.shape, device=self.device)
        else:
            latent, action = state
        embed = self.wm.encoder(self.wm.preprocess(obs))
        should_sample = (not eval_mode) or (not self.cfg.eval_state_mean)
        post, prior = self.wm.rssm.obs_step(latent, action, embed, obs['is_first'], should_sample)
        feat = self.wm.rssm.get_feat(post)

        # Sample policy trajectories
        num_pi_trajs = int(self.cfg.mpc_opt.mixture_coef * self.cfg.mpc_opt.num_samples)
        if num_pi_trajs > 0: 
            start = { k: v.repeat(num_pi_trajs, *list([1]*len(v.shape)) ) for k,v in post.items()}
            img_skill = torch.from_numpy(self.solved_meta['skill']).repeat(num_pi_trajs, 1).to(self.device)
            seq = self.wm.imagine(self._task_behavior.actor, start, None, self.cfg.mpc_opt.horizon, task_cond=img_skill) 
            pi_actions = seq['action'][1:]
            
        # Initialize state and parameters
        start = { k: v.repeat(self.cfg.mpc_opt.num_samples+num_pi_trajs, *list([1]*len(v.shape)) ) for k,v in post.items()}
        mean = torch.zeros(self.cfg.mpc_opt.horizon, self.act_dim, device=self.device)
        std = 2*torch.ones(self.cfg.mpc_opt.horizon, self.act_dim, device=self.device)
        if not t0 and hasattr(self, '_prev_mean'):
            mean[:-1] = self._prev_mean[1:]

        # Iterate CEM
        for i in range(self.cfg.mpc_opt.iterations):
            actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
                torch.randn(self.cfg.mpc_opt.horizon, self.cfg.mpc_opt.num_samples, self.act_dim, device=std.device), -1, 1)
            if num_pi_trajs > 0:
                actions = torch.cat([actions, pi_actions], dim=1)

            # Compute elite actions
            value = self.estimate_value(start, actions, self.cfg.mpc_opt.horizon)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.mpc_opt.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cfg.mpc_opt.temperature*(elite_value - max_value))
            score /= score.sum(0)
            _mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            _std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9))
            _std = _std.clamp_(self.cfg.mpc_opt.min_std, 2)
            mean, std = self.cfg.mpc_opt.momentum * mean + (1 - self.cfg.mpc_opt.momentum) * _mean, _std

        # Outputs
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        mean, std = actions[0], _std[0]
        a = mean
        if not eval_mode:
            a += std * torch.randn(self.act_dim, device=std.device)
        new_state = (post, a.unsqueeze(0))
        return a.cpu().numpy(), new_state