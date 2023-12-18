import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path
import pandas as pd
import hydra
import numpy as np
import torch
import wandb
from dm_env import specs

import envs
import utils
from logger import Logger
from dreamer_replay import ReplayBuffer, make_replay_loader
from agent.dreamer import DreamerAgent
from hydra.utils import get_original_cwd, to_absolute_path

torch.backends.cudnn.benchmark = True

def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)

def make_dreamer_agent(obs_space, action_spec, cur_config, cfg):
    from copy import deepcopy
    cur_config = deepcopy(cur_config)
    del cur_config.agent
    return hydra.utils.instantiate(cfg, cfg=cur_config, obs_space=obs_space, act_spec=action_spec)

class Workspace:
    def __init__(self, cfg, savedir=None, workdir=None):
        self.workdir = Path.cwd() if workdir is None else workdir
        print(f'workspace: {self.workdir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # create logger
        self.logger = Logger(self.workdir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)
        # create envs
        frame_stack = 1
        img_size = 64

        self.train_env = envs.make(cfg.task, cfg.obs_type, frame_stack, 
                                  cfg.action_repeat, cfg.seed, img_size=img_size) 
        self.eval_env = envs.make(cfg.task, cfg.obs_type, frame_stack, 
                                 cfg.action_repeat, cfg.seed, img_size=img_size)

        obs_space = self.train_env.obs_space
        act_spec = self.train_env.action_spec()

        # create agent 
        exceptions = ['diayn_dreamer', 'aps_dreamer'] # skill approaches
        if cfg.agent.name in exceptions:
            self.agent = make_dreamer_agent(self.train_env.obs_space,
                    self.train_env.action_spec(), cfg, cfg.agent)
            self.agent.finetune_mode()
        else:
            self.agent = DreamerAgent(cfg.agent.name, cfg, obs_space, act_spec) 

        # initialize from pretrained
        if cfg.snapshot_ts > 0:
            pretrained_agent = self.load_snapshot()['agent']
            self.agent.init_from(pretrained_agent)

        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        # create data storage
        self.replay_storage = ReplayBuffer(data_specs, meta_specs,
                                                  self.workdir / 'buffer',
                                                  length=cfg.batch_length, **cfg.replay,
                                                  device=cfg.device)

        if self.cfg.save_eval_episodes:
            self.eval_storage = ReplayBuffer(data_specs, meta_specs,
                                                    self.workdir / 'eval_episodes',
                                                    length=cfg.batch_length, **cfg.replay,
                                                    device=cfg.device)
        # create replay buffer
        self.replay_loader = make_replay_loader(self.replay_storage,
                                                cfg.batch_size, # 
                                                cfg.replay_buffer_num_workers)
        self._replay_iter = None

        # Globals
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self.savedir = savedir

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward, ep_rew = 0, 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        meta = self.agent.init_meta()
        episode_rewards = []
        while eval_until_episode(episode):
            dreamer_obs = self.eval_env.reset()
            data = dreamer_obs
            agent_state = None
            if self.cfg.save_eval_episodes: self.eval_storage.add(data, meta) 
            while not bool(dreamer_obs['is_last']):
                with torch.no_grad(), utils.eval_mode(self.agent):
                    if self.cfg.mpc:
                        action, agent_state = self.agent.plan(dreamer_obs,
                                                meta,
                                                self.global_step,
                                                eval_mode=True,
                                                state=agent_state,
                                                t0=bool(dreamer_obs['is_first']))
                    else:
                        action, agent_state = self.agent.act(dreamer_obs, 
                                                meta,
                                                self.global_step,
                                                eval_mode=True,
                                                state=agent_state)
                dreamer_obs = self.eval_env.step(action)
                data = dreamer_obs
                if self.cfg.save_eval_episodes: self.eval_storage.add(data, meta) 
                total_reward += dreamer_obs['reward']
                ep_rew += dreamer_obs['reward']
                step += 1
                if not bool(dreamer_obs['is_last']): meta = self.agent.update_meta(meta, step, dreamer_obs)

            episode_rewards.append(ep_rew)
            ep_rew = 0
            episode += 1

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)
        train_every_n_steps = self.cfg.train_every_actions // self.cfg.action_repeat 
        should_train_step = utils.Every(train_every_n_steps * self.cfg.action_repeat,  
                                      self.cfg.action_repeat)
        should_log_scalars = utils.Every(self.cfg.log_every_frames, 
                                      self.cfg.action_repeat)
        should_log_recon = utils.Every(self.cfg.recon_every_frames, 
                                        self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        dreamer_obs = self.train_env.reset()
        data = dreamer_obs
        agent_state = None
        meta = self.agent.init_meta()
        self.replay_storage.add(data, meta) 
        metrics = None
        while train_until_step(self.global_step):
            if bool(dreamer_obs['is_last']):
                self._global_episode += 1
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage)) 
                        log('step', self.global_step)
                # save last model
                self.save_last_model()

                # reset env
                dreamer_obs = self.train_env.reset()
                data = dreamer_obs
                agent_state = None # Resetting agent's latent state
                meta = self.agent.init_meta()
                self.replay_storage.add(data, meta) 
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if self.global_step > 0 \
                or self.agent.name == 'dreamer' \
                or self.cfg.num_train_frames < 1000:
                if eval_every_step(self.global_step):
                    self.logger.log('eval_total_time', self.timer.total_time(),
                                    self.global_frame)
                    self.eval()

            if hasattr(self.agent, "regress_meta"):
                repeat = self.cfg.action_repeat
                update_meta_every_step = getattr(self.agent, 'update_task_every_step', None) or getattr(self.agent, 'update_skill_every_step', None)
                every = update_meta_every_step // repeat
                init_step = self.agent.num_init_frames // repeat
                if self.global_step > init_step and self.global_step % every == 0:
                    meta = self.agent.regress_meta(self.replay_iter, self.global_step)

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                if self.cfg.mpc and not seed_until_step(self.global_step - 1000 // self.cfg.action_repeat): # some steps to bootstrap rew_pred
                    action, agent_state = self.agent.plan(dreamer_obs,
                                            meta,
                                            self.global_step,
                                            eval_mode=False,
                                            state=agent_state,
                                            t0=bool(dreamer_obs['is_first']))
                else:
                    action, agent_state = self.agent.act(dreamer_obs, 
                                            meta,
                                            self.global_step,
                                            eval_mode=False,
                                            state=agent_state)

            # try to update the agent
            if not seed_until_step(self.global_step):
                if should_train_step(self.global_step):
                    metrics = self.agent.update(next(self.replay_iter), self.global_step)[1] # , self.global_step)
                if should_log_scalars(self.global_step):
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')
                if self.global_step > 0 and should_log_recon(self.global_step):
                    videos = self.agent.report(next(self.replay_iter))
                    self.logger.log_video(videos, self.global_frame)

            # take env step
            dreamer_obs = self.train_env.step(action)
            data = dreamer_obs
            episode_reward += dreamer_obs['reward']
            self.replay_storage.add(data, meta) 
            episode_step += 1
            self._global_step += 1
            if not bool(dreamer_obs['is_last']): meta = self.agent.update_meta(meta, self.global_step, dreamer_obs)
        if self.cfg.save_ft_model:
            self.save_finetuned_model()

    def load_snapshot(self):
        snapshot_base_dir = Path(self.cfg.snapshot_base_dir) 
        domain, _ = self.cfg.task.split('_', 1)
        snapshot_dir = snapshot_base_dir / self.cfg.obs_type / domain / self.cfg.agent.name
        if self.cfg.custom_snap_dir != 'none':
            snapshot_dir = Path(self.cfg.custom_snap_dir) 
        snapshot = snapshot_dir / str(
            self.cfg.seed) / f'snapshot_{self.cfg.snapshot_ts}.pt'
            
        def try_load(seed):
            if not snapshot.exists():
                return None
            with snapshot.open('rb') as f:
                payload = torch.load(f)
            return payload

        # try to load current seed
        payload = try_load(self.cfg.seed)
        if payload is not None:
            print(f"Snapshot loaded from: {snapshot}")
            return payload
        else:
            raise Exception(f"Snapshot not found at: {snapshot}")

    @utils.retry
    def save_finetuned_model(self):
        root_dir = Path.cwd()
        snapshot = root_dir / 'finetuned_snapshot.pt'
        keys_to_save = ['agent', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)
    
    @utils.retry
    def save_last_model(self):
        root_dir = Path.cwd()
        snapshot = root_dir / 'last_snapshot.pt'
        if snapshot.is_file():
            temp = Path(str(snapshot).replace("last_snapshot.pt", "second_last_snapshot.pt"))
            os.replace(snapshot, temp)
        keys_to_save = ['agent', '_global_step', '_global_episode']
        if self.cfg.use_wandb: 
            keys_to_save.append('wandb_run_id')
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_last_model(self):
        root_dir = Path.cwd()
        try:
            snapshot = root_dir / 'last_snapshot.pt'
            with snapshot.open('rb') as f:
                payload = torch.load(f)
        except:
            snapshot = root_dir / 'second_last_snapshot.pt'
            with snapshot.open('rb') as f:
                payload = torch.load(f)
        for k,v in payload.items():
            setattr(self, k, v)
            if k == 'wandb_run_id':
                assert wandb.run is None
                cfg = self.cfg
                exp_name = '_'.join([
                    cfg.experiment, cfg.agent.name, cfg.task, cfg.obs_type,
                    str(cfg.seed)
                ])
                wandb.init(project=cfg.project_name + "_finetune", group=cfg.agent.name, name=exp_name, id=v, resume="must")

    def setup_wandb(self):
        cfg = self.cfg
        exp_name = '_'.join([
            cfg.experiment, cfg.agent.name, cfg.task, cfg.obs_type,
            str(cfg.seed)
        ])
        wandb.init(project=cfg.project_name + "_finetune", group=cfg.agent.name, name=exp_name)
        wandb.config.update(cfg)
        self.wandb_run_id = wandb.run.id

@hydra.main(config_path='.', config_name='dreamer_finetune')
def main(cfg):
    from dreamer_finetune import Workspace as W
    root_dir = Path.cwd()
    cfg.snapshot_base_dir = str(Path(get_original_cwd()) / cfg.snapshot_base_dir)
    workspace = W(cfg)
    snapshot = root_dir / 'last_snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_last_model()
    if cfg.use_wandb and wandb.run is None:
        # otherwise it was resumed
        workspace.setup_wandb()
    workspace.train()

if __name__ == '__main__':
    main()
