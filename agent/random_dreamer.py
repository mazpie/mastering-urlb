import torch.nn as nn
import torch

import utils
from collections import OrderedDict
import numpy as np

from agent.dreamer import DreamerAgent, WorldModel, stop_gradient
import agent.dreamer_utils as common

Module = nn.Module 

class RandomDreamerAgent(DreamerAgent):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def act(self, obs, meta, step, eval_mode, state):
    return torch.zeros(self.act_spec.shape).uniform_(-1.0, 1.0).numpy(), None

  def update(self, data, step):
    metrics = {}
    state, outputs, mets = self.wm.update(data, state=None)
    metrics.update(mets)
    return state, metrics