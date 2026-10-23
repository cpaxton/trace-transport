"""Categorical receiver controls with an exactly representable strong baseline.

The action bank is shared with the headroom diagnostic. Initial policy assigns
90% probability to batch-8/25-ms, rather than learning basic batching from zero.
"""
import math

import torch
from torch import nn

from .learning import HistoryTransformer,ModelConfig

ACTIONS=((1,0.),)+tuple((batch,delay) for batch in (2,4,8) for delay in (.001,.005,.01,.025))
WIDE_ACTIONS=ACTIONS+tuple((batch,delay) for batch in (8,16,32) for delay in (.025,.1,.25) if (batch,delay) not in ACTIONS)
BASELINE_INDEX=ACTIONS.index((8,.025))


class CategoricalReceiver(HistoryTransformer):
    feedback_actions=ACTIONS

    def __init__(self,config=None,*,actions=ACTIONS,baseline_action=(8,.025),prior_mass=.9):
        super().__init__(config or ModelConfig(history=8,width=32,heads=4,layers=1,action_dimensions=2))
        self.feedback_actions=tuple(actions)
        self.actor=nn.Linear(self.config.width,len(self.feedback_actions))
        del self.log_std
        nn.init.zeros_(self.actor.weight)
        prior=torch.full((len(self.feedback_actions),),(1-prior_mass)/(len(self.feedback_actions)-1))
        prior[self.feedback_actions.index(baseline_action)]=prior_mass
        with torch.no_grad():
            self.actor.bias.copy_(prior.log())

    def sample(self,history,valid):
        logits,value=self(history,valid)
        distribution=torch.distributions.Categorical(logits=logits)
        action=distribution.sample()
        return action,action,distribution.log_prob(action),value

    def evaluate(self,history,valid,action):
        logits,value=self(history,valid)
        distribution=torch.distributions.Categorical(logits=logits)
        return distribution.log_prob(action),distribution.entropy(),value


def make_categorical(seed=0,history=8,wide=False):
    torch.manual_seed(seed)
    return CategoricalReceiver(ModelConfig(history=history,width=32,heads=4,layers=1,action_dimensions=2),
        actions=WIDE_ACTIONS if wide else ACTIONS,baseline_action=(8,.25) if wide else (8,.025),
        prior_mass=.08 if wide else .9)
