#!/usr/bin/env python
"""
Created by Indraneel on 02/23/2026

Reimplementation of the Soft Actor Critic Policy
"""



from lerobot.policies.pretrained import PreTrainedPolicy
from policies.sac.configuration_sac import SACConfig


class SACPolicy(
    PreTrainedPolicy
):
    config_class = SACConfig
    