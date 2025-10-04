from typing import Any, Dict, Optional, Union
import abc

from etils import epath
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.ego_sim.grippers.floating_gripper import FloatingGripper
#



class RUMGripper(FloatingGripper):
    def __init__(
        self,
        config,
        mj_model,
        mjx_model
    ):
        super().__init__(config, mj_model, mjx_model)
        self.grasping_state = jp.array([-255.0])
        self.non_grasping_state = jp.array([0.0])

    def post_init(self):
        return super().post_init()
    
    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return 7