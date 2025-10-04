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



class RobotiqGripper(FloatingGripper):
    def __init__(
        self,
        config,
        mj_model,
        mjx_model
    ):
        super().__init__(config, mj_model, mjx_model)
        self.name = "robotiq"
        self.base_pos = jp.array([0.0, -0.55, 0.9])
        self.base_rot = jp.array([0.5, -0.867, 0.0, 0.0])

        pass

    def gripper_action_to_ctrl(self, action):
        ctrl_grasp = jp.clip(
            action * 255.0, jp.array([0.0]), jp.array([255.0])
        )
        return ctrl_grasp

    def post_init(self):
        return super().post_init()
    
    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return 7