# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Bring a box to a target and orientation."""

from typing import Any, Dict, Optional, Union
import jax.lax as lax

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np
from scipy.spatial.transform import Rotation as R

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.ego_sim.grippers import rum
from mujoco_playground._src.mjx_env import State  # pylint: disable=g-importing-member
from mujoco_playground._src.manipulation.ego_sim.utils import euler_to_mat, mat_to_quat


def default_config() -> config_dict.ConfigDict:
    """Returns the default config for bring_to_target tasks."""
    config = config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=600,
        action_repeat=1,
        action_scale=0.04,
        reward_config=config_dict.create(
            scales=config_dict.create(
                # Gripper goes to the box.
                gripper_box=4.0,
                # Box goes to the target mocap.
                box_target=20.0,
                # Do not collide the gripper with the table.
                no_table_collision=0.25,
                # Arm stays close to target pose.
                robot_target_qpos=0.3,
            )
        ),
        impl="jax",
        nconmax=24 * 8192,
        njmax=128,
    )
    return config


class RUMPickCube(rum.RUMGripper):
    """Bring a box to a target."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
        sample_orientation: bool = False,
    ):
        xml_path = (
            mjx_env.ROOT_PATH / "manipulation" / "ego_sim" / "xmls" / "pick_scene.xml"
        )
        super().__init__(
            xml_path,
            config,
            config_overrides,
        )
        self._post_init()
        self._sample_orientation = sample_orientation

        self._reward_scale = jp.array(5.0)
        self._distance_threshold = jp.array(0.02)

    def reset(self, rng: jax.Array) -> State:
        rng, rng_box = jax.random.split(rng, 2)

        object_pos = jax.random.uniform(
            rng_box,
            (3,),
            minval=jp.array([-0.20, 0.12, 0.78]),
            maxval=jp.array([0.20, 0.35, 0.78]),
        )

        target_pos = object_pos.at[2].add(0.05)

        init_q = (
            jp.array(self._init_q)
            .at[self._obj_qposadr : self._obj_qposadr + 3]
            .set(object_pos)
        )

        data = mjx_env.make_data(
            self._mj_model,
            qpos=init_q,
            qvel=jp.zeros(self._mjx_model.nv, dtype=float),
            impl=self._mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )

        gripper_pos = data.site_xpos[self._gripper_site].copy()

        metrics = {
            "out_of_bounds": jp.array(0.0, dtype=float),
            **{k: 0.0 for k in self._config.reward_config.scales.keys()},
        }
        info = {
            "rng": rng,
            "reached_box": 0.0,
            "initial_object_pos": object_pos,
            "target_pos": target_pos,
            "gripper_pos": gripper_pos,
            "current_grasp": 0.0,
        }

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        state = State(data, obs, reward, done, metrics, info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        # Delta Action Application - optimized for JAX without matrix operations
        delta_action = jp.clip(
            action[:6] * self._action_scale, self._lower_deltas, self._upper_deltas
        )

        # current_pos = state.info["gripper_pos"]
        current_pos = state.data.site_xpos[self._gripper_site].copy()

        new_position = current_pos + delta_action[:3]

        # Update mocap data in one operation
        data = state.data.replace(
            mocap_pos=state.data.mocap_pos.at[self._mocap_controller, :].set(
                new_position
            ),
        )

        ctrl_grasp = jp.clip(
            state.info["current_grasp"] + action[-1:] * -0.1,
            self._lower_grasp,
            self._upper_grasp,
        )
        state.info.update({"current_grasp": jp.squeeze(ctrl_grasp)})

        data = mjx_env.step(self._mjx_model, data, ctrl_grasp, self.n_substeps)

        raw_rewards = self._get_reward(data, state.info)

        reward_values = jp.array(
            [
                raw_rewards[k] * self._config.reward_config.scales[k]
                for k in raw_rewards.keys()
            ]
        )
        reward = jp.clip(jp.sum(reward_values), -1e4, 1e4)

        box_pos = data.xpos[self._obj_body]
        out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
        out_of_bounds |= box_pos[2] < 0.0
        done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = done.astype(float)

        # Get observations
        obs = self._get_obs(data, state.info)
        state = State(data, obs, reward, done, state.metrics, state.info)

        return state

    def _get_reward(self, data: mjx.Data, info: Dict[str, Any]):
        target_pos = info["target_pos"]
        box_pos = data.xpos[self._obj_body]
        gripper_pos = data.site_xpos[self._gripper_site]
        gripper_box = 1 - jp.tanh(5 * jp.linalg.norm(box_pos - gripper_pos))
        info["reached_box"] = 1.0 * jp.maximum(
            info["reached_box"],
            (jp.linalg.norm(box_pos - gripper_pos) < self._distance_threshold),
        )
        box_target = 1 - jp.tanh(5 * jp.linalg.norm(target_pos - box_pos))
        return {
            "gripper_box": gripper_box,
            "box_target": box_target * info["reached_box"],
        }

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        gripper_pos = data.site_xpos[self._gripper_site]
        obj_pos = data.xpos[self._obj_body]
        rel = obj_pos - gripper_pos
        target_rel = info["target_pos"] - data.xpos[self._obj_body]
        current_grasp = jp.array([info["current_grasp"]])
        obs = jp.concatenate([gripper_pos, obj_pos, rel, target_rel, current_grasp])
        return obs


class RUMPickCubeOrientation(RUMPickCube):
    """Bring a box to a target and orientation."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides, sample_orientation=True)
