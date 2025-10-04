from typing import Any, Dict, Optional, Union
import mujoco
import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from scipy.spatial.transform import Rotation as R

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.ego_sim.grippers.specific_grippers.rum import RUMGripper
from mujoco_playground._src.mjx_env import State
from mujoco_playground._src.manipulation.ego_sim.utils import euler_to_mat, mat_to_quat


def default_config() -> config_dict.ConfigDict:
    config = config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=400,
        action_repeat=1,
        action_scale=0.05,
        reward_config=config_dict.create(
            scales=config_dict.create(
                gripper_box=3.0,
                box_target=6.0,
                no_table_collision=0.25,
                robot_target_qpos=0.3,
            ),
            lifted_reward=1.0,
            success_reward=3.0,
        ),
        success_threshold=0.03,
        impl="jax",
        nconmax=24 * 8192,
        njmax=700,
    )
    return config


def get_assets() -> Dict[str, bytes]:
    assets = {}
    path = mjx_env.ROOT_PATH / "manipulation" / "ego_sim" / "xmls"
    mjx_env.update_assets(assets, path, "*.xml")
    path = mjx_env.ROOT_PATH / "manipulation" / "ego_sim" / "xmls" / "floating_grippers" / "rum"
    mjx_env.update_assets(assets, path, "*.xml")
    path = mjx_env.ROOT_PATH / "manipulation" / "ego_sim" / "xmls" / "textures"
    mjx_env.update_assets(assets, path, "*.png")
    path = mjx_env.ROOT_PATH / "manipulation" / "ego_sim" / "xmls" / "floating_grippers" / "rum" / "meshes"
    mjx_env.update_assets(assets, path, "*.stl")
    return assets

class EgoPick(mjx_env.MjxEnv):
    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        self.config = config
        xml_path = (
            mjx_env.ROOT_PATH / "manipulation" / "ego_sim" / "xmls" / "pick_scene.xml"
        )
        xml = xml_path.read_text()
        self._model_assets = get_assets()
        mj_model = mujoco.MjModel.from_xml_string(xml, assets=self._model_assets)
        mj_model.opt.timestep = self.config.sim_dt
        super().__init__(
            config,
            config_overrides,
        )
        self._mj_model = mj_model
        self._mjx_model = mjx.put_model(mj_model, impl=self.config.impl)
        self.gripper = RUMGripper(config, mj_model, self._mjx_model)
        
    def reset(self, rng: jax.Array) -> State:
        rng, rng_box = jax.random.split(rng, 2)

        self._obj_body = self._mj_model.body("object_body").id
        self._obj_geom = self._mj_model.geom("object_geom").id
        self._obj_qposadr = self._mj_model.jnt_qposadr[
            self._mj_model.body("object_body").jntadr[0]
        ]
        object_pos = jax.random.uniform(
            rng_box,
            (3,),
            minval=jp.array([-0.20, 0.12, 0.78]),
            maxval=jp.array([0.20, 0.35, 0.78]),
        )
        self._init_q = self.gripper.mj_model.keyframe("home").qpos
        init_q = (
            jp.array(self._init_q)
            .at[self._obj_qposadr : self._obj_qposadr + 3]
            .set(object_pos)
        )
        mocap_pos = jp.array([0.0, -0.55, 0.85])
        mocap_rot = R.from_euler("xyz", [0, 0, 0]).as_quat()
        mocap_quat = jp.array([mocap_rot[3], mocap_rot[0], mocap_rot[1], mocap_rot[2]])

        data = mjx_env.make_data(
            self._mj_model,
            qpos=init_q,
            qvel=jp.zeros(self._mjx_model.nv, dtype=float),
            impl=self._mjx_model.impl.value,
            nconmax=self._config.nconmax,
            njmax=self._config.njmax,
            mocap_pos=mocap_pos,
            mocap_quat=mocap_quat,
        )

        ctrl_grasp = jp.clip(
            0.0,
            self.gripper.grasping_state,
            self.gripper.non_grasping_state,
        )

        data = mjx_env.step(self._mjx_model, data, ctrl_grasp, self.n_substeps)

        box_pos = data.xpos[self._obj_body].copy()
        target_pos = box_pos.at[2].add(0.05)

        gripper_pos = self.gripper.get_eef_pose(data)[:3, 3]

        metrics = {
            "out_of_bounds": jp.array(0.0, dtype=float),
            **{f"reward/{k}": 0.0 for k in self._config.reward_config.scales.keys()},
            "reward/lifted": jp.array(0.0, dtype=float),
            "reward/success": jp.array(0.0, dtype=float),
        }
        info = {
            "rng": rng,
            "initial_object_pos": object_pos,
            "target_pos": target_pos,
            "gripper_pos": gripper_pos,
            "current_grasp": 0.0,
            "prev_reward": jp.array(0.0, dtype=float),
            "_steps": jp.array(0, dtype=int),
        }

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        state = State(data, obs, reward, done, metrics, info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        newly_reset = state.info["_steps"] == 0
        state.info["prev_reward"] = jp.where(
            newly_reset, 0.0, state.info["prev_reward"]
        )

        data = self.gripper.step(state.data, action, self.n_substeps)
        raw_rewards = self.get_reward(data, state.info)
        rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in raw_rewards.items()
        }
        total_reward = jp.clip(sum(rewards.values()), -1e4, 1e4)
        box_pos = data.xpos[self._obj_body]
        initial_z = state.info["initial_object_pos"][2]
        lifted = (
            box_pos[2] > initial_z + 0.02
        ) * self._config.reward_config.lifted_reward
        total_reward += lifted

        success = self._get_success(data, state.info)
        total_reward += (
            success.astype(float) * self._config.reward_config.success_reward
        )
        reward = jp.maximum(
            total_reward - state.info["prev_reward"], jp.zeros_like(total_reward)
        )
        state.info["prev_reward"] = jp.maximum(total_reward, state.info["prev_reward"])
        reward = jp.where(newly_reset, 0.0, reward)

        state.metrics.update({f"reward/{k}": v for k, v in raw_rewards.items()})
        state.metrics.update(
            {
                "reward/lifted": lifted,
                "reward/success": success.astype(float),
            }
        )
        box_pos = data.xpos[self._obj_body]
        out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
        out_of_bounds |= box_pos[2] < 0.0
        done = (
            out_of_bounds
            | jp.isnan(data.qpos).any()
            | jp.isnan(data.qvel).any()
            | success
        )
        state.info["_steps"] = jp.where(
            done | (state.info["_steps"] >= self._config.episode_length),
            0,
            state.info["_steps"] + self._config.action_repeat,
        )

        done = done.astype(float)
        obs = self._get_obs(data, state.info)
        state = State(data, obs, reward, done, state.metrics, state.info)

        return state

    def get_reward(self, data: mjx.Data, info: Dict[str, Any]):
        target_pos = info["target_pos"]
        box_pos = data.xpos[self._obj_body]
        gripper_pose = self.gripper.get_eef_pose(data)
        gripper_pos = gripper_pose[:3, 3]

        box_target_dist = jp.linalg.norm(target_pos - box_pos)
        box_target_err = jp.clip(box_target_dist, min=1e-6)
        box_target = 1 - jp.tanh(5 * box_target_err)

        gripper_box_dist = jp.linalg.norm(box_pos - gripper_pos)
        gripper_box_err = jp.clip(gripper_box_dist, min=1e-6)
        gripper_box = 1 - jp.tanh(5 * gripper_box_err)

        return {
            "gripper_box": gripper_box,
            "box_target": box_target,
        }

    def _get_success(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        box_pos = data.xpos[self._obj_body]
        target_pos = info["target_pos"]
        dist = jp.linalg.norm(box_pos - target_pos)
        return dist < self._config.success_threshold

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        gripper_pos = self.gripper.get_eef_pose(data)[:3, 3]
        obj_pos = data.xpos[self._obj_body]
        rel = obj_pos - gripper_pos
        target_rel = info["target_pos"] - data.xpos[self._obj_body]
        current_grasp = jp.array([info["current_grasp"]])
        obs = jp.concatenate([gripper_pos, obj_pos, rel, target_rel, current_grasp])
        return obs

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self.gripper.action_size

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
