from typing import Any, Dict, Optional, Union

from etils import epath
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env


def get_assets() -> Dict[str, bytes]:
    assets = {}
    path = mjx_env.ROOT_PATH / "manipulation" / "ego_sim" / "xmls"
    mjx_env.update_assets(assets, path, "*.xml")
    path = mjx_env.ROOT_PATH / "manipulation" / "ego_sim" / "xmls" / "rum"
    mjx_env.update_assets(assets, path, "*.xml")
    path = mjx_env.ROOT_PATH / "manipulation" / "ego_sim" / "xmls" / "textures"
    mjx_env.update_assets(assets, path, "*.png")
    path = mjx_env.ROOT_PATH / "manipulation" / "ego_sim" / "xmls" / "rum" / "meshes"
    mjx_env.update_assets(assets, path, "*.stl")
    path = (
        mjx_env.ROOT_PATH
        / "manipulation"
        / "ego_sim"
        / "xmls"
        / "objects"
        / "apple"
        / "apple_0"
    )
    mjx_env.update_assets(assets, path, "*.xml")
    mjx_env.update_assets(assets, path / "collision")
    mjx_env.update_assets(assets, path / "visual")
    return assets


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=400,
        action_repeat=1,
        action_scale=0.02,
        impl="jax",
        nconmax=12 * 8192,
        njmax=44,
    )


class RUMGripper(mjx_env.MjxEnv):
    """Base environment for RUM Gripper."""

    def __init__(
        self,
        xml_path: epath.Path,
        config: config_dict.ConfigDict,
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(config, config_overrides)

        self._xml_path = xml_path.as_posix()
        xml = xml_path.read_text()
        self._model_assets = get_assets()
        mj_model = mujoco.MjModel.from_xml_string(xml, assets=self._model_assets)
        mj_model.opt.timestep = self.sim_dt

        self._mj_model = mj_model
        self._mjx_model = mjx.put_model(mj_model, impl=self._config.impl)
        self._action_scale = config.action_scale

    def _post_init(self):
        self._gripper_site = self._mj_model.site("gripper").id
        self._gripper_base = self._mj_model.body("base").id
        # self._left_finger_geom = self._mj_model.geom("left_finger_pad").id
        # self._right_finger_geom = self._mj_model.geom("right_finger_pad").id
        # self._gripper_body_geom = self._mj_model.geom("gripper_body_collision").id
        # self._iphone_geom = self._mj_model.geom("iphone_collision").id
        # self._hand_geom = self._mj_model.geom("gripper_body_collision").id
        self._obj_body = self._mj_model.body("object_body").id
        self._obj_geom = self._mj_model.geom("object_geom").id
        self._obj_qposadr = self._mj_model.jnt_qposadr[
            self._mj_model.body("object_body").jntadr[0]
        ]
        self._mocap_controller = self._mj_model.body("target_ee_pose").mocapid
        self._table_geom = self._mj_model.geom("table_top").id
        self._lower_deltas = jp.array([-0.05] * 6)
        self._upper_deltas = jp.array([0.05] * 6)
        self._lower_grasp = jp.array([-0.8])
        self._upper_grasp = jp.array([0.0])
        self._init_q = self._mj_model.keyframe("home").qpos

    def get_pose(self, state):
        """Get gripper pose as position and rotation matrix separately for efficiency."""
        pos = state.data.site_xpos[self._gripper_site]
        rot_mat = state.data.site_xmat[self._gripper_site]
        return pos, rot_mat

    def get_pose_matrix(self, state):
        """Get full 4x4 pose matrix only when needed."""
        pos = state.data.site_xpos[self._gripper_site]
        rot_mat = state.data.site_xmat[self._gripper_site]
        # Construct pose matrix directly without identity matrix
        pose = jp.zeros((4, 4))
        pose = pose.at[:3, :3].set(rot_mat)
        pose = pose.at[:3, 3].set(pos)
        pose = pose.at[3, 3].set(1.0)
        return pose

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return 7

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
