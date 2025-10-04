import abc
import jax.numpy as jp
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.ego_sim.utils import euler_to_mat, mat_to_quat

class FloatingGripper():
    def __init__(
        self,
        config,
        _mj_model,
        _mjx_model
    ):
        self._config = config
        self._mj_model = _mj_model
        self._mjx_model = _mjx_model
        self.lower_deltas = jp.array([-0.05]*6)
        self.upper_deltas = jp.array([0.05]*6)
        self.post_init()
        return
    
    def post_init(self):
        self.mocap_controller = self._mj_model.body("target_ee_pose").mocapid
        self._gripper_site = self._mj_model.site("grasping_center").id

    def get_eef_pose(self, data):
        pos = data.site_xpos[self._gripper_site]
        rot_mat = data.site_xmat[self._gripper_site]
        pose = jp.zeros((4, 4))
        pose = pose.at[:3, :3].set(rot_mat)
        pose = pose.at[:3, 3].set(pos)
        pose = pose.at[3, 3].set(1.0)
        return pose

    def step(self, data, action, n_substeps):
        # pose step
        delta_action = jp.clip(
            action[:6] * self._config.action_scale, self.lower_deltas, self.upper_deltas
        )
        current_pose = self.get_eef_pose(data)
        current_pos = current_pose[:3, 3]
        current_rot = current_pose[:3, :3]
        new_pos = current_pos + delta_action[:3]
        new_rot = current_rot @ euler_to_mat(delta_action[3:6])
        new_quat = mat_to_quat(new_rot)
        # grasp step
        grasp_range_float = (self.grasping_state - self.non_grasping_state)
        ctrl_grasp = jp.clip(
            action[6:] * grasp_range_float, self.grasping_state, self.non_grasping_state
        )
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self.mocap_controller, :].set(
                new_pos
            ),
            mocap_quat=data.mocap_quat.at[self.mocap_controller, :].set(
                new_quat
            ),
        )
        return mjx_env.step(self._mjx_model, data, ctrl_grasp, n_substeps)




    @property
    @abc.abstractmethod
    def xml_path(self) -> str:
        """Path to the XML file for the gripper."""
        pass

    @property
    def action_size(self) -> int:
        """Size of the action space for the gripper."""
        return 7

    @property
    @abc.abstractmethod
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    @abc.abstractmethod
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
