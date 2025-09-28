import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path(
    "mujoco_playground/_src/manipulation/ego_sim/xmls/pick_scene.xml"
)
data = mujoco.MjData(model)
with mujoco.viewer.launch_passive(model, data) as viewer:
    while True:
        mujoco.mj_step(model, data)
        viewer.sync()
