from pathlib import Path

import mujoco


model_root = Path(__file__).resolve().parents[2] / "robot_models/deeprobotics/Lite3"
urdf_path = model_root / "Lite3_urdf/urdf/Lite3.urdf"

# 转换结果仅供检查，不能覆盖用于仿真部署的官方 Lite3.xml。
output_path = model_root / "Lite3_mjcf/mjcf/Lite3_converted.xml"
model = mujoco.MjModel.from_xml_path(str(urdf_path))
mujoco.mj_saveLastXML(str(output_path), model)
print(f"Saved MJCF to {output_path}")
