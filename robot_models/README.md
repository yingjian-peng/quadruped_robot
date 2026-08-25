# robot_models

训练与部署共享的机器人模型资产。

## Layout

```text
robot_models/
├── b2w/
├── b2w_z1/
├── go2_arx/
└── go2w/
```

每个机器人目录整体保存 URDF、网格和生成的 USD，避免破坏 URDF 中的相对网格路径。

## USD conversion

B2W、Go2W 和 B2W-Z1 的 USD 可通过训练侧转换工具生成。请在仓库根目录运行：

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  rl_training/scripts/tools/convert_wheel_quadruped_urdf.py b2w --headless

TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  rl_training/scripts/tools/convert_wheel_quadruped_urdf.py go2w --headless

TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  rl_training/scripts/tools/convert_wheel_quadruped_urdf.py b2w_z1 --headless
```

转换工具默认不会覆盖已有 USD；只有明确传入 `--force` 才会替换目标文件。

## Version control

DAE、STL、USD 和 OBJ 文件（包括大小写扩展名）由仓库根目录的 Git LFS 规则管理。首次提交前应确认 `git lfs ls-files` 能列出这些资产。

## Licensing

在公开发布前，应为每组模型补充来源、作者和许可证信息。第三方模型的使用与再分发条件可能不同于本仓库代码。
