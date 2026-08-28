# robot_models

训练与部署共享的机器人模型资产。

## Layout

```text
robot_models/
├── deeprobotics/
│   ├── lite3_description/
│   └── m20_description/
└── unitree/
    ├── a1_description/
    ├── b2_description/
    ├── b2w_description/
    ├── g1_description/
    ├── go2_description/
    └── go2w_description/
```

每个机器人目录整体保存 URDF、网格和生成的 USD，避免破坏 URDF 中的相对网格路径。

## USD conversion

轮足机器人的 USD 可通过训练侧转换工具生成。请在仓库根目录运行：

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  rl_training/scripts/tools/convert_wheel_quadruped_urdf.py unitree_b2w --headless

TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  rl_training/scripts/tools/convert_wheel_quadruped_urdf.py unitree_go2w --headless

TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  rl_training/scripts/tools/convert_wheel_quadruped_urdf.py deeprobotics_m20 --headless
```

转换工具默认不会覆盖已有 USD；只有明确传入 `--force` 才会替换目标文件。

## Version control

DAE、STL、USD 和 OBJ 文件（包括大小写扩展名）由仓库根目录的 Git LFS 规则管理。首次提交前应确认 `git lfs ls-files` 能列出这些资产。

## Licensing

在公开发布前，应为每组模型补充来源、作者和许可证信息。第三方模型的使用与再分发条件可能不同于本仓库代码。
