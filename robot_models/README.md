# robot_models

训练与部署共享的机器人模型资产。

## Layout

```text
robot_models/
├── deeprobotics/
│   └── Lite3/
│       ├── Lite3_mjcf/
│       ├── Lite3_urdf/
│       └── Lite3_usd/
└── unitree/
    ├── a1_description/
    ├── b2_description/
    └── go2_description/
```

Lite3 保留供应商给出的三种原生格式：MuJoCo 使用 `Lite3_mjcf/mjcf/Lite3.xml`，
Isaac Lab 训练使用 `Lite3_usd/Lite3.usd`；`Lite3_urdf` 仅保留作互操作/重新转换的源文件。

## USD conversion

腿足机器人的 USD 可通过训练侧转换工具生成。请在仓库根目录运行：

```bash
TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  rl_training/scripts/tools/convert_wheel_quadruped_urdf.py unitree_b2 --headless

TERM=xterm /home/robot/isaacsim/IsaacLab/isaaclab.sh -p \
  rl_training/scripts/tools/convert_wheel_quadruped_urdf.py unitree_go2 --headless

```

转换工具默认不会覆盖已有 USD；只有明确传入 `--force` 才会替换目标文件。

## Version control

DAE、STL、USD 和 OBJ 文件（包括大小写扩展名）由仓库根目录的 Git LFS 规则管理。首次提交前应确认 `git lfs ls-files` 能列出这些资产。

## Licensing

在公开发布前，应为每组模型补充来源、作者和许可证信息。第三方模型的使用与再分发条件可能不同于本仓库代码。
