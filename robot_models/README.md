# robot_models

训练、Sim-to-Sim 与 Sim-to-Real 共享的 DeepRobotics LITE3 模型资产。

## 目录

```text
robot_models/
└── deeprobotics/
    └── Lite3/
        ├── Lite3_mjcf/
        │   ├── meshes/
        │   └── mjcf/
        ├── Lite3_urdf/
        │   ├── meshes/
        │   └── urdf/
        └── Lite3_usd/
            └── configuration/
```

## 格式用途

- `Lite3_usd/Lite3.usd`：Isaac Lab 训练直接加载的供应商 USD 资产。
- `Lite3_mjcf/mjcf/Lite3.xml`：MuJoCo Sim-to-Sim 和模型可视化使用的原生 MJCF。
- `Lite3_urdf/urdf/Lite3.urdf`：用于互操作、参数核对和重新转换的源模型。

三种格式必须保持关节命名、关节方向、默认姿态、限位和执行器顺序一致。修改模型后，应同时检查
Isaac Lab 中的站姿和 `rl_deploy/sim_to_sim/view_robot.py` 的 MuJoCo 显示结果。

## 版本控制

STL 与 USD 文件由仓库根目录的 Git LFS 规则管理。首次拉取或提交前执行：

```bash
git lfs install
git lfs ls-files
```

## 许可

公开发布前应补充模型来源、作者和许可证信息，并确认供应商资产的使用与再分发条件。
