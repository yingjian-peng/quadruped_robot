#include "lite3_test_policy_runner_onnx.h"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: lite3_policy_runner_smoke <policy.onnx>\n";
        return 2;
    }

    Lite3TestPolicyRunnerONNX runner("lite3_policy_smoke", argv[1]);
    runner.OnEnter();

    RobotBasicState state;
    state.base_omega = Vec3f::Zero();
    state.base_rot_mat = Mat3f::Identity();
    state.joint_pos = VecXf::Zero(12);
    state.joint_vel = VecXf::Zero(12);

    for (int step = 0; step < 40; ++step) {
        state.cmd_vel_normlized = step < 20 ? Vec3f(0.25f, 0.0f, 0.0f) : Vec3f::Zero();
        const RobotAction action = runner.GetRobotAction(state);
        if (action.goal_joint_pos.size() != 12 || !action.goal_joint_pos.allFinite() ||
            !action.goal_joint_vel.allFinite() || !action.kp.allFinite() || !action.kd.allFinite()) {
            std::cerr << "Invalid action at policy step " << step << "\n";
            return 1;
        }
    }

    runner.DisplayPolicyInfo();
    std::cout << "Policy runner smoke test passed for " << argv[1] << "\n";
    return 0;
}
