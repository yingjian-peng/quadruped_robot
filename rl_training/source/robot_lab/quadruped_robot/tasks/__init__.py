"""Register the migrated DeepRobotics and Unitree task configurations."""

from .manager_based.locomotion.velocity.config.quadruped import deeprobotics_lite3
from .manager_based.locomotion.velocity.config.quadruped import unitree_a1
from .manager_based.locomotion.velocity.config.quadruped import unitree_b2
from .manager_based.locomotion.velocity.config.quadruped import unitree_go2

__all__ = [
	"deeprobotics_lite3",
	"unitree_a1",
	"unitree_b2",
	"unitree_go2",
]
