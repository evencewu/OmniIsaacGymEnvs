import math

import numpy as np
import torch
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.balance import Balance

class BalanceTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:        
        self.update_config(sim_config)
        self._max_episode_length = 500

        self._num_observations = 10
        self._num_actions = 2

        RLTask.__init__(self, name, env)
        return

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._balance_positions = torch.tensor([0.0, 0.0, 2.0])

        self._max_push_effort = self._task_cfg["env"]["maxEffort"]

    def set_up_scene(self, scene) -> None:
        # first create a single environment
        self.get_balance()

        # call the parent class to clone the single environment
        super().set_up_scene(scene)

        # construct an ArticulationView object to hold our collection of environments
        self._balances = ArticulationView(
            prim_paths_expr="/World/envs/.*/Balance", name="balance_view", reset_xform_properties=False
        )

        # register the ArticulationView object to the world, so that it can be initialized
        scene.add(self._balances)

    def get_balance(self):
        # add a single robot to the stage
        balance = Balance(
            prim_path=self.default_zero_env_path + "/Balance", name="Balance", translation=self._balance_positions
        )

        # applies articulation settings from the task configuration yaml file
        self._sim_config.apply_articulation_settings(
            "Balance", get_prim_at_path(balance.prim_path), self._sim_config.parse_actor_config("Balance")
        )
    
    def post_reset(self):
        # retrieve cart and pole joint indices
        self._joint1_dof_idx = self._balances.get_dof_index("joint1")
        self._joint2_dof_idx = self._balances.get_dof_index("joint2")

        # randomize all envs
        indices = torch.arange(self._balances.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)
    
    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        # randomize DOF positions
        dof_pos = torch.zeros((num_resets, self._balances.num_dof), device=self._device)
        dof_pos[:, self._joint1_dof_idx] = 0.125 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        dof_pos[:, self._joint2_dof_idx] = 0.125 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # randomize DOF velocities
        dof_vel = torch.zeros((num_resets, self._balances.num_dof), device=self._device)
        dof_vel[:, self._joint1_dof_idx] = 0.25 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        dof_vel[:, self._joint2_dof_idx] = 0.25 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # apply randomized joint positions and velocities to environments
        indices = env_ids.to(dtype=torch.int32)
        self._balances.set_joint_positions(dof_pos, indices=indices)
        self._balances.set_joint_velocities(dof_vel, indices=indices)

        # reset the reset buffer and progress buffer after applying reset
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
    
    def pre_physics_step(self, actions) -> None:
        # make sure simulation has not been stopped from the UI
        if not self._env._world.is_playing():
            return

        # extract environment indices that need reset and reset them
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # make sure actions buffer is on the same device as the simulation
        actions = actions.to(self._device)

        # compute forces from the actions
        forces = torch.zeros((self._balances.count, self._balances.num_dof), dtype=torch.float32, device=self._device)
        forces[:, self._joint1_dof_idx] = self._max_push_effort * actions[:, 0]
        forces[:, self._joint2_dof_idx] = self._max_push_effort * actions[:, 1]

        # apply actions to all of the environments
        indices = torch.arange(self._balances.count, dtype=torch.int32, device=self._device)
        self._balances.set_joint_efforts(forces, indices=indices)

    def get_observations(self) -> dict:
        # retrieve joint positions and velocities
        dof_pos = self._balances.get_joint_positions(clone=False)
        dof_vel = self._balances.get_joint_velocities(clone=False)

        # extract joint states for the cart and pole joints
        joint1_pos = dof_pos[:, self._joint1_dof_idx]
        joint1_vel = dof_vel[:, self._joint1_dof_idx]
        joint2_pos = dof_pos[:, self._joint2_dof_idx]
        joint2_vel = dof_vel[:, self._joint2_dof_idx]

        # populate the observations buffer
        self.obs_buf[:, 0] = joint1_pos
        self.obs_buf[:, 1] = joint1_vel
        self.obs_buf[:, 2] = joint2_pos
        self.obs_buf[:, 3] = joint2_vel

        # construct the observations dictionary and return
        observations = {self._balances.name: {"obs_buf": self.obs_buf}}
        return observations

    def calculate_metrics(self) -> None:
        # use states from the observation buffer to compute reward
        joint1_pos = self.obs_buf[:, 0]
        joint1_vel = self.obs_buf[:, 1]
        joint2_pos = self.obs_buf[:, 2]
        joint2_vel = self.obs_buf[:, 3]

        # define the reward function based on pole angle and robot velocities
        reward = 1.0 - joint2_angle * joint2_angle - 0.01 * torch.abs(joint1_vel) - 0.5 * torch.abs(joint2_vel)
        # penalize the policy if the cart moves too far on the rail
        reward = torch.where(torch.abs(joint1_pos) > self._reset_dist, torch.ones_like(reward) * -2.0, reward)
        # penalize the policy if the pole moves beyond 90 degrees
        reward = torch.where(torch.abs(joint2_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)

        # assign rewards to the reward buffer
        self.rew_buf[:] = reward
    
    def is_done(self) -> None:
        #cart_pos = self.obs_buf[:, 0]
        #pole_pos = self.obs_buf[:, 2]

        # check for which conditions are met and mark the environments that satisfy the conditions
        #resets = torch.where(torch.abs(cart_pos) > self._reset_dist, 1, 0)
        #resets = torch.where(torch.abs(pole_pos) > math.pi / 2, 1, resets)
        #resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)

        # assign the resets to the reset buffer
        #self.reset_buf[:] = resets