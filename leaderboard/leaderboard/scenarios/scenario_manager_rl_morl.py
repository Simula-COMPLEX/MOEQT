#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the ScenarioManager implementations.
It must not be modified and is for reference only!
"""

from __future__ import print_function

import math
import random
import signal
import sys
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import py_trees
import carla
import torch
from sympy.solvers import solve
from sympy import Symbol

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import ChangeNoiseParameters
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog
from srunner.scenariomanager.traffic_events import TrafficEventType

from leaderboard.autoagents.agent_wrapper import AgentWrapper, AgentError
from leaderboard.envs.sensor_interface import SensorReceivedNoData
from leaderboard.utils.result_writer import ResultOutputProvider

from RL.MORL_baselines.weights import random_weights

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()


class ScenarioManager(object):
    """
    Basic scenario manager class. This class holds all functionality
    required to start, run and stop a scenario.

    The user must not modify this class.

    To use the ScenarioManager:
    1. Create an object via manager = ScenarioManager()
    2. Load a scenario via manager.load_scenario()
    3. Trigger the execution of the scenario manager.run_scenario()
       This function is designed to explicitly control start and end of
       the scenario execution
    4. If needed, cleanup with manager.stop_scenario()
    """

    def __init__(self, timeout, traffic_manager, debug_mode=False):
        """
        Setups up the parameters, which will be filled at load_scenario()
        """
        self.elapsed_time = None
        self.count_tick = None
        self.scenario = None
        self.scenario_tree = None
        self.scenario_class = None
        self.ego_vehicles = None
        self.other_actors = None

        self.pre_percentage_route_completed = 0

        self.traffic_manager = traffic_manager
        self._debug_mode = debug_mode
        self._agent = None
        self._running = False
        self._timestamp_last_run = 0.0
        self._timeout = float(timeout)

        # Used to detect if the simulation is down
        watchdog_timeout = max(5, self._timeout - 2)
        self._watchdog = Watchdog(watchdog_timeout)

        # Avoid the agent from freezing the simulation
        agent_timeout = watchdog_timeout - 1
        self._agent_watchdog = Watchdog(agent_timeout)

        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        self.end_game_time = None

        # Register the scenario tick as callback for the CARLA world
        # Use the callback_id inside the signal handler to allow external interrupts
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        self._running = False

    def cleanup(self):
        """
        Reset all parameters
        """
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        self.end_game_time = None

    def load_scenario(self, scenario, agent, rep_number):
        """
        Load a new scenario
        """

        GameTime.restart()
        self._agent = AgentWrapper(agent)
        self.scenario_class = scenario
        self.scenario = scenario.scenario
        self.scenario_tree = self.scenario.scenario_tree
        self.ego_vehicles = scenario.ego_vehicles
        self.other_actors = scenario.other_actors
        self.repetition_number = rep_number

        # To print the scenario tree uncomment the next line
        # py_trees.display.render_dot_tree(self.scenario_tree)

        self._agent.setup_sensors(self.ego_vehicles[0], self._debug_mode)

    def observe(self):
        ego_trans = self.ego_vehicles[0].get_transform()
        ego_velocity = self.ego_vehicles[0].get_velocity()
        ego_acc = self.ego_vehicles[0].get_acceleration()
        ego_av = self.ego_vehicles[0].get_angular_velocity()
        state = [ego_trans.location.x, ego_trans.location.y, ego_trans.location.z,
                 ego_trans.rotation.pitch, ego_trans.rotation.yaw, ego_trans.rotation.roll,
                 ego_velocity.x, ego_velocity.y, ego_velocity.z,
                 ego_acc.x, ego_acc.y, ego_acc.z,
                 ego_av.x, ego_av.y, ego_av.z
                 ]
        return state

    def scenario_tick_json(self, tick):
        tick_data = {f"tick_{tick}": {}}
        ego_trans = self.ego_vehicles[0].get_transform()
        ego_velocity = self.ego_vehicles[0].get_velocity()
        ego_acc = self.ego_vehicles[0].get_acceleration()
        ego_av = self.ego_vehicles[0].get_angular_velocity()
        tick_data[f"tick_{tick}"]["Ego"] = {
            "location": {
                "x": ego_trans.location.x,
                "y": ego_trans.location.y,
                "z": ego_trans.location.z
            },
            "rotation": {
                "pitch": ego_trans.rotation.pitch,
                "yaw": ego_trans.rotation.yaw,
                "roll": ego_trans.rotation.roll
            },
            "velocity": {
                "x": ego_velocity.x,
                "y": ego_velocity.y,
                "z": ego_velocity.z
            },
            "acceleration": {
                "x": ego_acc.x,
                "y": ego_acc.y,
                "z": ego_acc.z
            },
            "angular_velocity": {
                "x": ego_av.x,
                "y": ego_av.y,
                "z": ego_av.z
            }
        }
        npc_num = 0
        for id, actor in CarlaDataProvider.get_actors():
            if actor.id == self.ego_vehicles[0].id:
                continue
            if 'vehicle' in actor.type_id:
                trans = actor.get_transform()
                velocity = actor.get_velocity()
                acc = actor.get_acceleration()
                av = actor.get_angular_velocity()
                tick_data[f"tick_{tick}"][f"NPC_{npc_num}"] = {
                    "id": actor.id,
                    "type": "vehicle",
                    "location": {
                        "x": trans.location.x,
                        "y": trans.location.y,
                        "z": trans.location.z
                    },
                    "rotation": {
                        "pitch": trans.rotation.pitch,
                        "yaw": trans.rotation.yaw,
                        "roll": trans.rotation.roll
                    },
                    "velocity": {
                        "x": velocity.x,
                        "y": velocity.y,
                        "z": velocity.z
                    },
                    "acceleration": {
                        "x": acc.x,
                        "y": acc.y,
                        "z": acc.z
                    },
                    "angular_velocity": {
                        "x": av.x,
                        "y": av.y,
                        "z": av.z
                    }
                }
            elif 'walker' in actor.type_id:
                trans = actor.get_transform()
                velocity = actor.get_velocity()
                tick_data[f"tick_{tick}"][f"NPC_{npc_num}"] = {
                    "id": actor.id,
                    "type": "pedestrian",
                    "location": {
                        "x": trans.location.x,
                        "y": trans.location.y,
                        "z": trans.location.z
                    },
                    "rotation": {
                        "pitch": trans.rotation.pitch,
                        "yaw": trans.rotation.yaw,
                        "roll": trans.rotation.roll
                    },
                    "velocity": {
                        "x": velocity.x,
                        "y": velocity.y,
                        "z": velocity.z
                    }
                }
            npc_num += 1

        return tick_data

    def run_scenario_envelope(self, scenario_id, agent, episode):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        print("envelope")
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        self._watchdog.start()
        self._running = True
        self.count_tick = 0
        self.pre_percentage_route_completed = 0

        step = 0
        self.elapsed_time = 0
        total_reward = 0

        state = self.observe()
        w = random_weights(agent.reward_dim, 1, dist="gaussian", rng=agent.np_random)
        tensor_w = torch.tensor(w).float().to(agent.device)

        while self._running:
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                # print('select action ...', self.count_tick)
                if agent.global_step < agent.learning_starts:
                    action = int(np.random.choice(agent.action_dim, 1)[0])
                else:
                    action = agent.act(torch.as_tensor(state).float().to(agent.device), tensor_w)
                action_ = agent.action_space[action]
                if action < agent.v_actions:
                    self.spawn_npc(action_[0], action_[1], action_[2], color='44, 44, 40')
                else:
                    self.spawn_pedestrian(action_[0], action_[1], action_[2], action_[3], action_[4])

                min_ttc_npc = 1000
                for i in range(40):
                    world = CarlaDataProvider.get_world()
                    if world:
                        snapshot = world.get_snapshot()
                        if snapshot:
                            timestamp = snapshot.timestamp
                    self.elapsed_time = timestamp.elapsed_seconds
                    dis_npc, ttc_npc = self._tick_scenario(timestamp, scenario_id)
                    min_ttc_npc = min(ttc_npc, min_ttc_npc)
                    if self.count_tick >= 240:
                        self._running = False
                    if not self._running:
                        break

                route_completed_percentage = 0
                for criterion in self.scenario.get_criteria():
                    if criterion.name == "RouteCompletionTest":
                        route_completed_percentage = criterion.percentage_route_completed - self.pre_percentage_route_completed
                        print("RouteCompletionTest", route_completed_percentage)
                        self.pre_percentage_route_completed = criterion.percentage_route_completed

                collision, collision_type = self.analyze_collision_step()
                new_state = self.observe()
                if collision:
                    reward_ttc = 1
                else:
                    reward_ttc = 1 / (1 + np.log(min_ttc_npc + 1))
                    reward_ttc = (reward_ttc - 1 / (1 + np.log(1000 + 1))) / (0.7 - 1 / (1 + np.log(1000 + 1)))
                reward_completion = 1 - route_completed_percentage / 45.6090385 if route_completed_percentage != 0 else 0
                reward_dict = {
                    "time_to_collision": reward_ttc,
                    "completion": reward_completion
                }
                reward = [reward_dict[objective] for objective in agent.multi_objective]
                done = collision or not self._running

                total_reward += reward_dict[agent.multi_objective[0]] * 0.5 + reward_dict[agent.multi_objective[-1]] * 0.5

                next_state = new_state

                agent.global_step += 1
                step += 1

                # Store the transition in memory
                agent.replay_buffer.add(state, action, reward, next_state, done)

                # Perform one step of the optimization (on the policy network)
                if agent.global_step >= agent.learning_starts and agent.replay_buffer.size >= agent.batch_size:
                    critic_losses = agent.update()
                else:
                    critic_losses = [100]

                end_system_time = time.time()
                end_game_time = GameTime.get_time()
                scenario_duration_system = end_system_time - self.start_system_time
                scenario_duration_game = end_game_time - self.start_game_time

                value_dict = {
                    "time_to_collision": min_ttc_npc,
                    "completion": route_completed_percentage
                }
                agent.write_to_file([episode, step - 1, action, value_dict[agent.multi_objective[0]],
                                     value_dict[agent.multi_objective[-1]], collision, collision_type,
                                     reward_dict[agent.multi_objective[0]], reward_dict[agent.multi_objective[-1]],
                                     total_reward, critic_losses[0], done, self.count_tick, scenario_duration_system,
                                     scenario_duration_game])

                if done:
                    self._running = False
                else:
                    # Move to the next state
                    state = next_state

        print('number of ticks: ', self.count_tick)
        print('timestamp.elapsed_seconds', self.elapsed_time)

        return total_reward

    def run_scenario_envelope_eval(self, scenario_id, agent, episode):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        print("envelope eval")
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        self._watchdog.start()
        self._running = True
        self.count_tick = 0
        self.pre_percentage_route_completed = 0

        step = 0
        self.elapsed_time = 0
        total_reward = 0

        all_tick_scenario = {f"episode_{episode}": {}}

        state = self.observe()
        w = random_weights(agent.reward_dim, 1, dist="gaussian", rng=agent.np_random)
        tensor_w = torch.tensor(w).float().to(agent.device)

        while self._running:
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                # print('select action ...', self.count_tick)
                action = agent.act(torch.as_tensor(state).float().to(agent.device), tensor_w)
                action_ = agent.action_space[action]
                if action < agent.v_actions:
                    self.spawn_npc(action_[0], action_[1], action_[2], color='44, 44, 40')
                else:
                    self.spawn_pedestrian(action_[0], action_[1], action_[2], action_[3], action_[4])

                all_tick_scenario[f"episode_{episode}"].update(self.scenario_tick_json(self.count_tick))

                min_ttc_npc = 1000
                for i in range(40):
                    world = CarlaDataProvider.get_world()
                    if world:
                        snapshot = world.get_snapshot()
                        if snapshot:
                            timestamp = snapshot.timestamp
                    self.elapsed_time = timestamp.elapsed_seconds
                    dis_npc, ttc_npc = self._tick_scenario(timestamp, scenario_id)
                    min_ttc_npc = min(ttc_npc, min_ttc_npc)
                    if self.count_tick % 2 == 0:
                        all_tick_scenario[f"episode_{episode}"].update(self.scenario_tick_json(self.count_tick))
                    if self.count_tick >= 240:
                        self._running = False
                    if not self._running:
                        break

                route_completed_percentage = 0
                for criterion in self.scenario.get_criteria():
                    if criterion.name == "RouteCompletionTest":
                        route_completed_percentage = criterion.percentage_route_completed - self.pre_percentage_route_completed
                        print("RouteCompletionTest", route_completed_percentage)
                        self.pre_percentage_route_completed = criterion.percentage_route_completed

                collision, collision_type = self.analyze_collision_step()
                new_state = self.observe()
                if collision:
                    reward_ttc = 1
                else:
                    reward_ttc = 1 / (1 + np.log(min_ttc_npc + 1))
                    reward_ttc = (reward_ttc - 1 / (1 + np.log(1000 + 1))) / (0.7 - 1 / (1 + np.log(1000 + 1)))
                reward_completion = 1 - route_completed_percentage / 45.6090385 if route_completed_percentage != 0 else 0
                reward_dict = {
                    "time_to_collision": reward_ttc,
                    "completion": reward_completion
                }
                reward = [reward_dict[objective] for objective in agent.multi_objective]
                done = collision or not self._running

                total_reward += reward_dict[agent.multi_objective[0]] * 0.5 + reward_dict[agent.multi_objective[-1]] * 0.5

                next_state = new_state

                step += 1

                end_system_time = time.time()
                end_game_time = GameTime.get_time()
                scenario_duration_system = end_system_time - self.start_system_time
                scenario_duration_game = end_game_time - self.start_game_time

                value_dict = {
                    "time_to_collision": min_ttc_npc,
                    "completion": route_completed_percentage
                }
                agent.write_to_file([episode, step - 1, action, value_dict[agent.multi_objective[0]],
                                     value_dict[agent.multi_objective[-1]], collision, collision_type,
                                     reward_dict[agent.multi_objective[0]], reward_dict[agent.multi_objective[-1]],
                                     total_reward, None, done, self.count_tick, scenario_duration_system,
                                     scenario_duration_game])

                if done:
                    self._running = False
                else:
                    # Move to the next state
                    state = next_state

        agent.update_all_episode_scenario(all_tick_scenario)
        print('number of ticks: ', self.count_tick)
        print('timestamp.elapsed_seconds', self.elapsed_time)

        return total_reward

    def run_scenario_deepcollision(self, scenario_id, agent, episode):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        print("single")
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        self._watchdog.start()
        self._running = True
        self.count_tick = 0
        self.pre_percentage_route_completed = 0

        step = 0
        self.elapsed_time = 0
        total_reward = 0

        state = self.observe()
        state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)

        while self._running:
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                # print('select action ...', self.count_tick)
                if agent.global_step < agent.learning_starts:
                    action = torch.tensor([[random.randrange(agent.action_dim)]], device=agent.device, dtype=torch.long)
                else:
                    action = agent.act(state)
                action_ = agent.action_space[action]
                if action < agent.v_actions:
                    self.spawn_npc(action_[0], action_[1], action_[2], color='44, 44, 40')
                else:
                    self.spawn_pedestrian(action_[0], action_[1], action_[2], action_[3], action_[4])

                min_ttc_npc = 1000
                for i in range(40):
                    world = CarlaDataProvider.get_world()
                    if world:
                        snapshot = world.get_snapshot()
                        if snapshot:
                            timestamp = snapshot.timestamp
                    self.elapsed_time = timestamp.elapsed_seconds
                    dis_npc, ttc_npc = self._tick_scenario(timestamp, scenario_id)
                    min_ttc_npc = min(ttc_npc, min_ttc_npc)
                    if self.count_tick >= 240:
                        self._running = False
                    if not self._running:
                        break

                route_completed_percentage = 0
                for criterion in self.scenario.get_criteria():
                    if criterion.name == "RouteCompletionTest":
                        route_completed_percentage = criterion.percentage_route_completed - self.pre_percentage_route_completed
                        print("RouteCompletionTest", route_completed_percentage)
                        self.pre_percentage_route_completed = criterion.percentage_route_completed

                collision, collision_type = self.analyze_collision_step()
                new_state = self.observe()
                if collision:
                    reward_ttc = 1
                else:
                    reward_ttc = 1 / (1 + np.log(min_ttc_npc + 1))
                    reward_ttc = (reward_ttc - 1 / (1 + np.log(1000 + 1))) / (0.7 - 1 / (1 + np.log(1000 + 1)))
                reward_completion = 1 - route_completed_percentage / 45.6090385 if route_completed_percentage != 0 else 0
                if agent.single_objective == 'time_to_collision':
                    reward = reward_ttc
                elif agent.single_objective == 'completion':
                    reward = reward_completion
                elif agent.single_objective == 'time_to_collision+completion':
                    reward = reward_completion * 0.5 + reward_ttc * 0.5
                done = collision or not self._running

                total_reward += reward

                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(new_state, dtype=torch.float32, device=agent.device).unsqueeze(0)

                agent.global_step += 1
                step += 1

                # Store the transition in memory
                agent.replay_buffer.push(state, action, next_state, torch.tensor([reward], device=agent.device))

                # Perform one step of the optimization (on the policy network)
                if agent.global_step >= agent.learning_starts:
                    critic_losses = agent.update()
                else:
                    critic_losses = [100]

                end_system_time = time.time()
                end_game_time = GameTime.get_time()
                scenario_duration_system = end_system_time - self.start_system_time
                scenario_duration_game = end_game_time - self.start_game_time

                agent.write_to_file([episode, step - 1, action, min_ttc_npc, route_completed_percentage,
                                     collision, collision_type, reward, reward_ttc, reward_completion,
                                     critic_losses[0],
                                     done, self.count_tick, scenario_duration_system, scenario_duration_game])

                if done:
                    self._running = False
                else:
                    # Move to the next state
                    state = next_state

        print('number of ticks: ', self.count_tick)
        print('timestamp.elapsed_seconds', self.elapsed_time)

        return total_reward

    def run_scenario_deepcollision_eval(self, scenario_id, agent, episode):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        print("single eval")
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        self._watchdog.start()
        self._running = True
        self.count_tick = 0
        self.pre_percentage_route_completed = 0

        step = 0
        self.elapsed_time = 0
        total_reward = 0

        all_tick_scenario = {f"episode_{episode}": {}}

        state = self.observe()
        state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)

        while self._running:
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                # print('select action ...', self.count_tick)
                action = agent.act(state)
                action_ = agent.action_space[action]
                if action < agent.v_actions:
                    self.spawn_npc(action_[0], action_[1], action_[2], color='44, 44, 40')
                else:
                    self.spawn_pedestrian(action_[0], action_[1], action_[2], action_[3], action_[4])

                all_tick_scenario[f"episode_{episode}"].update(self.scenario_tick_json(self.count_tick))

                min_ttc_npc = 1000
                for i in range(40):
                    world = CarlaDataProvider.get_world()
                    if world:
                        snapshot = world.get_snapshot()
                        if snapshot:
                            timestamp = snapshot.timestamp
                    self.elapsed_time = timestamp.elapsed_seconds
                    dis_npc, ttc_npc = self._tick_scenario(timestamp, scenario_id)
                    min_ttc_npc = min(ttc_npc, min_ttc_npc)
                    if self.count_tick % 2 == 0:
                        all_tick_scenario[f"episode_{episode}"].update(self.scenario_tick_json(self.count_tick))
                    if self.count_tick >= 240:
                        self._running = False
                    if not self._running:
                        break

                route_completed_percentage = 0
                for criterion in self.scenario.get_criteria():
                    if criterion.name == "RouteCompletionTest":
                        route_completed_percentage = criterion.percentage_route_completed - self.pre_percentage_route_completed
                        print("RouteCompletionTest", route_completed_percentage)
                        self.pre_percentage_route_completed = criterion.percentage_route_completed

                collision, collision_type = self.analyze_collision_step()
                new_state = self.observe()
                if collision:
                    reward_ttc = 1
                else:
                    reward_ttc = 1 / (1 + np.log(min_ttc_npc + 1))
                    reward_ttc = (reward_ttc - 1 / (1 + np.log(1000 + 1))) / (0.7 - 1 / (1 + np.log(1000 + 1)))
                reward_completion = 1 - route_completed_percentage / 45.6090385 if route_completed_percentage != 0 else 0
                if agent.single_objective == 'time_to_collision':
                    reward = reward_ttc
                elif agent.single_objective == 'completion':
                    reward = reward_completion
                elif agent.single_objective == 'time_to_collision+completion':
                    reward = reward_completion * 0.5 + reward_ttc * 0.5
                done = collision or not self._running

                total_reward += reward

                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(new_state, dtype=torch.float32, device=agent.device).unsqueeze(0)

                step += 1

                end_system_time = time.time()
                end_game_time = GameTime.get_time()
                scenario_duration_system = end_system_time - self.start_system_time
                scenario_duration_game = end_game_time - self.start_game_time

                agent.write_to_file([episode, step - 1, action, min_ttc_npc, route_completed_percentage,
                                     collision, collision_type, reward, reward_ttc, reward_completion,
                                     None,
                                     done, self.count_tick, scenario_duration_system, scenario_duration_game])

                if done:
                    self._running = False
                else:
                    # Move to the next state
                    state = next_state

        agent.update_all_episode_scenario(all_tick_scenario)
        print('number of ticks: ', self.count_tick)
        print('timestamp.elapsed_seconds', self.elapsed_time)

        return total_reward

    def run_scenario_random(self, scenario_id, agent, episode):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        print("random")
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        self._watchdog.start()
        self._running = True
        self.count_tick = 0
        self.pre_percentage_route_completed = 0

        step = 0
        self.elapsed_time = 0
        total_reward = 0

        all_tick_scenario = {f"episode_{episode}": {}}

        while self._running:
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            if timestamp:
                # print('select action ...', self.count_tick)
                action = int(np.random.choice(agent.action_dim, 1)[0])
                action_ = agent.action_space[action]
                if action < agent.v_actions:
                    self.spawn_npc(action_[0], action_[1], action_[2], color='44, 44, 40')
                else:
                    self.spawn_pedestrian(action_[0], action_[1], action_[2], action_[3], action_[4])

                all_tick_scenario[f"episode_{episode}"].update(self.scenario_tick_json(self.count_tick))

                min_ttc_npc = 1000
                for i in range(40):
                    world = CarlaDataProvider.get_world()
                    if world:
                        snapshot = world.get_snapshot()
                        if snapshot:
                            timestamp = snapshot.timestamp
                    self.elapsed_time = timestamp.elapsed_seconds
                    dis_npc, ttc_npc = self._tick_scenario(timestamp, scenario_id)
                    min_ttc_npc = min(ttc_npc, min_ttc_npc)
                    if self.count_tick % 2 == 0:
                        all_tick_scenario[f"episode_{episode}"].update(self.scenario_tick_json(self.count_tick))
                    if self.count_tick >= 240:
                        self._running = False
                    if not self._running:
                        break

                route_completed_percentage = 0
                for criterion in self.scenario.get_criteria():
                    if criterion.name == "RouteCompletionTest":
                        route_completed_percentage = criterion.percentage_route_completed - self.pre_percentage_route_completed
                        print("RouteCompletionTest", route_completed_percentage)
                        self.pre_percentage_route_completed = criterion.percentage_route_completed

                collision, collision_type = self.analyze_collision_step()
                if collision:
                    reward_ttc = 1
                else:
                    reward_ttc = 1 / (1 + np.log(min_ttc_npc + 1))
                    reward_ttc = (reward_ttc - 1 / (1 + np.log(1000 + 1))) / (0.7 - 1 / (1 + np.log(1000 + 1)))
                reward_completion = 1 - route_completed_percentage / 45.6090385 if route_completed_percentage != 0 else 0
                done = collision or not self._running

                total_reward += reward_ttc * 0.5 + reward_completion * 0.5

                step += 1

                end_system_time = time.time()
                end_game_time = GameTime.get_time()
                scenario_duration_system = end_system_time - self.start_system_time
                scenario_duration_game = end_game_time - self.start_game_time

                agent.write_to_file(
                    [episode, step - 1, action, min_ttc_npc, route_completed_percentage,
                     collision, collision_type, reward_ttc, reward_completion, total_reward,
                     done, self.count_tick, scenario_duration_system, scenario_duration_game])

                if done:
                    self._running = False

        agent.update_all_episode_scenario(all_tick_scenario)
        print('number of ticks: ', self.count_tick)
        print('timestamp.elapsed_seconds', self.elapsed_time)

        return None

    def _tick_scenario(self, timestamp, scenario_id):
        """
        Run next tick of scenario and the agent and tick the world.
        """
        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds
            self.count_tick += 1

            self._watchdog.update()
            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()

            try:
                agent_out = self._agent()
                ego_action = agent_out[0]

            # Special exception inside the agent that isn't caused by the agent
            except SensorReceivedNoData as e:
                raise RuntimeError(e)

            except Exception as e:
                raise AgentError(e)

            if 50 < self.count_tick < 100 and scenario_id in ['scenario_6']:
                ego_action = self.change_control(ego_action)

            self.ego_vehicles[0].apply_control(ego_action)

            # judge_stop_walker
            actors = CarlaDataProvider.get_actors()
            for actor_id, actor in actors:
                if 'walker' in actor.type_id:
                    self.judge_stop_walker(actor)

            # Tick scenario
            self.scenario_tree.tick_once()

            if self._debug_mode:
                print("\n")
                py_trees.display.print_ascii_tree(
                    self.scenario_tree, show_status=True)
                sys.stdout.flush()

            if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                self._running = False

            spectator = CarlaDataProvider.get_world().get_spectator()
            ego_trans = self.ego_vehicles[0].get_transform()
            spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=40),
                                                    carla.Rotation(pitch=-90)))

        if self._running and self.get_running_status():
            CarlaDataProvider.get_world().tick(self._timeout)

        min_dis_npc = 50
        min_ttc_npc = 1000
        if self.count_tick % 2 == 0:
            try:
                # min_dis_npc = self.calculate_distance()
                min_ttc_npc = self.calculate_TTC()
            except:
                min_dis_npc = 50
                min_ttc_npc = 1000

        return min_dis_npc, min_ttc_npc
        # return min_ttc_npc

    def calculate_acceleration(self):
        acceleration_squared = self.ego_vehicles[0].get_acceleration().x ** 2
        acceleration_squared += self.ego_vehicles[0].get_acceleration().y ** 2
        return math.sqrt(acceleration_squared)

    def calculate_speed(self, velocity):
        return math.sqrt(velocity.x ** 2 + velocity.y ** 2)

    def get_distance(self, actor, x, y):
        return math.sqrt((actor.get_location().x - x) ** 2 + (actor.get_location().y - y) ** 2)

    def get_line_y_x(self, actor):
        actor_location_x = actor.get_location().x
        actor_location_y = actor.get_location().y

        actor_velocity_x = actor.get_velocity().x if actor.get_velocity().x > 0.01 else 0.0
        # actor_velocity_y = actor.get_velocity().y if actor.get_velocity().y != 0 else 0.0001
        actor_velocity_y = 0.0001 if actor.get_velocity().y == 0.0 else actor.get_velocity().y if actor.get_velocity().y > 0.01 else 0.0001

        return actor_velocity_x / actor_velocity_y, actor_location_x - (actor_velocity_x / actor_velocity_y) * actor_location_y

    def judge_same_line_y_x(self, actor1, actor2, k1, k2):
        judge = False
        direction_vector = (actor1.get_location().y - actor2.get_location().y,
                            actor1.get_location().x - actor2.get_location().x)
        distance = self.get_distance(actor1, actor2.get_location().x, actor2.get_location().y)

        if abs(k1 - k2) < 0.2:
            if abs((actor1.get_location().x - actor2.get_location().x) /
                   ((actor1.get_location().y - actor2.get_location().y) if (actor1.get_location().y - actor2.get_location().y) != 0 else 0.0001)
                   - (k1 + k2) / 2) < 0.05:
                judge = True

        if not judge:
            return judge, 100000

        actor1_velocity = actor1.get_velocity()
        actor2_velocity = actor2.get_velocity()
        actor1_speed = self.calculate_speed(actor1_velocity)
        actor2_speed = self.calculate_speed(actor2_velocity)
        if direction_vector[0] * actor1_velocity.y >= 0 and direction_vector[1] * actor1_velocity.x >= 0:
            TTC = distance / ((actor2_speed - actor1_speed) if (actor2_speed - actor1_speed) != 0 else 0.0001)
        else:
            TTC = distance / ((actor1_speed - actor2_speed) if (actor1_speed - actor2_speed) != 0 else 0.0001)
        if TTC < 0:
            TTC = 100000

        return judge, TTC

    def calculate_TTC(self):
        trajectory_ego_k, trajectory_ego_b = self.get_line_y_x(self.ego_vehicles[0])
        ego_speed = self.calculate_speed(self.ego_vehicles[0].get_velocity())
        ego_speed = ego_speed if ego_speed > 0.01 else 0.01

        actors = CarlaDataProvider.get_actors()
        TTC = 100000

        for actor_id, actor in actors:
            if actor.id == self.ego_vehicles[0].id:
                continue
            trajectory_actor_k, trajectory_actor_b = self.get_line_y_x(actor)
            actor_speed = self.calculate_speed(actor.get_velocity())
            actor_speed = actor_speed if actor_speed > 0.01 else 0.01

            same_lane, ttc = self.judge_same_line_y_x(self.ego_vehicles[0], actor, trajectory_ego_k, trajectory_actor_k)
            if same_lane:
                TTC = min(TTC, ttc)
            else:
                trajectory_ego_k = trajectory_ego_k if trajectory_ego_k != 0 else trajectory_ego_k + 0.0001
                trajectory_actor_k = trajectory_actor_k if trajectory_actor_k != 0 else trajectory_actor_k + 0.0001
                trajectory_actor_k = trajectory_actor_k if trajectory_ego_k - trajectory_actor_k != 0 else trajectory_actor_k + 0.0001

                collision_point_y = (trajectory_actor_b - trajectory_ego_b) / (trajectory_ego_k - trajectory_actor_k)
                collision_point_x = ((trajectory_ego_k * trajectory_actor_b - trajectory_actor_k * trajectory_ego_b) /
                                     (trajectory_ego_k - trajectory_actor_k))

                ego_distance = self.get_distance(self.ego_vehicles[0], collision_point_x, collision_point_y)
                actor_distance = self.get_distance(actor, collision_point_x, collision_point_y)
                time_ego = ego_distance / ego_speed
                time_actor = actor_distance / actor_speed
                if ego_speed == 0.01 and actor_speed == 0.01:
                    TTC = min(TTC, 100000)
                else:
                    if abs(time_ego - time_actor) < 1:
                        TTC = min(TTC, (time_ego + time_actor) / 2)

        return TTC

    @staticmethod
    def judge_stop_walker(walker):
        walker_control = walker.get_control()
        if walker_control.speed != 0:
            actors = CarlaDataProvider.get_actors()
            for actor_id, actor in actors:
                if 'vehicle' in actor.type_id:
                    dis = math.sqrt(
                        (actor.get_location().x - walker.get_location().x) ** 2 + (
                                actor.get_location().y - walker.get_location().y) ** 2)
                    # print('walker distance ...', dis)
                    if dis < 5.0:
                        # print('stop walker ...')
                        walker_control.speed = 0  # {0.94,1.43} https://www.fhwa.dot.gov/publications/research/safety/pedbike/05085/chapt8.cfm
                        walker_control.jump = False
                        walker.apply_control(walker_control)
                        break

    def calculate_distance(self):
        actors = CarlaDataProvider.get_actors()
        ego_location = self.ego_vehicles[0].get_location()
        min_dis_npc = 1000

        for actor_id, actor in actors:
            if actor.id == self.ego_vehicles[0].id:
                continue
            actor_location = actor.get_location()
            dis = math.sqrt(
                (ego_location.x - actor_location.x) ** 2 + (ego_location.y - actor_location.y) ** 2)
            min_dis_npc = min(dis, min_dis_npc)

        return min_dis_npc

    @staticmethod
    def distance_constraints(wp):
        for actor in CarlaDataProvider.get_world().get_actors():
            dis = math.sqrt(
                (actor.get_transform().location.x - wp.location.x) ** 2 + (
                        actor.get_transform().location.y - wp.location.y) ** 2)
            if dis < 10:
                return False
            else:
                continue
        return True

    def spawn_npc(self, vertical: float, horizontal: float, behavior: float, color: str):
        ego_transform = CarlaDataProvider.get_hero_actor().get_transform()

        waypoint_t = carla.Transform(ego_transform.location +
                                     vertical * ego_transform.get_forward_vector() +
                                     horizontal * ego_transform.get_right_vector(),
                                     ego_transform.rotation)
        waypoint = CarlaDataProvider.get_world().get_map().get_waypoint(waypoint_t.location, project_to_road=True,
                                                                        lane_type=carla.LaneType.Driving)

        npc_vehicle = None
        while npc_vehicle is None:
            npc_transform = carla.Transform(carla.Location(waypoint.transform.location.x,
                                                           waypoint.transform.location.y, ego_transform.location.z),
                                            waypoint.transform.rotation)

            try_spawn = self.distance_constraints(npc_transform)

            if try_spawn:
                npc_vehicle = CarlaDataProvider.request_new_actor(
                    'vehicle.tesla.model3', npc_transform, rolename='scenario',
                    color=color
                )

            try:
                waypoint = waypoint.next(0.5)[0]
            except:
                return npc_vehicle

        if 0 <= behavior < 0.1:
            pass
        else:
            self.traffic_manager.force_lane_change(npc_vehicle, 0.1 <= behavior < 0.2)
        CarlaDataProvider.get_world().tick()

        npc_vehicle.set_autopilot(True, self.traffic_manager.get_port())
        return npc_vehicle

    def spawn_pedestrian(self, vertical, horizontal, direction_x, direction_y, speed):
        ego_transform = CarlaDataProvider.get_hero_actor().get_transform()

        waypoint_t = carla.Transform(ego_transform.location +
                                     vertical * ego_transform.get_forward_vector() +
                                     horizontal * ego_transform.get_right_vector(),
                                     ego_transform.rotation)
        # carla.LaneType.Any = 0xFFFFFFFE
        waypoint = CarlaDataProvider.get_world().get_map().get_waypoint(waypoint_t.location, project_to_road=True,
                                                                        lane_type=0xFFFFFFFE)

        walker = None
        while walker is None:
            walker_transform = carla.Transform(carla.Location(waypoint.transform.location.x,
                                                              waypoint.transform.location.y, ego_transform.location.z),
                                               waypoint.transform.rotation)

            try_spawn = self.distance_constraints(walker_transform)

            if try_spawn:
                walker = CarlaDataProvider.request_new_actor(
                    'walker.pedestrian.0001', walker_transform, rolename='scenario'
                )
            try:
                waypoint = waypoint.next(0.5)[0]
            except:
                return walker

        walker_control = carla.WalkerControl()
        if abs(ego_transform.get_right_vector().x) > 0.5:
            if ego_transform.get_right_vector().x * -1 * direction_x > 0:
                direction_x = 1
            elif ego_transform.get_right_vector().x * -1 * direction_x < 0:
                direction_x = -1
            else:
                direction_x = 0
            if ego_transform.get_forward_vector().y * direction_y > 0:
                direction_y = 1
            elif ego_transform.get_forward_vector().y * direction_y < 0:
                direction_y = -1
            else:
                direction_y = 0
            walker_control.direction = carla.Vector3D(direction_x, direction_y, 0)
        else:
            if ego_transform.get_right_vector().y * -1 * direction_x > 0:
                direction_x = 1
            elif ego_transform.get_right_vector().y * -1 * direction_x < 0:
                direction_x = -1
            else:
                direction_x = 0
            if ego_transform.get_forward_vector().x * direction_y > 0:
                direction_y = 1
            elif ego_transform.get_forward_vector().x * direction_y < 0:
                direction_y = -1
            else:
                direction_y = 0
            walker_control.direction = carla.Vector3D(direction_y, direction_x, 0)
        walker_control.speed = speed  # {0.94,1.43} https://www.fhwa.dot.gov/publications/research/safety/pedbike/05085/chapt8.cfm
        walker_control.jump = False

        walker.apply_control(walker_control)
        return walker

    @staticmethod
    def generate_noise():
        _noise_mean = 0  # Mean value of steering noise
        _noise_std = 0.01  # Std. deviation of steering noise
        _dynamic_mean_for_steer = 0.001
        _dynamic_mean_for_throttle = 0.045
        _abort_distance_to_intersection = 10
        _current_steer_noise = [0]  # This is a list, since lists are mutable
        _current_throttle_noise = [0]
        turn = ChangeNoiseParameters(_current_steer_noise, _current_throttle_noise,
                                     _noise_mean, _noise_std, _dynamic_mean_for_steer,
                                     _dynamic_mean_for_throttle)  # Mean value of steering noise

        turn.update()

        # print(turn._new_steer_noise[0])
        # print(turn._new_throttle_noise[0])
        return turn

    def change_control(self, control):
        """
        This is a function that changes the control based on the scenario determination
        :param control: a carla vehicle control
        :return: a control to be changed by the scenario.
        """
        turn = self.generate_noise()
        control.steer += turn._new_steer_noise[0]
        control.throttle += turn._new_throttle_noise[0]

        return control

    def get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        """
        return self._watchdog.get_status()

    def stop_scenario(self):
        """
        This function triggers a proper termination of a scenario
        """
        self._watchdog.stop()

        self.end_system_time = time.time()
        self.end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = self.end_game_time - self.start_game_time

        if self.get_running_status():
            if self.scenario is not None:
                self.scenario.terminate()

            if self._agent is not None:
                self._agent.cleanup()
                self._agent = None

            self.analyze_scenario()

    def analyze_scenario(self):
        """
        Analyzes and prints the results of the route
        """
        global_result = '\033[92m' + 'SUCCESS' + '\033[0m'

        for criterion in self.scenario.get_criteria():
            if criterion.test_status != "SUCCESS":
                global_result = '\033[91m' + 'FAILURE' + '\033[0m'

        if self.scenario.timeout_node.timeout:
            global_result = '\033[91m' + 'FAILURE' + '\033[0m'

        ResultOutputProvider(self, global_result)

    def analyze_collision_step(self):
        collision = False
        collision_type = None
        for node in self.scenario.get_criteria():
            if node.list_traffic_events:
                for event in node.list_traffic_events:
                    if event.get_type() == TrafficEventType.COLLISION_STATIC:
                        collision = True
                        collision_type = 'static'
                    elif event.get_type() == TrafficEventType.COLLISION_PEDESTRIAN:
                        collision = True
                        collision_type = 'pedestrian'
                    elif event.get_type() == TrafficEventType.COLLISION_VEHICLE:
                        collision = True
                        collision_type = 'vehicle'
        return collision, collision_type
