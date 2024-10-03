import json
import os
import time
from typing import Optional, Union

import pandas as pd
import torch

from RL.MORL_baselines.morl_algorithm import MOAgent


class RandomAlgorithm(MOAgent):

    def __init__(
            self,
            n_states: int = None,
            n_rewards: int = None,
            scenarios: str = '',
            evaluations: int = None,
            experiment_name: str = "Random",
            seed: Optional[int] = None,
            device: Union[torch.device, str] = "auto",
    ):
        """Random algorithm.

        Args:
            experiment_name: The name of the experiment.
            seed: The seed for the random number generator.
            device: The device to use for training.
        """

        MOAgent.__init__(self, n_states=n_states, n_rewards=n_rewards, device=device, seed=seed)

        self.evaluations = evaluations
        self.experiment_name = experiment_name

        self.log_file_n = f"log_{int(time.time())}_{self.experiment_name}_{scenarios}.csv"
        print("log_file", self.log_file_n)
        self.header = ['episode', 'step', 'action', 'time_to_collision', 'route_completed_percentage',
                       'collision', 'collision_type', 'reward_ttc', 'reward_completion', 'reward_total',
                       'done', 'tick', 'system_time', 'game_time']
        self.df = pd.DataFrame(columns=self.header)
        self.all_episode_scenario = {}

    def get_config(self):
        return {
            "seed": self.seed,
        }

    def write_to_file(self, train_results):
        if len(train_results) == len(self.header):
            train_results_cpu = [value.cpu().numpy() if isinstance(value, torch.Tensor) else value for value in train_results]
            self.df = self.df.append(pd.Series(train_results_cpu, index=self.df.columns), ignore_index=True)
        else:
            print("the length of the train_results is not the same as the columns")
            print(train_results)

    def save_to_file(self, save_dir: str = "train_results/"):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.df.to_csv(save_dir + "/" + self.log_file_n, index=False)

    def update_all_episode_scenario(self, all_tick_scenario):
        self.all_episode_scenario.update(all_tick_scenario)

    def save_all_episode_scenario(self, save_dir: str = "train_results/"):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        with open(save_dir + "/" + self.log_file_n.split('.csv')[0] + "_all_episode_scenario.json", 'w') as f:
            json.dump(self.all_episode_scenario, f, indent=4)
