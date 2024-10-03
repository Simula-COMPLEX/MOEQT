import argparse
import json
import os
import optuna
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime

import gc

from optuna.trial import TrialState

sys.path.insert(0, '/MOEQT/scenario_runner')
sys.path.insert(0, '/MOEQT/leaderboard')
sys.path.insert(0, '/MOEQT/carla/PythonAPI/carla')
sys.path.insert(0, '/MOEQT')

from RL.MORL_baselines.envelope import Envelope
from RL.MORL_baselines.deepcollision import DeepCollision
from RL.MORL_baselines.random_algorithm import RandomAlgorithm

from leaderboard.leaderboard_evaluator_rl_morl import LeaderboardEvaluator

from leaderboard.utils.route_indexer import RouteIndexer
from leaderboard.utils.statistics_manager import StatisticsManager

PATH = "/MOEQT"

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class RLGeneration:
    def __init__(self, evaluator, args):
        """
        Problem Initialization.

        :param int n_variables: the problem size. The number of float numbers in the solution vector (list).
        """

        if not args.optuna:
            n_states = 15
            tau = 1.0
            target_net_update_freq = 30  # 1000,  # 500 reduce by gradient updates
            gradient_updates = 1
            gamma = 0.99

            objectives = args.objective.split('+')  # distance+time_to_collision
            multi_objective = [objectives[0], objectives[-1]]

            if args.algorithm == 'envelope':
                self.agent = Envelope(
                    n_states=n_states,
                    n_rewards=2,
                    multi_objective=multi_objective,
                    scenarios=args.scenarios.split('/')[-1].split('.')[0],
                    eval=args.eval,
                    evaluations=args.evaluations,
                    learning_rate=args.lr,
                    initial_epsilon=1.0,
                    final_epsilon=0.05,
                    epsilon_decay_steps=args.decay_steps,
                    tau=tau,
                    target_net_update_freq=target_net_update_freq,  # 1000,  # 500 reduce by gradient updates
                    buffer_size=args.memory,
                    net_arch=[17 * 16, 17 * 32, 17 * 64, 17 * 32],
                    drop_rate=args.drop_rate,
                    batch_size=args.batch_size,
                    learning_starts=args.learning_starts,
                    gradient_updates=gradient_updates,
                    gamma=gamma,
                    max_grad_norm=args.max_grad_norm,
                    envelope=True,
                    num_sample_w=args.num_sample_w,
                    initial_homotopy_lambda=0.0,
                    final_homotopy_lambda=1.0,
                    homotopy_decay_steps=args.decay_steps,
                )
            elif args.algorithm == 'single':
                self.agent = DeepCollision(
                    n_states=n_states,
                    single_objective=args.objective,
                    scenarios=args.scenarios.split('/')[-1].split('.')[0],
                    eval=args.eval,
                    evaluations=args.evaluations,
                    learning_rate=args.lr,
                    initial_epsilon=1.0,
                    final_epsilon=0.05,
                    epsilon_decay_steps=args.decay_steps,
                    tau=tau,
                    target_net_update_freq=target_net_update_freq,  # 1000,  # 500 reduce by gradient updates
                    buffer_size=args.memory,
                    net_arch=[32, 64, 128, 64],
                    drop_rate=args.drop_rate,
                    batch_size=args.batch_size,
                    learning_starts=args.learning_starts,
                    gradient_updates=gradient_updates,
                    gamma=gamma,
                    max_grad_norm=1.0,
                )
            elif args.algorithm == 'random':
                self.agent = RandomAlgorithm(
                    n_states=n_states,
                    scenarios=args.scenarios.split('/')[-1].split('.')[0],
                    evaluations=args.evaluations,
                )
            print("Configuration==============================================================")
            print(self.agent.get_config())
            print("Configuration==============================================================")

        self.evaluator = evaluator
        self.args = args
        self.set_route()

    def set_route(self):
        self.route_indexer = RouteIndexer(self.args.routes, self.args.scenarios, self.args.repetitions)

        if self.args.resume:
            self.route_indexer.resume(self.args.checkpoint)
            self.evaluator.statistics_manager.resume(self.args.checkpoint)
        else:
            self.evaluator.statistics_manager.clear_record(self.args.checkpoint)
            self.route_indexer.save_state(self.args.checkpoint)
        self.config = self.route_indexer.next()

    def train(self):
        for i in range(self.args.evaluations):
            self.evaluator._load_and_run_scenario(self.args, self.config, self.agent, episode=i)
            if i % 9 == 0:
                self.agent.save_to_file()
            if i == 0 or i == 999 or i == 999 + 50 or i == 999 + 100 or i == 999 + 150 or i == 999 + 200:
                self.agent.save(filename=f"{self.agent.log_file_n.split('.csv')[0]}_episode_{i+1}")
        self.agent.save_to_file()
        self.agent.save(filename=f"{self.agent.log_file_n.split('.csv')[0]}")
        gc.collect()

    def eval(self):
        if self.args.algorithm != 'random':
            all_files = os.listdir("eval_results/")
            for file in all_files:
                if '.tar' in file and self.args.scenario_id in file and file.split('_')[2].lower() == self.args.algorithm:
                    self.agent.load(path=f"eval_results/{file}")
                    print(f"load {self.args.algorithm} model: eval_results/{file}")
                    break
        for i in range(self.args.evaluations):
            self.evaluator._load_and_run_scenario(self.args, self.config, self.agent, episode=i)
            if i % 9 == 0:
                self.agent.save_to_file(save_dir="eval_results/")
        self.agent.save_to_file(save_dir="eval_results/")
        self.agent.save_all_episode_scenario(save_dir="eval_results/")
        gc.collect()

    def objective(self, trial, num_episodes):
        n_states = 15
        tau = 1.0
        target_net_update_freq = 30  # 1000,  # 500 reduce by gradient updates
        gradient_updates = 1
        gamma = 0.99
        objectives = self.args.objective.split('+')  # distance+time_to_collision
        multi_objective = [objectives[0], objectives[-1]]
        learning_rate = trial.suggest_categorical('learning_rate', [1e-4, 1e-3, 1e-2, 5e-4, 5e-3, 5e-2])
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
        learning_starts = trial.suggest_categorical("learning_starts",
                                                    [8 * 8, 16 * 4, 16 * 8, 32 * 4, 32 * 8, 64 * 4, 64 * 8])
        if self.args.algorithm == 'envelope':
            max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
            num_sample_w = trial.suggest_categorical("num_sample_w", [2, 4, 8, 16, 32, 64])
            self.agent = Envelope(
                n_states=n_states,
                n_rewards=2,
                multi_objective=multi_objective,
                evaluations=self.args.evaluations,
                learning_rate=learning_rate,
                initial_epsilon=1.0,
                final_epsilon=0.05,
                epsilon_decay_steps=self.args.decay_steps,
                tau=tau,
                target_net_update_freq=target_net_update_freq,  # 1000,  # 500 reduce by gradient updates
                buffer_size=int(1000),
                net_arch=[17 * 16, 17 * 32, 17 * 64, 17 * 32],
                batch_size=batch_size,
                learning_starts=learning_starts,
                gradient_updates=gradient_updates,
                gamma=gamma,
                max_grad_norm=max_grad_norm,
                envelope=True,
                num_sample_w=num_sample_w,
                initial_homotopy_lambda=0.0,
                final_homotopy_lambda=1.0,
                homotopy_decay_steps=self.args.decay_steps,
            )
        elif self.args.algorithm == 'single':
            epsilon_decay_steps = trial.suggest_categorical("epsilon_decay_steps", [800, 1000, 1200, 1500])
            self.agent = DeepCollision(
                n_states=n_states,
                single_objective=self.args.objective,
                evaluations=self.args.evaluations,
                learning_rate=learning_rate,
                initial_epsilon=1.0,
                final_epsilon=0.05,
                epsilon_decay_steps=epsilon_decay_steps,
                tau=tau,
                target_net_update_freq=target_net_update_freq,  # 1000,  # 500 reduce by gradient updates
                buffer_size=int(1000),
                net_arch=[32, 64, 128, 64],
                batch_size=batch_size,
                learning_starts=learning_starts,
                gradient_updates=gradient_updates,
                gamma=gamma,
                max_grad_norm=1.0,
            )
        total_reward = 0
        for i in range(num_episodes):
            total_reward += self.evaluator._load_and_run_scenario(self.args, self.config, self.agent, episode=i)
        avg_reward = total_reward / num_episodes
        return avg_reward


def main(args):
    """
    start
    """
    statistics_manager = StatisticsManager()

    L = 10

    try:
        leaderboard_evaluator = LeaderboardEvaluator(args, statistics_manager)

        rl_alg = RLGeneration(evaluator=leaderboard_evaluator, args=args)
        if args.optuna:
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: rl_alg.objective(trial, num_episodes=args.initial_num_episodes), n_trials=args.initial_n_trials)
            for trial in study.best_trials[:10]:
                study.optimize(lambda trial: rl_alg.objective(trial, num_episodes=args.final_num_episodes), n_trials=args.final_n_trials)
            pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
            complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
            print("Study statistics: ")
            print("  Number of finished trials: ", len(study.trials))
            print("  Number of pruned trials: ", len(pruned_trials))
            print("  Number of complete trials: ", len(complete_trials))
            print("Best trial:")
            trial = study.best_trial
            print("  Value: ", trial.value)
            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
            with open(f"{rl_alg.agent.log_file_n}_best_params.json", "w") as f:
                json.dump(study.best_trial.params, f, indent=4)
        elif args.eval:
            rl_alg.eval()
        else:
            rl_alg.train()

    except Exception as e:
        traceback.print_exc()
    finally:
        del leaderboard_evaluator


def str_to_bool(value):
    if value.lower() in ('true', '1', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'no'):
        return False


def int_none(value):
    if value.lower() == 'none':
        return None
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value for decay_steps: {value}")


if __name__ == '__main__':
    description = "CARLA AD Leaderboard Evaluation: evaluate your Agent in CARLA scenarios\n"

    # general parameters
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default='2000', required=False, help='TCP port to listen to (default: 2000)')
    parser.add_argument('--trafficManagerPort', required=False, default='8000',
                        help='Port to use for the TrafficManager (default: 8000)')
    parser.add_argument('--trafficManagerSeed', default='1',
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--carlaProviderSeed', default='2000',
                        help='Seed used by the CarlaProvider (default: 2000)')
    parser.add_argument('--debug', type=int, help='Run with debug output', default=0)
    parser.add_argument('--record', type=str, default='',
                        help='Use CARLA recording feature to create a recording of the scenario')
    parser.add_argument('--timeout', default="120.0",
                        help='Set the CARLA client timeout value in seconds')

    # simulation setup
    parser.add_argument('--routes',
                        default='{}/leaderboard/data/test_routes/scenario_4.xml'.format(PATH),
                        help='Name of the route to be executed. Point to the route_xml_file to be executed.',
                        # required=True
                        )
    parser.add_argument('--scenarios',
                        default='{}/leaderboard/data/test_routes/scenario_4.json'.format(PATH),
                        help='Name of the scenario annotation file to be mixed with the route.',
                        # required=True
                        )
    parser.add_argument('--repetitions',
                        type=int,
                        default=1,
                        help='Number of repetitions per route.')

    # agent-related options
    parser.add_argument("-a", "--agent",
                        default='{}/leaderboard/team_code/interfuser_agent.py'.format(PATH),
                        type=str, help="Path to Agent's py file to evaluate",
                        # required=True
                        )
    parser.add_argument("--agent-config",
                        default='{}/leaderboard/team_code/interfuser_config.py'.format(PATH),
                        type=str, help="Path to Agent's configuration file")

    parser.add_argument("--track", type=str, default='SENSORS', help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=bool, default=False, help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str,
                        default='./simulation_results.json',
                        # '{}/simulation_results_{}.json'.format(args.checkpoint, str(int(time.time())))
                        help="Path to checkpoint used for saving statistics and resuming")
    parser.add_argument("--run", type=int, default=0, required=False, help="Experiment repetition")
    parser.add_argument("--evaluations", type=int, default=1200, required=False, help="Evaluations")
    parser.add_argument("--scenario_id", type=str, default='scenario_4', required=False, help="Scenario ID")

    parser.add_argument("--algorithm", type=str, default='envelope', required=False, help="algorithm")
    parser.add_argument("--objective", type=str, default='time_to_collision+completion', required=False, help="objective")
    parser.add_argument("--optuna", type=str_to_bool, default=False, required=False, help="optuna")
    parser.add_argument("--eval", type=str_to_bool, default=False, required=False, help="eval")

    parser.add_argument("--initial_num_episodes", type=int, default=50, required=False, help="initial_num_episodes")
    parser.add_argument("--initial_n_trials", type=int, default=20, required=False, help="initial_n_trials")
    parser.add_argument("--final_num_episodes", type=int, default=200, required=False, help="final_num_episodes")
    parser.add_argument("--final_n_trials", type=int, default=10, required=False, help="final_n_trials")

    parser.add_argument("--lr", type=float, default=0.0001, required=False, help="lr")
    parser.add_argument("--batch_size", type=int, default=16, required=False, help="batch_size")
    parser.add_argument("--learning_starts", type=int, default=512, required=False, help="learning_starts")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, required=False, help="max_grad_norm")
    parser.add_argument("--num_sample_w", type=int, default=16, required=False, help="num_sample_w")
    parser.add_argument("--decay_steps", type=int_none, default=4000, required=False, help="decay_steps")
    parser.add_argument("--memory", type=int, default=2000, required=False, help="memory")
    parser.add_argument("--drop_rate", type=float, default=0.0, required=False, help="drop_rate")

    arguments = parser.parse_args()

    os.system('kill $(lsof -t -i:{})'.format(int(arguments.port)))
    os.system('kill $(lsof -t -i:{})'.format(int(arguments.trafficManagerPort)))
    time.sleep(10)
    subprocess.Popen(
        ['cd {}/carla/ && DISPLAY= ./CarlaUE4.sh --world-port={} -opengl'.format(PATH, int(arguments.port))],
        stdout=subprocess.PIPE, universal_newlines=True, shell=True)

    logs = f'''sbatch slurm
    --port={arguments.port}
    --trafficManagerPort={arguments.trafficManagerPort}
    --evaluations={arguments.evaluations}
    --algorithm={arguments.algorithm}
    --objective={arguments.objective}
    --optuna={arguments.optuna}
    --lr={arguments.lr}
    --batch_size={arguments.batch_size}
    --learning_starts={arguments.learning_starts}
    --max_grad_norm={arguments.max_grad_norm}
    --num_sample_w={arguments.num_sample_w}
    --decay_steps={arguments.decay_steps}
    --memory={arguments.memory}
    --drop_rate={arguments.drop_rate}
    --{arguments.scenarios.split('/')[-1].split('.')[0]}'''
    print('===================================')
    print(logs)
    print('===================================')
    f = open('./logs.md', mode='a', encoding='utf-8')
    f.writelines(logs + '\n')

    time.sleep(20)
    # try:
    main(arguments)
    # except:
    #     os.system('kill $(lsof -t -i:{})'.format(int(arguments.port)))
    #     os.system('kill $(lsof -t -i:{})'.format(int(arguments.trafficManagerPort)))
    # finally:
    #     os.system('kill $(lsof -t -i:{})'.format(int(arguments.port)))
    #     os.system('kill $(lsof -t -i:{})'.format(int(arguments.trafficManagerPort)))
