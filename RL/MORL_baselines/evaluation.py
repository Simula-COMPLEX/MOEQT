import os
import pickle
import re

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as stats


def create_floder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_id(data_, episode):
    end_id = -1
    for k in range(len(data_)):
        if int(data_['episode'][k]) == episode:
            end_id = k
            break
    if end_id == -1:
        end_id = len(data_)
    return end_id


def converge_random(data, file_name):
    episode_start = 0
    episode_end = data['episode'][len(data) - 1]
    episode = episode_start
    ttc, comp, rew_ttc, rew_comp, collision = [], [], [], [], []
    ttc_episode, comp_episode, reward_ttc, reward_comp, coll = [], [], [], [], []
    col_comp_num = 0
    both_ttc, both_comp = [], []
    for i in range(data[data['episode'] == episode_start].index.min(), data[data['episode'] == episode_end].index.max() + 2):
        if len(data) == i or data['episode'][i] != episode:
            ttc.append(np.array(ttc_episode).mean())
            comp.append(np.array(comp_episode).mean())
            if data['episode'][i - 1] >= episode_end - 99:
                if data['collision'][i - 1] == True:
                    if "scenario_5" in file_name:
                        if abs(sum(comp_episode) - 90) > 0.1:
                            col_comp_num += 1
                            both_ttc.append(np.array(ttc_episode).mean())
                            both_comp.append(sum(comp_episode))
                    else:
                        if abs(sum(comp_episode) - 100) > 0.1:
                            col_comp_num += 1
                            both_ttc.append(np.array(ttc_episode).mean())
                            both_comp.append(sum(comp_episode))
            rew_ttc.append(np.array(reward_ttc).mean())
            rew_comp.append(np.array(reward_comp).mean())
            collision.append(max(np.array(coll)))
            if len(data) == i:
                break
            episode = data['episode'][i]
            ttc_episode, comp_episode, reward_ttc, reward_comp, coll = [], [], [], [], []
        if data['step'][i] == 0:
            continue
        if data['collision'][i] == True:
            col_data = 1
            ttc_data = 0
        else:
            col_data = 0
            if data['time_to_collision'][i] > 20:
                ttc_data = 20
            else:
                ttc_data = data['time_to_collision'][i]
        ttc_episode.append(ttc_data)
        comp_episode.append(data['route_completed_percentage'][i])
        reward_ttc.append(data['reward_ttc'][i])
        reward_comp.append(data['reward_completion'][i])
        coll.append(col_data)

    create_floder(f'{folder_path}/{file_name}')

    x = np.arange(episode_start, episode_end + 1, 1)

    plt.figure(1)
    plt.title(f'Time to Collision Convergence Trend')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.plot(x, ttc, label='Original', color='#9abbda')
    plt.plot(x, tensorboard_smoothing(ttc), label='EWMA Smoothed', color='#4f70a4')
    plt.legend(loc='upper right')
    plt.savefig(f'{folder_path}/{file_name}/ttc.png', format="png")
    plt.clf()

    plt.figure(1)
    plt.title(f'Route Completed Percentage Convergence Trend')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.plot(x, comp, label='Original', color='#9abbda')
    plt.plot(x, tensorboard_smoothing(comp), label='EWMA Smoothed', color='#4f70a4')
    plt.legend(loc='upper right')
    plt.savefig(f'{folder_path}/{file_name}/comp.png', format="png")
    plt.clf()

    plt.figure(1)
    plt.title(f'Time to Collision Reward Convergence Trend')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.plot(x, rew_ttc, label='Original', color='#9abbda')
    plt.plot(x, tensorboard_smoothing(rew_ttc), label='EWMA Smoothed', color='#4f70a4')
    plt.legend(loc='upper right')
    plt.savefig(f'{folder_path}/{file_name}/rew_ttc.png', format="png")
    plt.clf()

    plt.figure(1)
    plt.title(f'Route Completed Percentage Reward Convergence Trend')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.plot(x, rew_comp, label='Original', color='#9abbda')
    plt.plot(x, tensorboard_smoothing(rew_comp), label='EWMA Smoothed', color='#4f70a4')
    plt.legend(loc='upper right')
    plt.savefig(f'{folder_path}/{file_name}/rew_comp.png', format="png")
    plt.clf()

    plt.figure(1)
    plt.title(f'Collision Convergence Trend')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.plot(x, collision, color='#4f70a4')
    plt.savefig(f'{folder_path}/{file_name}/collision.png', format="png")
    plt.clf()

    with open(f'{folder_path}/{file_name}/collision.txt', 'w') as file:
        file.write(f"col_comp_num num: {col_comp_num}\n")
        file.write(f"{len(both_ttc)} both ttc average: {None if col_comp_num == 0 else sum(both_ttc) / len(both_ttc)}\n")
        file.write(f"{len(both_comp)} both comp average: {None if col_comp_num == 0 else sum(both_comp) / len(both_comp)}\n")


def converge_deepcollision(data, file_name):
    episode_start = 0
    episode_end = data['episode'][len(data)-1]
    episode = episode_start
    loss, ttc, comp, rew_ttc, rew_comp, rew_tol, collision = [], [], [], [], [], [], []
    loss_episode, ttc_episode, comp_episode, reward_ttc, reward_comp, reward_tol, coll = [], [], [], [], [], [], []
    col_comp_num = 0
    both_ttc, both_comp = [], []
    for i in range(data[data['episode'] == episode_start].index.min(),
                   data[data['episode'] == episode_end].index.max() + 2):
        if len(data) == i or data['episode'][i] != episode:
            loss.append(np.array(loss_episode).mean())
            ttc.append(np.array(ttc_episode).mean())
            comp.append(np.array(comp_episode).mean())
            rew_ttc.append(np.array(reward_ttc).mean())
            rew_comp.append(np.array(reward_comp).mean())
            rew_tol.append(np.array(reward_tol).mean())
            if data['episode'][i - 1] >= episode_end - 99:
                action_list, action_episode = [], []
                if data['collision'][i - 1] == True:
                    if "scenario_5" in file_name:
                        if abs(sum(comp_episode) - 90) > 0.1:
                            col_comp_num += 1
                            both_ttc.append(np.array(ttc_episode).mean())
                            both_comp.append(sum(comp_episode))
                    else:
                        if abs(sum(comp_episode) - 100) > 0.1:
                            col_comp_num += 1
                            both_ttc.append(np.array(ttc_episode).mean())
                            both_comp.append(sum(comp_episode))
            collision.append(max(np.array(coll)))
            if len(data) == i:
                break
            episode = data['episode'][i]
            loss_episode, ttc_episode, comp_episode, reward_ttc, reward_comp, reward_tol, coll = [], [], [], [], [], [], []
        if data['step'][i] == 0:
            continue
        if data['collision'][i] == True:
            col_data = 1
            ttc_data = 0
        else:
            col_data = 0
            if data['time_to_collision'][i] > 20:
                ttc_data = 20
            else:
                ttc_data = data['time_to_collision'][i]
        loss_episode.append(data['loss'][i])
        ttc_episode.append(ttc_data)
        comp_episode.append(data['completion'][i])
        reward_ttc.append(data[f'reward_time_to_collision'][i])
        reward_comp.append(data[f'reward_completion'][i])
        reward_tol.append(data[f'reward_merged'][i])
        coll.append(col_data)

    create_floder(f'{folder_path}/{file_name}')

    x = np.arange(episode_start, episode_end + 1, 1)

    plt.figure(1)
    plt.title(f'Loss Convergence Trend')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.plot(x, loss, label='Original', color='#9abbda')
    plt.plot(x, tensorboard_smoothing(loss), label='EWMA Smoothed', color='#4f70a4')
    plt.legend(loc='upper right')
    plt.savefig(f'{folder_path}/{file_name}/loss.png', format="png")
    plt.clf()

    plt.figure(1)
    plt.title(f'Time To Collision Convergence Trend')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.plot(x, ttc, label='Original', color='#9abbda')
    plt.plot(x, tensorboard_smoothing(ttc), label='EWMA Smoothed', color='#4f70a4')
    plt.legend(loc='upper right')
    plt.savefig(f'{folder_path}/{file_name}/ttc.png', format="png")
    plt.clf()

    plt.figure(1)
    plt.title(f'Completion Convergence Trend')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.plot(x, comp, label='Original', color='#9abbda')
    plt.plot(x, tensorboard_smoothing(comp), label='EWMA Smoothed', color='#4f70a4')
    plt.legend(loc='upper right')
    plt.savefig(f'{folder_path}/{file_name}/comp.png', format="png")
    plt.clf()

    plt.figure(1)
    plt.title(f'Time To Collision Reward Convergence Trend')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.plot(x, rew_ttc, label='Original', color='#9abbda')
    plt.plot(x, tensorboard_smoothing(rew_ttc), label='EWMA Smoothed', color='#4f70a4')
    plt.legend(loc='upper right')
    plt.savefig(f'{folder_path}/{file_name}/rew_ttc.png', format="png")
    plt.clf()

    plt.figure(1)
    plt.title(f'Completion Reward Convergence Trend')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.plot(x, rew_comp, label='Original', color='#9abbda')
    plt.plot(x, tensorboard_smoothing(rew_comp), label='EWMA Smoothed', color='#4f70a4')
    plt.legend(loc='upper right')
    plt.savefig(f'{folder_path}/{file_name}/rew_comp.png', format="png")
    plt.clf()

    plt.figure(1)
    plt.title(f'Total Reward Convergence Trend')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.plot(x, rew_tol, label='Original', color='#9abbda')
    plt.plot(x, tensorboard_smoothing(rew_tol), label='EWMA Smoothed', color='#4f70a4')
    plt.legend(loc='upper right')
    plt.savefig(f'{folder_path}/{file_name}/rew_total.png', format="png")
    plt.clf()

    plt.figure(1)
    plt.title(f'Collision Convergence Trend')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.plot(x, collision, color='#4f70a4')
    plt.savefig(f'{folder_path}/{file_name}/collision.png', format="png")
    plt.clf()

    with open(f'{folder_path}/{file_name}/collision.txt', 'w') as file:
        file.write(f"col_comp_num num: {col_comp_num}\n")
        file.write(f"{len(both_ttc)} both ttc average: {None if col_comp_num == 0 else sum(both_ttc) / len(both_ttc)}\n")
        file.write(f"{len(both_comp)} both comp average: {None if col_comp_num == 0 else sum(both_comp) / len(both_comp)}\n")

    data_to_save = {
        'ttc': ttc,
        'comp': comp,
        'episode': x
    }
    with open(f'{folder_path}/{file_name}/3D.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)


def converge_envelope(data, file_name):
    episode_start = 0
    episode_end = data['episode'][len(data) - 1]
    episode = episode_start
    loss, obj_1, obj_2, rew_obj_1, rew_obj_2, rew_total, collision = [], [], [], [], [], [], []
    loss_episode, obj_1_episode, obj_2_episode, reward_obj_1, reward_obj_2, reward_total, coll = [], [], [], [], [], [], []
    col_comp_num = 0
    both_ttc, both_comp = [], []
    for i in range(data[data['episode'] == episode_start].index.min(),
                   data[data['episode'] == episode_end].index.max() + 2):
        if len(data) == i or data['episode'][i] != episode:
            loss.append(np.array(loss_episode).mean())
            obj_1.append(np.array(obj_1_episode).mean())
            obj_2.append(np.array(obj_2_episode).mean())
            rew_obj_1.append(np.array(reward_obj_1).mean())
            rew_obj_2.append(np.array(reward_obj_2).mean())
            if data.columns[4] == 'completion' and data['episode'][i - 1] >= episode_end - 99:
                if data['collision'][i - 1] == True:
                    if "scenario_5" in file_name:
                        if abs(sum(obj_2_episode) - 90) > 0.1:
                            col_comp_num += 1
                            both_ttc.append(np.array(obj_1_episode).mean())
                            both_comp.append(sum(obj_2_episode))
                    else:
                        if abs(sum(obj_2_episode) - 100) > 0.1:
                            col_comp_num += 1
                            both_ttc.append(np.array(obj_1_episode).mean())
                            both_comp.append(sum(obj_2_episode))
            rew_total.append(data['reward_total'][i - 1] / data['step'][i - 1])
            collision.append(max(np.array(coll)))
            if len(data) == i:
                break
            episode = data['episode'][i]
            loss_episode, obj_1_episode, obj_2_episode, reward_obj_1, reward_obj_2, reward_total, coll = [], [], [], [], [], [], []
        if data['step'][i] == 0:
            continue
        if data['collision'][i] == True:
            col_data = 1
            ttc_data = 0
        else:
            col_data = 0
            if 'time_to_collision' in data:
                if data['time_to_collision'][i] > 20:
                    ttc_data = 20
                else:
                    ttc_data = data['time_to_collision'][i]
        loss_episode.append(data['loss'][i])
        if data.columns[3] == 'time_to_collision':
            obj_1_episode.append(ttc_data)
            reward_obj_1.append(data['reward_time_to_collision'][i])
        elif data.columns[3] == 'completion':
            obj_1_episode.append(data['completion'][i])
            reward_obj_1.append(data['reward_completion'][i])
        if data.columns[4] == 'time_to_collision':
            obj_2_episode.append(ttc_data)
            reward_obj_2.append(data['reward_time_to_collision'][i])
        elif data.columns[4] == 'completion':
            obj_2_episode.append(data['completion'][i])
            reward_obj_2.append(data['reward_completion'][i])
        coll.append(col_data)

    create_floder(f'{folder_path}/{file_name}')

    x = np.arange(episode_start, episode_end + 1, 1)

    plt.figure(1)
    plt.title(f'Loss Convergence Trend')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.plot(x, loss, label='Original', color='#9abbda')
    plt.plot(x, tensorboard_smoothing(loss), label='EWMA Smoothed', color='#4f70a4')
    plt.legend(loc='upper right')
    plt.savefig(f'{folder_path}/{file_name}/loss.png', format="png")
    plt.clf()

    plt.figure(1)
    plt.title(f'{data.columns[3]} Convergence Trend')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.plot(x, obj_1, label='Original', color='#9abbda')
    plt.plot(x, tensorboard_smoothing(obj_1), label='EWMA Smoothed', color='#4f70a4')
    plt.legend(loc='upper right')
    plt.savefig(f'{folder_path}/{file_name}/{data.columns[3]}.png', format="png")
    plt.clf()

    plt.figure(1)
    plt.title(f'{data.columns[4]} Convergence Trend')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.plot(x, obj_2, label='Original', color='#9abbda')
    plt.plot(x, tensorboard_smoothing(obj_2), label='EWMA Smoothed', color='#4f70a4')
    plt.legend(loc='upper right')
    plt.savefig(f'{folder_path}/{file_name}/{data.columns[4]}.png', format="png")
    plt.clf()

    plt.figure(1)
    plt.title(f'{data.columns[3]} Reward Convergence Trend')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.plot(x, rew_obj_1, label='Original', color='#9abbda')
    plt.plot(x, tensorboard_smoothing(rew_obj_1), label='EWMA Smoothed', color='#4f70a4')
    plt.legend(loc='upper right')
    plt.savefig(f'{folder_path}/{file_name}/rew_{data.columns[3]}.png', format="png")
    plt.clf()

    plt.figure(1)
    plt.title(f'{data.columns[4]} Reward Convergence Trend')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.plot(x, rew_obj_2, label='Original', color='#9abbda')
    plt.plot(x, tensorboard_smoothing(rew_obj_2), label='EWMA Smoothed', color='#4f70a4')
    plt.legend(loc='upper right')
    plt.savefig(f'{folder_path}/{file_name}/rew_{data.columns[4]}.png', format="png")
    plt.clf()

    plt.figure(1)
    plt.title(f'Total Reward Convergence Trend')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.plot(x, rew_total, label='Original', color='#9abbda')
    plt.plot(x, tensorboard_smoothing(rew_total), label='EWMA Smoothed', color='#4f70a4')
    plt.legend(loc='upper right')
    plt.savefig(f'{folder_path}/{file_name}/rew_total.png', format="png")
    plt.clf()

    plt.figure(1)
    plt.title(f'Collision Convergence Trend')
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.plot(x, collision, color='#4f70a4')
    plt.savefig(f'{folder_path}/{file_name}/collision.png', format="png")
    plt.clf()

    with open(f'{folder_path}/{file_name}/collision.txt', 'w') as file:
        file.write(f"col_comp_num num: {col_comp_num}\n")
        file.write(f"{len(both_ttc)} both ttc average: {None if col_comp_num == 0 else sum(both_ttc) / len(both_ttc)}\n")
        file.write(f"{len(both_comp)} both comp average: {None if col_comp_num == 0 else sum(both_comp) / len(both_comp)}\n")

    data_to_save = {
        'ttc': obj_1,
        'comp': obj_2,
        'episode': x
    }
    with open(f'{folder_path}/{file_name}/3D.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)


def tensorboard_smoothing(x, smooth=0.99):
    x = x.copy()
    weight = smooth
    for i in range(1,len(x)):
        x[i] = (x[i-1] * weight + x[i]) / (weight + 1)
        weight = (weight + 1) * smooth
    return x


def both_statistic():
    scenario_list = ['scenario_1', 'scenario_2', 'scenario_3', 'scenario_4', 'scenario_5', 'scenario_6']

    with open(f'{folder_path}/fisher_exact.txt', 'w') as file:
        file.write("")

    for scenario in scenario_list:
        random_violation = None
        envelope_violation = None
        for entry in os.scandir(folder_path):
            if entry.is_dir() and 'Random' in entry.name and scenario in entry.name:
                file_path = os.path.join(entry.path, 'collision.txt')
                with open(file_path, 'r') as file:
                    for line in file:
                        match = re.search(r'col_comp_num num:\s*(\d+)', line)
                        if match:
                            random_violation = int(match.group(1))
            if entry.is_dir() and 'Envelope' in entry.name and scenario in entry.name:
                file_path = os.path.join(entry.path, 'collision.txt')
                with open(file_path, 'r') as file:
                    for line in file:
                        match = re.search(r'col_comp_num num:\s*(\d+)', line)
                        if match:
                            envelope_violation = int(match.group(1))

        print("envelope_violation", envelope_violation, "random_violation", random_violation, scenario, "scenario")
        data = np.array([[envelope_violation, 100-envelope_violation], [random_violation, 100-random_violation]])
        oddsratio, p_fisher = stats.fisher_exact(data)
        if p_fisher < 0.01:
            p_fisher_sign = '<0.01'
        elif p_fisher < 0.05:
            p_fisher_sign = '<0.05'
        else:
            p_fisher_sign = '>=0.05'

        with open(f'{folder_path}/fisher_exact.txt', 'a') as file:
            file.write(f"{scenario}\n")
            file.write(f"envelope_violation {envelope_violation}\n")
            file.write(f"random_violation {random_violation}\n")
            file.write(f"fisher_exact {p_fisher} {p_fisher_sign} {oddsratio}\n\n")


def plot_3D():
    scenario_list = ['scenario_1', 'scenario_2', 'scenario_3', 'scenario_4', 'scenario_5', 'scenario_6']

    for scenario in scenario_list:
        for entry in os.scandir(folder_path):
            if entry.is_dir() and 'Single' in entry.name and scenario in entry.name:
                with open(os.path.join(entry.path, '3D.pkl'), 'rb') as f:
                    loaded_data = pickle.load(f)
                deep_ttc = loaded_data['ttc']
                deep_comp = loaded_data['comp']
                deep_episode = loaded_data['episode']
            if entry.is_dir() and 'Envelope' in entry.name and scenario in entry.name:
                with open(os.path.join(entry.path, '3D.pkl'), 'rb') as f:
                    loaded_data = pickle.load(f)
                envelope_ttc = loaded_data['ttc']
                envelope_comp = loaded_data['comp']
                envelope_episode = loaded_data['episode']

        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')
        if scenario == 'scenario_1':
            ax.plot(deep_episode, tensorboard_smoothing(deep_ttc), tensorboard_smoothing(deep_comp), color='#E76254')
            ax.plot(envelope_episode, tensorboard_smoothing(envelope_ttc), tensorboard_smoothing(envelope_comp),
                    label='MOEQT', color='#376795')
        elif scenario == 'scenario_2':
            ax.plot(deep_episode, tensorboard_smoothing(deep_ttc), tensorboard_smoothing(deep_comp), label='SORLW',
                    color='#E76254')
            ax.plot(envelope_episode, tensorboard_smoothing(envelope_ttc), tensorboard_smoothing(envelope_comp),
                    color='#376795')
        else:
            ax.plot(deep_episode, tensorboard_smoothing(deep_ttc), tensorboard_smoothing(deep_comp), color='#E76254')
            ax.plot(envelope_episode, tensorboard_smoothing(envelope_ttc), tensorboard_smoothing(envelope_comp), color='#376795')
        ax.set_ylim(ax.get_ylim()[::-1])
        if scenario == 'scenario_5':
            ax.set_xlabel('Episode', fontweight='bold', fontsize=14, labelpad=-1)
            ax.set_ylabel('TTC', fontweight='bold', fontstyle='italic', fontsize=14, labelpad=5)
            ax.set_zlabel('RC', fontweight='bold', fontstyle='italic', fontsize=14, labelpad=5.5)
        elif scenario == 'scenario_6':
            ax.set_xlabel('Episode', fontweight='bold', fontsize=14, labelpad=-2.5)
            ax.set_ylabel('TTC', fontweight='bold', fontstyle='italic', fontsize=14, labelpad=0)
            ax.set_zlabel('RC', fontweight='bold', fontstyle='italic', fontsize=14, labelpad=6)
        else:
            ax.set_xlabel('Episode', fontweight='bold', fontsize=14, labelpad=-2.5)
            ax.set_ylabel('TTC', fontweight='bold', fontstyle='italic', fontsize=14, labelpad=0)
            ax.set_zlabel('RC', fontweight='bold', fontstyle='italic', fontsize=14, labelpad=-2)

        if scenario == 'scenario_5':
            ax.tick_params(axis='x', labelsize=10, width=2, pad=-3)
            ax.tick_params(axis='y', labelsize=10, width=2, pad=-1)
            ax.tick_params(axis='z', labelsize=10, width=2, pad=2)
        elif scenario == 'scenario_6':
            ax.tick_params(axis='x', labelsize=10, width=2, pad=-4)
            ax.tick_params(axis='y', labelsize=10, width=2, pad=-4.5)
            ax.tick_params(axis='z', labelsize=10, width=2, pad=2)
        else:
            ax.tick_params(axis='x', labelsize=10, width=2, pad=-4)
            ax.tick_params(axis='y', labelsize=10, width=2, pad=-4.5)
            ax.tick_params(axis='z', labelsize=10, width=2, pad=-1.5)
        for tick in ax.xaxis.get_ticklabels():
            tick.set_fontweight('bold')
        for tick in ax.yaxis.get_ticklabels():
            tick.set_fontweight('bold')
        for tick in ax.zaxis.get_ticklabels():
            tick.set_fontweight('bold')

        ax.w_xaxis._axinfo['grid']['linestyle'] = '-'
        ax.w_yaxis._axinfo['grid']['linestyle'] = '-'
        ax.w_zaxis._axinfo['grid']['linestyle'] = '-'

        if scenario == 'scenario_1' or scenario == 'scenario_2':
            ax.legend(prop={'weight': 'bold', 'style': 'italic', 'size': 12})

        plt.savefig(f'{folder_path}/3D_{scenario}.pdf', format="pdf", bbox_inches='tight', pad_inches=0, transparent=True)
        plt.savefig(f'{folder_path}/3D_{scenario}.png', format="png", bbox_inches='tight', pad_inches=0, transparent=True)
        plt.clf()


if __name__ == '__main__':
    folder_path = './eval_results'  # train_results  eval_results
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    for file_name in csv_files:
        f_name = os.path.splitext(file_name)[0]
        fr_path = os.path.join(folder_path, f_name)
        # if os.path.isdir(fr_path):
        #     continue
        data = pd.read_csv(filepath_or_buffer=f'{folder_path}/{file_name}', sep=',')
        if 'Random' in file_name:
            continue
            # converge_random(data, file_name.split('.csv')[0])
        elif 'Single' in file_name:
            continue
            # converge_deepcollision(data, file_name.split('.csv')[0])
        elif 'Envelope' in file_name:
            continue
            # converge_envelope(data, file_name.split('.csv')[0])

    # both_statistic()
    # plot_3D()
