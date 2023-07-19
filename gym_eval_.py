from __future__ import division
import os

os.environ["OMP_NUM_THREADS"] = "1"
import argparse
import torch
from environment import Environment
from utils import read_config, setup_logger
from model import A3Clstm
from player_util import Agent
import gym
import logging
import time
from Config import Config
# from gym.configuration import undo_logger_setup

import matplotlib.pyplot as plt
import skimage.transform
import os
import cv2
import numpy as np

filename = "attention/"

Config.GIF = False  # True / False

BLUE = np.array((60, 150, 255))
RED_PROBE = np.array((230, 90, 80))
YELLOW = np.array((235, 226, 80))

if not os.path.exists(filename):
    os.makedirs(filename)

# undo_logger_setup()
parser = argparse.ArgumentParser(description='A3C_EVAL')
parser.add_argument(
    '--env',
    default='Wafer',
    metavar='ENV',
    help='environment to train on (default: Wafer)')
parser.add_argument(
    '--env-config',
    default='config.json',
    metavar='EC',
    help='environment to crop and resize info (default: config.json)')
parser.add_argument(
    '--num-episodes',
    type=int,
    default=10,
    metavar='NE',
    help='how many episodes in evaluation (default: 100)')
parser.add_argument(
    '--load-model-dir',
    default='trained_models/',
    metavar='LMD',
    help='folder to load trained models from')
parser.add_argument(
    '--log-dir', default='logs/', metavar='LG', help='folder to save logs')
parser.add_argument(
    '--render',
    default=False,
    metavar='R',
    help='Watch game as it being played')
parser.add_argument(
    '--render-freq',
    type=int,
    default=1,
    metavar='RF',
    help='Frequency to watch rendered game play')
parser.add_argument(
    '--max-episode-length',
    type=int,
    default=Config.TRAINING_STEPS,
    metavar='M',
    help='maximum length of an episode (default: Config.TRAINING_STEPS)')
parser.add_argument(
    '--gpu-id',
    type=int,
    default=-1,
    help='GPU to use [-1 CPU only] (default: -1)')
parser.add_argument(
    '--skip-rate',
    type=int,
    default=4,
    metavar='SR',
    help='frame skip rate (default: 4)')
parser.add_argument(
    '--seed',
    type=int,
    default=1,
    metavar='S',
    help='random seed (default: 1)')
parser.add_argument(
    '--new-gym-eval',
    default=False,
    metavar='NGE',
    help='Create a gym evaluation for upload')
args = parser.parse_args()

setup_json = read_config(args.env_config)
env_conf = setup_json["Default"]
for i in setup_json.keys():
    if i in args.env:
        env_conf = setup_json[i]

gpu_id = args.gpu_id

results_logger = open(Config.RESULTS_FILENAME, 'a')

torch.manual_seed(args.seed)
if gpu_id >= 0:
    torch.cuda.manual_seed(args.seed)

saved_state = torch.load(
    '{0}{1}.dat'.format(args.load_model_dir, args.env),
    map_location=lambda storage, loc: storage)

log = {}
setup_logger('{}_mon_log'.format(args.env), r'{0}{1}_mon_log'.format(
    args.log_dir, args.env))
log['{}_mon_log'.format(args.env)] = logging.getLogger('{}_mon_log'.format(
    args.env))

d_args = vars(args)
for k in d_args.keys():
    log['{}_mon_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

env = Environment(True)  # atari_env(True)
# env = atari_env("{}".format(args.env), env_conf, args)
num_tests = 0
start_time = time.time()
reward_total_sum = 0
player = Agent(None, env, args, None)

num_actions = env.get_num_actions()
player.model = A3Clstm(Config.STACKED_FRAMES,
                       num_actions)

player.gpu_id = gpu_id
if gpu_id >= 0:
    with torch.cuda.device(gpu_id):
        player.model = player.model.cuda()

if gpu_id >= 0:
    with torch.cuda.device(gpu_id):
        player.model.load_state_dict(saved_state)
else:
    player.model.load_state_dict(saved_state)

player.model.eval()
for i_episode in range(args.num_episodes):
    player.state, _ = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
    # player.eps_len += 1
    reward_sum = 0
    while True:
        player.action_test()
        reward_sum += player.reward

        if player.done and not player.info:
            state, _ = player.env.reset()
            player.eps_len += 1
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
        elif player.info:
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            episode_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
            log['{}_mon_log'.format(args.env)].info(
                "Time {0}, episode {1}, reward {2}, Step {3}, reward mean {4:.4f}".
                    format(
                    episode_time, num_tests,
                    reward_sum, player.eps_len, reward_mean))
            results_logger.write('%s, %d, %10.4f, %d, %10.4f, %10.4f\n' % (
            episode_time, num_tests, reward_sum, player.eps_len, player.envs_mean, player.envs_std))
            results_logger.flush()
            # print(player.action_all)
            # player.action_all.clear()

            if Config.GIF:
                for i in range(len(player.x)):
                    fig, (ax1, ax2) = plt.subplots(1, 2)
                    ax1.set_title('x_reshape')
                    # x_resize = skimage.transform.pyramid_expand(player.x[i], upscale=1, sigma=0)
                    x = cv2.cvtColor(player.x[i], cv2.COLOR_GRAY2RGB)
                    for h in range(16):
                        for w in range(16):
                            if (x[h, w, 0]) == 1:
                                x[h, w] = BLUE / 255
                            elif (x[h, w, 0]) > 0.8:
                                x[h, w] = RED_PROBE / 255
                            elif 0.67 > (x[h, w, 0]) > 0.66:
                                x[h, w] = YELLOW / 255
                    ax1.imshow(x)
                    ax2.set_title('alpha_reshape')
                    alpha_resize = cv2.cvtColor(player.alpha[i], cv2.COLOR_GRAY2RGB)
                    alpha_resize = skimage.transform.pyramid_expand(alpha_resize, upscale=8, sigma=0)
                    merge = x * 0.1 + alpha_resize * 0.9
                    merge = alpha_resize
                    ax2.imshow(merge, cmap='gray')
                    folder = filename + str(i_episode)
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    plt.savefig(folder + "/len_%02d" % (i,) + '.png')
                    plt.close('all')
                    print("saved:", i_episode, "-", i)

            time.sleep(1)
            player.eps_len = 0
            player.x = []
            player.alpha = []
            break
results_logger.close()
