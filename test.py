# add result.txt
from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
from environment import Environment
from utils import setup_logger
from model import A3Clstm
from player_util import Agent
from torch.autograd import Variable
import time
import logging
from Config import Config
import sys

if sys.version_info >= (3, 0):
    from queue import Queue as queueQueue
else:
    from Queue import Queue as queueQueue


def test(args, shared_model, env_conf):
    ptitle('Test Agent')
    gpu_id = args.gpu_ids[-1]
    log = {}
    setup_logger('{}_log'.format(args.env), r'{0}{1}_log'.format(
        args.log_dir, args.env))
    log['{}_log'.format(args.env)] = logging.getLogger('{}_log'.format(
        args.env))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))

    torch.manual_seed(args.seed)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed)
    env = Environment(Config.SHOW_MODE)  # (True) or False
    reward_sum = 0
    start_time = time.time()
    num_tests = 0
    reward_total_sum = 0
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id
    num_actions = env.get_num_actions()

    player.model = A3Clstm(Config.STACKED_FRAMES,
                           num_actions)

    player.state, available = player.env.reset()
    # player.eps_len += 1
    player.state = torch.from_numpy(player.state).float()
    player.available = torch.from_numpy(available).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.model = player.model.cuda()
            player.state = player.state.cuda()
            player.available = player.available.cuda()
    flag = True
    max_score = 0
    results_logger = open(Config.RESULTS_FILENAME, 'a')
    rolling_frame_count = 0
    rolling_reward = 0
    results_q = queueQueue(maxsize=Config.STAT_ROLLING_MEAN_WINDOW)
    while True:
        if flag:  # first load state
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.model.load_state_dict(shared_model.state_dict())
            else:
                player.model.load_state_dict(shared_model.state_dict())
            player.model.eval()
            flag = False

        player.action_test()
        reward_sum += player.reward

        if player.done and not player.info:
            state, available = player.env.reset()
            player.state = torch.from_numpy(state).float()
            player.available = torch.from_numpy(available).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
                    player.available = player.available.cuda()
        elif player.info:
            flag = True
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests

            rolling_frame_count += player.eps_len
            rolling_reward += reward_sum

            if results_q.full():
                old_length, old_reward = results_q.get()
                rolling_frame_count -= old_length
                rolling_reward -= old_reward
            results_q.put((player.eps_len, reward_sum))

            episode_time = int(time.time() - start_time)
            log['{}_log'.format(args.env)].info(
                "Time {0:10d}, episode {1}, reward {2}, Step {3}, reward mean {4:.4f}, Rstep {5:.4f}, Rreward {6:.4f}".format(
                    episode_time, num_tests,
                    reward_sum, player.eps_len, reward_mean, (rolling_frame_count / results_q.qsize()),
                    (rolling_reward / results_q.qsize())))
            results_logger.write('%d, %d, %10.4f, %d, %10.4f, %10.4f\n' % (
            episode_time, num_tests, reward_sum, player.eps_len, player.envs_mean, player.envs_std))
            results_logger.flush()

            if args.save_max and reward_sum >= max_score:
                max_score = reward_sum
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        state_to_save = player.model.state_dict()
                        torch.save(state_to_save, '{0}{1}.dat'.format(
                            args.save_model_dir, args.env))
                else:
                    state_to_save = player.model.state_dict()
                    torch.save(state_to_save, '{0}{1}.dat'.format(
                        args.save_model_dir, args.env))

            reward_sum = 0
            player.eps_len = 0
            state, available = player.env.reset()
            time.sleep(1)
            player.state = torch.from_numpy(state).float()
            player.available = torch.from_numpy(available).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
                    player.available = player.available.cuda()
    results_logger.close()
