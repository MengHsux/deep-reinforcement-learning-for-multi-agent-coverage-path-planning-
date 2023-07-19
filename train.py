from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
import torch.optim as optim
from environment import Environment
from utils import ensure_shared_grads
from model import A3Clstm
from player_util import Agent
from torch.autograd import Variable
from Config import Config


def train(rank, args, shared_model, optimizer, env_conf):
    ptitle('Training Agent: {}'.format(rank))
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        torch.cuda.manual_seed(args.seed + rank)
    env = Environment()  # creat env
    if optimizer is None:
        if args.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(shared_model.parameters(), lr=args.lr)
        if args.optimizer == 'Adam':
            optimizer = optim.Adam(
                shared_model.parameters(), lr=args.lr, amsgrad=args.amsgrad)
    # env.seed(args.seed + rank)
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id
    num_actions = env.get_num_actions()

    player.model = A3Clstm(Config.STACKED_FRAMES,
                           num_actions)

    player.state, available = player.env.reset()
    player.state = torch.from_numpy(player.state).float()
    player.available = torch.from_numpy(available).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.state = player.state.cuda()
            player.model = player.model.cuda()
            player.available = player.available.cuda()
    player.model.train()
    player.eps_len += 1
    while True:
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                player.model.load_state_dict(shared_model.state_dict())
        else:
            player.model.load_state_dict(shared_model.state_dict())
        if player.done:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.cx = Variable(torch.zeros(1, 512).cuda())
                    player.hx = Variable(torch.zeros(1, 512).cuda())
            else:
                player.cx = Variable(torch.zeros(1, 512))
                player.hx = Variable(torch.zeros(1, 512))
        else:
            player.cx = Variable(player.cx.data)
            player.hx = Variable(player.hx.data)

        for step in range(args.num_steps):
            player.action_train()
            if player.done:
                break

        if player.done:
            state, available = player.env.reset()
            player.state = torch.from_numpy(state).float()
            player.available = torch.from_numpy(available).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
                    player.available = player.available.cuda()

        R = torch.zeros(1, 1)
        if not player.done:
            value, _, _, _, _, _ = player.model((Variable(player.state.unsqueeze(0)),
                                                 (player.hx, player.cx)))
            R = value.data

        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                R = R.cuda()

        player.values.append(Variable(R))
        policy_loss = 0
        policy_loss1 = 0
        policy_loss2 = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        if gpu_id >= 0:
            with torch.cuda.device(gpu_id):
                gae = gae.cuda()
        R = Variable(R)
        for i in reversed(range(len(player.rewards))):
            R = args.gamma * R + player.rewards[i] + player.rewards1[i] + player.rewards2[i]
            advantage = R - player.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = player.rewards[i] + player.rewards1[i] + player.rewards2[i] + args.gamma * \
                      player.values[i + 1].data - player.values[i].data

            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                          player.log_probs[i] * \
                          Variable(gae) - 0.01 * player.entropies[i]

            policy_loss1 = policy_loss1 - \
                           player.log_probs1[i] * \
                           Variable(gae) - 0.01 * player.entropies1[i]

            policy_loss2 = policy_loss2 - \
                           player.log_probs2[i] * \
                           Variable(gae) - 0.01 * player.entropies2[i]

        player.model.zero_grad()
        (policy_loss + policy_loss1 + policy_loss2 + 0.5 * value_loss).backward()
        ensure_shared_grads(player.model, shared_model, gpu=gpu_id >= 0)
        optimizer.step()
        player.clear_actions()
