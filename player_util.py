from __future__ import division
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from Config import Config


class Agent(object):
    def __init__(self, model, env, args, state):
        self.model = model
        self.env = env
        self.state = state
        self.available = None
        self.hx = None
        self.cx = None
        self.eps_len = 0
        self.args = args
        self.values = []
        self.log_probs = []
        self.log_probs1 = []
        self.log_probs2 = []
        self.rewards = []
        self.rewards1 = []
        self.rewards2 = []
        self.entropies = []
        self.entropies1 = []
        self.entropies2 = []
        self.done = True
        self.info = None
        self.reward = 0
        self.reward1 = 0
        self.reward2 = 0
        self.gpu_id = -1
        self.alpha = []
        self.x = []
        self.envs_mean = None
        self.envs_std = None
        self.log_epsilon = Config.LOG_EPSILON
        # self.action_all = []

    def action_train(self):

        value, logit, logit1, logit2, (self.hx, self.cx), _ = self.model((Variable(
            self.state.unsqueeze(0)), (self.hx, self.cx)))

        prob = F.softmax(logit, dim=1)
        prob1 = F.softmax(logit1, dim=1)
        prob2 = F.softmax(logit2, dim=1)

        log_prob = F.log_softmax(prob, dim=1)
        log_prob1 = F.log_softmax(prob1, dim=1)
        log_prob2 = F.log_softmax(prob2, dim=1)

        entropy = -(log_prob * prob).sum(1)
        entropy1 = -(log_prob1 * prob1).sum(1)
        entropy2 = -(log_prob2 * prob2).sum(1)

        self.entropies.append(entropy)
        self.entropies1.append(entropy1)
        self.entropies2.append(entropy2)

        action = prob.multinomial(1).data  # choose action
        action1 = prob1.multinomial(1).data  # choose action
        action2 = prob2.multinomial(1).data  # choose action

        log_prob = log_prob.gather(1, Variable(action))
        log_prob1 = log_prob1.gather(1, Variable(action1))
        log_prob2 = log_prob2.gather(1, Variable(action2))

        state, self.reward, self.reward1, self.reward2, self.done, available, self.envs_mean, self.envs_std = self.env.step(
            action.cpu().numpy(), action1.cpu().numpy(), action2.cpu().numpy())

        self.info = self.envs_mean  # ##

        self.state = torch.from_numpy(state).float()
        self.available = torch.from_numpy(available).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
                self.available = self.available.cuda()

        self.reward = max(min(self.reward, 1), -1)  # limit reward
        self.reward1 = max(min(self.reward1, 1), -1)  # limit reward
        self.reward2 = max(min(self.reward2, 1), -1)  # limit reward

        self.values.append(value)

        self.log_probs.append(log_prob)
        self.log_probs1.append(log_prob1)
        self.log_probs2.append(log_prob2)

        self.rewards.append(self.reward)
        self.rewards1.append(self.reward1)
        self.rewards2.append(self.reward2)
        return self

    def action_test(self):
        with torch.no_grad():
            if self.done:
                if self.gpu_id >= 0:
                    with torch.cuda.device(self.gpu_id):
                        self.cx = Variable(
                            torch.zeros(1, 512).cuda())
                        self.hx = Variable(
                            torch.zeros(1, 512).cuda())
                else:
                    self.cx = Variable(torch.zeros(1, 512))
                    self.hx = Variable(torch.zeros(1, 512))
            else:
                self.cx = Variable(self.cx.data)
                self.hx = Variable(self.hx.data)
            value, logit, logit1, logit2, (self.hx, self.cx), alpha_reshape = self.model((Variable(
                self.state.unsqueeze(0)), (self.hx, self.cx)))
        if Config.GIF:
            self.alpha.append(alpha_reshape.data.cpu().numpy())
            self.x.append(self.state.view(22, 22).data.cpu().numpy())

        prob = F.softmax(logit, dim=1)
        prob1 = F.softmax(logit1, dim=1)
        prob2 = F.softmax(logit2, dim=1)

        #  choose random action
        action = prob.multinomial(1).data
        action1 = prob1.multinomial(1).data
        action2 = prob2.multinomial(1).data
        state, self.reward, self.reward1, self.reward2, self.done, available, self.envs_mean, self.envs_std = self.env.step(
            action.cpu().numpy(), action1.cpu().numpy(), action2.cpu().numpy())

        self.info = self.envs_mean  # print information

        self.state = torch.from_numpy(state).float()
        self.available = torch.from_numpy(available).float()
        if self.gpu_id >= 0:
            with torch.cuda.device(self.gpu_id):
                self.state = self.state.cuda()
                self.available = self.available.cuda()
        self.eps_len += 1
        return self

    def clear_actions(self):
        self.values = []

        self.log_probs = []
        self.rewards = []
        self.entropies = []

        self.log_probs1 = []
        self.rewards1 = []
        self.entropies1 = []

        self.log_probs2 = []
        self.rewards2 = []
        self.entropies2 = []

        return self
