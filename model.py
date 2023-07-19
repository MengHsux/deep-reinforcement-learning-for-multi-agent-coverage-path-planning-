## has avaliable
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import norm_col_init, weights_init
from torch.autograd import Variable
from Config import Config
import math

ctx_dim = (math.ceil(Config.IMAGE_WIDTH / 4) ** 2, 512)


class A3Clstm(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(A3Clstm, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        # critic
        self.att_conv_c1 = nn.Conv2d(64, 1, 1, stride=1, padding=0)
        self.sigmoid_c = nn.Sigmoid()
        self.c_conv = nn.Conv2d(64, 512, 1, stride=1, padding=0)
        self.Wai = nn.Linear(ctx_dim[1], ctx_dim[1], bias=False)
        self.Wh = nn.Linear(512, ctx_dim[1], bias=False)
        self.att = nn.Linear(ctx_dim[1], 1)

        # self.fc = nn.Linear(ctx_dim[1] * 4 * 4, 256)
        self.lstm = nn.LSTMCell(ctx_dim[1], 512)
        self.critic_linear = nn.Linear(512, 1)

        # actor
        self.att_conv_a1 = nn.Conv2d(64, 1, 1, stride=1, padding=0)
        self.sigmoid_a = nn.Sigmoid()
        self.a_conv = nn.Conv2d(64, 512, 1, stride=1, padding=0)
        self.Wai = nn.Linear(ctx_dim[1], ctx_dim[1], bias=False)
        self.Wh = nn.Linear(512, ctx_dim[1], bias=False)
        self.att = nn.Linear(ctx_dim[1], 1)

        self.lstm = nn.LSTMCell(ctx_dim[1], 512)
        self.actor_linear = nn.Linear(512, num_outputs)
        self.actor_linear1 = nn.Linear(512, num_outputs)
        self.actor_linear2 = nn.Linear(512, num_outputs)

        self.apply(weights_init)
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)
        self.a_conv.weight.data.mul_(relu_gain)
        self.c_conv.weight.data.mul_(relu_gain)
        self.att_conv_a1.weight.data.mul_(relu_gain)
        self.att_conv_c1.weight.data.mul_(relu_gain)

        self.Wai.weight.data = norm_col_init(self.Wai.weight.data, 1.0)

        self.Wh.weight.data = norm_col_init(self.Wh.weight.data, 1.0)

        self.att.weight.data = norm_col_init(self.att.weight.data, 1.0)
        self.att.bias.data.fill_(0)

        self.actor_linear.weight.data = norm_col_init(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)

        self.actor_linear1.weight.data = norm_col_init(self.actor_linear1.weight.data, 0.01)
        self.actor_linear1.bias.data.fill_(0)

        self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.actor_linear2.weight.data = norm_col_init(self.actor_linear2.weight.data, 0.01)
        self.actor_linear2.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Critic
        c_x = F.relu(self.c_conv(x))
        att_v_feature = self.att_conv_c1(x)
        self.att_v = self.sigmoid_c(att_v_feature)  # mask-attention
        self.att_v_sig5 = self.sigmoid_c(att_v_feature * 5.0)
        c_mask_x = c_x * self.att_v  # mask processing
        c_x = c_mask_x

        filter_size = (c_x.size()[2])
        ai = c_x.view(-1, ctx_dim[1], ctx_dim[0])
        ai = ai.transpose_(1, 2)
        ai_ = ai.view(-1, ctx_dim[1])
        Lai = self.Wai(ai_)
        Lai = Lai.view(-1, ctx_dim[0], ctx_dim[1])

        Uh_ = self.Wh(hx)
        Uh_ = torch.unsqueeze(Uh_, 1)

        Lai_Uh_ = (torch.add(Lai, Uh_)).view(-1, ctx_dim[1])
        att_ = self.att(torch.tanh(Lai_Uh_))

        alpha_ = F.softmax(att_.view(-1, ctx_dim[0]), dim=1)
        zt = torch.sum(torch.mul(ai, torch.unsqueeze(alpha_, 2)), 1)

        alpha_reshape = alpha_.view(filter_size, filter_size)

        hx, cx = self.lstm(zt, (hx, cx))

        x_c = hx

        # Actor
        a_x = F.relu(self.a_conv(x))
        att_p_feature = self.att_conv_a1(x)
        self.att_p = self.sigmoid_a(att_p_feature)  # mask-attention
        self.att_p_sig5 = self.sigmoid_a(att_p_feature * 5.0)
        a_mask_x = a_x * self.att_p  # mask processing
        a_x = a_mask_x

        filter_size = (a_x.size()[2])
        ai_a = a_x.view(-1, ctx_dim[1], ctx_dim[0])
        ai_a = ai_a.transpose_(1, 2)
        ai_a_ = ai_a.view(-1, ctx_dim[1])
        Lai_a = self.Wai(ai_a_)
        Lai_a = Lai_a.view(-1, ctx_dim[0], ctx_dim[1])

        Uh_a_ = self.Wh(hx)
        Uh_a_ = torch.unsqueeze(Uh_a_, 1)

        Lai_Uh_a = (torch.add(Lai_a, Uh_a_)).view(-1, ctx_dim[1])
        att_a = self.att(torch.tanh(Lai_Uh_a))

        alpha_ = F.softmax(att_a.view(-1, ctx_dim[0]), dim=1)
        zt = torch.sum(torch.mul(ai_a, torch.unsqueeze(alpha_, 2)), 1)

        alpha_reshape = alpha_.view(filter_size, filter_size)

        hx, cx = self.lstm(zt, (hx, cx))

        x_a = hx

        return self.critic_linear(x_c), self.actor_linear(x_a), self.actor_linear1(x_a), self.actor_linear2(x_a), (
        hx, cx), alpha_reshape
