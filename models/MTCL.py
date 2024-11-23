import math
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from layers.MoE import MoE
from layers.Layer import WeightGenerator, CustomLinear
from layers.RevIN import RevIN
from functools import reduce
from operator import mul


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.layer_nums = configs.layer_nums
        self.num_nodes = configs.num_nodes
        self.pre_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.k = configs.k
        self.num_experts_list = configs.num_experts_list
        self.patch_size_list = configs.patch_size_list
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.residual_connection = configs.residual_connection
        self.revin = configs.revin
        if self.revin:
            self.revin_layer = RevIN(num_features=configs.num_nodes, affine=False, subtract_last=False)

        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)
        self.AMS_lists = nn.ModuleList()
        self.device = torch.device('cuda:{}'.format(configs.gpu))
        self.batch_norm = configs.batch_norm
        self.temp = configs.temp
        self.no_inter_atten = configs.no_inter_atten
        self.no_intra_atten = configs.no_intra_atten

        for num in range(self.layer_nums):
            self.AMS_lists.append(
                MoE(self.seq_len, self.seq_len, self.num_experts_list[num], self.device, k=self.k,
                    num_nodes=self.num_nodes, patch_size=self.patch_size_list[num], noisy_gating=True,
                    d_model=self.d_model, d_ff=self.d_ff, layer_number=num + 1,
                    residual_connection=self.residual_connection, batch_norm=self.batch_norm, temp=self.temp,
                    no_inter_atten=self.no_inter_atten, no_intra_atten=self.no_intra_atten))
        self.projections = nn.Sequential(
            nn.Linear(self.seq_len * self.d_model, self.pre_len)
        )

    def forward(self, x):
        # x: [bz, seq, d]
        balance_loss = 0
        contrastive_loss = []
        # norm
        if self.revin:
            x = self.revin_layer(x, 'norm')
        out = self.start_fc(x.unsqueeze(-1))  # out: [bz, seq, d, d_model]

        batch_size = x.shape[0]

        for layer in self.AMS_lists:
            out, aux_loss, con_loss = layer(out)
            balance_loss += aux_loss
            contrastive_loss.append(con_loss)

        out = out.permute(0,2,1,3).reshape(batch_size, self.num_nodes, -1)
        out = self.projections(out).transpose(2, 1)

        # denorm
        if self.revin:
            out = self.revin_layer(out, 'denorm')

        contrastive_loss = torch.stack(contrastive_loss)  # [num_layers, ...]
        contrastive_loss = torch.mean(contrastive_loss)

        return out, balance_loss, contrastive_loss

