import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

from utils import quantile_loss


class Encoder(nn.Module):

    def __init__(self, input_size, embed_size, hidden_size):
        """
        Init Encoder
        :param input_size (int): size of data vector (y)
        :param embed_size (int): size of exogenous covariates (x)
        :param hidden_size (int): hidden Size of LSTM (dimensionality)
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size + embed_size, hidden_size, batch_first=True)


    def forward(self, y, x):
        """
        Forward pass of encoder
        :param y: data vector  # dimensions: (batch, len, input_size)
        :param x: exogenous covariates  # dimensions: (batch, len, embed_size)
        :return: RNN output - outputs, (last_hidden_state, last_cell_state)
        """
        input = torch.cat((y, x), 2)
        return self.lstm(input)


class Decoder(nn.Module):

    def __init__(self, input_size, embed_size, horizon, context_size, quantiles):
        """ Init Decoder
        :param input_size (int): size of data received from Encoder (h_t)
        :param embed_size (int): size of exogenous covariates (x)
        :param horizon (int): number of predictions (K)
        :param context_size (int): size of contextual output (c_t)
        :param quantiles (list[int]): list of requested quantiles for regression ([q_i])
        """
        super().__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.horizon = horizon
        self.context_size = context_size
        self.quantiles = quantiles
        self.global_mlp = nn.Linear(input_size + embed_size * horizon, context_size * (horizon + 1))
        self.local_mlp = nn.Linear(context_size * 2 + embed_size, len(quantiles))


    def forward(self, hidden_states, x_f):
        """
        Forward pass of Decoder
        :param hidden_states: hidden states (not only last) returned from encoder  # dimensions: (batch, len(hidden_states), hidden_size)
        :param x_f: future exogenous covariates                                    # dimensions: (batch, horizon, embed_size)
        :return: quantiles estimation, according to list given in initialization   # dimensions: (batch, len, horizon, len(quantiles))
        """
        hidden_states = hidden_states[:, 0, :].unsqueeze(1)  # dimensions: (batch, 1, hidden_size) - currently until we decide how to cut, the forking (T) length is 1
        x_f = x_f.reshape(x_f.shape[0], -1).unsqueeze(1)     # dimensions: (batch, 1, horizon * embed_size)
        global_input = torch.cat((hidden_states, x_f), 2)    # dimensions: (batch, 1, hidden_size + horizon * embed_size)

        contexts = self.global_mlp(global_input)   #  (c_t+1,...,c_t+K, c_a)      # dimensions: (batch, 1, context_size * (horizon + 1))
        batch_size, seq_len = contexts.shape[0], contexts.shape[1]
        c_t, c_a = contexts[:, :, :self.context_size * self.horizon], contexts[:, :, self.context_size * self.horizon:]
        c_t = c_t.view(batch_size, seq_len, self.horizon, self.context_size)     # dimensions: (batch, 1, horizon, context_size)
        c_a = c_a.unsqueeze(2).expand(batch_size, seq_len, self.horizon, self.context_size)   # dimensions: (batch, 1, horizon, context_size)
        x_f = x_f.view(batch_size, seq_len, self.horizon, self.embed_size) # dimensions: (batch, 1, horizon, embed_size)
        local_input = torch.cat((c_t, c_a, x_f), dim=-1)    # dimensions: (batch, 1, horizon, 2 * context_size + embed_size)
        quantiles = self.local_mlp(local_input)             # dimensions: (batch, 1, horizon, len(quantiles))
        return quantiles


class MQRNN(nn.Module):

    def __init__(self, input_size, embed_size, hidden_size, context_size, horizon, quantiles):
        """
        Init MQRNN Module
        :param embed_size (int): size of exogenous covariates (x)
        :param hidden_size (int): hidden Size of LSTM (dimensionality)
        :param context_size (int): size of contextual output (c_t)
        :param horizon (int): number of predictions (K)
        :param quantiles (list[int]): list of requested quantiles for regression ([q_i])
        """
        super().__init__()
        self.quantiles = quantiles
        self.encoder = Encoder(input_size, embed_size, hidden_size)
        self.decoder = Decoder(hidden_size, embed_size, horizon, context_size, quantiles)

    def forward(self, y_e, x_e, y_d, x_d):
        """
        Forward pass of MQRNN
        :param y_e: data vector  # dimensions: (batch, len, input_size)
        :param x_e: exogenous covariates  # dimensions: (batch, len, embed_size)
        :param y_d: data vector  # dimensions: (batch, horizon, input_size)
        :param x_d: exogenous covariates  # dimensions: (batch, horizon, embed_size)
        :return: quantile loss.
        """
        # currently len(hidden_states) = 1
        hidden_states, (last_h, last_c) = self.encoder(y_e, x_e)     # dimensions: (batch, len(hidden_states), hidden_size)
        predictions = self.decoder(hidden_states, x_d)               # dimensions: (batch, len(hidden_states), horizon, len(quantiles))
        y_d = y_d.unsqueeze(1)                          # dimensions: (batch, 1, horizon, 1)
        y_d = y_d.expand(*predictions.shape)                         # dimensions: (batch, len(hidden_states), horizon, len(quantiles))
        return quantile_loss(predictions, y_d, self.quantiles)