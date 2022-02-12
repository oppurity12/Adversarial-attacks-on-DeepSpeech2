import torch
import torch.nn as nn
import math


class GRUCellAlpha(nn.Module):
    def __init__(self, input_size, hidden_size, alpha, bias=True):
        super(GRUCellAlpha, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

        self.residual_h = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.proj = nn.Linear(input_size, hidden_size, bias=bias)
        self.alpha = alpha
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward_one_step(self, x, hidden):
        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))
        new_hidden = newgate - hidden

        if x.size(1) != hidden.size(1):
             new_hidden = (self.proj(x) + self.alpha * new_hidden)
        else:
             new_hidden = x + self.alpha * new_hidden

        new_hidden = (1 - inputgate) * new_hidden

        return new_hidden

    def forward(self, x, hidden):
        return hidden + self.forward_one_step(x, hidden)


class GRUModelAlpha(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, alpha=1, batch_first=False):
        super(GRUModelAlpha, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.batch_first = batch_first
        self.rnn = []
        for i in range(layer_dim):
            if i == 0:
                self.rnn.append(GRUCellAlpha(input_dim, hidden_dim, alpha))
                continue
            self.rnn.append(GRUCellAlpha(hidden_dim, hidden_dim, alpha))

        self.rnn = nn.ModuleList(self.rnn)

    def forward(self, x):
        # x shape = (seq_length, batch_size, input_dim)
        if self.batch_first:
            # x shape =  (batch_size, seq_length, input_dim) -> (seq_length, batch_size, input_dim)
            x = x.permute(1, 0, 2)

        h0 = torch.zeros(self.layer_dim, x.size(1), self.hidden_dim).to(x.device)
        for idx, layer in enumerate(self.rnn):
            hn = h0[idx, :, :]
            outs = []

            for seq in range(x.size(0)):
                hn = layer(x[seq, :, :], hn)
                # print(hn.shape)
                outs.append(hn)

            x = torch.stack(outs, 0)

        if self.batch_first:
            x = x.permute(1, 0, 2)

        return x, x[-1]
