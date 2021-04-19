import torch
import torch.nn.functional as F
from torch import nn


class MyRNN(nn.Module):

    def __init__(self, input_size=256, hidden_size=256, num_layers=3, nonlinearity='relu', bias=True, max_points=8):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.max_points = max_points

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            bias=bias,
            batch_first=False
        )

        self.linear = nn.Linear(input_size, 3)

    def forward(self, transformer_output: torch.Tensor):

        dec_layers, batch_size, num_queries, model_dim = transformer_output.shape

        transformer_output = transformer_output.reshape(dec_layers*batch_size*num_queries, -1)\
            .unsqueeze(1)\
            .expand(-1, self.max_points, -1)\
            .permute(1, 0, 2)  # [120, 8, 256]

        rnn_output = self.rnn(transformer_output)[0]
        rnn_output = rnn_output.view(self.max_points, dec_layers, batch_size, num_queries, -1).permute(1, 2, 3, 0, 4)

        return self.linear(rnn_output)
