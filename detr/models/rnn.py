import torch
import torch.nn.functional as F
from torch import nn


class MyRNN(nn.Module):

    def __init__(self, input_size=256, hidden_size=256, num_layers=8, nonlinearity='relu', bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            bias=bias,
            batch_first=False
        )

    def forward(self, transformer_output: torch.Tensor):

        dec_layers, batch_size, num_queries, model_dim = transformer_output.shape
        print("dec_layers", dec_layers)
        print("batch_size", batch_size)
        print("num_queries", num_queries)
        print("model_dim", model_dim)

        transformer_output = transformer_output.reshape(dec_layers*batch_size*num_queries, -1)

        print(transformer_output.shape)

        return transformer_output