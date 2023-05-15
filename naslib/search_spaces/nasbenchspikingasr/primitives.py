# defines the NASBench-ASR primitives, move to core in future
# Copyright @ NB-ASR authors
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
from naslib.search_spaces.core import primitives as core_ops
from naslib.search_spaces.core.primitives import AbstractPrimitive

def get_loss():
    def loss(output, output_len, targets, targets_len):
        output_trans = output.permute(1, 0, 2) # needed by the CTCLoss
        loss = F.ctc_loss(output_trans, targets, output_len, targets_len, reduction='none', zero_infinity=True)
        loss = loss / output_len
        loss = loss.mean()
        return loss

    return loss


class ASRPrimitive(AbstractPrimitive):

    def get_op_name(self):
        if hasattr(self, 'name'):
            return self.name
        return super().get_op_name()

    def get_embedded_ops(self):
        return None

    def forward(self, x, edge_data=None):
        raise NotImplementedError()


class Relu(ASRPrimitive):
    def __init__(self, in_features, out_features, dropout_rate=0, name='Relu'):
        super().__init__(locals())
        self.name = name
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, edge_data=None):
        # x = x.permute(0,2,1)
        x = self.relu(x)
        x = torch.clamp(x, max=20)
        x = self.dropout(x)
        # x = x.permute(0,2,1)
        return x

    def __repr__(self):
        return f'{self.__class__}({self.relu})'


class Sigmoid(ASRPrimitive):
    def __init__(self, in_features, out_features, dropout_rate=0, name='Sigmoid'):
        super().__init__(locals())
        self.name = name
        # self.relu = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, edge_data=None):
        # x = x.permute(0,2,1)
        x = self.sigmoid(x)
        x = torch.clamp(x, max=20)
        x = self.dropout(x)
        # x = x.permute(0,2,1)
        return x

    def __repr__(self):
        return f'{self.__class__}({self.sigmoid})'


class Tanh(ASRPrimitive):
    def __init__(self, in_features, out_features, dropout_rate=0, name='Tanh'):
        super().__init__(locals())
        self.name = name
        # self.relu = nn.ReLU(inplace=False)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, edge_data=None):
        # x = x.permute(0,2,1)
        x = self.tanh(x)
        x = torch.clamp(x, max=20)
        x = self.dropout(x)
        # x = x.permute(0,2,1)
        return x

    def __repr__(self):
        return f'{self.__class__}({self.tanh})'


class Absolute(ASRPrimitive):
    def __init__(self, in_features, out_features, dropout_rate=0, name='Absolute'):
        super().__init__(locals())
        self.name = name
        self.abs = torch.abs
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, edge_data=None):
        # x = x.permute(0,2,1)
        x = self.abs(x)
        x = torch.clamp(x, max=20)
        x = self.dropout(x)
        # x = x.permute(0,2,1)
        return x

    def __repr__(self):
        return f'{self.__class__}({self.abs})'
    

class Linear(ASRPrimitive):
    def __init__(self, in_features, out_features, dropout_rate=0, name='Linear'):
        # print('Linear', in_features, out_features)
        super().__init__(locals())
        self.name = name

        self.linear = nn.Linear(in_features, out_features)
        # self.linear = nn.Linear(100, 100)
        # self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, edge_data=None):
        # print('Linear', x.shape, self.linear)
        x = x.permute(0,2,1)
        x = self.linear(x)
        # x = self.relu(x)
        # x = torch.clamp(x, max=20)
        x = self.dropout(x)
        x = x.permute(0,2,1)
        return x

    def __repr__(self):
        return f'{self.__class__}({self.linear})'


class CellLayerNorm(ASRPrimitive):

    def __init__(self, filters, eps=0.001):
        super().__init__(locals())
        self.norm_layer = nn.LayerNorm(filters, eps)

    def forward(self, x, edge_data=None):
        # print("self.norm_layer, x.shape", self.norm_layer, x.shape)
        output = x.permute(0,2,1)
        output = self.norm_layer(output)
        output = output.permute(0,2,1)
        return output


class Head(ASRPrimitive):

    def __init__(self, dropout_rate, filters, num_classes):
        super().__init__(locals())
        self.layers = nn.ModuleList([
            nn.Dropout(dropout_rate),
            nn.LSTM(input_size=filters, hidden_size=500, batch_first=True, dropout=0.0),
            nn.Linear(in_features=500, out_features=num_classes)
        ])

    def forward(self, x, edge_data=None):
        output = self.layers[0](x)
        output = output.permute(0,2,1)
        output = self.layers[1](output)[0]
        output = self.layers[2](output)
        return output


ops = {
    'linear': Linear,
    'relu': Relu,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
    'absolute': Absolute,
    'zero': lambda *args, **kwargs: core_ops.Zero(stride=1)
}
