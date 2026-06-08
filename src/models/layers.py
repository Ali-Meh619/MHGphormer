import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPLayer(nn.Module):
    def __init__(self, in_features, out_features1, out_features2):
        super(MLPLayer, self).__init__()
        self.in_features = in_features
        self.out_features1 = out_features1
        self.out_features2 = out_features2
        
        if out_features2 == 1:
            self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features1))
            self.bias = nn.Parameter(torch.FloatTensor(out_features1))
        else:
            self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features1, out_features2))
            self.bias = nn.Parameter(torch.FloatTensor(out_features1, out_features2))
            
        self.reset_parameters()

    def reset_parameters(self):
        def xavier_uniform_(tensor, gain=1.):
            fan_in, fan_out = tensor.size()[-2:]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std  
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)

        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.weight, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        if self.out_features2 == 1:
            output = torch.einsum('bmi,io->bmo', input, self.weight) + self.bias
        else:
            output = torch.einsum('bmi,ipo->bmpo', input, self.weight) + self.bias
        return output
    

class MLPLayer_comp(nn.Module):
    def __init__(self, in_features, out_features1, out_features2):
        super(MLPLayer_comp, self).__init__()
        self.in_features = in_features
        self.out_features1 = out_features1
        self.out_features2 = out_features2
        
        if out_features2 == 1:
            self.weight = nn.Parameter(torch.randn([in_features, out_features1], dtype=torch.cfloat))
            self.bias = nn.Parameter(torch.randn([out_features1], dtype=torch.cfloat))
        else:
            self.weight = nn.Parameter(torch.randn([in_features, out_features1, out_features2], dtype=torch.cfloat))
            self.bias = nn.Parameter(torch.randn([out_features1, out_features2], dtype=torch.cfloat))
            
        self.reset_parameters()

    def reset_parameters(self):
        def xavier_uniform_(tensor, gain=1.):
            fan_in, fan_out = tensor.size()[-2:]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std  
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)

        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.weight, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        if self.out_features2 == 1:
            output = torch.einsum('bmi,io->bmo', input, self.weight) + self.bias
        else:
            output = torch.einsum('bmi,ipo->bmpo', input, self.weight) + self.bias
        return output    


class Transformer(nn.Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels, num_heads=1, att_drop=0., act='none'):
        super(Transformer, self).__init__()
        self.n_channels = n_channels
        self.num_heads = num_heads
        self.hid = self.n_channels // 4
        
        self.query = nn.Linear(self.n_channels, self.n_channels // 4)
        self.key   = nn.Linear(self.n_channels, self.n_channels // 4)
        self.value = nn.Linear(self.n_channels, self.n_channels)
        
        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.att_drop = nn.Dropout(att_drop)
        
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)

    def reset_parameters(self):
        def xavier_uniform_(tensor, gain=1.):
            fan_in, fan_out = tensor.size()[-2:]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std  
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)

        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        xavier_uniform_(self.query.weight, gain=gain)
        xavier_uniform_(self.key.weight, gain=gain)
        xavier_uniform_(self.value.weight, gain=gain)
        nn.init.zeros_(self.query.bias)
        nn.init.zeros_(self.key.bias)
        nn.init.zeros_(self.value.bias)

    def forward(self, x, mask=None):
        B, N, C = x.size()
        
        f = self.query(x)
        g = self.key(x)  
        h = self.value(x)
        
        beta = F.softmax(torch.einsum('bij,bjk->bik', f, g.permute(0, 2, 1)) / (math.sqrt(self.hid)), dim=2)
        beta = self.att_drop(beta)
        
        aa = h[:, :, :, None]
        b = beta.permute(0, 2, 1)
        bb = b[:, :, None, :]

        aa = aa.repeat(1, 1, 1, N)
        bb = bb.repeat(1, 1, C, 1)

        c = bb * aa
        d = torch.sum(c, 1)
        d = d.permute(0, 2, 1)
        
        return self.gamma * d + x
