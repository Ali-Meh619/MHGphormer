import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import MLPLayer, MLPLayer_comp, Transformer

class SeHGNN(nn.Module):
    def __init__(self, args, node_1, feature_1, node_2, feature_2, node_3, feature_3, hidden, dropout):
        super(SeHGNN, self).__init__()
        
        self.args = args
        self.node1 = node_1
        self.node2 = node_2
        self.node3 = node_3

        self.layers_1 = nn.Sequential(
            MLPLayer(feature_1, hidden, 1),
            nn.LayerNorm([hidden]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self.layers_2 = nn.Sequential(
            MLPLayer(feature_2, hidden, 1),
            nn.LayerNorm([hidden]),
            nn.RReLU(),
            nn.Dropout(dropout),
        )
        
        self.layers_3 = nn.Sequential(
            MLPLayer(feature_3, hidden, 1),
            nn.LayerNorm([hidden]),
            nn.RReLU(),
            nn.Dropout(dropout),
        )

        self.layer_mid = Transformer(hidden, num_heads=1)
        
        # Beamforming
        self.layer_1p = MLPLayer_comp(hidden, args["BS_antenna"] * args["sub_bands"] * args["stream"], 1)
        self.layer_2p = MLPLayer_comp(args["BS_antenna"] * args["sub_bands"] * args["stream"], 
                                      args["BS_antenna"] * args["sub_bands"] * args["stream"], 1)
        
        # Phase shift amplitude
        self.layer_1phi = MLPLayer(hidden, args["IRS_elements"], 1)
        self.layer_norm_1phi = nn.LayerNorm([1, args["IRS_elements"]])
        self.layer_2phi = MLPLayer(args["IRS_elements"], args["IRS_elements"], 1)
        
        # Bandwidth
        self.layer_1b = MLPLayer(hidden, args["sub_bands"], 1)
        self.layer_norm_1b = nn.LayerNorm([1, args["sub_bands"]])
        self.layer_2b = MLPLayer(args["sub_bands"], args["sub_bands"], 1)
        self.layer_norm_2b = nn.LayerNorm([1, args["sub_bands"]])

        self.sig = nn.Sigmoid()

    def forward(self, x1, x2, x3, e):
        features1 = self.layers_1(x1)
        features2 = self.layers_2(x2)
        features3 = self.layers_3(x3)
        
        feature = torch.concat([features1, features2, features3], dim=1)
        features = self.layer_mid(feature)
        
        p = features[:, 0:self.node1, :]
        p0 = torch.zeros_like(p)
        c = torch.stack((p, p0), dim=3)
        p = torch.view_as_complex(c)
        
        phi = features[:, self.node1:self.node1+self.node2, :]
        b = features[:, self.node1+self.node2:self.node1+self.node2+self.node3, :]
        
        # Beamforming
        p_f = self.layer_2p(self.layer_1p(p))
        batch_sz = p_f.shape[0]
        p_fr = torch.reshape(p_f, [batch_sz, self.args["num_users"] * self.args["BS_antenna"] * self.args["sub_bands"] * self.args["stream"]])
        
        denom = torch.sqrt(torch.sum(torch.abs(p_fr).pow(2), dim=1))
        denom = denom[:, None]
        p_rf = denom.expand(-1, self.args["num_users"] * self.args["BS_antenna"] * self.args["sub_bands"] * self.args["stream"])
        p_ff = math.sqrt(self.args["P_max"]) * torch.div(p_fr, p_rf)
        
        beamforming = torch.reshape(p_ff, [batch_sz, self.args["num_users"], self.args["BS_antenna"], self.args["stream"], self.args["sub_bands"]])
        
        # Phase shift and amplitude
        phi_h = self.layer_2phi(F.relu(self.layer_norm_1phi(self.layer_1phi(phi))))
        phi_h1 = torch.reshape(phi_h, [batch_sz, 1, self.args["IRS_elements"], 1])
        phi_h1[:, :, :, 0] = 2 * math.pi * self.sig(phi_h1[:, :, :, 0])
        
        # Bandwidth
        b0 = F.relu(self.layer_norm_1b(self.layer_1b(b)))
        b_f = F.relu(self.layer_norm_2b(self.layer_2b(b0)))
        
        cons = (self.args["f_end"] - self.args["f_start"] - self.args["b_g"] * (self.args["sub_bands"] - 1))
        b_fi = self.args["b_max"] * self.sig(b_f)
        b_ff = torch.sum(b_fi[:, 0, :], dim=1)
        b_ff = b_ff[:, None]
        b_ff = b_ff.expand(-1, self.args["sub_bands"])
        b_final = cons * torch.div(b_fi[:, 0, :], b_ff)
        b_final = self.args["b_max"] - F.relu(self.args["b_max"] - b_final)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return beamforming, phi_h1, b_final[:, None, :]
