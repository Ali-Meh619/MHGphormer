import torch
import math
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.config import ARGS, DEVICE

def dist(a, b):
    return torch.sqrt(torch.sum((a - b) * (a - b)))

def generate_datasets_and_args(args):
    # Set up locations and antennas
    d_BR = dist(args["loc_BS"], args["loc_IRS"])
    cos_phi_cos_theta_BR = abs(args["loc_IRS"][0] - args["loc_BS"][0]) / d_BR
    sin_phi_cos_theta_BR = abs(args["loc_IRS"][1] - args["loc_BS"][1]) / d_BR
    sin_theta_BR = abs(args["loc_IRS"][2] - args["loc_BS"][2]) / d_BR

    user_antennas = range(args["user_antenna"])
    angles_R = torch.zeros(args["IRS_elements"])
    angles_B = torch.zeros(args["BS_antenna"])

    for l in range(args["IRS_elements"]):
        angles_R[l] = (l % args["IRS_elements_row"]) * sin_phi_cos_theta_BR + math.floor(l / args["IRS_elements_row"]) * sin_theta_BR
        
    for n in range(args["BS_antenna"]):
        angles_B[n] = (n % args["BS_row"]) * sin_phi_cos_theta_BR + math.floor(n / args["BS_col"]) * sin_theta_BR

    args["angles_R"] = angles_R.to(DEVICE)
    args["angles_B"] = angles_B.to(DEVICE)

    user_pos = torch.zeros(args["samples"], args["num_users"], 3)
    X_user = torch.zeros(args["samples"], args["num_users"], 3)
    
    angles_uR = torch.zeros(args["samples"], args["num_users"], args["IRS_elements"])
    angles_uB = torch.zeros(args["samples"], args["num_users"], args["BS_antenna"])
    angles_uBR = torch.zeros(args["samples"], args["num_users"], 2, args["user_antenna"])
    
    dist_uB = torch.zeros(args["samples"], args["num_users"], 1)
    dist_uR = torch.zeros(args["samples"], args["num_users"], 1)

    for k in range(args["samples"]):
        user_pos[k, :, 2] = args["user_range_z"]
        user_pos[k, :, 0] = (args["user_range_x1"][1] - args["user_range_x1"][0]) * torch.rand(1, int(args["num_users"])) + args["user_range_x1"][0]
        user_pos[k, :, 1] = (args["user_range_y"][1] - args["user_range_y"][0]) * torch.rand(1, int(args["num_users"])) + args["user_range_y"][0]

        for u in range(args["num_users"]):
            d_uR = dist(user_pos[k, u, :], args["loc_IRS"])
            d_uB = dist(user_pos[k, u, :], args["loc_BS"])
            
            dist_uB[k, u, 0] = d_uB
            dist_uR[k, u, 0] = d_uR
            X_user[k, u, 0] = d_uB
            X_user[k, u, 1] = d_uR
            X_user[k, u, 2] = args["r_u_min"] / 1e9
            
            sin_phi_cos_theta_uR = abs(user_pos[k, u, 1] - args["loc_IRS"][1]) / d_uR
            sin_theta_uR = abs(user_pos[k, u, 2] - args["loc_IRS"][2]) / d_uR
            
            sin_phi_cos_theta_uB = abs(user_pos[k, u, 1] - args["loc_BS"][1]) / d_uB
            sin_theta_uB = abs(user_pos[k, u, 2] - args["loc_BS"][2]) / d_uB
            
            cos_phi_cos_theta_uB = abs(user_pos[k, u, 0] - args["loc_BS"][0]) / d_uB
            cos_phi_cos_theta_uBR0 = abs(user_pos[k, u, 0] - args["loc_IRS"][0]) / d_uR
            
            angles_uBR[k, u, 0, :] = cos_phi_cos_theta_uB * torch.tensor(user_antennas)
            angles_uBR[k, u, 1, :] = cos_phi_cos_theta_uBR0 * torch.tensor(user_antennas)
            
            for l in range(args["IRS_elements"]):
                angles_uR[k, u, l] = (l % args["IRS_elements_row"]) * sin_phi_cos_theta_uR + math.floor(l / args["IRS_elements_row"]) * sin_theta_uR
            
            for n in range(args["BS_antenna"]):
                angles_uB[k, u, n] = (n % args["BS_row"]) * sin_phi_cos_theta_uB + math.floor(n / args["BS_col"]) * sin_theta_uB

    args["angles_uR"] = angles_uR.to(DEVICE)
    args["angles_uB"] = angles_uB.to(DEVICE)
    args["dist_ub"] = dist_uB
    args["dist_ur"] = dist_uR
    args["dist_br"] = d_BR
    args["stream"] = math.floor(min(args["BS_antenna"], args["num_users"] * args["user_antenna"]) / args["num_users"])

    adj_user_BS = F.softmax(dist_uB.pow(-1), dim=1)
    adj_user_IRS = F.softmax(dist_uR.pow(-1), dim=1)
    adj_BS_IRS = torch.zeros(1, 1)
    adj_BS_IRS[0, 0] = 1 / d_BR
    adj_BS_IRS = F.softmax(adj_BS_IRS, dim=1)
    
    dist_BR = d_BR * torch.ones(args["samples"], 1)
    
    X_BS = torch.cat((dist_uB[:, :, 0], dist_BR), 1)
    X_BS = X_BS / torch.max(X_BS)
    
    X_IRS = torch.cat((dist_uR[:, :, 0], dist_BR), 1)
    X_IRS = X_IRS / torch.max(X_IRS)
    
    X_BS = X_BS[:, None, :]
    X_IRS = X_IRS[:, None, :]

    # Meta_paths
    X_user_U = X_user / torch.max(X_user)
    X_user_UR = torch.einsum('bij,bjk->bik', adj_user_IRS, X_IRS)
    X_user_UB = torch.einsum('bij,bjk->bik', adj_user_BS, X_BS)
    X_IRS_RB = torch.einsum('ij,bjk->bik', adj_BS_IRS, X_BS)
    X_user_URB = torch.einsum('bij,bjk->bik', adj_user_IRS, X_IRS_RB)
    X_BS_BR = torch.einsum('ij,bjk->bik', adj_BS_IRS, X_IRS)
    X_user_UBR = torch.einsum('bij,bjk->bik', adj_user_BS, X_BS_BR)
    
    X_BS_B = X_BS
    X_BS_BU = torch.einsum('bij,bjk->bik', torch.permute(adj_user_BS, (0, 2, 1)), X_user)
    X_IRS_RU = torch.einsum('bij,bjk->bik', torch.permute(adj_user_IRS, (0, 2, 1)), X_user)
    X_BS_BRU = torch.einsum('ij,bjk->bik', adj_BS_IRS, X_IRS_RU)
    X_BS_BUR = torch.einsum('bij,bjk->bik', torch.permute(adj_user_BS, (0, 2, 1)), X_user_UR)
    
    X_IRS_R = X_IRS
    X_IRS_RUB = torch.einsum('bij,bjk->bik', torch.permute(adj_user_IRS, (0, 2, 1)), X_user_UB)
    X_IRS_RBU = torch.einsum('bij,bjk->bik', torch.permute(adj_user_IRS, (0, 2, 1)), X_BS_BU)

    X_user_f = torch.cat((X_user_U, X_user_UR, X_user_UB, X_user_URB, X_user_UBR), 2)
    X_BS_f = torch.cat((X_BS_B, X_BS_BR, X_BS_BU, X_BS_BRU, X_BS_BUR), 2)
    X_IRS_f = torch.cat((X_IRS_R, X_IRS_RU, X_IRS_RB, X_IRS_RUB, X_IRS_RBU), 2)

    args["feature_user"] = X_user_f.shape[2]
    args["feature_BS"] = X_BS_f.shape[2]
    args["feature_IRS"] = X_IRS_f.shape[2]

    X_U_F = torch.reshape(X_user_f, [args["samples"], args['num_users'] * args["feature_user"]])
    X_B_F = torch.reshape(X_BS_f, [args["samples"], 1 * args["feature_BS"]])
    X_IRS_F = torch.reshape(X_IRS_f, [args["samples"], 1 * args["feature_IRS"]])

    X_final1 = torch.cat((X_U_F, X_B_F, X_IRS_F), 1)

    angle_ur = torch.reshape(angles_uR, [args["samples"], args['num_users'] * args["IRS_elements"]])
    angle_ub = torch.reshape(angles_uB, [args["samples"], args['num_users'] * args["BS_antenna"]])
    dist_ur_reshape = torch.reshape(dist_uR, [args["samples"], args['num_users'] * 1])
    dist_ub_reshape = torch.reshape(dist_uB, [args["samples"], args['num_users'] * 1])
    angle_ubr = torch.reshape(angles_uBR, [args["samples"], args['num_users'] * 2 * args["user_antenna"]])

    X_final = torch.cat((X_final1, angle_ur, angle_ub, dist_ur_reshape, dist_ub_reshape, angle_ubr), 1)

    X_train = X_final[0:args["train_s"], :]
    X_valid = X_final[args["train_s"]:args["train_s"] + args["validation_s"], :]
    X_test = X_final[args["train_s"] + args["validation_s"]:args["train_s"] + args["validation_s"] + args["test_s"], :]

    train_loader = DataLoader(X_train.to(DEVICE), batch_size=args["batch"], shuffle=False, drop_last=True)
    valid_loader = DataLoader(X_valid.to(DEVICE), batch_size=args["batch"], shuffle=False, drop_last=True)
    test_loader = DataLoader(X_test.to(DEVICE), batch_size=args["batch"], shuffle=False, drop_last=True)

    return args, train_loader, valid_loader, test_loader
