import torch
import random
import numpy as np
import torch.utils.data
import os
import logging
from datetime import datetime
import torch.nn.functional as F
import torch.nn as nn
import math
import os
import time
import copy

import sys
from scipy.special import comb

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import special as sp
from torch.utils.data import DataLoader
from scipy.integrate import quad


def dBm_watt(x):
    
    return 10**(x/10)/1000

args={"IRS_elements":64,"IRS_elements_row":8,"BS_row":4,"BS_col":8,"BS_antenna":32,"num_users":6,"P_max":dBm_watt(30),"noise_pow":dBm_watt(-174),
      "loc_BS":torch.tensor((25,-20,-5)),"loc_IRS":torch.tensor((0,0,0)),"user_range_x1":(0,15),"user_range_x2":(-2.5,-10),
      "user_range_y":(0,25),"node_types":3,"sub_bands":5,"f_start":0.380e12,"f_end":0.4e12,"r_u_min":13e9,"user_antenna":2,
      "hidden":1024,"dropout":0.005,"eta_1":111.48,"eta_2":-2.97e-10,"eta_3":0.01,"int_samp":100000,"lam1":11e9,"lam2":11e9,
      "user_range_z":-10,"Rician_factor":1,"b_g":0.75e9,"b_max":4e9,"antenna_space":395e-6,
      "IRS_space":395e-6,"inp_dim":2,"hid_dim":4,"num_metapath":5,"train_s":12000,"validation_s":260,
      "test_s":2700,"samples":14960,"batch":250,"lr_init":0.0005,"epochs":250,"log_step":25,"lr_decay":True}



#half of users one side, the other half on the other side (always even)

#x~[5,30],y~[-15,25],z=-10

#x~[-5,-15],y~[-15,25],z=-10


def dist(a,b):
    
    return torch.sqrt(torch.sum((a-b)*(a-b)))



d_BR=dist(args.get("loc_BS"),args.get("loc_IRS"))

cos_phi_cos_theta_BR=abs(args.get("loc_IRS")[0]-args.get("loc_BS")[0])/d_BR

sin_phi_cos_theta_BR=abs(args.get("loc_IRS")[1]-args.get("loc_BS")[1])/d_BR

sin_theta_BR=abs(args.get("loc_IRS")[2]-args.get("loc_BS")[2])/d_BR

BS_antennas=range(args.get("BS_antenna"))

user_antennas=range(args.get("user_antenna"))

angles_R=torch.zeros(args.get("IRS_elements"))

angles_B=torch.zeros(args.get("BS_antenna"))

#angles_B=cos_phi_cos_theta_BR*torch.tensor(BS_antennas)

for l in range(args.get("IRS_elements")):
            
    angles_R[l]=(l%args.get("IRS_elements_row"))*sin_phi_cos_theta_BR+math.floor(l/args.get("IRS_elements_row"))*sin_theta_BR

    
for n in range(args.get("BS_antenna")):
            
    angles_B[n]=(n%args.get("BS_row"))*sin_phi_cos_theta_BR+math.floor(n/args.get("BS_col"))*sin_theta_BR



args["angles_R"]=angles_R.cuda()

args["angles_B"]=angles_B.cuda()










user_pos=torch.zeros(args.get("samples"),args.get("num_users"),3)


X_user=torch.zeros(args.get("samples"),args.get("num_users"),3)
### angles user-to-irs    U*L^2


angles_uR=torch.zeros(args.get("samples"),args.get("num_users"),args.get("IRS_elements"))

### angles user-to-bs    U*N_t

angles_uB=torch.zeros(args.get("samples"),args.get("num_users"),args.get("BS_antenna"))

angles_uBR=torch.zeros(args.get("samples"),args.get("num_users"),2,args.get("user_antenna"))

dist_uB=torch.zeros(args.get("samples"),args.get("num_users"),1)

dist_uR=torch.zeros(args.get("samples"),args.get("num_users"),1)

for k in range(args.get("samples")):
    
    
    
    user_pos[k,:,2]=args.get("user_range_z")
    
    user_pos[k,:,0]=(args.get("user_range_x1")[1]-args.get("user_range_x1")[0])*torch.rand(1,int(args.get("num_users")))+args.get("user_range_x1")[0]
    
    #user_pos[k,int(args.get("num_users")/2):,0]=(args.get("user_range_x2")[1]-args.get("user_range_x2")[0])*torch.rand(1,int(args.get("num_users")/2))+args.get("user_range_x2")[0]
    
    user_pos[k,:,1]=(args.get("user_range_y")[1]-args.get("user_range_y")[0])*torch.rand(1,int(args.get("num_users")))+args.get("user_range_y")[0]

    
    for u in range(args.get("num_users")):
        
        
        d_uR=dist(user_pos[k,u,:],args.get("loc_IRS"))
        
        d_uB=dist(user_pos[k,u,:],args.get("loc_BS"))
        
        dist_uB[k,u,0]=d_uB
        
        dist_uR[k,u,0]=d_uR
        
        X_user[k,u,0]=d_uB
        
        X_user[k,u,1]=d_uR

        X_user[k,u,2]=args.get("r_u_min")/1e9
        
        sin_phi_cos_theta_uR=abs(user_pos[k,u,1]-args.get("loc_IRS")[1])/d_uR
        
        sin_theta_uR=abs(user_pos[k,u,2]-args.get("loc_IRS")[2])/d_uR
        
        
        sin_phi_cos_theta_uB=abs(user_pos[k,u,1]-args.get("loc_BS")[1])/d_uB
        
        sin_theta_uB=abs(user_pos[k,u,2]-args.get("loc_BS")[2])/d_uB
        
        
        cos_phi_cos_theta_uB=abs(user_pos[k,u,0]-args.get("loc_BS")[0])/d_uB
        
        
        cos_phi_cos_theta_uBR0=abs(user_pos[k,u,0]-args.get("loc_IRS")[0])/d_uR
        
        
        angles_uBR[k,u,0,:]=cos_phi_cos_theta_uB*torch.tensor(user_antennas)
        
        angles_uBR[k,u,1,:]=cos_phi_cos_theta_uBR0*torch.tensor(user_antennas)
        
        for l in range(args.get("IRS_elements")):
            
            
            angles_uR[k,u,l]=(l%args.get("IRS_elements_row"))*sin_phi_cos_theta_uR+math.floor(l/args.get("IRS_elements_row"))*sin_theta_uR
        
        
        for n in range(args.get("BS_antenna")):
            
            angles_uB[k,u,n]=(n%args.get("BS_row"))*sin_phi_cos_theta_uB+math.floor(n/args.get("BS_col"))*sin_theta_uB
        
        
        
        #angles_uB[k,u,:]=cos_phi_cos_theta_uB*torch.tensor(BS_antennas)
        
        
args["angles_uR"]=angles_uR.cuda()
args["angles_uB"]=angles_uB.cuda()

args["dist_ub"]=dist_uB
args["dist_ur"]=dist_uR
args["dist_br"]=d_BR

args["stream"]=math.floor(min(args.get("BS_antenna"),args.get("num_users")*args.get("user_antenna"))/args.get("num_users"))



adj_user_BS=F.softmax(dist_uB.pow_(-1),dim=1)

adj_user_IRS=F.softmax(dist_uR.pow_(-1),dim=1)

adj_BS_IRS=torch.zeros(1,1)

adj_BS_IRS[0,0]=1/d_BR

adj_BS_IRS=F.softmax(adj_BS_IRS,dim=1)

dist_BR=d_BR*torch.ones(args.get("samples"),1)


X_BS=torch.cat((dist_uB[:,:,0],dist_BR),1)

X_BS=X_BS/torch.max(X_BS)

X_IRS=torch.cat((dist_uR[:,:,0],dist_BR),1)

X_IRS=X_IRS/torch.max(X_IRS)

X_BS=X_BS[:,None,:]

X_IRS=X_IRS[:,None,:]




#meta_paths



#1 U
X_user_U=X_user/torch.max(X_user)


#2 UR

X_user_UR=torch.einsum('bij,bjk->bik',adj_user_IRS, X_IRS)

#3 UB

X_user_UB=torch.einsum('bij,bjk->bik',adj_user_BS, X_BS)

#4 RB (irs)

X_IRS_RB=torch.einsum('ij,bjk->bik',adj_BS_IRS, X_BS)

#5 URB

X_user_URB=torch.einsum('bij,bjk->bik',adj_user_IRS, X_IRS_RB)

#6 BR (BS)

X_BS_BR=torch.einsum('ij,bjk->bik',adj_BS_IRS, X_IRS)

#7 UBR

X_user_UBR=torch.einsum('bij,bjk->bik',adj_user_BS, X_BS_BR)



#8 B
X_BS_B=X_BS

# 9 BU

X_BS_BU=torch.einsum('bij,bjk->bik',torch.permute(adj_user_BS,(0,2,1)), X_user)

# 10 RU (IRS)

X_IRS_RU=torch.einsum('bij,bjk->bik',torch.permute(adj_user_IRS,(0,2,1)), X_user)

# 11 BRU

X_BS_BRU=torch.einsum('ij,bjk->bik',adj_BS_IRS, X_IRS_RU)

#12 BUR

X_BS_BUR=torch.einsum('bij,bjk->bik',torch.permute(adj_user_BS,(0,2,1)), X_user_UR)


X_BS_f=torch.cat((X_BS_B,X_BS_BR,X_BS_BU,X_BS_BRU,X_BS_BUR),2)

#13 R

X_IRS_R=X_IRS

# 14 RUB

X_IRS_RUB=torch.einsum('bij,bjk->bik',torch.permute(adj_user_IRS,(0,2,1)), X_user_UB)

# 15 RBU

X_IRS_RBU=torch.einsum('bij,bjk->bik',torch.permute(adj_user_IRS,(0,2,1)), X_BS_BU)




X_user_f=torch.cat((X_user_U,X_user_UR,X_user_UB,X_user_URB,X_user_UBR),2)

X_BS_f=torch.cat((X_BS_B,X_BS_BR,X_BS_BU,X_BS_BRU,X_BS_BUR),2)

X_IRS_f=torch.cat((X_IRS_R,X_IRS_RU,X_IRS_RB,X_IRS_RUB,X_IRS_RBU),2)


feature_user=X_user_f.shape[2]

feature_BS=X_BS_f.shape[2]

feature_IRS=X_IRS_f.shape[2]

args["feature_user"]=feature_user

args["feature_BS"]=feature_BS

args["feature_IRS"]=feature_IRS

X_U_F=torch.reshape(X_user_f,[args["samples"],args.get('num_users')*feature_user])

X_B_F=torch.reshape(X_BS_f,[args["samples"],1*feature_BS])

X_IRS_F=torch.reshape(X_IRS_f,[args["samples"],1*feature_IRS])

X_final1=torch.cat((X_U_F,X_B_F,X_IRS_F),1)



angle_ur=torch.reshape(angles_uR,[args["samples"],args.get('num_users')*args.get("IRS_elements")])

angle_ub=torch.reshape(angles_uB,[args["samples"],args.get('num_users')*args.get("BS_antenna")])



dist_ur=torch.reshape(dist_uR,[args["samples"],args.get('num_users')*1])
dist_ub=torch.reshape(dist_uB,[args["samples"],args.get('num_users')*1])

angle_ubr=torch.reshape(angles_uBR,[args["samples"],args.get('num_users')*2*args.get("user_antenna")])

X_final=torch.cat((X_final1,angle_ur,angle_ub,dist_ur,dist_ub,angle_ubr),1)

X_train=X_final[0:args.get("train_s"),:]

#X_train2=X_final_user2[0:args.get("train_s"),:,:,:]



X_valid=X_final[args.get("train_s"):args.get("train_s")+args.get("validation_s"),:]

#X_valid2=X_final_user2[args.get("train_s"):args.get("train_s")+args.get("validation_s"),:,:,:]

X_test=X_final[args.get("train_s")+args.get("validation_s"):args.get("train_s")+args.get("validation_s")+args.get("test_s"),:]

#X_test2=X_final_user2[args.get("train_s")+args.get("validation_s"):args.get("train_s")+args.get("validation_s")+args.get("test_s"),:,:,:]



train = DataLoader(X_train.cuda(), batch_size=args["batch"], shuffle=False,drop_last=True)

#train2 = DataLoader(X_train2, batch_size=args["batch"], shuffle=False)

valid = DataLoader(X_valid.cuda(), batch_size=args["batch"], shuffle=False,drop_last=True)
#valid2 = DataLoader(X_valid2, batch_size=args["batch"], shuffle=False)

test = DataLoader(X_test.cuda(), batch_size=args["batch"], shuffle=False,drop_last=True)

