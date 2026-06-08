import os
import time
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class Trainer(object):
    def __init__(self, model, optimizer, train_loader, val_loader, test_loader, args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        
        self.lam1 = args["lam1"]
        self.lam2 = args["lam2"]

    def channel_gain(self, f, int_samp, power, phase, bs, dist_ur, dist_ub, ang_ur, ang_ub, batch, ang_uBR):
        eta1 = self.args["eta_1"]
        eta2 = self.args["eta_2"]
        eta3 = self.args["eta_3"]
        c = 3e8
        
        j = torch.view_as_complex(torch.FloatTensor([0, 1]).to(power.device))
        
        a_bs1 = j * self.args["angles_B"] * 2 * math.pi * self.args["antenna_space"] / c
        f_bs = f[:, :, :, None].repeat(1, 1, 1, self.args["BS_antenna"])
        a_bs1 = a_bs1[None, None, None, :].repeat(batch, self.args["sub_bands"], int_samp, 1)
        
        a_bs = (1 / math.sqrt(self.args["BS_antenna"])) * torch.exp(f_bs * a_bs1)
        
        a_irs = j * self.args["angles_R"] * 2 * math.pi * self.args["IRS_space"] / c
        f_irs = f[:, :, :, None].repeat(1, 1, 1, self.args["IRS_elements"])
        a_irs1 = a_irs[None, None, None, :].repeat(batch, self.args["sub_bands"], int_samp, 1)
        
        a_irs = (1 / self.args["IRS_elements"]) * torch.exp(f_irs * a_irs1)
        
        f_br = f[:, :, :, None, None].repeat(1, 1, 1, self.args["IRS_elements"], self.args["BS_antenna"])
        k_abs = torch.exp(eta1 + eta2 * f_br) + eta3
        
        f_inv = torch.div(1, f_br)
        alpha_br = torch.exp(-0.5 * k_abs * self.args["dist_br"]) * (f_inv * (c / (4 * math.pi * self.args["dist_br"])))
        
        delay = random.uniform(0, 30) * 1e-9
        comp = -2 * math.pi * j * delay
        
        outer_br = torch.einsum('bsij,bsik->bsijk', a_irs, torch.conj(a_bs)) * torch.exp(comp * f_br)
        
        h_rb = alpha_br * outer_br
        
        ang_ur = ang_ur[:, None, None, :, :].repeat(1, self.args["sub_bands"], int_samp, 1, 1)
        dist_ur = dist_ur[:, None, None, :, None, :].repeat(1, self.args["sub_bands"], int_samp, 1, self.args["user_antenna"], self.args["IRS_elements"])
        
        f_ur = f[:, :, :, None, None].repeat(1, 1, 1, self.args["num_users"], self.args["IRS_elements"])
        
        a_ur = j * ang_ur * 2 * math.pi * self.args["IRS_space"] / c
        a_ur = (1 / self.args["IRS_elements"]) * torch.exp(a_ur * f_ur)
        
        f_ur2 = f[:, :, :, None, None].repeat(1, 1, 1, self.args["num_users"], self.args["user_antenna"])
        a_ur1 = j * ang_uBR[:, :, 1, :] * 2 * math.pi * self.args["antenna_space"] / c
        a_ur1 = a_ur1[:, None, None, :, :].repeat(1, self.args["sub_bands"], int_samp, 1, 1)
        
        a_ur1 = (1 / math.sqrt(self.args["user_antenna"])) * torch.exp(a_ur1 * f_ur2)
        outer_ur = torch.einsum('bsoij,bsoik->bsoijk', a_ur1, torch.conj(a_ur))
        
        f_ur3 = f[:, :, :, None, None, None].repeat(1, 1, 1, self.args["num_users"], self.args["user_antenna"], self.args["IRS_elements"])
        
        k_abs_ur = torch.exp(eta1 + eta2 * f_ur3) + eta3
        alpha_ur1 = torch.exp(-0.5 * k_abs_ur * dist_ur)
        alhpa_ur2 = 4 * math.pi * f_ur3 * dist_ur
        alpha_ur = alpha_ur1 * (c * alhpa_ur2.pow(-1))
        
        delay = random.uniform(0, 30) * 1e-9
        comp = -2 * math.pi * j * delay
        h_ur = alpha_ur * outer_ur * torch.exp(comp * f_ur3)
        
        ang_ub = ang_ub[:, None, None, :, :].repeat(1, self.args["sub_bands"], int_samp, 1, 1)
        dist_ub = dist_ub[:, None, None, :, None, :].repeat(1, self.args["sub_bands"], int_samp, 1, self.args["user_antenna"], self.args["BS_antenna"])
        
        f_ub = f[:, :, :, None, None].repeat(1, 1, 1, self.args["num_users"], self.args["BS_antenna"])
        a_ub = j * ang_ub * 2 * math.pi * self.args["antenna_space"] / c
        a_ub = (1 / math.sqrt(self.args["BS_antenna"])) * torch.exp(a_ub * f_ub)
        
        f_ub2 = f[:, :, :, None, None].repeat(1, 1, 1, self.args["num_users"], self.args["user_antenna"])
        a_ub1 = j * ang_uBR[:, :, 0, :] * 2 * math.pi * self.args["antenna_space"] / c
        a_ub1 = a_ub1[:, None, None, :, :].repeat(1, self.args["sub_bands"], int_samp, 1, 1)
        a_ub1 = (1 / math.sqrt(self.args["user_antenna"])) * torch.exp(a_ub1 * f_ub2)
        
        f_ub3 = f[:, :, :, None, None, None].repeat(1, 1, 1, self.args["num_users"], self.args["user_antenna"], self.args["BS_antenna"])
        
        k_abs_ub = torch.exp(eta1 + eta2 * f_ub3) + eta3
        alpha_ub1 = torch.exp(-0.5 * k_abs_ub * dist_ub)
        alhpa_ub2 = 4 * math.pi * f_ub3 * dist_ub
        alpha_ub = alpha_ub1 * (c * alhpa_ub2.pow(-1))
        
        outer_ub = torch.einsum('bsoij,bsoik->bsoijk', a_ub1, torch.conj(a_ub))
        
        delay = random.uniform(0, 30) * 1e-9
        comp = -2 * math.pi * j * delay
        h_ub = alpha_ub * outer_ub * torch.exp(comp * f_ub3)
        
        gr = torch.exp(j * phase[:, 0, :, 0])
        b = torch.eye(gr.size(1), device=power.device)
        c1 = gr.unsqueeze(2).expand(*gr.size(), gr.size(1))
        G_r = c1 * b
        
        h_rb = h_rb[:, :, :, None, :, :].repeat(1, 1, 1, self.args["num_users"], 1, 1)
        h_u_1 = torch.einsum('bsiurl,blm->bsiurm', h_ur, G_r)
        h_u_all = h_ub + torch.einsum('bsiurl,bsiult->bsiurt', h_u_1, h_rb)
            
        noise = self.args["noise_pow"] * bs
        noise = noise[:, :, :, :, None, None].repeat(1, 1, 1, 1, self.args["user_antenna"], self.args["user_antenna"])
        
        power = torch.permute(power, (0, 4, 1, 2, 3))
        rate_s_u1 = torch.einsum('bsiurt,bsutm->bsiurm', h_u_all, power)
        rate_s_u = torch.einsum('bsiurm,bsiump->bsiurp', rate_s_u1, torch.conj(torch.permute(rate_s_u1, (0, 1, 2, 3, 5, 4))))
        
        I = torch.eye(self.args["num_users"], device=power.device) + 0 * j
        I = I[None, None, None, :, :].repeat(batch, self.args["sub_bands"], int_samp, 1, 1)
        
        nu = torch.einsum('bsiku,bsiurt->bsikrt', I, rate_s_u)
        de = noise + torch.einsum('bsiku,bsiurt->bsikrt', (1 - I), rate_s_u)
        
        I_r = torch.eye(self.args["user_antenna"], device=power.device)
        I_r = I_r[None, None, None, None, :, :].repeat(batch, self.args["sub_bands"], int_samp, self.args["num_users"], 1, 1) + 0 * j
        
        # Add a small epsilon to the diagonal to ensure numerical stability and prevent singular matrices
        eps = 1e-10 * torch.eye(self.args["user_antenna"], device=power.device)
        eps = eps[None, None, None, None, :, :]
        
        rate_f = torch.log2(torch.linalg.det(I_r + torch.einsum('bsiurt,bsiutp->bsiurp', nu, torch.linalg.inv(de + eps))).real)
        
        del h_u_all, h_rb, h_u_1, G_r, h_ub, h_ur, I, f_bs, a_bs1, f_br, k_abs
        del f_inv, c1, a_irs, a_irs1, alpha_br, a_bs, outer_br, f_ur, f_ub, de, nu
        del f_ur2, f_ub2, f_ub3, f_ur3, outer_ur, a_ub1, outer_ub, noise, I_r, rate_s_u, rate_s_u1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return rate_f

    def optimization_problem(self, power, phase, band, dist_ur, dist_ub, ang_ur, ang_ub, ang_uBR):
        batch = power.size()[0]
        device = power.device
        fs = self.args["f_start"] * torch.ones(batch, self.args["sub_bands"], device=device)
        int_samp = self.args["int_samp"]
        
        bs = band[:, 0, :]
        bs1 = bs[:, :, None].repeat(1, 1, int_samp)
        bs2 = bs[:, :, None].repeat(1, 1, self.args["num_users"])
        bs3 = bs[:, :, None, None].repeat(1, 1, int_samp, self.args["num_users"])
        
        start = torch.ones(self.args["sub_bands"], self.args["sub_bands"], device=device)
        start = start - torch.triu(start)
        
        bsg = bs + self.args["b_g"]
        f_st = fs + torch.einsum('ij,bj->bi', start, bsg)
        f_st = f_st[:, :, None].repeat(1, 1, int_samp)
        
        ran = torch.tensor(range(int_samp), device=device)
        ran = ran[None, None, :].repeat(batch, self.args["sub_bands"], 1)
        
        ff = f_st + (bs1 / (int_samp - 1)) * ran
        r = self.channel_gain(ff, int_samp, power, phase, bs3, dist_ur, dist_ub, ang_ur, ang_ub, batch, ang_uBR)
        r_s_u = bs2 * torch.sum(r, dim=2) * (1 / int_samp)
        
        cons_rate = F.relu(self.args["r_u_min"] - torch.sum(r_s_u, dim=1))
        cons = (self.args["f_end"] - self.args["f_start"] - self.args["b_g"] * (self.args["sub_bands"] - 1))
        cons_band = F.relu(cons - torch.sum(band[:, 0, :], dim=1))
        
        loss = (-torch.sum(r_s_u) + self.lam1 * torch.sum(cons_rate) + self.lam2 * torch.sum(cons_band)) / (batch * 1e10)
        
        r_s_u = r_s_u.detach()
        band = band.detach()
        
        self.lam1 = F.relu(self.lam1 + torch.sum(self.args["r_u_min"] - torch.sum(r_s_u, dim=1)) / batch)
        self.lam2 = F.relu(self.lam2 + torch.sum(cons - torch.sum(band[:, 0, :], dim=1)) / batch)
        
        with open('new_mhgphormer_lam1.txt', 'a') as f:
            f.write("{}\n".format(self.lam1.item() if isinstance(self.lam1, torch.Tensor) else self.lam1))
            
        del bs1, bs2, bs3, ff, r, ran, f_st, start
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return loss, r_s_u, torch.sum(r_s_u) / batch

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, data in enumerate(val_dataloader):
                batch = data.size()[0]
                
                x_user = data[:, 0:self.args["num_users"] * self.args["feature_user"]]
                x_user = torch.reshape(x_user, [batch, self.args["num_users"], self.args["feature_user"]])
                
                x_bs = data[:, self.args["num_users"] * self.args["feature_user"]:self.args["num_users"] * self.args["feature_user"] + self.args["feature_BS"]]
                x_bs = torch.reshape(x_bs, [batch, 1, self.args["feature_BS"]])
                
                x_irs = data[:, self.args["num_users"] * self.args["feature_user"] + self.args["feature_BS"]:self.args["num_users"] * self.args["feature_user"] + self.args["feature_BS"] + self.args["feature_IRS"]]
                x_irs = torch.reshape(x_irs, [batch, 1, self.args["feature_IRS"]])
                
                ind = self.args["num_users"] * self.args["feature_user"] + self.args["feature_BS"] + self.args["feature_IRS"]
                ind2 = ind
                
                ang_ur1 = data[:, ind2:ind2 + self.args["num_users"] * self.args["IRS_elements"]]
                ang_ur = torch.reshape(ang_ur1, [batch, self.args["num_users"], self.args["IRS_elements"]])
                
                ang_ub1 = data[:, ind2 + self.args["num_users"] * self.args["IRS_elements"]:ind2 + self.args["num_users"] * self.args["IRS_elements"] + self.args["num_users"] * self.args["BS_antenna"]]
                ang_ub = torch.reshape(ang_ub1, [batch, self.args["num_users"], self.args["BS_antenna"]])
                
                ind3 = ind2 + self.args["num_users"] * self.args["IRS_elements"] + self.args["num_users"] * self.args["BS_antenna"]
                
                dist_ur1 = data[:, ind3:ind3 + self.args["num_users"]]
                dist_ur = torch.reshape(dist_ur1, [batch, self.args["num_users"], 1])
                
                dist_ub1 = data[:, ind3 + self.args["num_users"]:ind3 + 2 * self.args["num_users"]]
                dist_ub = torch.reshape(dist_ub1, [batch, self.args["num_users"], 1])
                
                ind4 = ind3 + 2 * self.args["num_users"]
                
                ang_uBR = data[:, ind4:ind4 + 2 * self.args["num_users"] * self.args["user_antenna"]]
                ang_uBR = torch.reshape(ang_uBR, [batch, self.args["num_users"], 2, self.args["user_antenna"]])
                
                power, phase, band = self.model(x_user, x_bs, x_irs, epoch)
                
                loss, rate, sum_rate = self.optimization_problem(power, phase, band, dist_ur, dist_ub, ang_ur, ang_ub, ang_uBR)
                
                if not torch.isnan(sum_rate):
                    total_val_loss += sum_rate.item()
                    
        val_rate = total_val_loss / len(val_dataloader)
        print('**********Val Epoch {}: average Rate: {:.6f}'.format(epoch, val_rate))
        return val_rate

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_sum = 0
        for batch_idx, data in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            batch = data.size()[0]
                
            x_user = data[:, 0:self.args["num_users"] * self.args["feature_user"]]
            x_user = torch.reshape(x_user, [batch, self.args["num_users"], self.args["feature_user"]])
                
            x_bs = data[:, self.args["num_users"] * self.args["feature_user"]:self.args["num_users"] * self.args["feature_user"] + self.args["feature_BS"]]
            x_bs = torch.reshape(x_bs, [batch, 1, self.args["feature_BS"]])
                
            x_irs = data[:, self.args["num_users"] * self.args["feature_user"] + self.args["feature_BS"]:self.args["num_users"] * self.args["feature_user"] + self.args["feature_BS"] + self.args["feature_IRS"]]
            x_irs = torch.reshape(x_irs, [batch, 1, self.args["feature_IRS"]])
                
            ind = self.args["num_users"] * self.args["feature_user"] + self.args["feature_BS"] + self.args["feature_IRS"]
            ind2 = ind
                
            ang_ur1 = data[:, ind2:ind2 + self.args["num_users"] * self.args["IRS_elements"]]
            ang_ur = torch.reshape(ang_ur1, [batch, self.args["num_users"], self.args["IRS_elements"]])
                
            ang_ub1 = data[:, ind2 + self.args["num_users"] * self.args["IRS_elements"]:ind2 + self.args["num_users"] * self.args["IRS_elements"] + self.args["num_users"] * self.args["BS_antenna"]]
            ang_ub = torch.reshape(ang_ub1, [batch, self.args["num_users"], self.args["BS_antenna"]])
                
            ind3 = ind2 + self.args["num_users"] * self.args["IRS_elements"] + self.args["num_users"] * self.args["BS_antenna"]
                
            dist_ur1 = data[:, ind3:ind3 + self.args["num_users"]]
            dist_ur = torch.reshape(dist_ur1, [batch, self.args["num_users"], 1])
                
            dist_ub1 = data[:, ind3 + self.args["num_users"]:ind3 + 2 * self.args["num_users"]]
            dist_ub = torch.reshape(dist_ub1, [batch, self.args["num_users"], 1])
                
            ind4 = ind3 + 2 * self.args["num_users"]
                
            ang_uBR = data[:, ind4:ind4 + 2 * self.args["num_users"] * self.args["user_antenna"]]
            ang_uBR = torch.reshape(ang_uBR, [batch, self.args["num_users"], 2, self.args["user_antenna"]])
                
            power, phase, band = self.model(x_user, x_bs, x_irs, epoch)
            loss, rate, sum_rate = self.optimization_problem(power, phase, band, dist_ur, dist_ub, ang_ur, ang_ub, ang_uBR)
                
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_sum += sum_rate.item()
            
            if batch_idx % self.args['log_step'] == 0:
                print('Train Epoch {}: {}/{} Loss: {:.6f}'.format(epoch, batch_idx, self.train_per_epoch, sum_rate.item()))
                
        train_epoch_loss = total_loss / self.train_per_epoch
        sum_rate_t = total_sum / self.train_per_epoch
        print('**********Train Epoch {}: averaged Rate: {:.6f}'.format(epoch, sum_rate_t))
        
        with open('new_mhgphormer_rate.txt', 'a') as f:
            f.write("{}\n".format(sum_rate_t))
        
        if self.args.get('lr_decay') and self.lr_scheduler is not None:
            self.lr_scheduler.step()
            
        return train_epoch_loss

    def train(self):
        best_rate = -float('inf')
        start_time = time.time()
        for epoch in range(1, self.args['epochs'] + 1):
            train_epoch_loss = self.train_epoch(epoch)
            
            val_dataloader = self.test_loader if self.val_loader is None else self.val_loader
            val_epoch_rate = self.val_epoch(epoch, val_dataloader)

            if val_epoch_rate > best_rate:
                best_rate = val_epoch_rate

        training_time = time.time() - start_time
        print("Total training time: {:.4f}min, best rate: {:.6f}".format((training_time / 60), best_rate))

    def test(self):
        self.model.eval()
        power_out = []
        phase_out = []
        band_out = []
        rate_out = []
        total_sum = []
        
        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                batch = data.size()[0]
                
                x_user = data[:, 0:self.args["num_users"] * self.args["feature_user"]]
                x_user = torch.reshape(x_user, [batch, self.args["num_users"], self.args["feature_user"]])
                
                x_bs = data[:, self.args["num_users"] * self.args["feature_user"]:self.args["num_users"] * self.args["feature_user"] + self.args["feature_BS"]]
                x_bs = torch.reshape(x_bs, [batch, 1, self.args["feature_BS"]])
                
                x_irs = data[:, self.args["num_users"] * self.args["feature_user"] + self.args["feature_BS"]:self.args["num_users"] * self.args["feature_user"] + self.args["feature_BS"] + self.args["feature_IRS"]]
                x_irs = torch.reshape(x_irs, [batch, 1, self.args["feature_IRS"]])
                
                ind = self.args["num_users"] * self.args["feature_user"] + self.args["feature_BS"] + self.args["feature_IRS"]
                ind2 = ind
                
                ang_ur1 = data[:, ind2:ind2 + self.args["num_users"] * self.args["IRS_elements"]]
                ang_ur = torch.reshape(ang_ur1, [batch, self.args["num_users"], self.args["IRS_elements"]])
                
                ang_ub1 = data[:, ind2 + self.args["num_users"] * self.args["IRS_elements"]:ind2 + self.args["num_users"] * self.args["IRS_elements"] + self.args["num_users"] * self.args["BS_antenna"]]
                ang_ub = torch.reshape(ang_ub1, [batch, self.args["num_users"], self.args["BS_antenna"]])
                
                ind3 = ind2 + self.args["num_users"] * self.args["IRS_elements"] + self.args["num_users"] * self.args["BS_antenna"]
                
                dist_ur1 = data[:, ind3:ind3 + self.args["num_users"]]
                dist_ur = torch.reshape(dist_ur1, [batch, self.args["num_users"], 1])
                
                dist_ub1 = data[:, ind3 + self.args["num_users"]:ind3 + 2 * self.args["num_users"]]
                dist_ub = torch.reshape(dist_ub1, [batch, self.args["num_users"], 1])
                
                ind4 = ind3 + 2 * self.args["num_users"]
                
                ang_uBR = data[:, ind4:ind4 + 2 * self.args["num_users"] * self.args["user_antenna"]]
                ang_uBR = torch.reshape(ang_uBR, [batch, self.args["num_users"], 2, self.args["user_antenna"]])
                
                power, phase, band = self.model(x_user, x_bs, x_irs, 100)
                loss, rate, sum_rate = self.optimization_problem(power, phase, band, dist_ur, dist_ub, ang_ur, ang_ub, ang_uBR)
                
                rate_out.append(rate)
                power_out.append(power)
                phase_out.append(phase)
                band_out.append(band)
                total_sum.append(sum_rate)
                
        power_out = torch.cat(power_out, dim=0)
        phase_out = torch.cat(phase_out, dim=0)
        band_out = torch.cat(band_out, dim=0)
        rate_out = torch.cat(rate_out, dim=0)
        
        return total_sum, rate_out, power_out, phase_out, band_out

def save_model(model, model_dir, epoch=None):
    if model_dir is None:
        return
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    epoch_str = str(epoch) if epoch else ""
    file_name = os.path.join(model_dir, epoch_str + "_mhgnn_u_6.pt")
    with open(file_name, "wb") as f:
        torch.save(model, f)
