class Trainer(object):
    def __init__(self, model, optimizer, train_loader,val_loader, test_loader,
                  args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.model = model
        
        
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        
        self.la=nn.Parameter(torch.FloatTensor(7))
        
        
        nn.init.constant_(self.la, 0.1)
        
        self.lam1=args.get("lam1")
        
        self.lam2=args.get("lam2")
        
        
        #log
        
        #if not args.debug:
        #self.logger.info("Argument: %r", args)
        # for arg, value in sorted(vars(args).items()):
        #     self.logger.info("Argument %s: %r", arg, value)
    
    
    
    
    
    
    def channel_gain(self,f,int_samp,power,phase,bs,dist_ur,dist_ub,ang_ur,ang_ub,batch,ang_uBR):
        
        eta1=self.args.get("eta_1")
        eta2=self.args.get("eta_2")
        eta3=self.args.get("eta_3")
        
        c=3e8
        
        #k_abs=torch.exp(eta1+eta2*f)+eta3
        
        #f_inv=torch.div(1,f)
        
        
        
        
        #lamb=c*f_inv
        
        
        j=torch.view_as_complex(torch.FloatTensor([0,1]))
        
        a_bs1=j*self.args.get("angles_B")*2*torch.tensor(math.pi)*self.args.get("antenna_space")/c
        
        f_bs=f[:,:,:,None]
        f_bs=f_bs.repeat(1,1,1,self.args.get("BS_antenna"))
        a_bs1=a_bs1[None,None,None,:]
        a_bs1=a_bs1.repeat(batch,self.args.get("sub_bands"),int_samp,1)
        
        
        
        
        a_bs=(1/math.sqrt(self.args.get("BS_antenna")))*torch.exp(f_bs*a_bs1) #B S INT_SAMP N_t
        
        a_irs=j*self.args.get("angles_R")*2*torch.tensor(math.pi)*self.args.get("IRS_space")/c
        
        f_irs=f[:,:,:,None]
        f_irs=f_irs.repeat(1,1,1,self.args.get("IRS_elements"))
        a_irs1=a_irs[None,None,None,:]
        a_irs1=a_irs1.repeat(batch,self.args.get("sub_bands"),int_samp,1)
        
        a_irs=(1/self.args.get("IRS_elements"))*torch.exp(f_irs*a_irs1) #B S INT_SAMP L^2
        
        f_br=f[:,:,:,None,None]
        f_br=f_br.repeat(1,1,1,self.args.get("IRS_elements"),self.args.get("BS_antenna"))
        k_abs=torch.exp(eta1+eta2*f_br)+eta3
        
        f_inv=torch.div(1,f_br)
        alpha_br=torch.exp(-0.5*k_abs*self.args.get("dist_br"))*(f_inv*(c/(4*math.pi*self.args.get("dist_br"))))
        
        delay=random.uniform(0,30)*1e-9
        comp=-2*torch.tensor(math.pi)*j*delay
        
        outer_br=torch.einsum('bsij,bsik->bsijk',a_irs,torch.conj(a_bs))*torch.exp(comp*f_br)
        
        #h_rb=alpha_br*((math.sqrt(self.args.get("Rician_factor")/(1+self.args.get("Rician_factor"))))*outer_br)+(math.sqrt(1/(1+self.args.get("Rician_factor"))))*torch.randn(batch,self.args.get("sub_bands"),int_samp,self.args.get("BS_antenna"),self.args.get("IRS_elements")).cuda()
        
        #print(k_abs)
        #print(outer_br[0,0,0,:,:])
        
        h_rb=alpha_br*outer_br
        
        
        ang_ur=ang_ur[:,None,None,:,:]
        ang_ur=ang_ur.repeat(1,self.args.get("sub_bands"),int_samp,1,1)
        
        dist_ur=dist_ur[:,None,None,:,None,:]
        dist_ur=dist_ur.repeat(1,self.args.get("sub_bands"),int_samp,1,self.args.get("user_antenna"),self.args.get("IRS_elements"))
        
        f_ur=f[:,:,:,None,None]
        f_ur=f_ur.repeat(1,1,1,self.args.get("num_users"),self.args.get("IRS_elements"))
        
        
        
        
        a_ur=j*ang_ur*2*torch.tensor(math.pi)*self.args.get("IRS_space")/c
        
        a_ur=(1/self.args.get("IRS_elements"))*torch.exp(a_ur*f_ur)
        
        #new
        f_ur2=f[:,:,:,None,None]
        f_ur2=f_ur2.repeat(1,1,1,self.args.get("num_users"),self.args.get("user_antenna"))
        
        a_ur1=j*ang_uBR[:,:,1,:]*2*torch.tensor(math.pi)*self.args.get("antenna_space")/c
        
        a_ur1=a_ur1[:,None,None,:,:]
        
        a_ur1=a_ur1.repeat(1,self.args.get("sub_bands"),int_samp,1,1)
        
        #print(f_ur2.shape)
        
        #print(a_ur1.shape)
        
        a_ur1=(1/math.sqrt(self.args.get("user_antenna")))*torch.exp(a_ur1*f_ur2)
        
        
        outer_ur=torch.einsum('bsoij,bsoik->bsoijk',a_ur1,torch.conj(a_ur)) #dimension N_r * L
        
        f_ur3=f[:,:,:,None,None,None]
        f_ur3=f_ur3.repeat(1,1,1,self.args.get("num_users"),self.args.get("user_antenna"),self.args.get("IRS_elements"))
        
        k_abs_ur=torch.exp(eta1+eta2*f_ur3)+eta3
        
        alpha_ur1=torch.exp(-0.5*k_abs_ur*dist_ur)
        
        alhpa_ur2=4*math.pi*f_ur3*dist_ur
        
        alpha_ur=alpha_ur1*(c*alhpa_ur2.pow(-1))
        
        #alpha_ur=alpha_ur.repeat(1,1,self.args.get("IRS_elements"))
        
        delay=random.uniform(0,30)*1e-9
        comp=-2*torch.tensor(math.pi)*j*delay
        
        h_ur=alpha_ur*outer_ur*torch.exp(comp*f_ur3)
        
        #h_ur=alpha_ur*((math.sqrt(self.args.get("Rician_factor")/(1+self.args.get("Rician_factor"))))*a_ur)+(math.sqrt(1/(1+self.args.get("Rician_factor"))))*torch.randn(batch,self.args.get("sub_bands"),int_samp,self.args.get("num_users"),self.args.get("IRS_elements")).cuda()
        
        
        ang_ub=ang_ub[:,None,None,:,:]
        ang_ub=ang_ub.repeat(1,self.args.get("sub_bands"),int_samp,1,1)
        
        dist_ub=dist_ub[:,None,None,:,None,:]
        dist_ub=dist_ub.repeat(1,self.args.get("sub_bands"),int_samp,1,self.args.get("user_antenna"),self.args.get("BS_antenna"))
        
        f_ub=f[:,:,:,None,None]
        f_ub=f_ub.repeat(1,1,1,self.args.get("num_users"),self.args.get("BS_antenna"))
        
        a_ub=j*ang_ub*2*torch.tensor(math.pi)*self.args.get("antenna_space")/c
        a_ub=(1/math.sqrt(self.args.get("BS_antenna")))*torch.exp(a_ub*f_ub)
        
        
        #new
        f_ub2=f[:,:,:,None,None]
        f_ub2=f_ub2.repeat(1,1,1,self.args.get("num_users"),self.args.get("user_antenna"))
        
        a_ub1=j*ang_uBR[:,:,0,:]*2*torch.tensor(math.pi)*self.args.get("antenna_space")/c
        
        a_ub1=a_ub1[:,None,None,:,:]
        
        a_ub1=a_ub1.repeat(1,self.args.get("sub_bands"),int_samp,1,1)
        
        a_ub1=(1/math.sqrt(self.args.get("user_antenna")))*torch.exp(a_ub1*f_ub2)
        
        
        f_ub3=f[:,:,:,None,None,None]
        f_ub3=f_ub3.repeat(1,1,1,self.args.get("num_users"),self.args.get("user_antenna"),self.args.get("BS_antenna"))
        
        
        k_abs_ub=torch.exp(eta1+eta2*f_ub3)+eta3
        
        alpha_ub1=torch.exp(-0.5*k_abs_ub*dist_ub)
        
        alhpa_ub2=4*math.pi*f_ub3*dist_ub
        
        alpha_ub=alpha_ub1*(c*alhpa_ub2.pow(-1))
        
        
        outer_ub=torch.einsum('bsoij,bsoik->bsoijk',a_ub1,torch.conj(a_ub)) #dimension N_r * N_t
        
        delay=random.uniform(0,30)*1e-9
        comp=-2*torch.tensor(math.pi)*j*delay
        
        h_ub=alpha_ub*outer_ub*torch.exp(comp*f_ub3)
        
        #h_ub=alpha_ub*((math.sqrt(self.args.get("Rician_factor")/(1+self.args.get("Rician_factor"))))*a_ub)+(math.sqrt(1/(1+self.args.get("Rician_factor"))))*torch.randn(batch,self.args.get("sub_bands"),int_samp,self.args.get("num_users"),self.args.get("BS_antenna")).cuda()
        
        
        
        
        gr=torch.exp(j*phase[:,0,:,0])
        
        #gt=torch.sqrt(1-phase[:,0,:,2])*torch.exp(j*phase[:,0,:,0])
        
        
        b = torch.eye(gr.size(1)).cuda()
        
        c1 = gr.unsqueeze(2).expand(*gr.size(), gr.size(1))
        G_r = c1* b #B L^2 L^2
        

        
        
        
        # G_r B L^2 L^2
        
        # h_ub B sub_band int_samp U N_r N_t
        
        # h_ur B sub_band int_samp U N_r L
        
        # h_br B sub_band int_samp  N_t L
        
        h_rb=h_rb[:,:,:,None,:,:]
        h_rb=h_rb.repeat(1,1,1,self.args.get("num_users"),1,1)
        
        # h_br B sub_band int_samp U N_t L
        
        
        
        
        h_u_1=torch.einsum('bsiurl,blm->bsiurm',h_ur,G_r)
        
        h_u_all=h_ub+torch.einsum('bsiurl,bsiult->bsiurt',h_u_1,h_rb)
            
        #h_u_tra1=torch.einsum('bSIij,bjk->bSIik',torch.conj(h_ur[:,:,:,ind:,:]),G_t)
            
        #h_u_transmission=torch.conj(h_ub[:,:,:,ind:,:])+torch.einsum('bSIij,bSIjk->bSIik',h_u_tra1,torch.permute(h_rb,(0,1,2,4,3)))
        
        
        
        
        
        
        
        
        
        #for u in range(self.args.get("num_users")):
        
        #print(h_u_all[0,0,0,0,:])
        
        #print( h_u_transmission.shape)
            
        noise=self.args.get("noise_pow")*bs
        
        #print(noise.shape)
        
        noise=noise[:,:,:,:,None,None]
        
        noise=noise.repeat(1,1,1,1,self.args.get("user_antenna"),self.args.get("user_antenna"))
        
        power=torch.permute(power,(0,4,1,2,3))

        rate_s_u1=torch.einsum('bsiurt,bsutm->bsiurm',h_u_all,power)
        
        rate_s_u=torch.einsum('bsiurm,bsiump->bsiurp',rate_s_u1,torch.conj(torch.permute(rate_s_u1,(0,1,2,3,5,4)))) #U N_r N_r
        
        
        
        I=torch.eye(self.args.get("num_users")).cuda()+0*j
        
        I=I[None,None,None,:,:]
        
        I=I.repeat(batch,self.args.get("sub_bands"),int_samp,1,1)
        
        nu=torch.einsum('bsiku,bsiurt->bsikrt',I,rate_s_u)
        
        de=noise+torch.einsum('bsiku,bsiurt->bsikrt',(1-I),rate_s_u)
        
        #print(de)
        
        I_r=torch.eye(self.args.get("user_antenna")).cuda()
        
        I_r=I_r[None,None,None,None,:,:]
        
        I_r=I_r.repeat(batch,self.args.get("sub_bands"),int_samp,self.args.get("num_users"),1,1)+0*j
        
        #rate_f1=I_r+torch.einsum('bsiurt,bsiutp->bsiurp',nu,torch.linalg.inv(de))
        #print(nu[0,0,0,:])
        #print(de[0,0,0,:])
        
        #print((de == 0).nonzero())
        
        rate_f=torch.log2(torch.linalg.det(I_r+torch.einsum('bsiurt,bsiutp->bsiurp',nu,torch.linalg.inv(de+1e-10))).real)
        
        del h_u_all
       # del h_u_transmission
       # del h_u_tra1
        #del h_u_reflection
        #del h_u_ref1
        del h_rb
        del h_u_1
       # del G_t
        del G_r
        del h_ub
        del h_ur
        #del f_rate
        del I
        del f_bs
        del a_bs1
        del f_br
        del k_abs
        
        
        del f_inv
        del c1
        #del c2
        del a_irs
        del a_irs1
        del alpha_br
        
        del a_bs
        
        del outer_br
        del f_ur
        del f_ub
        del de
        del nu
        del f_ur2
        del f_ub2
        del f_ub3
        del f_ur3
        del outer_ur
        del a_ub1
        del outer_ub
        del noise
        del I_r
        #del I
        del rate_s_u
        del rate_s_u1
        
        
        
        torch.cuda.empty_cache()
        
        return rate_f
        
        
            
            #for uu in range(self.args.get("num_users")):
                
             #   if uu!=u:
                    
        #denum=denum+torch.sum(torch.abs(torch.einsum('j,bj->b',h_u_all[uuu,:],power[bbb,:uuu,:,s])).pow(2))
        
        #denum=denum+torch.sum(torch.abs(torch.einsum('j,bj->b',h_u_all[uuu,:],power[bbb,uuu+1:,:,s])).pow(2))
                
            
        #rate_s_u=bs*torch.log2(torch.div(num,denum))
            
            
        
        
    
    
    def optimization_problem(self,power,phase,band,dist_ur,dist_ub,ang_ur,ang_ub,ang_uBR):
        
        #A1=torch.einsum('ij,bjk->bik', args.get("adj_rrh_user"),prb)
        #A2=torch.einsum('ij,bjk->bik', args.get("adj_rrh_user"),power)
        
        batch=power.size()[0]
        
        #ps=prb.size()[2]
        
        #print(power[0,0,:,:])
        
        #print(band[5,:,:])
        
        #print(phase[0,0,0,:])
        
        
        
        
            
            
        fs=self.args.get("f_start")*torch.ones(batch,self.args.get("sub_bands")).cuda()
            
        int_samp=self.args.get("int_samp")    
        
        #for s in range(self.args.get("sub_bands")):
            
            
        bs=band[:,0,:]
            
            #if s!=0:
                
            #    fs=fs+self.args.get("b_g")+band[:,0,s-1]/2+band[:,0,s]/2
                   
            
        bs1=bs[:,:,None]
        bs1=bs1.repeat(1,1,int_samp)
        
        bs2=bs[:,:,None]
        bs2=bs2.repeat(1,1,self.args.get("num_users"))
        #bs2=torch.div(1,bs2)
        
        
        bs3=bs[:,:,None,None]
        bs3=bs3.repeat(1,1,int_samp,self.args.get("num_users"))
        
        start=torch.ones(self.args.get("sub_bands"),self.args.get("sub_bands")).cuda()
        start=start-torch.triu(start)
        
        bsg=bs+self.args.get("b_g")
        
        f_st=fs+torch.einsum('ij,bj->bi',start,bsg)
        
        f_st=f_st[:,:,None]
        f_st=f_st.repeat(1,1,int_samp)
        
        
        
        
            
        ran=torch.tensor(range(int_samp)).cuda()
        ran=ran[None,None,:]
        ran=ran.repeat(batch,self.args.get("sub_bands"),1)
            #r_s_u[b,s,u] =quad(self.channel_gain, fs-band[b,0,s]/2, fs+band[b,0,s]/2, args=(power,phase,bs,dist_ur,dist_ub,ang_ur,ang_ub,batch,b,u,s))[0]
            
            
        ff=f_st+(bs1/(int_samp-1))*ran
                    #r_s_u[b,s,u]=self.channel_gain(0.5e12,power,phase,bs,dist_ur,dist_ub,ang_ur,ang_ub,batch,b,u,s)
            
                    
            
                    
        r=self.channel_gain(ff,int_samp,power,phase,bs3,dist_ur,dist_ub,ang_ur,ang_ub,batch,ang_uBR)
                    
        r_s_u=bs2*torch.sum(r,dim=2)*(1/int_samp)
            
        
        #print(torch.sum(r_s_u,dim=1)[0,:])
        
        cons_rate=F.relu(self.args.get("r_u_min")-torch.sum(r_s_u,dim=1))
        
        #self.lam=F.relu(self.lam+torch.sum(self.args.get("r_u_min")-torch.sum(r_s_u,dim=1))/batch)
        
        
        cons=(self.args.get("f_end")-self.args.get("f_start")-self.args.get("b_g")*(self.args.get("sub_bands")-1))
        
        cons_band=F.relu(cons-torch.sum(band[:,0,:],dim=1))
        
        
        
        
        #with open('MHGNN_IRS_MIMO_THz_lam2.txt', 'a') as f:
        #    f.write("{}".format(self.lam2))
        #    f.write('\n')
        
        loss=(-torch.sum(r_s_u)+self.lam1*torch.sum(cons_rate))/(batch*1e10)
        
        
        r_s_u = r_s_u.detach()
        
        band=band.detach()
        
        self.lam1=F.relu(self.lam1+torch.sum(self.args.get("r_u_min")-torch.sum(r_s_u,dim=1))/batch)
        
        #self.lam2=F.relu(self.lam2+torch.sum(cons-torch.sum(band[:,0,:],dim=1))/batch)
        
        
        with open('new_mhgphormer_lam1.txt', 'a') as f:
            f.write("{}".format(self.lam1))
            f.write('\n')
        #print(loss)
        
        del bs1
        del bs2
        del bs3
        del ff
        del r
        #del r_s_u
        #del cons_rate
        del ran
        del f_st
        del start
        torch.cuda.empty_cache()
        
        #print(band)
        
        return loss,r_s_u,torch.sum(r_s_u)/batch
        
        
    
    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, data in enumerate(val_dataloader):
                #data = data[..., :self.args.get('input_dim')]
                #label = target[..., :self.args.get('output_dim')]
                batch=data.size()[0]
                
                user=data.size()[1]
                
                x_user=data[:,0:self.args.get("num_users")*self.args.get("feature_user")]
                
                x_user=torch.reshape(x_user,[batch,self.args.get("num_users"),self.args.get("feature_user")])
                
                
                
                x_bs=data[:,self.args.get("num_users")*self.args.get("feature_user"):self.args.get("num_users")*self.args.get("feature_user")+self.args.get("feature_BS")]
                
                x_bs=torch.reshape(x_bs,[batch,1,self.args.get("feature_BS")])
                
                
                
                x_irs=data[:,self.args.get("num_users")*self.args.get("feature_user")+self.args.get("feature_BS"):self.args.get("num_users")*self.args.get("feature_user")+self.args.get("feature_BS")+self.args.get("feature_IRS")]
                
                x_irs=torch.reshape(x_irs,[batch,1,self.args.get("feature_IRS")])
                
                
                
                ind=self.args.get("num_users")*self.args.get("feature_user")+self.args.get("feature_BS")+self.args.get("feature_IRS")
                
                
                ang_r=1
                
                
                ang_b=2
                
                
                ind2=ind
                
                
                ang_ur1=data[:,ind2:ind2+self.args.get("num_users")*self.args.get("IRS_elements")]
                
                ang_ur=torch.reshape(ang_ur1,[batch,self.args.get("num_users"),self.args.get("IRS_elements")])
                
                
                
                ang_ub1=data[:,ind2+self.args.get("num_users")*self.args.get("IRS_elements"):ind2+self.args.get("num_users")*self.args.get("IRS_elements")+self.args.get("num_users")*self.args.get("BS_antenna")]
                
                ang_ub=torch.reshape(ang_ub1,[batch,self.args.get("num_users"),self.args.get("BS_antenna")])
                
                
                ind3=ind2+self.args.get("num_users")*self.args.get("IRS_elements")+self.args.get("num_users")*self.args.get("BS_antenna")
                
                dist_ur1=data[:,ind3:ind3+self.args.get("num_users")]
                
                dist_ur=torch.reshape(dist_ur1,[batch,self.args.get("num_users"),1])
                
                
                dist_ub1=data[:,ind3+self.args.get("num_users"):ind3+2*self.args.get("num_users")]
                
                dist_ub=torch.reshape(dist_ur1,[batch,self.args.get("num_users"),1])
                
                ind4=ind3+2*self.args.get("num_users")
                
                ang_uBR=data[:,ind4:ind4+2*self.args.get("num_users")*self.args.get("user_antenna")]
                
                ang_uBR=torch.reshape(ang_uBR,[batch,self.args.get("num_users"),2,self.args.get("user_antenna")])
                
                power,phase,band = self.model(x_user,x_bs,x_irs,epoch)
                
                
                loss,rate,sum_rate=self.optimization_problem(power,phase,band,dist_ur,dist_ub,ang_ur,ang_ub,ang_uBR)
                
                   
                #if self.args.get('real_value'):
                    #label = self.scaler.inverse_transform(label)
                #loss = self.loss(output.cuda(), label)
                #a whole batch of Metr_LA is filtered
                if not torch.isnan(sum_rate):
                    total_val_loss += sum_rate
        val_loss = total_val_loss / len(val_dataloader)
        print('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, sum_rate))
        
        #with open('STAR-IRS-MISO-THz.txt', 'a') as f:
         #   f.write('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
         #   f.write('\n\n')
        
        return val_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_sum=0
        cr_t=0
        cb_t=0
        for batch_idx, (data) in enumerate(self.train_loader):
            #data = data[..., :self.args.get('input_dim')]
            #label = target[..., :self.args.get('output_dim')]  # (..., 1)
            self.optimizer.zero_grad()

            #teacher_forcing for RNN encoder-decoder model
            #if teacher_forcing_ratio = 1: use label as input in the decoder for all steps
            batch=data.size()[0]
                
            user=data.size()[1]
                
            x_user=data[:,0:self.args.get("num_users")*self.args.get("feature_user")]
                
            x_user=torch.reshape(x_user,[batch,self.args.get("num_users"),self.args.get("feature_user")])
                
                
                
            x_bs=data[:,self.args.get("num_users")*self.args.get("feature_user"):self.args.get("num_users")*self.args.get("feature_user")+self.args.get("feature_BS")]
                
            x_bs=torch.reshape(x_bs,[batch,1,self.args.get("feature_BS")])
                
                
                
            x_irs=data[:,self.args.get("num_users")*self.args.get("feature_user")+self.args.get("feature_BS"):self.args.get("num_users")*self.args.get("feature_user")+self.args.get("feature_BS")+self.args.get("feature_IRS")]
                
            x_irs=torch.reshape(x_irs,[batch,1,self.args.get("feature_IRS")])
                
                
                
            ind=self.args.get("num_users")*self.args.get("feature_user")+self.args.get("feature_BS")+self.args.get("feature_IRS")
                
                
            ang_r=1
                
                
            ang_b=2
                
                
            ind2=ind
                
                
            ang_ur1=data[:,ind2:ind2+self.args.get("num_users")*self.args.get("IRS_elements")]
                
            ang_ur=torch.reshape(ang_ur1,[batch,self.args.get("num_users"),self.args.get("IRS_elements")])
                
                
                
            ang_ub1=data[:,ind2+self.args.get("num_users")*self.args.get("IRS_elements"):ind2+self.args.get("num_users")*self.args.get("IRS_elements")+self.args.get("num_users")*self.args.get("BS_antenna")]
                
            ang_ub=torch.reshape(ang_ub1,[batch,self.args.get("num_users"),self.args.get("BS_antenna")])
                
                
            ind3=ind2+self.args.get("num_users")*self.args.get("IRS_elements")+self.args.get("num_users")*self.args.get("BS_antenna")
                
            dist_ur1=data[:,ind3:ind3+self.args.get("num_users")]
                
            dist_ur=torch.reshape(dist_ur1,[batch,self.args.get("num_users"),1])
                
                
            dist_ub1=data[:,ind3+self.args.get("num_users"):ind3+2*self.args.get("num_users")]
            
            
                
            dist_ub=torch.reshape(dist_ur1,[batch,self.args.get("num_users"),1])
                
            ind4=ind3+2*self.args.get("num_users")
                
            ang_uBR=data[:,ind4:ind4+2*self.args.get("num_users")*self.args.get("user_antenna")]
            
            ang_uBR=torch.reshape(ang_uBR,[batch,self.args.get("num_users"),2,self.args.get("user_antenna")])
                
            power,phase,band = self.model(x_user,x_bs,x_irs,epoch)
                
                
            loss,rate,sum_rate=self.optimization_problem(power,phase,band,dist_ur,dist_ub,ang_ur,ang_ub,ang_uBR)
                
                
                
                
            
            
            
            
            loss.backward()

            # add max grad clipping
            
            self.optimizer.step()
            total_loss += loss.item()
            
            total_sum=total_sum+sum_rate
            
            
            
            

            #log information
            if batch_idx % self.args.get('log_step') == 0:
                print('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, sum_rate))
        train_epoch_loss = total_loss/self.train_per_epoch
        sum_rate_t=total_sum/self.train_per_epoch
        print('**********Train Epoch {}: averaged Loss: {:.6f}'.format(epoch, sum_rate_t))
        
        with open('new_mhgphormer_rate.txt', 'a') as f:
            f.write("{}".format(sum_rate_t))
            f.write('\n')
        
        
        #learning rate decay
        if self.args.get('lr_decay'):
            self.lr_scheduler.step()
        return train_epoch_loss

    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in range(1, self.args.get('epochs') + 1):
            #epoch_time = time.time()
            train_epoch_loss = self.train_epoch(epoch)
            #print(time.time()-epoch_time)
            #exit()
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)

            #print('LR:', self.optimizer.param_groups[0]['lr'])
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            #if train_epoch_loss > 1e6:
            #    print('Gradient explosion detected. Ending...')
            #    break
            #if self.val_loader == None:
            #val_epoch_loss = train_epoch_loss
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.get('early_stop'):
                if not_improved_count == self.args.get('early_stop_patience'):
                    print("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.get('early_stop_patience')))
                    break
            # save the best state
            

        training_time = time.time() - start_time
        print("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))
        
        #with open('MHGNN_IRS_MIMO-THz_training_time.txt', 'a') as f:
        #    f.write("{}".format(training_time))
        #    f.write('\n')

        #save the best model to file
        

        #test
        #self.model.load_state_dict(best_model)
        #self.val_epoch(self.args.epochs, self.test_loader)
        #y1,y2=self.test(self.model, self.args, self.test_loader,  self.logger)

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        self.logger.info("Saving current best model to " + self.best_path)

    
    def test(self):
        
        self.model.eval()
        power_out = []
        phase_out = []
        band_out=[]
        rate_out=[]
        total_sum=[]
        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_loader):
                batch=data.size()[0]
                
                user=data.size()[1]
                
                x_user=data[:,0:args.get("num_users")*args.get("feature_user")]
                
                x_user=torch.reshape(x_user,[batch,args.get("num_users"),args.get("feature_user")])
                
                
                
                x_bs=data[:,args.get("num_users")*args.get("feature_user"):args.get("num_users")*args.get("feature_user")+args.get("feature_BS")]
                
                x_bs=torch.reshape(x_bs,[batch,1,args.get("feature_BS")])
                
                
                
                x_irs=data[:,args.get("num_users")*args.get("feature_user")+args.get("feature_BS"):args.get("num_users")*args.get("feature_user")+args.get("feature_BS")+args.get("feature_IRS")]
                
                x_irs=torch.reshape(x_irs,[batch,1,args.get("feature_IRS")])
                
                
                
                ind=args.get("num_users")*args.get("feature_user")+args.get("feature_BS")+args.get("feature_IRS")
                
                
                ang_r=1
                
                
                ang_b=2
                
                
                ind2=ind
                
                
                ang_ur1=data[:,ind2:ind2+args.get("num_users")*args.get("IRS_elements")]
                
                ang_ur=torch.reshape(ang_ur1,[batch,args.get("num_users"),args.get("IRS_elements")])
                
                
                
                ang_ub1=data[:,ind2+args.get("num_users")*args.get("IRS_elements"):ind2+args.get("num_users")*args.get("IRS_elements")+args.get("num_users")*args.get("BS_antenna")]
                
                ang_ub=torch.reshape(ang_ub1,[batch,args.get("num_users"),args.get("BS_antenna")])
                
                
                ind3=ind2+args.get("num_users")*args.get("IRS_elements")+args.get("num_users")*args.get("BS_antenna")
                
                dist_ur1=data[:,ind3:ind3+args.get("num_users")]
                
                dist_ur=torch.reshape(dist_ur1,[batch,args.get("num_users"),1])
                
                
                dist_ub1=data[:,ind3+args.get("num_users"):ind3+2*args.get("num_users")]
                
                dist_ub=torch.reshape(dist_ur1,[batch,args.get("num_users"),1])
                
                ind4=ind3+2*self.args.get("num_users")
                
                ang_uBR=data[:,ind4:ind4+2*self.args.get("num_users")*self.args.get("user_antenna")]
                
                ang_uBR=torch.reshape(ang_uBR,[batch,self.args.get("num_users"),2,self.args.get("user_antenna")])
                
                
                
                power,phase,band = model(x_user,x_bs,x_irs,100)
                
                loss,rate,sum_rate=self.optimization_problem(power,phase,band,dist_ur,dist_ub,ang_ur,ang_ub,ang_uBR)
                
                
                
                
                
                
                rate_out.append(rate)
                power_out.append(power)
                
                phase_out.append(phase)
                
                band_out.append(band)
                total_sum.append(sum_rate)
                
                
                
        #y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        power_out = torch.cat(power_out, dim=0)
        phase_out = torch.cat(phase_out, dim=0)
        band_out=torch.cat(band_out, dim=0)
        rate_out=torch.cat(rate_out,dim=0)
        #total_sum=torch.cat(total_sum)
        #if not args.get('real_value'):
        #    y_pred = torch.cat(y_pred, dim=0)
        #else:
           # y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
        #np.save('./{}_true.npy'.format(args.get('dataset')), y_true.cpu().numpy())
        #np.save('./{}_pred.npy'.format(args.get('dataset')), y_pred.cpu().numpy())
        #for t in range(y_true.shape[1]):
        #    mae, rmse, mape, _ = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
        #                                        args.get('mae_thresh'), args.get('mape_thresh'))
        #    logger.info("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
        #        t + 1, mae, rmse, mape*100))
        #mae, rmse, mape, _ = All_Metrics(y_pred, y_true, args.get('mae_thresh'), args.get('mape_thresh'))
        #logger.info("Average Horizon, MAE: {:.4f}, MSE: {:.4f}, MAPE: {:.4f}%".format(
        #            mae, rmse, mape*100))
        return total_sum,rate_out,power_out,phase_out,band_out