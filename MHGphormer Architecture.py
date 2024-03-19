
class MLPLayer(nn.Module):
    def __init__(self, in_features, out_features1,out_features2):
        super(MLPLayer, self).__init__()
        self.in_features = in_features
        self.out_features1 = out_features1
        
        self.out_features2 = out_features2
        
        if out_features2==1:
            
            
            self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features1)).cuda()
        
            self.bias = nn.Parameter(torch.FloatTensor(out_features1)).cuda()
            
        else:
            
            self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features1,out_features2)).cuda()
        
            self.bias = nn.Parameter(torch.FloatTensor(out_features1,out_features2)).cuda()
            
            
            
        #self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):

        def xavier_uniform_(tensor, gain=1.):
            fan_in, fan_out = tensor.size()[-2:]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)

        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.weight, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        
        
        #w1=torch.einsum('ni,io->no', e1*e2, self.weight)
        #w2=torch.einsum('ni,io->no', w1,e2.transpose(0,1))
        
        if self.out_features2==1:
                               
            output = torch.einsum('bmi,io->bmo', input, self.weight)+self.bias
        else:
            
            output = torch.einsum('bmi,ipo->bmpo', input, self.weight)+self.bias
        #bias=torch.matmul(e1*e2,self.bias)
        
        return output
    

    
class MLPLayer_comp(nn.Module):
    def __init__(self, in_features, out_features1,out_features2):
        super(MLPLayer_comp, self).__init__()
        
        self.in_features = in_features
        self.out_features1 = out_features1
        
        self.out_features2 = out_features2
        
        if out_features2==1:
            
            
            self.weight = nn.Parameter(torch.randn([in_features, out_features1],dtype=torch.cfloat)).cuda()
            
            
        
            self.bias = nn.Parameter(torch.randn([out_features1],dtype=torch.cfloat)).cuda()
            
        else:
            
            self.weight = nn.Parameter(torch.randn([in_features, out_features1,out_features2],dtype=torch.cfloat)).cuda()
        
            self.bias = nn.Parameter(torch.randn([out_features1,out_features2],dtype=torch.cfloat)).cuda()
            
            
            
        #self.register_parameter('bias', None)
        #self.reset_parameters()

    def reset_parameters(self):

        def xavier_uniform_(tensor, gain=1.):
            fan_in, fan_out = tensor.size()[-2:]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)

        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.weight, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        
        
        #w1=torch.einsum('ni,io->no', e1*e2, self.weight)
        #w2=torch.einsum('ni,io->no', w1,e2.transpose(0,1))
        
        if self.out_features2==1:
            
            output = torch.einsum('bmi,io->bmo', input, self.weight)+self.bias
        else:
            
            output = torch.einsum('bmi,ipo->bmpo', input, self.weight)+self.bias
        #bias=torch.matmul(e1*e2,self.bias)
        
        return output    



class Transformer(nn.Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels, num_heads=1, att_drop=0., act='none'):
        super(Transformer, self).__init__()
        self.n_channels = n_channels
        self.num_heads = num_heads
        
        self.hid=self.n_channels//4
        

        self.query = nn.Linear(self.n_channels, self.n_channels//4)
        self.key   = nn.Linear(self.n_channels, self.n_channels//4)
        self.value = nn.Linear(self.n_channels, self.n_channels)
        
        
        self.q=nn.Parameter(torch.FloatTensor(self.n_channels, self.n_channels//4)).cuda()

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
            a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)

        gain = nn.init.calculate_gain('leaky_relu', 0.2)
        xavier_uniform_(self.query.weight, gain=gain)
        xavier_uniform_(self.key.weight, gain=gain)
        xavier_uniform_(self.value.weight, gain=gain)
        nn.init.zeros_(self.query.bias)
        nn.init.zeros_(self.key.bias)
        nn.init.zeros_(self.value.bias)

    def forward(self, x, mask=None):
        B, N, C = x.size() # batchsize, nodes, channels
        

        f = self.query(x)# 
        g = self.key(x)  # 
        h = self.value(x) #
        
        beta=F.softmax(torch.einsum('bij,bjk->bik', f, g.permute(0,2,1))/(math.sqrt(self.hid)),dim=2)

        
        beta = self.att_drop(beta)
        
        
        aa=h[:,:,:,None]

        b=beta.permute(0,2,1)

        bb=b[:,:,None,:]

        aa=aa.repeat(1,1,1,N)

        bb=bb.repeat(1,1,C,1)

        c=bb*aa

        d=torch.sum(c,1)

        d=d.permute(0,2,1)
        
        
        return self.gamma*d + x
    

class SeHGNN(nn.Module):
    def __init__(self, args,node_1,feature_1,node_2,feature_2,node_3,feature_3,hidden,dropout
                ):
        super(SeHGNN, self).__init__()
        
        self.args=args
        
        self.hidd=hidden
        
        self.node1=node_1
        
        self.node2=node_2
        
        self.node3=node_3
        
        
        

        self.layers_1 = nn.Sequential(
            MLPLayer(feature_1,hidden, 1),
            nn.LayerNorm([hidden]),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Conv1d1x1(num_channels, num_channels, hidden, bias=True, cformat='channel-last'),
            # nn.LayerNorm([num_channels, hidden]),
            # nn.PReLU(),
            
            # Conv1d1x1(num_channels, 4, 1, bias=True, cformat='channel-last'),
            # nn.LayerNorm([4, hidden]),
            # nn.PReLU(),
        )
        
        self.layers_2 = nn.Sequential(
            MLPLayer(feature_2,hidden, 1),
            nn.LayerNorm([hidden]),
            nn.RReLU(),
            nn.Dropout(dropout),
            # Conv1d1x1(num_channels, num_channels, hidden, bias=True, cformat='channel-last'),
            # nn.LayerNorm([num_channels, hidden]),
            # nn.PReLU(),
            
            # Conv1d1x1(num_channels, 4, 1, bias=True, cformat='channel-last'),
            # nn.LayerNorm([4, hidden]),
            # nn.PReLU(),
        )
        
        self.layers_3 = nn.Sequential(
            MLPLayer(feature_3,hidden, 1),
            nn.LayerNorm([hidden]),
            nn.RReLU(),
            nn.Dropout(dropout),
            # Conv1d1x1(num_channels, num_channels, hidden, bias=True, cformat='channel-last'),
            # nn.LayerNorm([num_channels, hidden]),
            # nn.PReLU(),
            
            # Conv1d1x1(num_channels, 4, 1, bias=True, cformat='channel-last'),
            # nn.LayerNorm([4, hidden]),
            # nn.PReLU(),
        )
        
        


        self.layer_mid = Transformer(hidden, num_heads=1)
            # self.layer_mid = Transformer(hidden, drop_metapath=drop_metapath)
            #self.layer_final = MLPLayer(num_metapath * hidden, nclass,1)
        
        
        
        
        
        # beamforming
        
        self.layer_1p=MLPLayer_comp(hidden,args.get("BS_antenna")*args.get("sub_bands")*args.get("stream"),1)
        #self.layer_norm_1p=nn.LayerNorm([args.get("num_users"),args.get("BS_antenna")*args.get("sub_bands")])
        self.layer_2p=MLPLayer_comp(args.get("BS_antenna")*args.get("sub_bands")*args.get("stream"),args.get("BS_antenna")*args.get("sub_bands")*args.get("stream"),1)
        #self.layer_norm_2p=nn.LayerNorm([args.get("num_users"),args.get("BS_antenna")*args.get("sub_bands")])
        #self.layer_3p=MLPLayer_comp(args.get("BS_antenna")*args.get("sub_bands")*args.get("stream"),args.get("BS_antenna")*args.get("sub_bands")*args.get("stream"),1)
        
        
        #phase shift amplitude
        self.layer_1phi=MLPLayer(hidden,args.get("IRS_elements"),1)
        self.layer_norm_1phi=nn.LayerNorm([1,self.args.get("IRS_elements")])
        self.layer_2phi=MLPLayer(args.get("IRS_elements"),args.get("IRS_elements"),1)
        self.layer_norm_2phi=nn.LayerNorm([1,self.args.get("IRS_elements")])
        self.layer_3phi=MLPLayer(args.get("IRS_elements"),args.get("IRS_elements"),1)
        
        #bandwidth
        self.layer_1b=MLPLayer(hidden,args.get("sub_bands"),1)
        self.layer_norm_1b=nn.LayerNorm([1,args.get("sub_bands")])
        self.layer_2b=MLPLayer(args.get("sub_bands"),args.get("sub_bands"),1)
        self.layer_norm_2b=nn.LayerNorm([1,args.get("sub_bands")])
        self.layer_3b=MLPLayer(args.get("sub_bands"),args.get("sub_bands"),1)
        #self.final_layer1 = nn.Sequential(
        #MLPLayer(hidden, nclass,1),
            #nn.BatchNorm1d([num_nodes,nclass], affine=False, track_running_stats=False),
            #F.relu())
         #   nn.ReLU())
        
        #self.final_layer2 = nn.Sequential(
        #MLPLayer2(num_metapath * hidden, nclass,1),
            #nn.BatchNorm1d([num_nodes,nclass], affine=False, track_running_stats=False),
            #F.relu())
        #    nn.RReLU())
         
            
        
        
            

        #if self.residual:
         #   self.res_fc = nn.Linear(nfeat, hidden, bias=False)

       

        self.prelu = nn.RReLU(0.1,0.4)
        self.dropout = nn.Dropout(dropout)
        #self.input_drop = nn.Dropout(input_drop)
        #self.att_drop = nn.Dropout(att_dropout)
        #self.label_drop = nn.Dropout(label_drop)
        
        self.sig=nn.Sigmoid()

        #self.reset_parameters()

    def reset_parameters(self):
        # for k, v in self.embeding:
        #     v.data.uniform_(-0.5, 0.5)
        # for k, v in self.labels_embeding:
        #     v.data.uniform_(-0.5, 0.5)

        #for layer in self.layers1:
            
        #    layer.reset_parameters()
            
        #for layer in self.layers2:
            
        #    layer.reset_parameters()

        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.final_layer.weight, gain=gain)
        nn.init.zeros_(self.final_layer.bias)
        
        

    def forward(self,x1,x2,x3,e):
        

        
        
        
        features1 = self.layers_1(x1)
        
        #print(features1)
        
        features2 = self.layers_2(x2)
        
        features3 = self.layers_2(x3)
        
        
        
        feature=torch.concat([features1,features2,features3],dim=1)
        
        
      
        features = self.layer_mid(feature)
        
        #print(features.size())
        
        #outt=torch.reshape(features,(B,N,self.hidd*self.num_meta))
        
        
        p=features[:,0:self.node1,:]
        
        p0=torch.zeros(p.shape[0],p.shape[1],p.shape[2]).cuda()
        
        
        c=torch.stack((p,p0),dim=3)
        
        p=torch.view_as_complex(c)
        
        
        
        phi=features[:,self.node1:self.node1+self.node2,:]
        
        b=features[:,self.node1+self.node2:self.node1+self.node2+self.node3,:]
        
        
        #p=torch.cat([p_t,p_1],dim=2)
        
        #phi=torch.cat([phi_t,phi_1],dim=2)
        
        #b=torch.cat([b_t,b_1],dim=2)
        
        
        #beamforming
        #p11=self.layer_2p(self.layer_1p(p))
        #p_f=self.layer_3p(p11)
        
        p_f=self.layer_2p(self.layer_1p(p))
        
        
        p_fr=torch.reshape(p_f,[self.args.get("batch"),self.args.get("num_users")*self.args.get("BS_antenna")*self.args.get("sub_bands")*args.get("stream")])
        
        denom=torch.sqrt(torch.sum(torch.abs(p_fr).pow(2),dim=1))
        
        denom=denom[:,None]
        
        p_rf=denom.expand(-1,self.args.get("num_users")*self.args.get("BS_antenna")*self.args.get("sub_bands")*args.get("stream"))
        
        p_ff=math.sqrt(self.args.get("P_max"))*torch.div(p_fr,p_rf)
        
        beamforming=torch.reshape(p_ff,[self.args.get("batch"),self.args.get("num_users"),self.args.get("BS_antenna"),args.get("stream"),self.args.get("sub_bands")])
        
        #phase shift and amplitude
        
        phi_h=self.layer_2phi(F.relu(self.layer_norm_1phi(self.layer_1phi(phi))))
        
        #phi_h=self.layer_3phi(F.relu(self.layer_norm_2phi(phi_h)))
        
        phi_h1=torch.reshape(phi_h,[self.args.get("batch"),1,self.args.get("IRS_elements"),1])
        
        phi_h1[:,:,:,0]=2*torch.tensor(math.pi)*self.sig(phi_h1[:,:,:,0])
        
       # phi_h1[:,:,:,2]=self.sig(phi_h1[:,:,:,2])
        
        #phi_f1=torch.reshape(phi_h1,[self.args.get("batch"),1,self.args.get("IRS_elements")*3])
        
        #phi_f=self.layer_2phi(phi_h2)
        
        #phi_f1=torch.reshape(phi_f,[args.get("batch"),1,self.args.get("IRS_elements"),3])
        
        #phi_f1[:,:,:,0:2]=2*torch.tensor(math.pi)*self.sig(phi_f1[:,:,:,0:2])
        
        #phi_f1[:,:,:,2]=self.sig(phi_f1[:,:,:,2])
        
        #phi_h3=torch.reshape(phi_f1,[self.args.get("batch"),1,self.args.get("IRS_elements")*3])
        
        #phi_f=self.layer_2phi(phi_h3)
        
        #phi_f1=torch.reshape(phi_f,[args.get("batch"),1,self.args.get("IRS_elements"),3])
        
        #phi_f1[:,:,:,0:2]=2*torch.tensor(math.pi)*self.sig(phi_f1[:,:,:,0:2])
        
        #phi_f1[:,:,:,2]=self.sig(phi_f1[:,:,:,2])
        
        #bandwidth
        b0=F.relu(self.layer_norm_1b(self.layer_1b(b)))
        b_f=F.relu(self.layer_norm_2b(self.layer_2b(b0)))
        
        #b_f=F.relu(self.layer_3b(b1))
        
        cons=(self.args.get("f_end")-self.args.get("f_start")-self.args.get("b_g")*(self.args.get("sub_bands")-1))
        
        #b_f=b_f[:,0,:]
        
        #bf1=torch.sum(b_f,dim=1)-cons
        #bf1=bf1[:,None]
        #bf1=bf1.repeat(1,self.args.get("sub_bands"))

        
        #b_final=F.relu(self.args.get("b_max")-F.relu(self.args.get("b_max")-b_f+(1/self.args.get("sub_bands"))*(bf1)))
        
        #1
        #cons=(self.args.get("f_end")-self.args.get("f_start")-self.args.get("b_g")*(self.args.get("sub_bands")-1))
        
        b_fi=self.args.get("b_max")*self.sig(b_f)
        #b_final=b_f
        #b_f=self.args.get("b_max")-F.relu(self.args.get("b_max")-b_f)
        
        b_ff=torch.sum(b_fi[:,0,:],dim=1)
        
        b_ff=b_ff[:,None]
        
        b_ff=b_ff.expand(-1,self.args.get("sub_bands"))
        
        b_final=cons*torch.div(b_fi[:,0,:],b_ff)
        
        
        b_final=self.args.get("b_max")-F.relu(self.args.get("b_max")-b_final)
        
        #minn=torch.min(b_final,0)[0]
        
        #minn=minn[None,:]
        
        
        
        #if e>=2 and e<=10: 
        
        #    b_final=F.relu(mm*minn)+b_final
        
        #b_final=(b_final*torch.div(1,b_final*mm+(1-mm)))*(mm*minn+(1-mm))
        
        
        
        
        
        #2
        #b_final=self.args.get("b_max")*nn.sig(self.args.get("f_end")-self.args.get("f_start")-(self.args.get("b_g")*(self.args.get("sub_bands")-1))*F.softmax(b_f,dim=2))
        del features1
        del features2
        del features3
        
        del feature
        del features
        del p
        del p0
        del c
        del phi
        del b
        #del p11
        del p_f
        del p_fr
        del denom
        del p_rf
        del p_ff
        del phi_h
        #del phi_h1
       # del phi_h2
        #del phi_f
        
        #del phi_h3
        #del b1
        #del b_f
        #del b_ff
        #del minn
        
        torch.cuda.empty_cache()
        
        #print(beamforming)
        
        
        return beamforming,phi_h1,b_final[:,None,:]