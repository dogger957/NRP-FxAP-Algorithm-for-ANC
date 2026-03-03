import torch.nn as nn
import torch
import torch.nn.functional as functional
from scipy.io import loadmat
from ptflops import get_model_complexity_info


import torch
import torch.nn as nn
import torchaudio
from scipy.io import loadmat


def mimo_wiener(ref,d,filter_len,secpath,beta,ref_test,d_test):
    device = ref.device
    secpath = secpath.to(device)
    time_len = ref.size(-1)
    ref_num = ref.size(-2)
    sec_num = secpath.size(-1)
    err_num = d.size(-2)
    spk_num = sec_num//err_num
    sec_len = secpath.size(-2)
    tmp = secpath.clone()
    batch_size = ref.size(0)
    sum_R = 0          
    sum_P = 0
    auto_corr = 0
    cross_corr = 0
    for i in range(err_num):
        for j in range(spk_num):
            secpath[:,i*spk_num+j] = tmp[:,j*err_num+i]
    for i in range(err_num):
        xx_list = []
        for j in range(spk_num):
            xx = nn.functional.pad(ref,(sec_len-1,0),mode='constant',value=0)
            sec = secpath[:,i*spk_num+j].unsqueeze(0).unsqueeze(0).expand(ref_num,-1,-1).to(device)
            xx_list.append(nn.functional.conv1d(xx,sec, bias=None, stride=1, padding=0, dilation=1, groups=ref_num))
        xx = torch.cat(xx_list,dim=1)
        [R,P,auto_corr_temp,cross_corr_temp] = correlation(filter_len-1,xx,d[:,i,:],beta)
        sum_R = sum_R + R
        sum_P = sum_P + P
        auto_corr = auto_corr + auto_corr_temp/err_num
        cross_corr = cross_corr + cross_corr_temp/err_num
    W,B = block_levinson(sum_P,sum_R)
    W = -W
    W = W.reshape(batch_size,filter_len,spk_num,ref_num).permute(0,2,3,1)
    W = W.flip(dims=[-1])   
    ref_test = nn.functional.pad(ref_test,(filter_len-1,0),mode='constant',value=0)
    secpath = secpath.transpose(0,1)
    secpath = secpath.reshape(spk_num,err_num,sec_len)
    err = torch.zeros((batch_size,err_num,time_len),dtype=torch.float32).to(device)
    for i in range(batch_size):
        con_signal = nn.functional.conv1d(ref_test[i,...],W[i,...], bias=None, stride=1, padding=0, dilation=1, groups=1)
        con_signal = nn.functional.pad(con_signal,(sec_len-1,0),mode='constant',value=0)
        err[i,...] = nn.functional.conv1d(con_signal,secpath, bias=None, stride=1, padding=0, dilation=1, groups=1)
    return err,auto_corr,cross_corr

def xcorr(x,y,N,biased=True):
    L = x.size(-1)+y.size(-1)-1
    Xf = torch.fft.rfft(x,n=L)
    Yf = torch.fft.rfft(y,n=L)
    out = torch.fft.irfft(Xf*torch.conj(Yf),n=L)

    result = torch.zeros((x.size(0), 2*N+1), dtype=torch.float32).to(x.device)
    result[:,0:N] = out[...,L-N:]
    result[:,N:] = out[...,0:N+1]
    if biased:
        result = result/(x.size(-1))
    return result


def correlation(N,x,d,beta):
    M = x.size(-2)
    batch_size = x.size(0)
    # R = torch.zeros((batch_size,M*(N+1),M),dtype=torch.float32).to(x.device)
    R_blocks = []
    auto_corr = []
    cross_corr = []
    power_avg = 0
    for m in range(M):
        rmi = xcorr(x[:,m,:],x[:,m,:],N)
        auto_corr.append(rmi)
        power_avg=power_avg+rmi[:,N].unsqueeze(-1).unsqueeze(-1)/M
    auto_corr = torch.stack(auto_corr, dim=1) 
    auto_corr =auto_corr/power_avg
    for m in range(M):
        Row_blocks = []
        for i in range(M):
            # rmi = xcorr(x[:,m,:],x[:,i,:],N)
            if m==i:
                rmi = xcorr(x[:,m,:],x[:,i,:],N)
                # auto_corr.append(rmi)
            else:
                rmi = xcorr(x[:,m,:],x[:,i,:],N)
                cross_corr.append(rmi/(power_avg+1e-8))
            rmi = torch.flip(rmi[:,0:N+1],dims=[-1])
            Row_blocks.append(rmi)

        R_blocks.append(torch.stack(Row_blocks, dim=-1))  # B,N+1,M2
    cross_corr = torch.stack(cross_corr, dim=1)

    R = torch.stack(R_blocks, dim=-2)  # B,N+1,M1,M2

    R = torch.reshape(R,(batch_size,M*(N+1),M))
    reg = torch.zeros((batch_size,M*(N+1),M),dtype=torch.float32).to(x.device)
    reg[:,0:M,:] = beta*torch.eye(M).to(x.device)
    R = R + reg
    P_blocks = []
    for i in range(M):
        pmi = xcorr(d,x[:,i,:],N)
        P_blocks.append(pmi[:,N:2*N+1].unsqueeze(-1))
    P = torch.stack(P_blocks, dim=-1)  # B,N+1,M
    P = P.reshape(batch_size,M*(N+1),1)
    return R,P,auto_corr,cross_corr


def block_levinson(y,L):
    batch_size,Nd,d = L.shape   #B,Nd,d
    N = Nd//d
    device = L.device
    Bmat = L.reshape(batch_size, N, d, d).permute(0, 2, 1, 3)  #B,d1,N,d2
    Bmat = torch.flip(Bmat, dims=[-2])
    Bmat = Bmat.reshape(batch_size, d, N*d)
    f = torch.inverse(L[:, :d, :])
    b = f
    x = torch.bmm(f, y[:, :d,:])
    for n in range(2, N + 1):
        pad_f = torch.cat([f, torch.zeros(batch_size, d, d, device=device)], dim=1)   # (B, 2d, d)
        ef = torch.bmm(Bmat[:, :, (N-n)*d:N*d], pad_f)                           # (B, d, d)

        pad_b = torch.cat([torch.zeros(batch_size, d, d, device=device), b], dim=1)
        eb = torch.bmm(L[:, :n*d, :].transpose(1, 2), pad_b)

        pad_x = torch.cat([x, torch.zeros(batch_size, d,1, device=device)], dim=1)
        ex = torch.bmm(Bmat[:, :, (N-n)*d:N*d], pad_x)
        E = torch.eye(d, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        A = torch.inverse(torch.cat([
            torch.cat([E, eb], dim=2),
            torch.cat([ef, E], dim=2)
        ], dim=1))
        fnb = torch.cat([f, torch.zeros(batch_size, d, d, device=device)], dim=1)
        bnb = torch.cat([torch.zeros(batch_size, d, d, device=device), b], dim=1)
        fn = torch.bmm(torch.cat([fnb, bnb], dim=2), A[:, :, :d])
        bn = torch.bmm(torch.cat([fnb, bnb], dim=2), A[:, :, d:])

        f = fn
        b = bn

        y_n = y[:, (n-1)*d:n*d,:]
        diff = (y_n - ex)
        x = torch.cat([x, torch.zeros(batch_size, d, 1,device=device)], dim=1) + torch.bmm(b, diff)
    return x,Bmat




class Causal_Conv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,dilation=1):
        super(Causal_Conv,self).__init__()
        self.kernel_size=kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=0,dilation=dilation)
    def forward(self,x):
        x = functional.pad(x,(self.dilation*(self.kernel_size-1),0),mode='constant',value=0)
        x = self.conv(x)
        return x  



class residual_block(nn.Module):
    def __init__(self,in_channels,kernel_size,stride,dilation):
        super(residual_block,self).__init__()
        self.in_channels=in_channels
        self.act1 = nn.Tanh()
        self.act2 = nn.Sigmoid()
        self.conv1 = Causal_Conv(in_channels=in_channels,out_channels=2*in_channels,kernel_size=kernel_size,stride=stride,dilation=dilation)
        self.conv2 = Causal_Conv(in_channels=in_channels,out_channels=2*in_channels,kernel_size=1,stride=1,dilation=1)
    def forward(self,x0):
        ## x0: [batch_size, in_channels, seq_len]
        x=x0
        x = self.conv1(x)
        x1 = x[:,0:self.in_channels,:]
        x2 = x[:,self.in_channels:,:]
        x1 = self.act1(x1)
        x2 = self.act2(x2)
        x = x1*x2
        x = self.conv2(x)
        res_x = x[:,0:self.in_channels,:]+x0
        skip_x = x[:,self.in_channels:,:]
        return res_x,skip_x
    
class DilatedStack(nn.Module):
    def __init__(self, in_channels,kernel_size,depth,stride=1):
        super(DilatedStack, self).__init__()
        residual_stack = [residual_block(in_channels,kernel_size,stride,dilation = 2**i)
                         for i in range(depth)]
        self.residual_stack = nn.ModuleList(residual_stack)
    def forward(self, x):
        skips = 0
        for layer in self.residual_stack:
            x, skip = layer(x)
            skips+=skip
        return x,skips


class Wavenet(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,res_channels,depth,num_stacks,res_ker,stride=1):
        super(Wavenet,self).__init__()
        self.conv_in = Causal_Conv(in_channels=in_channels,out_channels=res_channels,kernel_size=32,stride=stride,dilation=1)
        self.dilated_stacks = nn.ModuleList(
            [DilatedStack(res_channels,res_ker,depth)
             for i in range(num_stacks)]
        )

        self.conv1 = Causal_Conv(in_channels=res_channels,out_channels=res_channels//2,kernel_size=kernel_size)
        self.act = nn.Tanh()
        self.conv2 = Causal_Conv(in_channels=res_channels//2,out_channels=res_channels//4,kernel_size=kernel_size)
        self.conv_out = Causal_Conv(in_channels=res_channels//4,out_channels=out_channels,kernel_size=256)
    def forward(self,x0,d):
        x = self.conv_in(x0)
        skip_x = 0
        for layer in self.dilated_stacks:
            x, skip = layer(x)
            skip_x+=skip
        out = self.act(skip_x)
        out = self.act(self.conv1(out))
        out = self.act(self.conv2(out))
        out = self.conv_out(out)
        mat = loadmat('sec_path.mat')
        w2 = mat['sec_path']
        # w2[:,[1,2]]=w2[:,[2,1]]
        w2 = torch.tensor(w2,dtype=torch.float32)
        # w2 = torch.reshape(w2,(2,2,512))
        w2 = torch.flip(w2,dims=[-2])    
        secpath = w2
        err,auto_corr,cross_corr = mimo_wiener(out,d,512,secpath,2*1e-4,out,d)
        return err,auto_corr,cross_corr,out

net = Wavenet(42,6,3,128,10,3,3)
