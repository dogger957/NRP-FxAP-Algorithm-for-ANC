import torch
from torch import nn
import math
import torch.fft as F
import random
def filterA(x,fs):
    c1 = 1.871934400e+08
    c2 = 20.598997**2
    c3 = 107.65265**2
    c4 = 737.86223**2
    c5 = 12194.217**2
    xlen = x.size(-1)
    NumUniquePts = math.ceil((xlen+1)/2)
    X = F.fft(x)
    f = torch.arange(0,NumUniquePts)*fs/xlen
    f = f.pow(2)
    A = f.pow(2)/((c2+f)) / ((c3+f).pow(0.5)) / ((c4+f).pow(0.5)) / ((c5+f))*c1
    A=A.to(device=x.device)
    XA = X[...,:NumUniquePts]*A
    if xlen%2==1:
        XA = torch.cat((XA,torch.conj(XA[...,1:].flip(-1))), dim=-1)
        xA = torch.real(F.ifft(XA))
    else:
        XA = torch.cat((XA,torch.conj(XA[...,1:-1].flip(-1))), dim=-1)
        xA = torch.real(F.ifft(XA))
    return xA

class Loss_1_Aweight(nn.Module):
    def __init__(self,fs):
        super(Loss_1_Aweight, self).__init__()
        self.fs = fs
    def forward(self, x, y):

        err = x+y
        err_a = filterA(err,self.fs)
        y_a = filterA(y,self.fs)
        a = err_a.pow(2)
        a = a.mean(dim=(-1,-2))
        b = y_a.pow(2)
        b = b.mean(dim=(-1,-2))
        loss = 10*torch.log10((a/b)).mean()
        return loss


class Loss_Hybrid(nn.Module):
    def __init__(self,fs):
        super(Loss_Hybrid, self).__init__()
        self.fs = fs
    def forward(self, x, y,auto_corr,cross_corr):
        
        err = x+y
        err_a = filterA(err,self.fs)
        y_a = filterA(y,self.fs)
        a = err_a.pow(2)
        a = a.mean(dim=(-1))
        b = y_a.pow(2)
        b = b.mean(dim=(-1))
        loss_err = ((a/b)).mean()
        tmp = torch.zeros_like(auto_corr)
        L = tmp.size(-1)
        tmp[:,:,L//2]=1.0

        loss_auto = ((auto_corr-tmp).pow(2)).mean()*auto_corr.size(-1)
        loss_cross = ((cross_corr).pow(2)).mean()*cross_corr.size(-1)

        loss = 10*torch.log10(loss_err) + 0.1*loss_auto + 0.1*loss_cross

        return loss_err,loss_auto,loss_cross,loss



class Loss_Hybrid_new(nn.Module):
    def __init__(self,fs):
        super(Loss_Hybrid_new, self).__init__()
        self.fs = fs
    def forward(self, x, y,auto_corr,cross_corr,epoch):
        
        err = x+y
        err_a = filterA(err,self.fs)
        y_a = filterA(y,self.fs)
        a = err_a.pow(2)
        a = a.mean(dim=(-1))
        b = y_a.pow(2)
        b = b.mean(dim=(-1))
        loss_err = ((a/b)).mean()
        tmp = torch.zeros_like(auto_corr)
        L = tmp.size(-1)
        tmp[:,:,L//2]=1.0

        loss_auto = ((auto_corr-tmp).pow(2)).mean()*auto_corr.size(-1)
        loss_cross = ((cross_corr).pow(2)).mean()*cross_corr.size(-1)
        if epoch<30:
            loss = 10*torch.log10(loss_err)
        else:
            loss = 10*torch.log10(loss_err) + 0.1*loss_auto + 0.1*loss_cross
        return loss_err,loss_auto,loss_cross,loss

    

