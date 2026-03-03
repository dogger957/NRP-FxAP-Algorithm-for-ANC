
import torch
from models.wavenet import Wavenet
import numpy as np
from loss_factory import Loss_Hybrid
import torchaudio
import scipy.io as sio
from tqdm import tqdm
from dataset import MyDataset
import os
## Conduct tests on the specified test files and save the dimensionality reduction results as a mat file for subsequent use by the adaptive algorithm.
## In the list of test files here, each line represents two consecutive 5-minute recordings that have been recorded continuously.
# The latter file is selected as the test object, while the former file is only used for future training of the traditional Wiener solution to avoid overfitting.

loss_fn=Loss_Hybrid(fs = 4000)
device = 'cuda'

test_name = '2025-01-02-22-11-35.wav'


name = test_name.split('.')[0]
print(name)
model = Wavenet(42,6,3,128,10,3,3)
model_path = 'model_path/wavenet_6chn'+'.tar'

result_path = 'np_result'
output_path = result_path+'/'+name+'.mat'
if not os.path.exists(result_path):
    os.makedirs(result_path)

model.load_state_dict(torch.load(model_path)['model'])
model=model.to(device)
model.eval()

waveform,fs = torchaudio.load(test_name)
waveform = waveform.unsqueeze(0).to(device) # [1,1,fs*seg_len]
noise = waveform[:,[58,60],:]
length = waveform.size(-1)
seg_len=30          
## Inputting the entire recording directly is too large and may cause insufficient memory. 
## Therefore, it will be processed in segments. The network's receptive field is approximately 
## 6000 points. Thus, a transition section longer than 6000 points (here, 3 seconds) is left at the beginning of each segment.
seg_num = (length-1)//((seg_len)*fs)+1
restruct_all = []

with torch.no_grad():
    for i in range(seg_num):
        if i==0:
            start_idx = 0
        else:
            start_idx = i*(seg_len)*fs-3*fs

        end_idx = min((i+1)*seg_len*fs, length)
        ref = waveform[:, 0:42, start_idx:end_idx]
        d = waveform[:, [58,60], start_idx:end_idx]
        err,auto_corr,cross_corr,out = model(ref,d)
        loss1,loss2,loss3,loss0 = loss_fn(err,d,auto_corr,cross_corr)
        print('loss1:{:.4f}, loss2:{:.4f}, loss3:{:.4f}, loss0:{:.4f}'.format(loss1,loss2,loss3,loss0))
        if i==0:
            out = out
        else:
            out = out[...,fs*3:]  # Remove the first 3 seconds of the transition section
        print(out.shape)
        restruct_all.append(out) 
        
    restruct_all = torch.cat(restruct_all, dim=-1)
    print(restruct_all.shape)  # [1,1,L]
    print(length)


restruct_ref= restruct_all.squeeze(dim=0)
restruct_ref=restruct_ref.to('cpu')
restruct_ref = restruct_ref.t()
sio.savemat(output_path, {'data': restruct_ref.numpy()}) # Save the restructed reference signal as .mat file

