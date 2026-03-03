import torchaudio
import os
from torch.utils.data import Dataset
import pandas as pd
import torch
import glob


class MyDataset(Dataset):
    def __init__(self,annotations_file='/data/ssd0/yiming.he/dataset/RNC/ann/train.csv',wave_length=0,norm=0):
        self.annotations_file = pd.read_csv(os.path.join(annotations_file))
        self.wave_length = wave_length  
        self.norm = norm
    def __getitem__(self, index):
        audio_path = self._get_audio_sample_path(index)
        signal0,fs=torchaudio.load(audio_path,backend='soundfile')
        if self.wave_length!=0:
            signal0 = signal0[:,0:self.wave_length*fs]
        if self.norm!=0:
            signal0 = Normalizer(signal0)
        signal = signal0[0:42,:]
        desire = signal0[42:44,:]
        return signal,desire

    def _get_audio_sample_path(self,index):
        path = self.annotations_file.iloc[index,1]
        return path
    
    def _get_audio_sample_label(self,index):
        label = self.annotations_file.iloc[index,2]
        return label
    
    def __len__(self):
        return len(self.annotations_file)


def Normalizer(data):
    dim = torch.argmax(torch.tensor(data.shape))
    min = data.min(dim=dim,keepdim=True).values
    max = data.max(dim=dim,keepdim=True).values
    return data/(max-min)


if __name__ == "__main__":
    dataset = MyDataset()
    print(len(dataset))


