import os
import toml
import torch
import random
import argparse
import json
import numpy as np
import torch.distributed as dist

from trainer import Trainer
from models.wavenet import net
from dataset import MyDataset
from loss_factory import Loss_Hybrid_new

seed = 0
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic =True
def run(rank, config, args):
    
    args.rank = rank
    args.device = torch.device(rank)
    
    if args.world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12354'
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(rank)
        dist.barrier()

        train_dataset = MyDataset(**config['train_dataset'])
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, sampler=train_sampler,
                                                        **config['train_dataloader'], shuffle=False)
        
    else:
        train_dataset = MyDataset(**config['train_dataset'])
        train_sampler = None
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, **config['train_dataloader'], shuffle=True)
        
    model = net  
    model.to(args.device)

    if args.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['optimizer']['lr'])

    trainer = Trainer(config=config, model=model,optimizer=optimizer,scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2), loss_func=Loss_Hybrid_new(fs = 4000),
                      train_dataloader=train_dataloader,val_dataloader=train_dataloader,train_sampler=train_sampler, args=args)

    trainer.train()
    if args.world_size > 1:
        dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--config', default='cfg_train.toml')
    parser.add_argument('-D', '--device', default='0,1', help='The index of the available devices, e.g. 0,1,2,3')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    args.world_size = len(args.device.split(','))
    config = toml.load(args.config)
    
    if args.world_size > 1:
        torch.multiprocessing.spawn(
            run, args=(config, args,), nprocs=args.world_size, join=True)
    else:
        run(0, config, args)
