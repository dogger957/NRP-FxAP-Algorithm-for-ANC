import torch
import toml
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from glob import glob
from tqdm import tqdm
from utils import reduce_value
class Trainer:
    def __init__(self,config,model,optimizer,scheduler,loss_func,
                 train_dataloader,val_dataloader,train_sampler,args):
        self.epochs = config
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_func
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.train_sampler = train_sampler
        self.scheduler = scheduler
        self.rank = args.rank
        self.device = args.device
        self.world_size = args.world_size

        # training config
        config['DDP']['world_size'] = args.world_size
        self.trainer_config = config['trainer']
        self.epochs = self.trainer_config['epochs']
        self.save_checkpoint_interval = self.trainer_config['save_checkpoint_interval']
        self.clip_grad_norm_value = self.trainer_config['clip_grad_norm_value']
        self.resume = self.trainer_config['resume']
        if not self.resume:
            self.exp_path = self.trainer_config['exp_path'] + '_' + datetime.now().strftime("%Y-%m-%d-%Hh%Mm")
 
        else:
            self.exp_path = self.trainer_config['exp_path'] + '_' + self.trainer_config['resume_datetime']

        self.log_path = os.path.join(self.exp_path, 'logs')
        self.checkpoint_path = os.path.join(self.exp_path, 'checkpoints')
        self.sample_path = os.path.join(self.exp_path, 'val_samples')

        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.sample_path, exist_ok=True)
        # save the config
        # save the config
        if self.rank == 0:
            with open(
                os.path.join(
                    self.exp_path, 'config.toml'.format(datetime.now().strftime("%Y-%m-%d-%Hh%Mm"))), 'w') as f:

                toml.dump(config, f)

            self.writer = SummaryWriter(self.log_path)

        self.start_epoch = 1
        self.best_score = 1

        if self.resume:
            self._resume_checkpoint()
 

    def _set_train_mode(self):
        self.model.train()
    
    def _save_checkpoint(self, epoch, score):
        model_dict = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()
        state_dict = {'epoch': epoch,
                      'optimizer': self.optimizer.state_dict(),
                      'scheduler': self.scheduler.state_dict(),
                      'model': model_dict}

        torch.save(state_dict, os.path.join(self.checkpoint_path, f'model_{str(epoch).zfill(4)}.tar'))

        torch.save(model_dict, os.path.join(self.checkpoint_path, 'model.pth'))

        if score < self.best_score:
            self.state_dict_best = state_dict.copy()
            self.best_score = score

    def _resume_checkpoint(self):
        latest_checkpoints = sorted(glob(os.path.join(self.checkpoint_path, 'model_*.tar')))[-1]

        map_location = self.device
        checkpoint = torch.load(latest_checkpoints, map_location=map_location)

        self.start_epoch = checkpoint['epoch'] + 1
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        if self.world_size > 1:
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])

    def _train_epoch(self,epoch):
        train_loss = 0
        train_bar = tqdm(self.train_dataloader, ncols=150)
        for step,(input,noise) in enumerate(train_bar,1):
            input,noise = input.to(self.device),noise.to(self.device)
            anti_noise = self.model(input,noise)

            #loss   
            loss = self.loss_fn(anti_noise,noise)
            if self.world_size > 1:
                loss = reduce_value(loss)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)

            self.optimizer.step()

            train_loss += loss.item()

            train_bar.desc = '   train[{}/{}][{}]'.format(
                epoch, self.epochs + self.start_epoch-1, datetime.now().strftime("%Y-%m-%d-%H:%M"))
            
            train_bar.set_postfix({
            'train_loss': '{:.4f}'.format(train_loss / step),
            'lr': '{:.6f}'.format(self.optimizer.param_groups[0]['lr'])})

        if self.world_size > 1 and (self.device != torch.device("cpu")):
            torch.cuda.synchronize(self.device)

        if self.rank == 0:
            self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]['lr']}, epoch)
            self.writer.add_scalars('train_loss', {'train_loss': train_loss / step}, epoch)

        return train_loss / step                     
        
        

 
    def train(self):
        if self.resume:
            self._resume_checkpoint()

        
        for epoch in range(self.start_epoch,self.epochs + self.start_epoch):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            self._set_train_mode()
            score = self._train_epoch(epoch)
            self.scheduler.step()
            if (self.rank == 0) and (epoch % self.save_checkpoint_interval == 0):
                self._save_checkpoint(epoch, score)

        if self.rank == 0:
            torch.save(self.state_dict_best,
                    os.path.join(self.checkpoint_path,
                    'best_model_{}.tar'.format(str(self.state_dict_best['epoch']).zfill(4))))    

            print('------------Training for {} epochs has done!------------'.format(self.epochs))
            