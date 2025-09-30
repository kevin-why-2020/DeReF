import os
import numpy as np
from tqdm import tqdm

from sksurv.metrics import concordance_index_censored

import torch.optim
import torch.nn.parallel
from torch.utils.tensorboard import SummaryWriter

class Engine(object):
    def __init__(self, args, results_dir, fold):
        self.args = args
        self.results_dir = results_dir
        self.fold = fold
        # tensorboard
        if args.log_data:
            # from tensorboardX import SummaryWriter
            writer_dir = os.path.join(results_dir, 'fold_' + str(fold))
            if not os.path.isdir(writer_dir):
                os.mkdir(writer_dir)
            self.writer = SummaryWriter(writer_dir, flush_secs=15)
        self.best_score = 0
        self.best_epoch = 0
        self.filename_best = None

    def learning(self, model, train_loader, val_loader, criterion, optimizer, scheduler):
        if torch.cuda.is_available():
            model = model.cuda()

        if self.args.resume is not None:
            if os.path.isfile(self.args.resume):
                print("=> loading checkpoint '{}'".format(self.args.resume))
                checkpoint = torch.load(self.args.resume)
                self.best_score = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint (score: {})".format(checkpoint['best_score']))
            else:
                print("=> no checkpoint found at '{}'".format(self.args.resume))

        if self.args.evaluate:
            self.validate(val_loader, model, criterion)
            return

        for epoch in range(self.args.num_epoch):
            self.epoch = epoch

            # change lr
            curr_lr = self.args.lr * (1.0 - np.float32(self.epoch) / np.float32(self.args.num_epoch)) ** (
                0.9
            )
            for parm in optimizer.param_groups:
                parm["lr"] = curr_lr

            # train for one epoch
            self.train(train_loader, model, criterion, optimizer)
            # evaluate on validation set
            c_index = self.validate(val_loader, model, criterion)
            # remember best c-index and save checkpoint
            is_best = c_index > self.best_score
            if is_best:
                self.best_score = c_index
                self.best_epoch = self.epoch
                self.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_score': self.best_score})
            print(' *** best c-index={:.4f} at epoch {}'.format(self.best_score, self.best_epoch))
            if scheduler is not None:
                scheduler.step()
            print('>')
        return self.best_score, self.best_epoch

    def train(self, data_loader, model, criterion, optimizer):
        model.train()
        train_loss = 0.0
        ls_m = 0.0
        ls_p = 0.0
        ls_co = 0.0
        ls_new = 0.0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        dataloader = tqdm(data_loader, desc='Train Epoch: {}'.format(self.epoch))
        for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c) in enumerate(dataloader):

            if torch.cuda.is_available():
                data_WSI = data_WSI.cuda()
                data_omic1 = data_omic1.type(torch.FloatTensor).cuda()
                data_omic2 = data_omic2.type(torch.FloatTensor).cuda()
                data_omic3 = data_omic3.type(torch.FloatTensor).cuda()
                data_omic4 = data_omic4.type(torch.FloatTensor).cuda()
                data_omic5 = data_omic5.type(torch.FloatTensor).cuda()
                data_omic6 = data_omic6.type(torch.FloatTensor).cuda()
                label = label.type(torch.LongTensor).cuda()
                c = c.type(torch.FloatTensor).cuda()

            f_list, score, hazards, S, logits = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2,
                                                   x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)

            loss_sur = criterion[0](hazards=hazards, S=S, Y=label, c=c)
            loss_mse = criterion[1](f_list[0], f_list[2]) + criterion[1](f_list[1], f_list[2]) + \
                       criterion[1](f_list[0], f_list[3]) + criterion[1](f_list[1], f_list[3]) + \
                       criterion[1](f_list[2], f_list[3]) - criterion[1](f_list[0], f_list[1])

            loss = loss_sur + self.args.alpha * loss_mse
            print('loss',loss)
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.item()
            all_event_times[batch_idx] = event_time
            train_loss += loss.item()
            ls_m = ls_m + score[0]
            ls_p = ls_p + score[1]
            ls_co = ls_co + score[2]
            ls_new = ls_new + score[3]

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # calculate loss and error for epoch
        train_loss /= len(dataloader)
        ls_m /= len(dataloader)
        ls_p /= len(dataloader)
        ls_co /= len(dataloader)
        ls_new /= len(dataloader)

        c_index = concordance_index_censored((1-all_censorships).astype(bool),
                                             all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print('loss: {:.4f}, c_index: {:.4f}'.format(train_loss, c_index))

        if self.writer:
            self.writer.add_scalar('train/loss', train_loss, self.epoch)
            self.writer.add_scalar('train/c_index', c_index, self.epoch)
            self.writer.add_scalar('train/f_m_epoch', ls_m, self.epoch)
            self.writer.add_scalar('train/f_p_epoch', ls_p, self.epoch)
            self.writer.add_scalar('train/f_co_epoch', ls_co, self.epoch)
            self.writer.add_scalar('train/f_new_epoch', ls_new, self.epoch)


    def validate(self, data_loader, model, criterion):
        model.eval()
        val_loss = 0.0
        ls_m = 0.0
        ls_p = 0.0
        ls_co = 0.0
        ls_new = 0.0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        dataloader = tqdm(data_loader, desc='Test Epoch: {}'.format(self.epoch))
        for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c) in enumerate(dataloader):
            if torch.cuda.is_available():
                data_WSI = data_WSI.cuda()
                data_omic1 = data_omic1.type(torch.FloatTensor).cuda()
                data_omic2 = data_omic2.type(torch.FloatTensor).cuda()
                data_omic3 = data_omic3.type(torch.FloatTensor).cuda()
                data_omic4 = data_omic4.type(torch.FloatTensor).cuda()
                data_omic5 = data_omic5.type(torch.FloatTensor).cuda()
                data_omic6 = data_omic6.type(torch.FloatTensor).cuda()
                label = label.type(torch.LongTensor).cuda()
                c = c.type(torch.FloatTensor).cuda()

            with torch.no_grad():
                f_list, score, hazards, S, logits = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2,
                                                          x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5,
                                                          x_omic6=data_omic6)  # return hazards, S, Y_hat, A_raw, results_dict

            loss_sur = criterion[0](hazards=hazards, S=S, Y=label, c=c)
            loss_mse = criterion[1](f_list[0], f_list[2]) + criterion[1](f_list[1], f_list[2]) + \
                       criterion[1](f_list[0], f_list[3]) + criterion[1](f_list[1], f_list[3]) + \
                       criterion[1](f_list[2], f_list[3]) - criterion[1](f_list[0], f_list[1])

            loss = loss_sur + self.args.alpha * loss_mse

            risk = -torch.sum(S, dim=1).cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.cpu().numpy()
            all_event_times[batch_idx] = event_time
            val_loss += loss.item()
            ls_m = ls_m + score[0]
            ls_p = ls_p + score[1]
            ls_co = ls_co + score[2]
            ls_new = ls_new + score[3]


        val_loss /= len(dataloader)
        ls_m /= len(dataloader)
        ls_p /= len(dataloader)
        ls_co /= len(dataloader)
        ls_new /= len(dataloader)

        c_index = concordance_index_censored((1-all_censorships).astype(bool),
                                             all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print('loss: {:.4f}, c_index: {:.4f}'.format(val_loss, c_index))
        if self.writer:
            self.writer.add_scalar('val/loss', val_loss, self.epoch)
            self.writer.add_scalar('val/c-index', c_index, self.epoch)
            self.writer.add_scalar('val/f_m_epoch', ls_m, self.epoch)
            self.writer.add_scalar('val/f_p_epoch', ls_p, self.epoch)
            self.writer.add_scalar('val/f_co_epoch', ls_co, self.epoch)
            self.writer.add_scalar('val/f_new_epoch', ls_new, self.epoch)
        return c_index

    
    def save_checkpoint(self, state):
        if self.filename_best is not None:
            os.remove(self.filename_best)
        self.filename_best = os.path.join(self.results_dir,
                                          'fold_' + str(self.fold),
                                          'model_best_{score:.4f}_{epoch}.pth.tar'.format(score=state['best_score'], epoch=state['epoch']))
        print('save best model {filename}'.format(filename=self.filename_best))
        torch.save(state, self.filename_best)
