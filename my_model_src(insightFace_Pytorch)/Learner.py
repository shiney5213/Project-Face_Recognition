from data.data_pipe import de_preprocess, get_train_loader, get_val_data
from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
from verifacation import evaluate
import torch
from torch import optim
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from utils import get_time, gen_plot, hflip_batch, separate_bn_paras
from PIL import Image
from torchvision import transforms as trans
import math
import bcolz
import datetime

class face_learner(object):
    def __init__(self, conf, inference=False):
        print(conf)
        if conf.use_mobilfacenet:
            self.model = MobileFaceNet(conf.embedding_size).to(conf.device)
            print('MobileFaceNet model generated')
        else:
            self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode).to(conf.device)
            print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))
        
        if not inference:
            self.milestones = conf.milestones
            self.loader, self.class_num = get_train_loader(conf)  
            print('class_num:', self.class_num)      

            self.writer = SummaryWriter(conf.log_path)
            self.step = 0
            self.head = Arcface(embedding_size=conf.embedding_size, classnum=self.class_num).to(conf.device)

            print('two model heads generated')

            paras_only_bn, paras_wo_bn = separate_bn_paras(self.model)
            
            if conf.use_mobilfacenet:
                self.optimizer = optim.SGD([
                                    {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                                    {'params': [paras_wo_bn[-1]] + [self.head.kernel], 'weight_decay': 4e-4},
                                    {'params': paras_only_bn}
                                ], lr = conf.lr, momentum = conf.momentum)
            else:
                self.optimizer = optim.SGD([
                                    {'params': paras_wo_bn + [self.head.kernel], 'weight_decay': 5e-4},
                                    {'params': paras_only_bn}
                                ], lr = conf.lr, momentum = conf.momentum)
            print(self.optimizer)
            # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=40, verbose=True)

            print('optimizers generated')   
            # if conf.data_mode == 'small_vgg':
            #     self.board_loss_every = len(self.loader)
            #     print('len(loader', len(self.loader))
            #     self.evaluate_every = len(self.loader)
            #     self.save_every = len(self.loader)
            #     # self.lfw, self.lfw_issame = get_val_data(conf, conf.smallvgg_folder)  

            # else:
            #     self.board_loss_every = len(self.loader)
                

            #     self.evaluate_every = len(self.loader)//10
            #     self.save_every = len(self.loader)//5
            self.agedb_30, self.cfp_fp, self.lfw, self.kface, self.agedb_30_issame, self.cfp_fp_issame, self.lfw_issame, self.kface_issame = get_val_data(conf, self.loader.dataset.root.parent)
                
        else:
            self.threshold = conf.threshold
    
    def save_state(self, conf, accuracy, e, loss, to_save_folder=False, extra=None, model_only=False):
        if to_save_folder:
            save_path = conf.save_path

        else:
            save_path = conf.model_path

        model_name = f'model_e:{e+1}_acc:{accuracy}_loss:{loss}_{extra}.pth'

        torch.save(
            self.model.state_dict(), save_path /model_name)
        if not model_only:
            # 똥째로 저장 
            state = {
                'model': self.model.state_dict(),
                'head': self.head.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }
            torch.save(state,  save_path /model_name)
            # print('model saved: ', model_name)

            
            ## 따로따로 저장
            # torch.save(
            #     self.head.state_dict(), save_path /
            #     ('head_{}_accuracy:{}_epoch:{}_step:{}_{}.pth'.format(get_time(), accuracy, e, self.step, extra)))
            # torch.save(
            #     self.optimizer.state_dict(), save_path /
            #     ('optimizer_{}_accuracy:{}_epoch:{}_step:{}_{}.pth'.format(get_time(), accuracy, e, self.step, extra)))
    
    def load_state(self, conf, fixed_str, from_save_folder=False, model_only=False):
        if from_save_folder:
            save_path = conf.save_path
        else:
            save_path = conf.model_path     
        if model_only:       
            self.model.load_state_dict(torch.load(save_path/'model_{}'.format(fixed_str)))
        if not model_only:
            model_name = 'model_{}'.format(fixed_str)
            state = torch.load(save_path/model_name)
            self.model.load_state_dict(state['model'])
            self.head.load_state_dict(state['head'])
            self.optimizer.load_state_dict(state['optimizer'])

            # self.head.load_state_dict(torch.load(save_path/'head_{}'.format(fixed_str)))
            # self.optimizer.load_state_dict(torch.load(save_path/'optimizer_{}'.format(fixed_str)))
        
    def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor):
        self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_scalar('{}_best_threshold'.format(db_name), best_threshold, self.step)
        self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)
#         self.writer.add_scalar('{}_val:true accept ratio'.format(db_name), val, self.step)
#         self.writer.add_scalar('{}_val_std'.format(db_name), val_std, self.step)
#         self.writer.add_scalar('{}_far:False Acceptance Ratio'.format(db_name), far, self.step)
        
    def evaluate(self, conf, carray, issame, nrof_folds = 5, tta = False):
        self.model.eval()
        idx = 0
        embeddings = np.zeros([len(carray), conf.embedding_size])
        with torch.no_grad():
            while idx + conf.batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + conf.batch_size])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings[idx:idx + conf.batch_size] = l2_norm(emb_batch)
                else:
                    embeddings[idx:idx + conf.batch_size] = self.model(batch.to(conf.device)).cpu()
                idx += conf.batch_size
            if idx < len(carray):
                batch = torch.tensor(carray[idx:])            
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(conf.device)) + self.model(fliped.to(conf.device))
                    embeddings[idx:] = l2_norm(emb_batch)
                else:
                    embeddings[idx:] = self.model(batch.to(conf.device)).cpu()
        tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve)
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor
    
    def find_lr(self,
                conf,
                init_value=1e-8,
                final_value=10.,
                beta=0.98,
                bloding_scale=3.,
                num=None):
        if not num:
            num = len(self.loader)
        mult = (final_value / init_value)**(1 / num)
        lr = init_value
        for params in self.optimizer.param_groups:
            params['lr'] = lr
        self.model.train()
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        for i, (imgs, labels) in tqdm(enumerate(self.loader), total=num):

            imgs = imgs.to(conf.device)
            labels = labels.to(conf.device)
            batch_num += 1          

            self.optimizer.zero_grad()

            embeddings = self.model(imgs)
            thetas = self.head(embeddings, labels)
            loss = conf.ce_loss(thetas, labels)          
          
            #Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            self.writer.add_scalar('avg_loss', avg_loss, batch_num)
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            self.writer.add_scalar('smoothed_loss', smoothed_loss,batch_num)
            #Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > bloding_scale * best_loss:
                print('exited with best_loss at {}'.format(best_loss))
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses
            #Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            #Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            self.writer.add_scalar('log_lr', math.log10(lr), batch_num)
            #Do the SGD step
            #Update the lr for the next step

            loss.backward()
            self.optimizer.step()

            lr *= mult
            for params in self.optimizer.param_groups:
                params['lr'] = lr
            if batch_num > num:
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses    

    def train(self, conf, epochs):
        self.model.train()
        running_loss = 0.   
        time_ = datetime.datetime.now()


         # check parameter of model
        print("------------------------------------------------------------")
        total_params = sum(p.numel() for p in self.model.parameters())
        print("num of parameter : ", total_params)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("num of trainable_ parameter :", trainable_params)
        print("------------------------------------------------------------")

        
        # if conf.data_mode == 'small_vgg':
        for e in range(epochs):
            print('epoch {} started'.format(e))
            if e == self.milestones[0]:
                self.schedule_lr()
            if e == self.milestones[1]:
                self.schedule_lr()      
            if e == self.milestones[2]:
                self.schedule_lr()

            accuracy_list = []
            for iter_, (imgs, labels) in tqdm(enumerate(iter(self.loader))):
                # print('iter_', type(iter_))
                # print('step', self.step)
                imgs = imgs.to(conf.device)
                labels = labels.to(conf.device)
                self.optimizer.zero_grad()
                embeddings = self.model(imgs)
                thetas = self.head(embeddings, labels)
                loss = conf.ce_loss(thetas, labels)
                loss.backward()
                running_loss += loss.item()
                self.optimizer.step()

                if iter_ % conf.print_iter == 0 :
                    elapsed = datetime.datetime.now() - time_
                    expected = elapsed * (conf.batch_size / conf.print_iter)
                    _epoch = round(e + ((iter_ + 1) / conf.batch_size),2)
                    # print('_epoch', _epoch)
                    _loss = round(loss.item(), 5)
                    print(f'[{_epoch}/{conf.epochs}] loss:{_loss}, elapsed:{elapsed}, expected per epoch: {expected}')
                    time_ = datetime.datetime.now()

            self.step += 1

            # log 남기기
            board_loss_every = len(self.loader) / conf.print_iter
            print('board_loss_ebery', board_loss_every)
            loss_board = running_loss / board_loss_every
            self.writer.add_scalar('train_loss', loss_board, self.step)
            running_loss = 0.

            # validate 
            accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.agedb_30, self.agedb_30_issame)
            self.board_val('agedb_30', accuracy, best_threshold, roc_curve_tensor)
            accuracy_list.append(accuracy)

            accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.lfw, self.lfw_issame)
            self.board_val('lfw', accuracy, best_threshold, roc_curve_tensor)
            accuracy_list.append(accuracy)
            
            accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.cfp_fp, self.cfp_fp_issame)
            self.board_val('cfp_fp', accuracy, best_threshold, roc_curve_tensor)
            accuracy_list.append(accuracy)

            accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.cfp_fp, self.cfp_fp_issame)
            self.board_val('kface', accuracy, best_threshold, roc_curve_tensor)
            accuracy_list.append(accuracy)

            # save model, info
            # print(accuracy_list)
            accuracy_mean = round(sum(accuracy_list)/conf.testset_num, 5)
            loss_mean = round(loss_board, 5)
            self.save_state(conf,accuracy_mean,e, loss_mean, to_save_folder = False, extra = 'training')
            time_ = datetime.datetime.now()
            elapsed = datetime.datetime.now() - time_
            print(f'[epoch {e + 1}] acc: {accuracy_mean}, loss: {loss_mean}, elapsed: {elapsed}')
            # print('train_loss:', loss_board) 

        self.save_state(conf,accuracy_mean,e, loss_mean, to_save_folder = False, extra = 'final')



    def schedule_lr(self):
        for params in self.optimizer.param_groups:                 
            params['lr'] /= 10
        print('***self.optimizer:',self.optimizer)
    
    def infer(self, conf, faces, target_embs, tta=False):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        for img in faces:
            if tta:
                mirror = trans.functional.hflip(img)
                emb = self.model(conf.test_transform(img).to(conf.device).unsqueeze(0))
                emb_mirror = self.model(conf.test_transform(mirror).to(conf.device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:                        
                embs.append(self.model(conf.test_transform(img).to(conf.device).unsqueeze(0)))
        source_embs = torch.cat(embs)
        
        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1,0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1 # if no match, set idx to -1
        return min_idx, minimum               