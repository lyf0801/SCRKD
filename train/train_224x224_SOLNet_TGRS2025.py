#!/usr/bin/python3
#coding=utf-8
from misc import AvgMeter, check_mkdir
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root) 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


from model.SOLNet import SOLNet as Net
from ORSI_SOD_dataset import ORSI_SOD_Dataset
from evaluator import *

args = {
    'iter_num': 0,
    'epoch': 120, # lyf add: according to https://github.com/SpiritAshes/SOLNet/blob/main/config/SOLNet.yaml epoch: 120
    'train_batch_size': 32, # lyf add: according to https://github.com/SpiritAshes/SOLNet/blob/main/config/SOLNet.yaml batchsize: 32
    'last_iter': 0,
    'snapshot': ''
}

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

torch.manual_seed(2025)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean', eps=1e-10):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.eps = eps


    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        alpha = self.alpha
        eps = self.eps
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt + eps) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt + eps)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        elif self.reduction =='False':
            loss = loss
        return loss


def Fbeta_Measure(pred, target):

    b = pred.shape[0]
    precision_mean = 0.0
    recall_mean = 0.0
    Fbeta_mean = 0.0
    for i in range(0, b):
        #compute the IoU of the foreground
        precision = torch.sum(target[i, :, :, :] * pred[i, :, :, :]) / (torch.sum(pred[i, :, :, :]) + 1e-10)
        recall = torch.sum(target[i, :, :, :] * pred[i, :, :, :]) / (torch.sum(target[i, :, :, :]) + 1e-10)
        
        # precision = torch.sum(torch.logical_and(pred, target).float()) / (torch.sum(pred[i, :, :, :]) + 1e-10)
        # recall = torch.sum(torch.logical_and(pred, target).float()) / (torch.sum(target[i, :, :, :]) + 1e-10)

        Fbeta = (1.3 * precision * recall) / ((0.3 * precision + recall) + 1e-10)

        precision_mean += precision
        recall_mean += recall
        Fbeta_mean += Fbeta
    

    return precision_mean / b, recall_mean / b, Fbeta_mean / b

def IOU(pred, gt, extra=False, eps=1e-8):
    inter = (gt * pred).sum(dim=(1, 2, 3))
    union = (gt + pred).sum(dim=(1, 2, 3)) - inter
    iou_loss = 1 - inter / (union + eps)

    # background = (gt == 0).float()

    # background_inter = (background * pred).sum(dim=(1, 2, 3))
    # background_union = background.sum(dim=(1, 2, 3))

    # background_loss = background_inter / (background_union + eps)

    # loss = iou_loss + background_loss

    loss = iou_loss.mean()
    if extra:
        precision = inter / (pred.sum(dim=(1, 2, 3)) + 1e-10)
        recall = inter / (gt.sum(dim=(1, 2, 3)) + 1e-10)
        Fbeta = (1.3 * precision * recall) / ((0.3 * precision + recall) + 1e-10)
        return loss, precision.mean(), recall.mean(), Fbeta.mean()
    return loss

def MAE_Measure(pred, target):
    mae = F.l1_loss(pred, target, reduction='mean')
    return mae

    
def Structure_Measure(pred, gt):
    gt = gt.float()
    h, w = pred.size()[-2:]
    N = h * w
    x = pred.mean()
    y = gt.mean()
    sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
    sigma_y2 = ((gt - y) * (gt - y)).sum() / (N - 1 + 1e-20)
    sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-20)

    aplha = 4 * x * y * sigma_xy
    beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

    if aplha != 0:
        Q = aplha / (beta + 1e-20)
    elif aplha == 0 and beta == 0:
        Q = torch.tensor(1.0)
    else:
        Q = torch.tensor(0.0)
    return Q



BCE = torch.nn.BCEWithLogitsLoss()

def loss_func(feature_1, feature_2, feature_3, feature_4, feature_1_sig, feature_2_sig, feature_3_sig, feature_4_sig, gts):
    train_precision_4, train_recall_4, train_Fbeta_4 = Fbeta_Measure(feature_4_sig, gts)
    train_MAE = MAE_Measure(feature_4_sig, gts)
    train_Smearsure = Structure_Measure(feature_4_sig, gts)

    loss1 = BCE(feature_1, gts) + IOU(feature_1_sig, gts) + (1 - train_Fbeta_4)
    loss2 = BCE(feature_2, gts) + IOU(feature_2_sig, gts) + (1 - train_Fbeta_4)
    loss3 = BCE(feature_3, gts) + IOU(feature_3_sig, gts) + (1 - train_Fbeta_4)
    loss4 = BCE(feature_4, gts) + IOU(feature_4_sig, gts) + (1 - train_Fbeta_4)

    loss = loss1 + loss2 + loss3 + loss4
    return loss, loss4, train_precision_4, train_recall_4, train_Fbeta_4, train_MAE, train_Smearsure

def main(dataset):
    #define dataset
    data_path = '/data1/users/liuyanfeng/RSI_SOD/'+ dataset +' dataset/'
    train_dataset = ORSI_SOD_Dataset(root = data_path,  img_size = 224,  mode = 'train', aug = True)
    train_loader = DataLoader(train_dataset, batch_size = args['train_batch_size'], shuffle = True, num_workers = 8)
    test_set = ORSI_SOD_Dataset(root = data_path,  img_size = 224, mode = "test", aug = False) 
    test_loader = DataLoader(test_set, batch_size = 1,  num_workers = 1)
    args['iter_num'] = args['epoch'] * len(train_loader)

    if not os.path.exists("./data/SOLNet_224x224_"+ dataset):
        os.makedirs("./data/SOLNet_224x224_"+ dataset)
    
    #############################ResNet pretrained###########################
    #res18[2,2,2,2],res34[3,4,6,3],res50[3,4,6,3],res101[3,4,23,3],res152[3,8,36,3]
    model = Net()
    ##############################Optim setting###############################
    net = model.cuda().train()
    # lyf add: according to line 26 in https://github.com/SpiritAshes/SOLNet/blob/main/train.py
    optimizer = torch.optim.Adam(model.parameters(), 4e-4)
    from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(optimizer, step_size=80, gamma=0.1)
    #########################################################################
    max_fm = 0
    curr_iter = args['last_iter']
    for epoch in range(0, args['epoch']): 
        loss_sod_record,  total_loss_record = AvgMeter(),  AvgMeter()
        net.train() 
        for i, data in enumerate(train_loader):
            
            # data\binarizing\Variable
            inputs, labels, _, _ = data
            batch_size = inputs.size(0)

            inputs = inputs.type(torch.FloatTensor).cuda()
            labels = labels.type(torch.FloatTensor).cuda()

            optimizer.zero_grad()
            #feature_1, feature_2, feature_3, x, self.sigmoid(feature_1), self.sigmoid(feature_2), self.sigmoid(feature_3), self.sigmoid(x)
            feature_1, feature_2, feature_3, feature_4, feature_1_sig, feature_2_sig, feature_3_sig, feature_4_sig = net(inputs)
            ##########loss#############

            loss, loss4, train_precision_4, train_recall_4, train_Fbeta_4, train_MAE, train_Smearsure = loss_func(feature_1, feature_2, feature_3, feature_4, feature_1_sig, feature_2_sig, feature_3_sig, feature_4_sig, labels)
           
            total_loss = loss
            total_loss.backward()
            
            optimizer.step()
            

            loss_sod_record.update(loss.item(), batch_size)

            total_loss_record.update(total_loss.item(), batch_size)

            #############log###############
            curr_iter += 1
            if curr_iter % 20 == 0:
                log = '[epoch: %03d] [iter: %05d] [total loss %.5f]  [sod loss %.5f]   [lr %.13f] ' % \
                    (epoch, curr_iter, total_loss_record.avg,  loss_sod_record.avg,   optimizer.param_groups[0]['lr'])
                print(log)

        if epoch >= 80:   
            #eval
            thread = Eval_thread(epoch = epoch, model = net.eval(), loader = test_loader, method = "SOLNet_224x224", dataset = dataset, txt_name="SOLNet_224x224_{}.txt".format(dataset), output_dir = "./data/output", cuda=True)
            logg, fm = thread.run()
            print(logg)
            if fm > max_fm:
                max_fm = fm
                torch.save(net.state_dict(), "./data/SOLNet_224x224_"+ dataset +'/epoch_{}_{}.pth'.format(epoch,fm))

        scheduler.step()

if __name__ == '__main__':
    datasets = ["ORSSD","EORSSD", "ORS_4199"]
    for dataset in datasets:
        main(dataset)

