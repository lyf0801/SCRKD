#!/usr/bin/python3
#coding=utf-8
import torch.optim
from misc import AvgMeter, check_mkdir
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root) 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm

from model.ISAANet import ISAANet as Net
from ORSI_SOD_dataset import ORSI_SOD_Dataset
from evaluator import *

args = {
    'iter_num': 0,
    'epoch': 60, # lyf add: follow line 16 (parser.add_argument('--epoch', type=int, default=60, help='epoch number')) in https://github.com/YiuCK/ISAANet/blob/main/train.py
    'train_batch_size': 8,
    'last_iter': 0,
    'snapshot': ''
}

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

torch.manual_seed(2025)




def _iou(pred, target, size_average = True):

    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
        Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand1
        IoU1 = Iand1/Ior1

        #IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)

    return IoU/b

class IOU(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):

        return _iou(pred, target, self.size_average)


BCE = nn.BCEWithLogitsLoss().cuda()
DL = IOU(size_average=True).cuda()
"""
lyf add:
wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_large_100_ra-f55367f5.pth
"""
pretrained_cfg = timm.models.create_model('mobilenetv3_large_100').default_cfg
pretrained_cfg['file'] = r'./mobilenetv3_large_100_ra-f55367f5.pth'



def main(dataset):
    #define dataset
    data_path = '/data1/users/liuyanfeng/RSI_SOD/'+ dataset +' dataset/'
    train_dataset = ORSI_SOD_Dataset(root = data_path,  img_size = 112,  mode = 'train', aug = True)
    train_loader = DataLoader(train_dataset, batch_size = args['train_batch_size'], shuffle = True, num_workers = 8)
    test_set = ORSI_SOD_Dataset(root = data_path,  img_size = 112, mode = "test", aug = False) 
    test_loader = DataLoader(test_set, batch_size = 1,  num_workers = 1)
    args['iter_num'] = args['epoch'] * len(train_loader)

    if not os.path.exists("./data/ISAANet_112x112_"+ dataset):
        os.makedirs("./data/ISAANet_112x112_"+ dataset)
    
    #############################ResNet pretrained###########################
    #res18[2,2,2,2],res34[3,4,6,3],res50[3,4,6,3],res101[3,4,23,3],res152[3,8,36,3]
    model = Net(pretrained_cfg)
    ##############################Optim setting###############################
    net = model.cuda().train()
    # lyf add: follow line 69 in https://github.com/YiuCK/ISAANet/blob/main/train.py
    optimizer = torch.optim.Adam(params=filter(lambda x: x.requires_grad, model.parameters()), lr=1e-3, betas=(0.5, 0.999), eps=1e-08)
    from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
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
            #sal2, sal4, sal6
            s1, s2, s3 = net(inputs)
            ##########loss#############
            #parser.add_argument('--lanmb', type=float, default=3, help='lanmb1')
            loss1 = BCE(s1, labels) + 3 * DL(torch.sigmoid(s1), labels)
            loss2 = BCE(s2, labels) + 3 * DL(torch.sigmoid(s2), labels)
            loss3 = BCE(s3, labels) + 3 * DL(torch.sigmoid(s3), labels)

            total_loss = loss1 + loss2 + loss3
            total_loss.backward()
            
            optimizer.step()
            

            loss_sod_record.update(total_loss.item(), batch_size)

            total_loss_record.update(total_loss.item(), batch_size)

            #############log###############
            curr_iter += 1
            if curr_iter % 20 == 0:
                log = '[epoch: %03d] [iter: %05d] [total loss %.5f]  [sod loss %.5f]   [lr %.13f] ' % \
                    (epoch, curr_iter, total_loss_record.avg,  loss_sod_record.avg,   optimizer.param_groups[0]['lr'])
                print(log)

        if epoch >= 40:   
            #eval
            thread = Eval_thread(epoch = epoch, model = net.eval(), loader = test_loader, method = "ISAANet_112x112", dataset = dataset, txt_name="ISAANet_112x112_{}.txt".format(dataset), output_dir = "./data/output", cuda=True)
            logg, fm = thread.run()
            print(logg)
            if fm > max_fm:
                max_fm = fm
                torch.save(net.state_dict(), "./data/ISAANet_112x112_"+ dataset +'/epoch_{}_{}.pth'.format(epoch,fm))

        scheduler.step()

if __name__ == '__main__':
    datasets = ["ORSSD","EORSSD", "ORS_4199"]
    for dataset in datasets:
        main(dataset)

