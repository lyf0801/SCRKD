#!/usr/bin/python3
#coding=utf-8
from misc import AvgMeter, check_mkdir
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root) 
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random

from model.TransXNet import TransXNet as Net
from ORSI_SOD_dataset import ORSI_SOD_Dataset
from evaluator import *

args = {
    'iter_num': 0,
    'epoch': 100,
    'train_batch_size': 8,
    'last_iter': 0,
    'snapshot': ''
}



torch.manual_seed(2025)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(0)

"""
  smaps : BCE + wIOU
  edges: BCE
"""
def structure_loss(pred, mask):
    #mask = mask.detach()
    wbce  = F.binary_cross_entropy_with_logits(pred, mask)
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return wbce.mean()+wiou.mean()#



def main(dataset):
    #define dataset
    data_path = '/data1/users/liuyanfeng/RSI_SOD/'+ dataset +' dataset/'
    train_dataset = ORSI_SOD_Dataset(root = data_path,  img_size = 112,  mode = 'train', aug = True)
    train_loader = DataLoader(train_dataset, batch_size = args['train_batch_size'], shuffle = True, num_workers = 8)
    test_set = ORSI_SOD_Dataset(root = data_path,  img_size = 112, mode = "test", aug = False) 
    test_loader = DataLoader(test_set, batch_size = 1, shuffle = True, num_workers = 1)
    args['iter_num'] = args['epoch'] * len(train_loader)

    if not os.path.exists("./data/TransXNet_112x112_"+ dataset):
        os.makedirs("./data/TransXNet_112x112_"+ dataset)
    
    #############################ResNet pretrained###########################
    #res18[2,2,2,2],res34[3,4,6,3],res50[3,4,6,3],res101[3,4,23,3],res152[3,8,36,3]
    model = Net(image_size=112)
    ##############################Optim setting###############################
    net = model.cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5, weight_decay=1e-4)

    #########################################################################
    max_fm = 0
    curr_iter = args['last_iter']
    for epoch in range(0, args['epoch']): 
        loss_sod_record,  total_loss_record = AvgMeter(),  AvgMeter()
        net.train() 
        epoch_start = time.time()  # 单个epoch计时 
        for i, data in enumerate(train_loader):
            
            # data\binarizing\Variable
            inputs, labels, _, _ = data
            batch_size = inputs.size(0)

            inputs = inputs.type(torch.FloatTensor).cuda()
            labels = labels.type(torch.FloatTensor).cuda()

            optimizer.zero_grad()
            smap1, features = net(inputs)
            ##########loss#############

            loss_sod = structure_loss(smap1, labels)
           
            total_loss = loss_sod 
            total_loss.backward()
            
            optimizer.step()
            loss_sod_record.update(loss_sod.item(), batch_size)

            total_loss_record.update(total_loss.item(), batch_size)

            #############log###############
            curr_iter += 1
            if curr_iter % 20 == 0:
                log = '[epoch: %03d] [iter: %05d] [total loss %.5f]  [sod loss %.5f]   [lr %.13f] ' % \
                    (epoch, curr_iter, total_loss_record.avg,  loss_sod_record.avg,   optimizer.param_groups[0]['lr'])
                print(log)
        
        #打印训练一个epoch的时间
        if epoch == 0:
            epoch_time = time.time() - epoch_start
            log_text = f'\n TransXNet_112x112 Epoch training time: {epoch_time:.2f}s'
            # 使用标准open写入（追加模式）
            with open('training_time.txt', 'a', encoding='utf-8') as f:
                f.write(log_text)
            import sys
            sys.exit()  # 退出码默认为0

        if epoch >=70:
            #eval
            thread = Eval_thread(epoch = epoch, model = net.eval(), loader = test_loader, method = "TransXNet_112x112", dataset = dataset, txt_name="TransXNet_112x112_{}.txt".format(dataset), output_dir = "./data/output", cuda=True)
            logg, fm = thread.run()
            print(logg)
            if fm > max_fm:
                max_fm = fm
                torch.save(net.state_dict(), "./data/TransXNet_112x112_"+ dataset +'/epoch_{}_{}.pth'.format(epoch,fm))
    


if __name__ == '__main__':
    datasets = ["ORS_4199"]#"ORSSD","EORSSD",
    for dataset in datasets:
        main(dataset)

