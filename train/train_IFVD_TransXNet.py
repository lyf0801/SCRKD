#!/usr/bin/python3
#coding=utf-8
from misc import AvgMeter, check_mkdir
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root) 
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random

from model.TransXNet_IFVD import TransXNet_IFVD as Net
from ORSI_SOD_dataset_KD import ORSI_SOD_Dataset
from evaluator_KD import *

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
Distilling 224 teacher to 112 student
"""

def main(dataset):
    #define dataset
    data_path = '/data1/users/liuyanfeng/RSI_SOD/'+ dataset +' dataset/'
    train_dataset = ORSI_SOD_Dataset(root = data_path,  mode = 'train', aug = True)
    train_loader = DataLoader(train_dataset, batch_size = args['train_batch_size'], shuffle = True, num_workers = 8)
    test_set = ORSI_SOD_Dataset(root = data_path,  mode = "test", aug = False) 
    test_loader = DataLoader(test_set, batch_size = 1, shuffle = True, num_workers = 1)
    args['iter_num'] = args['epoch'] * len(train_loader)

    #############################ResNet pretrained###########################
    #res18[2,2,2,2],res34[3,4,6,3],res50[3,4,6,3],res101[3,4,23,3],res152[3,8,36,3]
    model = Net(img_size=112)
    if dataset == 'ORSSD':
        model.teacher.load_state_dict(torch.load("./data/TransXNet_224x224_ORSSD/epoch_97_0.9005411863327026.pth"))
    elif dataset == "EORSSD":
        model.teacher.load_state_dict(torch.load("./data/TransXNet_224x224_EORSSD/epoch_97_0.8603147864341736.pth"))
    elif dataset == "ORS_4199":
        model.teacher.load_state_dict(torch.load("./data/TransXNet_224x224_ORS_4199/epoch_90_0.8695002198219299.pth"))
    
    if not os.path.exists("./data/TransXNet_IFVD_224to112_"+ dataset):
        os.makedirs("./data/TransXNet_IFVD_224to112_"+ dataset)
    ##############################Optim setting###############################
    net = model.cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(optimizer,step_size=50, gamma=0.1)

    #########################################################################
    max_fm = 0
    curr_iter = args['last_iter']
    for epoch in range(0, args['epoch']): 
        loss_sod_record, loss_distillation_record, loss_discriminator_record, total_loss_record = AvgMeter(),  AvgMeter(), AvgMeter(),  AvgMeter()
        net.train() 
        epoch_start = time.time()  # 单个epoch计时 
        for i, data in enumerate(train_loader):
            
            # data\binarizing\Variable
            inputs_224x224 = data['img_224x224']
            inputs_112x112 = data['img_112x112']
            labels_112x112 = data['label_112x112']
            batch_size = inputs_224x224.size(0)

            inputs_224x224 = inputs_224x224.type(torch.FloatTensor).cuda()
            inputs_112x112 = inputs_112x112.type(torch.FloatTensor).cuda()
            labels_112x112 = labels_112x112.type(torch.FloatTensor).cuda()

            optimizer.zero_grad()
            loss_sod, loss_distillation, student_smap1, teacher_smap1 = net(inputs_112x112, inputs_224x224, labels_112x112)
            ##########loss#############

           
            total_loss = loss_sod + loss_distillation
            total_loss.backward()
            d_loss = net.Discriminator_backward(student_smap1, teacher_smap1)
            d_loss.backward()
            optimizer.step()
            

            loss_sod_record.update(loss_sod.item(), batch_size)
            loss_distillation_record.update(loss_distillation.item(), batch_size)
            total_loss_record.update(total_loss.item(), batch_size)
            loss_discriminator_record.update(d_loss.item(), batch_size)
            #############log###############
            curr_iter += 1
            if curr_iter % 20 == 0:
                log = '[epoch: %03d] [iter: %05d] [total loss %.5f]  [sod loss %.5f]  [distillation loss %.5f]  [discriminator loss %.5f]  [lr %.13f] ' % \
                    (epoch, curr_iter, total_loss_record.avg,  loss_sod_record.avg, loss_distillation_record.avg, loss_discriminator_record.avg, optimizer.param_groups[0]['lr'])
                print(log)

        #打印训练一个epoch的时间
        if epoch == 0:
            epoch_time = time.time() - epoch_start
            log_text = f'\n TransXNet_IFVD Epoch training time: {epoch_time:.2f}s'
            # 使用标准open写入（追加模式）
            with open('training_time.txt', 'a', encoding='utf-8') as f:
                f.write(log_text)
            import sys
            sys.exit()  # 退出码默认为0

        if epoch >= 70:    
            #eval
            thread = Eval_thread(epoch = epoch, model = net.eval(), loader = test_loader, method = "TransXNet_IFVD_224to112", student_size= 112, dataset = dataset, txt_name="TransXNet_IFVD_224to112_{}.txt".format(dataset), output_dir = "./data/output", cuda=True)
            logg, fm = thread.run()
            print(logg)
            if fm > max_fm:
                max_fm = fm
                torch.save(net.state_dict(), "./data/TransXNet_IFVD_224to112_"+ dataset +'/epoch_{}_{}.pth'.format(epoch,fm))

        scheduler.step()  # 每个epoch更新一次

if __name__ == '__main__':
    datasets = ["ORS_4199"]
    for dataset in datasets:
        main(dataset)

