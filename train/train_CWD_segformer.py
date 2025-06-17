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


from model.segformer_CWD import SegFormer_CWD as Net
from ORSI_SOD_dataset_KD import ORSI_SOD_Dataset
from evaluator_KD import *

args = {
    'iter_num': 0,
    'epoch': 100,
    'train_batch_size': 8,
    'last_iter': 0,
    'snapshot': ''
}

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

torch.manual_seed(2025)



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
    model = Net(img_size=(112, 112))
    if dataset == 'ORSSD':
        model.teacher.load_state_dict(torch.load("./data/SegFormer_224x224_ORSSD/epoch_94_0.8900975584983826.pth"))
    elif dataset == "EORSSD":
        model.teacher.load_state_dict(torch.load("./data/SegFormer_224x224_EORSSD/epoch_98_0.8526952862739563.pth"))
    elif dataset == "ORS_4199":
        model.teacher.load_state_dict(torch.load("./data/SegFormer_224x224_ORS_4199/epoch_88_0.8617156147956848.pth"))
    
    if not os.path.exists("./data/SegFormer_CWD_224to112_"+ dataset):
        os.makedirs("./data/SegFormer_CWD_224to112_"+ dataset)
    ##############################Optim setting###############################
    net = model.cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    #########################################################################
    max_fm = 0
    curr_iter = args['last_iter']
    for epoch in range(0, args['epoch']): 
        loss_sod_record, loss_distillation_record, total_loss_record = AvgMeter(),  AvgMeter(), AvgMeter()
        net.train() 
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
            loss_sod, loss_distillation = net(inputs_112x112, inputs_224x224, labels_112x112)
            ##########loss#############

           
            total_loss = loss_sod + loss_distillation
            total_loss.backward()
            
            optimizer.step()
            

            loss_sod_record.update(loss_sod.item(), batch_size)
            loss_distillation_record.update(loss_distillation.item(), batch_size)
            total_loss_record.update(total_loss.item(), batch_size)

            #############log###############
            curr_iter += 1
            if curr_iter % 20 == 0:
                log = '[epoch: %03d] [iter: %05d] [total loss %.5f]  [sod loss %.5f]  [distillation loss %.5f]  [lr %.13f] ' % \
                    (epoch, curr_iter, total_loss_record.avg,  loss_sod_record.avg,  loss_distillation_record.avg, optimizer.param_groups[0]['lr'])
                print(log)

        if epoch >= 60:
            #eval
            thread = Eval_thread(epoch = epoch, model = net.eval(), loader = test_loader, method = "SegFormer_CWD_224to112", student_size= 112, dataset = dataset, txt_name="SegFormer_CWD_224to112_{}.txt".format(dataset), output_dir = "./data/output", cuda=True)
            logg, fm = thread.run()
            print(logg)
            if fm > max_fm:
                max_fm = fm
                torch.save(net.state_dict(), "./data/SegFormer_CWD_224to112_"+ dataset +'/epoch_{}_{}.pth'.format(epoch,fm))

if __name__ == '__main__':
    datasets = ["EORSSD"]
    for dataset in datasets:
        main(dataset)

