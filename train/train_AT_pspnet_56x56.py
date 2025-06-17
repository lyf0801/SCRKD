#!/usr/bin/python3
#coding=utf-8
import torch.backends
from misc import AvgMeter, check_mkdir
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root) 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


from model.pspnet_AT import PSPNet_AT as Net
from ORSI_SOD_dataset_KD import ORSI_SOD_Dataset
from evaluator_KD import *
import random

args = {
    'iter_num': 0,
    'epoch': 100,
    'train_batch_size': 8,
    'last_iter': 0,
    'snapshot': ''
}

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

torch.manual_seed(2025)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

random.seed(0)


'''
Distilling 112 teacher to 56 student
'''

def main(scale):
    #define dataset
    data_path = '/data1/users/liuyanfeng/RSI_SOD/EORSSD dataset/'
    train_dataset = ORSI_SOD_Dataset(root = data_path,  mode = 'train', aug = True)
    train_loader = DataLoader(train_dataset, batch_size = args['train_batch_size'], shuffle = True, num_workers = 8)
    test_set = ORSI_SOD_Dataset(root = data_path,  mode = 'test', aug = False) 
    test_loader = DataLoader(test_set, batch_size = 1, shuffle = True, num_workers = 1)
    args['iter_num'] = args['epoch'] * len(train_loader)

    #############################ResNet pretrained###########################
    #res18[2,2,2,2],res34[3,4,6,3],res50[3,4,6,3],res101[3,4,23,3],res152[3,8,36,3]

    model = Net()
    if scale == '112':
        model.teacher.load_state_dict(torch.load('./data/PSPNet_112x112_EORSSD/epoch_97_0.7715491652488708.pth'))
    elif scale == '224':
        model.teacher.load_state_dict(torch.load('./data/PSPNet_224x224_EORSSD/epoch_99_0.8390894532203674.pth'))
    elif scale == '448':
        model.teacher.load_state_dict(torch.load('./data/PSPNet_448x448_EORSSD/epoch_81_0.8730648159980774.pth'))
    
    if not os.path.exists('./data/PSPNet_AT_' + scale + 'to56_EORSSD'):
        os.makedirs('./data/PSPNet_AT_' + scale + 'to56_EORSSD')
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
            if scale == '448':
                teacher_inputs = data['img_448x448']
            elif scale == '224':
                teacher_inputs = data['img_224x224']
            elif scale == '112':
                teacher_inputs = data['img_112x112']

            inputs_56x56 = data['img_56x56']
            labels_56x56 = data['label_56x56']
            batch_size = inputs_56x56.size(0)

            teacher_inputs = teacher_inputs.type(torch.FloatTensor).cuda()
            inputs_56x56 = inputs_56x56.type(torch.FloatTensor).cuda()
            labels_56x56 = labels_56x56.type(torch.FloatTensor).cuda()

            optimizer.zero_grad()
            loss_sod, loss_distillation = net(inputs_56x56, teacher_inputs, labels_56x56)
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
            
        #eval
        thread = Eval_thread(epoch = epoch, model = net.eval(), loader = test_loader, method = 'PSPNet_AT_' + scale + 'to56', student_size= 56, dataset = 'EORSSD', txt_name='PSPNet_AT_' + scale + 'to56_EORSSD.txt', output_dir = './data/output', cuda=True)
        logg, fm = thread.run()
        print(logg)
        if fm > max_fm:
            max_fm = fm
            torch.save(net.state_dict(), './data/PSPNet_AT_' + scale + 'to56_EORSSD/epoch_{}_{}.pth'.format(epoch,fm))

if __name__ == '__main__':
    teacher_scales = ['448', '224', '112']
    for scale in teacher_scales:
        main(scale)

