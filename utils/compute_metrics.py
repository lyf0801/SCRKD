from pickle import FALSE
import numpy as np
import os
from PIL import Image
from compute_evaluator import Eval_thread
from ORSI_SOD_dataset import ORSI_SOD_Dataset
from torch.utils.data import DataLoader
import os
import time
from torch import nn
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import cv2
from PIL import Image

if __name__ == '__main__':  
    method_names = ['TransXNet_SCRKD5_224to112']
    print("一共" + str(len(method_names)) + "种对比算法")
    dataset_name = ["ORSSD", "EORSSD", "ORS_4199"]#
    for method_name in method_names:
        for dataseti in dataset_name:

            root = '/data1/users/liuyanfeng/RSI_SOD/'+ dataseti +' dataset/'
            smap_path = "./data/output/predict_smaps_" + method_name + "_" + dataseti + "/"
            prefixes = [line.strip() for line in open(os.path.join(root, 'test.txt'))]

          
            test_set = ORSI_SOD_Dataset(root = root, mode = "test", img_size = 448, aug = False)
            test_loader = DataLoader(test_set, batch_size = 1, num_workers = 1)  

            thread = Eval_thread(smap_path=smap_path, loader = test_loader, method = method_name, dataset = dataseti, output_dir = "./data/", cuda=True)
            logg, fm = thread.run()
            print(logg)
