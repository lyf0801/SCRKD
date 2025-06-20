import os
import time
from torch import nn
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
import cv2
from PIL import Image

class Eval_thread():
    def __init__(self, smap_path, loader, method, dataset, output_dir, cuda = False):
        
        self.smap_path = smap_path
        self.loader = loader
        self.method = method
        self.dataset = dataset
        self.cuda = cuda
        self.output_dir = output_dir
        self.logfile = os.path.join(output_dir, 'results'  + '.txt')
        if not os.path.exists(self.logfile):
            open(self.logfile, mode = 'w',encoding='utf-8')
        self.transformation =  transforms.Compose([transforms.ToTensor()])
        
    def run(self):
        Res = {}
        start_time = time.time()
        mae = self.Eval_mae()
        Res['MAE'] = mae

        Fm, prec, recall = self.Eval_fmeasure()
        max_f = Fm.max().item()
        # mean_f = Fm.mean().item()
        prec = prec.cpu().numpy()
        recall = recall.cpu().numpy()
        # avg_p = self.Eval_AP(prec, recall)  # AP
        Fm = Fm.cpu().numpy()
        Res['MaxFm'] = max_f
        # Res['MeanFm'] = mean_f
        # Res['AP'] = avg_p
        Res['Prec'] = prec
        Res['Recall'] = recall
        Res['Fm'] = Fm

        # auc, TPR, FPR = self.Eval_auc()
        # TPR = TPR.cpu().numpy()
        # FPR = FPR.cpu().numpy()

        # Res['AUC'] = auc
        # Res['TPR'] = TPR
        # Res['FPR'] = FPR

        # Em = self.Eval_Emeasure()
        # max_e = Em.max().item()
        # mean_e = Em.mean().item()
        # Em = Em.cpu().numpy()
        # Res['MaxEm'] = max_e
        # Res['MeanEm'] = mean_e
        # Res['Em'] = Em

        s = self.Eval_Smeasure()
        Res['Sm'] = s
        
        os.makedirs(os.path.join(self.output_dir, 'Detail'), exist_ok=True)
        torch.save(
            Res,
            os.path.join(self.output_dir, 'Detail',
                          self.dataset + '_' + self.method + '.pth'))
        
        self.LOG(
            '{} ({}): {:.4f} mae || {:.4f} max-fm || {:.4f} S-measure.\n'
            .format( self.dataset, self.method, mae, max_f, s))
        
        return '[cost:{:.4f}s] {} ({}): {:.4f} mae || {:.4f} max-fm || {:.4f} S-measure.'.format(
            time.time() - start_time, self.dataset, self.method, mae, max_f, s), max_f

    def Eval_mae(self):
        print('eval[MAE]:{} dataset with {} method.'.format(
            self.dataset, self.method))
        avg_mae, img_num = 0.0, 0.0
        with torch.no_grad():
            
            for img, gt, edge, name in tqdm(self.loader):  
                
                
                path = self.smap_path + name[0] + "_" + self.method + ".png"
                
                
                pred = self.transformation(Image.open(path))

                if self.cuda:
                    pred = pred.cuda()
                    gt = gt.cuda()
                mea = torch.abs(pred - gt).mean()
                if mea == mea:  # for Nan
                    avg_mae += mea
                    img_num += 1.0
            avg_mae /= img_num
            return avg_mae.item()

    def Eval_fmeasure(self):
        print('eval[FMeasure]:{} dataset with {} method.'.format(self.dataset, self.method))
        beta2 = 0.3
        avg_f, avg_p, avg_r, img_num = 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            
            for img, gt, edge, name in tqdm(self.loader):
                
                path = self.smap_path + name[0] + "_" + self.method + ".png"
                
                
                pred = self.transformation(Image.open(path))
                
                pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                if self.cuda:
                    pred = pred.cuda()
                    gt = gt.cuda()

                prec, recall = self._eval_pr(pred, gt, 255)
                f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
                f_score[f_score != f_score] = 0  # for Nan
                avg_f += f_score
                avg_p += prec
                avg_r += recall
                img_num += 1.0
            Fm = avg_f / img_num
            avg_p = avg_p / img_num
            avg_r = avg_r / img_num
            return Fm, avg_p, avg_r

    def Eval_auc(self):
        print('eval[AUC]:{} dataset with {} method.'.format(
            self.dataset, self.method))

        avg_tpr, avg_fpr, avg_auc, img_num = 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            
            for img, gt, edge, name in tqdm(self.loader):
                
                path = self.smap_path + name[0] + "_" + self.method + ".png"
                
                
                pred = self.transformation(Image.open(path))
                
                
                pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                if self.cuda:
                    pred = pred.cuda()
                    gt = gt.cuda()
                TPR, FPR = self._eval_roc(pred, gt, 255)
                avg_tpr += TPR
                avg_fpr += FPR
                img_num += 1.0
            avg_tpr = avg_tpr / img_num
            avg_fpr = avg_fpr / img_num

            sorted_idxes = torch.argsort(avg_fpr)
            avg_tpr = avg_tpr[sorted_idxes]
            avg_fpr = avg_fpr[sorted_idxes]
            avg_auc = torch.trapz(avg_tpr, avg_fpr)

            return avg_auc.item(), avg_tpr, avg_fpr

    def Eval_Emeasure(self):
        print('eval[EMeasure]:{} dataset with {} method.'.format(
            self.dataset, self.method))
        avg_e, img_num = 0.0, 0.0
        with torch.no_grad():
            Em = torch.zeros(255)
            
            for img, gt, edge, name in tqdm(self.loader):
                
                path = self.smap_path + name[0] + "_" + self.method + ".png"
               
                
                pred = self.transformation(Image.open(path))
               
                pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                if self.cuda:
                    pred = pred.cuda()
                    gt = gt.cuda()
                Em += self._eval_e(pred, gt, 255)
                img_num += 1.0

            Em /= img_num
            return Em

    def Eval_Smeasure(self):
        print('eval[SMeasure]:{} dataset with {} method.'.format(
            self.dataset, self.method))
        alpha, avg_q, img_num = 0.5, 0.0, 0.0
        with torch.no_grad():
            
            for img, gt, edge, name in tqdm(self.loader):
               
                path = self.smap_path + name[0] + "_" + self.method + ".png"
                
               
                
                pred = self.transformation(Image.open(path))
            
                pred = (pred - torch.min(pred)) / (torch.max(pred) -
                                                       torch.min(pred) + 1e-20)
                if self.cuda:
                    pred = pred.cuda()
                    gt = gt.cuda()
                y = gt.mean()
                if y == 0:
                    x = pred.mean()
                    Q = 1.0 - x
                elif y == 1:
                    x = pred.mean()
                    Q = x
                else:
                    gt[gt >= 0.5] = 1
                    gt[gt < 0.5] = 0
                    Q = alpha * self._S_object(
                        pred, gt) + (1 - alpha) * self._S_region(pred, gt)
                    if Q.item() < 0:
                        Q = torch.FloatTensor([0.0])
                img_num += 1.0
                avg_q += Q.item()
            avg_q /= img_num
            return avg_q

    def LOG(self, output): # 追加的写 a
        with open(self.logfile, 'a') as f:
            f.write(output)

    def _eval_e(self, y_pred, y, num):
        if self.cuda:
            score = torch.zeros(num).cuda()
            thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
        else:
            score = torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_pred_th = (y_pred >= thlist[i]).float()
            fm = y_pred_th - y_pred_th.mean()
            gt = y - y.mean()
            align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
            enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
            score[i] = torch.sum(enhanced) / (y.numel() - 1 + 1e-20)
        return score

    def _eval_pr(self, y_pred, y, num):
        if y_pred.sum() == 0 and y.sum() == 0: # a negative sample
            y_pred = 1 - y_pred
            y = 1 - y
        if self.cuda:
            prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
            thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
        else:
            prec, recall = torch.zeros(num), torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() +
                                                                    1e-20)
        return prec, recall

    def _eval_roc(self, y_pred, y, num):
        if self.cuda:
            TPR, FPR = torch.zeros(num).cuda(), torch.zeros(num).cuda()
            thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
        else:
            TPR, FPR = torch.zeros(num), torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            fp = (y_temp * (1 - y)).sum()
            tn = ((1 - y_temp) * (1 - y)).sum()
            fn = ((1 - y_temp) * y).sum()

            TPR[i] = tp / (tp + fn + 1e-20)
            FPR[i] = fp / (fp + tn + 1e-20)

        return TPR, FPR

    def _S_object(self, pred, gt):
        fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
        bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
        o_fg = self._object(fg, gt)
        o_bg = self._object(bg, 1 - gt)
        u = gt.mean()
        Q = u * o_fg + (1 - u) * o_bg
        return Q

    def _object(self, pred, gt):
        temp = pred[gt == 1]
        x = temp.mean()
        sigma_x = temp.std()
        score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)

        return score

    def _S_region(self, pred, gt):
        X, Y = self._centroid(gt)
        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = self._divideGT(gt, X, Y)
        p1, p2, p3, p4 = self._dividePrediction(pred, X, Y)
        Q1 = self._ssim(p1, gt1)
        Q2 = self._ssim(p2, gt2)
        Q3 = self._ssim(p3, gt3)
        Q4 = self._ssim(p4, gt4)
        Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
        return Q

    def _centroid(self, gt):
        rows, cols = gt.size()[-2:] # rows 256   cols 256  
        gt = gt.view(rows, cols)
        if gt.sum() == 0:
            if self.cuda:
                X = torch.eye(1).cuda() * round(cols / 2)
                Y = torch.eye(1).cuda() * round(rows / 2)
            else:
                X = torch.eye(1) * round(cols / 2)
                Y = torch.eye(1) * round(rows / 2)
        else:
            total = gt.sum()
            if self.cuda:
                i = torch.from_numpy(np.arange(0, cols)).cuda().float()
                j = torch.from_numpy(np.arange(0, rows)).cuda().float()
            else:
                i = torch.from_numpy(np.arange(0, cols)).float()
                j = torch.from_numpy(np.arange(0, rows)).float()
            X = torch.round((gt.sum(dim=0) * i).sum() / total + 1e-20)
            Y = torch.round((gt.sum(dim=1) * j).sum() / total + 1e-20)
        return X.long(), Y.long()

    def _divideGT(self, gt, X, Y):
        h, w = gt.size()[-2:]
        area = h * w
        gt = gt.view(h, w)
        LT = gt[:Y, :X]
        RT = gt[:Y, X:w]
        LB = gt[Y:h, :X]
        RB = gt[Y:h, X:w]
        X = X.float()
        Y = Y.float()
        w1 = X * Y / area
        w2 = (w - X) * Y / area
        w3 = X * (h - Y) / area
        w4 = 1 - w1 - w2 - w3
        return LT, RT, LB, RB, w1, w2, w3, w4

    def _dividePrediction(self, pred, X, Y):
        h, w = pred.size()[-2:]
        pred = pred.view(h, w)
        LT = pred[:Y, :X]
        RT = pred[:Y, X:w]
        LB = pred[Y:h, :X]
        RB = pred[Y:h, X:w]
        return LT, RT, LB, RB

    def _ssim(self, pred, gt):
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
            Q = 1.0
        else:
            Q = 0
        return Q
    

    def Eval_AP(self, prec, recall):
        # Ref:
        # https://github.com/facebookresearch/Detectron/blob/05d04d3a024f0991339de45872d02f2f50669b3d/lib/datasets/voc_eval.py#L54
        print('eval[AP]:{} dataset with {} method.'.format(
            self.dataset, self.method))
        ap_r = np.concatenate(([0.], recall, [1.]))
        ap_p = np.concatenate(([0.], prec, [0.]))
        sorted_idxes = np.argsort(ap_r)
        ap_r = ap_r[sorted_idxes]
        ap_p = ap_p[sorted_idxes]
        count = ap_r.shape[0]

        for i in range(count - 1, 0, -1):
            ap_p[i - 1] = max(ap_p[i], ap_p[i - 1])

        i = np.where(ap_r[1:] != ap_r[:-1])[0]
        ap = np.sum((ap_r[i + 1] - ap_r[i]) * ap_p[i + 1])
        return ap