import numpy as np
import torch
import torch.utils.data as Data
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from ORSI_SOD_dataset import ORSI_SOD_Dataset
from tqdm import tqdm
#from model.segformer import get_segformer as Net 
#from model.segformer_proposed_SCRKD5 import Segformer_SCRKD as Net
#from model.segformer_PD import SegFormer_PD as Net
#from model.segformer_CMHRD import SegFormer_CMHRD as Net
#from model.pspnet_proposed_SCRKD52 import PSPNet_SCRKD as Net 

#from model.TransXNet import TransXNet as Net
#from model.TransXNet_logits import TransXNet_logits as Net
#from model.TransXNet_FitNet import TransXNet_FitNet as Net
#from model.TransXNet_AT import TransXNet_AT as Net
#from model.TransXNet_IFVD import TransXNet_IFVD as Net
#from model.TransXNet_CWD import TransXNet_CWD as Net
#from model.TransXNet_ReviewKD import TransXNet_ReviewKD as Net
#from model.TransXNet_SKD import TransXNet_SKD as Net
#from model.TransXNet_SRD import TransXNet_SRD as Net
#from model.TransXNet_LogitStdKD import TransXNet_LogitStdKD as Net
#from model.TransXNet_STONet import TransXNet_STONet as Net
#from model.TransXNet_PD import TransXNet_PD as Net
#from model.TransXNet_CMHRD import TransXNet_CMHRD as Net
from model.TransXNet_proposed_SCRKD5 import TransXNet_SCRKD5 as Net
from evaluator import Eval_thread
from PIL import Image
import time







def unload(x):
    y = x.squeeze().cpu().data.numpy()
    return y
def convert2img(x):
    return Image.fromarray(x*255).convert('L')
def min_max_normalization(x):
    x_normed = (x - np.min(x)) / (np.max(x)-np.min(x))
    return x_normed
def save_smap(smap, path, negative_threshold=0.25):
    # smap: [1, H, W]
    if torch.max(smap) <= negative_threshold:
        smap[smap<negative_threshold] = 0
        smap = convert2img(unload(smap))
    else:
        smap = convert2img(min_max_normalization(unload(smap)))
    if smap.size != (448, 448):
        smap = smap.resize((448,448),Image.Resampling.BICUBIC)
    smap.save(path)



def getsmaps(dataset_name):
    ##define dataset
    dataset_root  = '/data1/users/liuyanfeng/RSI_SOD/'+ dataset_name +' dataset/'
    #test_set = ORSI_SOD_Dataset(root = dataset_root, img_size=448,  mode = "test", aug = False)
    #test_set = ORSI_SOD_Dataset(root = dataset_root, img_size=224,  mode = "test", aug = False)
    test_set = ORSI_SOD_Dataset(root = dataset_root, img_size=112,  mode = "test", aug = False)
    #test_set = ORSI_SOD_Dataset(root = dataset_root, img_size=56,  mode = "test", aug = False)
    test_loader = DataLoader(test_set, batch_size = 1, num_workers = 1)
    
    ##define network and load weight 224 448 112
    net = Net(img_size=112).cuda().eval()  #img_size=(448, 448)
   
    if dataset_name == "ORSSD":
        #net.load_state_dict(torch.load("./data/PSPNet_224x224_ORSSD/epoch_94_0.8752317428588867.pth")) 
        #net.load_state_dict(torch.load("./data/PSPNet_448x448_ORSSD/epoch_96_0.9051483869552612.pth")) 
        #net.load_state_dict(torch.load("./data/PSPNet_112x112_ORSSD/epoch_99_0.8328226804733276.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_logits_224to112_ORSSD/epoch_91_0.8416323661804199.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_FitNet_224to112_ORSSD/epoch_63_0.8430297374725342.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_AT_224to112_ORSSD/epoch_91_0.850625216960907.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_SPKD_224to112_ORSSD/epoch_95_0.8434325456619263.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_SRD_224to112_ORSSD/epoch_98_0.8434983491897583.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_LogitStdKD_224to112_ORSSD/epoch_81_0.8394263982772827.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_FAMKD_224to112_ORSSD/epoch_75_0.8400411009788513.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_STONet_224to112_ORSSD/epoch_96_0.8454548716545105.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_logits_448to224_ORSSD/epoch_91_0.8870370388031006.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_AT_448to224_ORSSD/epoch_81_0.8901381492614746.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_FitNet_448to224_ORSSD/epoch_97_0.8886995315551758.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_SPKD_448to224_ORSSD/epoch_91_0.8796983361244202.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_SRD_448to224_ORSSD/epoch_83_0.884074330329895.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_LogitStdKD_448to224_ORSSD/epoch_99_0.8782950639724731.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_STONet_448to224_ORSSD/epoch_94_0.8853043913841248.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_logits_224to112_ORSSD/epoch_97_0.8367313742637634.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_FitNet_224to112_ORSSD/epoch_97_0.8403520584106445.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_AT_224to112_ORSSD/epoch_97_0.8370007276535034.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_SPKD_224to112_ORSSD/epoch_84_0.8318301439285278.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_CWD_224to112_ORSSD/epoch_87_0.8407173752784729.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_SRD_224to112_ORSSD/epoch_99_0.835822582244873.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_LogitStdKD_224to112_ORSSD/epoch_88_0.8401384353637695.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_STONet_224to112_ORSSD/epoch_99_0.8363576531410217.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_SKD_224to112_ORSSD/epoch_95_0.8357298374176025.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_APD_224to112_ORSSD/epoch_96_0.8336579203605652.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_SCRKD5_224to112_ORSSD/epoch_90_0.8520212173461914.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_IFVD_224to112_ORSSD/epoch_99_0.8301117420196533.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_PD_224to112_ORSSD/epoch_99_0.8354914784431458.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_CMHRD_224to112_ORSSD/epoch_95_0.8363062739372253.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_112x112_ORSSD/epoch_83_0.8338368535041809.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_448x448_ORSSD/epoch_90_0.9016013145446777.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_logits_448to112_ORSSD/epoch_91_0.8374702334403992.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_FitNet_448to112_ORSSD/epoch_79_0.8465874791145325.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_AT_448to112_ORSSD/epoch_81_0.8463174104690552.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_SPKD_448to112_ORSSD/epoch_91_0.8396198153495789.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_IFVD_448to112_ORSSD/epoch_90_0.8402890563011169.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_SKD_448to112_ORSSD/epoch_92_0.8436832427978516.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_CWD_448to112_ORSSD/epoch_98_0.8455255627632141.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_ReviewKD_448to112_ORSSD/epoch_91_0.8482306599617004.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_SRD_448to112_ORSSD/epoch_95_0.8371676206588745.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_LogitStdKD_448to112_ORSSD/epoch_95_0.845936119556427.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_STONet_448to112_ORSSD/epoch_95_0.8517595529556274.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_SCRKD5_448to112_ORSSD/epoch_96_0.8538699150085449.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_SCRKD5_448to224to112_ORSSD/epoch_99_0.8576335906982422.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_112x112_ORSSD/epoch_93_0.8468277454376221.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_224x224_ORSSD/epoch_97_0.9005411863327026.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_448x448_ORSSD/epoch_90_0.9096945524215698.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_logits_224to112_ORSSD/epoch_72_0.8614241480827332.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_FitNet_224to112_ORSSD/epoch_92_0.8651519417762756.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_AT_224to112_ORSSD/epoch_86_0.8651975989341736.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_IFVD_224to112_ORSSD/epoch_95_0.8618801832199097.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_CWD_224to112_ORSSD/epoch_74_0.8655902147293091.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_ReviewKD_224to112_ORSSD/epoch_98_0.8584254384040833.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_SKD_224to112_ORSSD/epoch_75_0.8664246201515198.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_SRD_224to112_ORSSD/epoch_97_0.8610344529151917.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_LogitStdKD_224to112_ORSSD/epoch_82_0.8629142045974731.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_STONet_224to112_ORSSD/epoch_96_0.8662665486335754.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_PD_224to112_ORSSD/epoch_94_0.8583172559738159.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_CMHRD_224to112_ORSSD/epoch_79_0.8686314225196838.pth"))
        net.load_state_dict(torch.load("./data/TransXNet_SCRKD5_224to112_ORSSD/epoch_96_0.8768681287765503.pth"))
    elif dataset_name == "EORSSD":
        #net.load_state_dict(torch.load("./data/PSPNet_224x224_EORSSD/epoch_99_0.8390894532203674.pth")) 
        #net.load_state_dict(torch.load("./data/PSPNet_448x448_EORSSD/epoch_93_0.8721891641616821.pth")) 
        #net.load_state_dict(torch.load("./data/PSPNet_112x112_EORSSD/epoch_97_0.7715491652488708.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_logits_224to112_EORSSD/epoch_90_0.7762871384620667.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_FitNet_224to112_EORSSD/epoch_97_0.781947910785675.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_AT_224to112_EORSSD/epoch_96_0.7847193479537964.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_SRD_224to112_EORSSD/epoch_89_0.7801363468170166.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_SRD_224to112_EORSSD/epoch_89_0.7801363468170166.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_LogitStdKD_224to112_EORSSD/epoch_94_0.774192750453949.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_FAMKD_224to112_EORSSD/epoch_81_0.783511221408844.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_STONet_224to112_EORSSD/epoch_89_0.783097505569458.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_logits_448to224_EORSSD/epoch_90_0.846024215221405.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_AT_448to224_EORSSD/epoch_96_0.8506535291671753.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_FitNet_448to224_EORSSD/epoch_92_0.8451589941978455.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_SPKD_448to224_EORSSD/epoch_97_0.8454791307449341.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_SRD_448to224_EORSSD/epoch_92_0.8443663120269775.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_LogitStdKD_448to224_EORSSD/epoch_93_0.8436920642852783.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_STONet_448to224_EORSSD/epoch_99_0.8484703302383423.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_logits_224to112_EORSSD/epoch_89_0.793080747127533.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_FitNet_224to112_EORSSD/epoch_96_0.7930741906166077.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_AT_224to112_EORSSD/epoch_95_0.791903018951416.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_SPKD_224to112_EORSSD/epoch_92_0.7877092957496643.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_CWD_224to112_EORSSD/epoch_99_0.7965458631515503.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_SRD_224to112_EORSSD/epoch_85_0.7946343421936035.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_LogitStdKD_224to112_EORSSD/epoch_96_0.7922381162643433.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_STONet_224to112_EORSSD/epoch_89_0.7945895195007324.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_SKD_224to112_EORSSD/epoch_95_0.7952609658241272.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_APD_224to112_EORSSD/epoch_89_0.7897936701774597.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_SCRKD5_224to112_EORSSD/epoch_96_0.8012802600860596.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_IFVD_224to112_EORSSD/epoch_99_0.795856237411499.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_PD_224to112_EORSSD/epoch_97_0.79045170545578.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_CMHRD_224to112_EORSSD/epoch_99_0.7881526947021484.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_112x112_EORSSD/epoch_79_0.7880675196647644.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_448x448_EORSSD/epoch_98_0.8810319304466248.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_logits_448to112_EORSSD/epoch_99_0.7752470374107361.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_FitNet_448to112_EORSSD/epoch_93_0.7786803841590881.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_AT_448to112_EORSSD/epoch_96_0.7814053893089294.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_SPKD_448to112_EORSSD/epoch_97_0.780590832233429.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_IFVD_448to112_EORSSD/epoch_91_0.7733801603317261.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_SKD_448to112_EORSSD/epoch_91_0.7775724530220032.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_CWD_448to112_EORSSD/epoch_97_0.7799797058105469.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_ReviewKD_448to112_EORSSD/epoch_89_0.7814944982528687.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_SRD_448to112_EORSSD/epoch_89_0.7778472304344177.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_LogitStdKD_448to112_EORSSD/epoch_98_0.7791224718093872.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_STONet_448to112_EORSSD/epoch_95_0.7790117263793945.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_SCRKD5_448to112_EORSSD/epoch_90_0.7908192873001099.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_SCRKD5_448to224to112_EORSSD/epoch_99_0.7926951050758362.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_SCRKD_1_448to112_EORSSD/epoch_92_0.7829902768135071.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_SCRKD_2_448to112_EORSSD/epoch_86_0.7876742482185364.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_SCRKD_3_448to112_EORSSD/epoch_70_0.7819944024085999.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_SCRKD_1_2_448to112_EORSSD/epoch_87_0.7898060083389282.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_SCRKD5_112to56_EORSSD/epoch_99_0.6661571860313416.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_SCRKD5_224to56_EORSSD/epoch_98_0.66356360912323.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_SCRKD5_448to56_EORSSD/epoch_95_0.6568228602409363.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_112x112_EORSSD/epoch_99_0.7900862693786621.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_224x224_EORSSD/epoch_97_0.8603147864341736.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_448x448_EORSSD/epoch_91_0.8854798674583435.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_logits_224to112_EORSSD/epoch_99_0.8017134070396423.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_FitNet_224to112_EORSSD/epoch_70_0.8053228259086609.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_AT_224to112_EORSSD/epoch_95_0.8031890392303467.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_IFVD_224to112_EORSSD/epoch_94_0.8029742240905762.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_CWD_224to112_EORSSD/epoch_96_0.8060512542724609.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_ReviewKD_224to112_EORSSD/epoch_97_0.8055668473243713.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_SKD_224to112_EORSSD/epoch_78_0.8054310083389282.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_SRD_224to112_EORSSD/epoch_94_0.8061962127685547.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_LogitStdKD_224to112_EORSSD/epoch_98_0.7990105748176575.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_STONet_224to112_EORSSD/epoch_98_0.8036390542984009.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_PD_224to112_EORSSD/epoch_99_0.8037005662918091.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_CMHRD_224to112_EORSSD/epoch_79_0.8063256740570068.pth"))
        net.load_state_dict(torch.load("./data/TransXNet_SCRKD5_224to112_EORSSD/epoch_89_0.811864972114563.pth"))
    elif dataset_name == "ORS_4199":
        #net.load_state_dict(torch.load("./data/PSPNet_224x224_ORS_4199/epoch_94_0.8520545959472656.pth")) 
        #net.load_state_dict(torch.load("./data/PSPNet_448x448_ORS_4199/epoch_80_0.8476475477218628.pth")) 
        #net.load_state_dict(torch.load("./data/PSPNet_112x112_ORS_4199/epoch_97_0.8303403854370117.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_logits_224to112_ORS_4199/epoch_95_0.8337389826774597.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_FitNet_224to112_ORS_4199/epoch_97_0.8351300954818726.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_AT_224to112_ORS_4199/epoch_89_0.8363099098205566.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_SPKD_224to112_ORS_4199/epoch_95_0.8354843854904175.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_SRD_224to112_ORS_4199/epoch_95_0.8318607807159424.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_LogitStdKD_224to112_ORS_4199/epoch_97_0.8353437185287476.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_FAMKD_224to112_ORS_4199/epoch_77_0.8373781442642212.pth"))
        #net.load_state_dict(torch.load("./data/PSPNet_STONet_224to112_ORS_4199/epoch_92_0.8360699415206909.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_logits_224to112_ORS_4199/epoch_77_0.8465359210968018.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_FitNet_224to112_ORS_4199/epoch_92_0.8441546559333801.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_AT_224to112_ORS_4199/epoch_99_0.8461555242538452.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_SPKD_224to112_ORS_4199/epoch_76_0.8426039218902588.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_CWD_224to112_ORS_4199/epoch_99_0.8466850519180298.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_SRD_224to112_ORS_4199/epoch_75_0.8462033271789551.pth"))
        #net.load_state_dict((torch.load("./data/SegFormer_LogitStdKD_224to112_ORS_4199/epoch_75_0.8460550308227539.pth")))
        #net.load_state_dict(torch.load("./data/SegFormer_STONet_224to112_ORS_4199/epoch_97_0.8437785506248474.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_SKD_224to112_ORS_4199/epoch_96_0.8425534963607788.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_APD_224to112_ORS_4199/epoch_83_0.8434704542160034.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_SCRKD5_224to112_ORS_4199/epoch_95_0.8499596118927002.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_IFVD_224to112_ORS_4199/epoch_98_0.8457165956497192.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_PD_224to112_ORS_4199/epoch_94_0.845474123954773.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_CMHRD_224to112_ORS_4199/epoch_97_0.8422139883041382.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_112x112_ORS_4199/epoch_89_0.8419054746627808.pth"))
        #net.load_state_dict(torch.load("./data/SegFormer_448x448_ORS_4199/epoch_83_0.862912654876709.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_112x112_ORS_4199/epoch_99_0.8418759703636169.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_224x224_ORS_4199/epoch_90_0.8695002198219299.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_448x448_ORS_4199/epoch_89_0.8702975511550903.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_logits_224to112_ORS_4199/epoch_93_0.8521995544433594.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_FitNet_224to112_ORS_4199/epoch_93_0.8513845205307007.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_AT_224to112_ORS_4199/epoch_80_0.8512827754020691.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_IFVD_224to112_ORS_4199/epoch_77_0.8537198305130005.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_CWD_224to112_ORS_4199/epoch_72_0.8529568910598755.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_ReviewKD_224to112_ORS_4199/epoch_93_0.8522476553916931.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_SKD_224to112_ORS_4199/epoch_81_0.8515921235084534.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_SRD_224to112_ORS_4199/epoch_92_0.8529736995697021.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_LogitStdKD_224to112_ORS_4199/epoch_93_0.852831244468689.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_STONet_224to112_ORS_4199/epoch_78_0.8523058295249939.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_PD_224to112_ORS_4199/epoch_95_0.8525421023368835.pth"))
        #net.load_state_dict(torch.load("./data/TransXNet_CMHRD_224to112_ORS_4199/epoch_70_0.854026734828949.pth"))
        net.load_state_dict(torch.load("./data/TransXNet_SCRKD5_224to112_ORS_4199/epoch_91_0.8558633923530579.pth"))
    ##save saliency map
    infer_time = 0
    for image, label, _, name in tqdm(test_loader):
    #for data in tqdm(test_loader):
        with torch.no_grad():
            image, label = image.cuda(), label.cuda()
            #image_lr = data['img_112x112']
            #image_hr = data['img_224x224']
            #image_lr = image_lr.cuda()
            #image_hr = image_hr.cuda() 
            #name = data['name']   
            t1 = time.time()
            smap = net(image)  
            #smap = net(image_lr, image_hr)
            t2 = time.time()
            infer_time += (t2 - t1)
                
            ##if not exist then define
            dirs = "./data/output/predict_smaps" +  "_TransXNet_SCRKD5_224to112_" + dataset_name
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            path = os.path.join(dirs, name[0] + "_TransXNet_SCRKD5_224to112" + '.png')  
            
            save_smap(smap, path)
            
    print(len(test_loader))
    print(infer_time)
    print(len(test_loader) / infer_time)  # inference speed (without I/O time),

if __name__ == "__main__":
    

    

    dataset = ["EORSSD"]# , , 
    for datseti in dataset:
        getsmaps(datseti)
