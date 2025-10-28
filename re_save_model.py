import torch
import torch.nn as nn
from torchvision import transforms, models
import torch.hub
import argparse
from model_bronze import AKG
from graph_loss import GraphLoss
# from train_test_bronze import train, save_feature
from bronze_dataset import BronzeWare_Dataset
import os
import random
import numpy as np
from collections import Counter
from evaluate_Metric_recall import EVAL_recall
from evaluate_Metric_precision import EVAL_precision
from evaluate_Metric_recall_independent import EVAL_recall_independent

# from draw_feature import draw_PCA, draw_TSNE, draw_PCA2, draw_TSNE2, draw_PaCMAP2, save_PaCMAP3_feature, draw_feature_scatter, save_PCA3_feature, save_TSNE3_feature

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
DATASET_ROOT = "/home/zhourixin/DATASET/Ding_and_Gui_Dataset"
Modelset = "/home/zhourixin/DATASET/Ding_and_Gui_Dataset/pretrained_resnet50"
exp_PATH = "/home/zhourixin/Bronze_proj/Run_ding_and_gui/EXP/nc_exp"
# exp2 ,优化层级连接结构；输入图像去背景；簋的结点处理

seed = 2022
np.random.seed(seed)
torch.manual_seed(seed) 
torch.cuda.manual_seed(seed) 
torch.cuda.manual_seed_all(seed) 
torch.backends.cudnn.benchmark = False 
torch.backends.cudnn.deterministic = True 
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# my_list = [150 + random.uniform(-2, 2) for _ in range(20)]
# print(my_list)

def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch Deployment')
    parser.add_argument('--worker', default=4, type=int, help='number of workers (default: 4)')
    parser.add_argument('--model', type=str, default='./pre-trained/resnet50-19c8e357.pth', help='Path of pre-trained model')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--proportion', type=float, default=1.0, help='Proportion of species label')  
    parser.add_argument('--epoch', default=128, type=int, help='Epochs')
    # parser.add_argument('--batch', type=int, default=32, help='batch size')      
    parser.add_argument('--dataset', type=str, default='bronze', help='dataset name')
    parser.add_argument('--lr_adjt', type=str, default='Cos', help='Learning rate schedual', choices=['Cos', 'Step'])
    parser.add_argument('--device', nargs='+', default='0', help='GPU IDs for DP training')
    parser.add_argument('--img_size', default=420, type=int, help='-')
    parser.add_argument('--input_size', default=400, type=int, help='-')
    parser.add_argument('--BATCH_SIZE', default=64, type=int, help='-')
    parser.add_argument('--lr', default=1e-4, type=float, help='-')
    parser.add_argument('--alph1', default=10, type=int)
    parser.add_argument('--alph2', default=10, type=int)

    parser.add_argument('--beta', default=1, type=float)
    parser.add_argument('--Lambda', default=1, type=float)
    parser.add_argument('--exp_name', default="five_fold_1_Ours_nc_82_cvpr23_new_tree_setting_exp4", type=str)
    parser.add_argument('--exp_path', default=exp_PATH, type=str)
    parser.add_argument('--sig_threshold', default=0.8, type=float)
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':

    for EXP_number in range(0,5):
        args = arg_parse()
        exp_ROOT = os.path.join(exp_PATH, args.exp_name)
        nb_epoch = args.epoch
        batch_size = args.BATCH_SIZE
        num_workers = args.worker
        save_name = args.dataset+'_'+str(args.epoch)+'_'+str(args.BATCH_SIZE)+'_'+str(args.img_size)+'_'+str(args.proportion)+'_ResNet-50_'+'_'+args.lr_adjt
        model = torch.load(exp_ROOT+'/bronze_pt/fold%s_model_%s.pt' % (str(EXP_number),save_name))

        torch.save(model.state_dict(), exp_ROOT+'/bronze_pt/fold{}_model_state_dict.pth'.format(str(EXP_number)))


