import torch
import torch.nn as nn
from torchvision import transforms, models
import torch.hub
import argparse
from model_bronze import AKG
from graph_loss import GraphLoss
from train_test_bronze import train, save_feature, save_ding_feature_for_analyse
from bronze_dataset import BronzeWare_Dataset, DingGuiXv_Dataset
from cal_nc_distance import cal_distance, cal_distance_noBG
import os
import random
import numpy as np
from collections import Counter
from evaluate_Metric_recall import EVAL_recall
from evaluate_Metric_precision import EVAL_precision
from evaluate_Metric_recall_independent import EVAL_recall_independent

from draw_feature import draw_PCA, draw_TSNE, draw_PCA2, draw_TSNE2, draw_PaCMAP2, save_PaCMAP3_feature, draw_feature_scatter, save_PCA3_feature, save_TSNE3_feature

from draw_feature_ding_gui_xv import save_feature_ding_gui_xv_reduce_dim, draw_feature_scatter_ding_gui_xv
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
DATASET_ROOT = "/home/zhourixin/DATASET/Ding_and_Gui_Dataset"
Modelset = "/home/zhourixin/DATASET/Ding_and_Gui_Dataset/pretrained_resnet50"
exp_PATH = "/home/zhourixin/Bronze_proj/Run_ding_and_gui/EXP/nc_exp"
# exp2 ,äŧååąįē§čŋæĨįģæīŧčžåĨåžååģčæ¯īŧį°įįģįšå¤į

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
    parser.add_argument('--exp_name', default="1_Ours_nc_82_cvpr23_new_tree_setting_exp4", type=str)
    parser.add_argument('--exp_path', default=exp_PATH, type=str)
    parser.add_argument('--sig_threshold', default=0.8, type=float)
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':

    for EXP_number in range(1,2):
        args = arg_parse()
        exp_ROOT = os.path.join(exp_PATH, args.exp_name)
        if os.path.exists(exp_ROOT) is False:
            os.makedirs(exp_ROOT)
        if os.path.exists(os.path.join(exp_ROOT, "bronze_pt")) is not True:
            os.mkdir(os.path.join(exp_ROOT, "bronze_pt"))

        print("parameter setting: alph1 = %.5f,alph2 = %.5f,beta = %.5f, lambda = %.6f" % (args.alph1, args.alph2, args.beta, args.Lambda))
        print('==> proportion: ', args.proportion)
        print('==> epoch: ', args.epoch)
        print('==> batch: ', args.BATCH_SIZE)
        print('==> dataset: ', args.dataset)
        print('==> img_size: ', args.img_size)
        print('==> device: ', args.device)
        print('==> Schedual: ', args.lr_adjt)

        nb_epoch = args.epoch
        batch_size = args.BATCH_SIZE
        num_workers = args.worker

        transform_train = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomCrop(args.input_size),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        transform_test = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            # transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        if args.dataset == 'bronze':
            levels = 3
            total_nodes = 17
            # level1:0,1 ; level2:2,3,4,5 ; level3:6,7,8,9,10,11,12,13,14,15,16
            trees = [
                [6, 2, 0],
                [7, 2, 0],
                [8, 3, 0],
                [9, 3, 0],
                [10, 3, 0],
                [11, 4, 0],
                [12, 4, 0],
                [13, 4, 0],
                [14, 5, 0],
                [15, 5, 0],
                [16, 5, 0],

                [6, 2, 1],
                [7, 2, 1],
                [8, 3, 1],
                [9, 3, 1],
                [10, 3, 1],
                [11, 4, 1],
                [12, 4, 1],
                [13, 4, 1],
                [14, 5, 1],
                [15, 5, 1],
                [16, 5, 1],
                ]
        
        # data_path = DATASET_ROOT+"/image"
        data_path = DATASET_ROOT+"/image_noBG"
        xml_path = DATASET_ROOT+"/xml"
        train_excel_path = DATASET_ROOT+"/excel_nc/ding_and_gui_train.xlsx"
        # val_excel_path = DATASET_ROOT+"/excel_nc/ding_and_gui_val.xlsx"
        # test_excel_path = DATASET_ROOT+"/excel_nc/ding_and_gui_test.xlsx"

        # train_excel_path = DATASET_ROOT+"/excel_nc/fold0_train.xlsx"
        # val_excel_path = DATASET_ROOT+"/excel_nc/fold0_test.xlsx"
        # test_excel_path = DATASET_ROOT+"/excel_nc/fold0_test.xlsx"

        trainset = BronzeWare_Dataset(data_path, xml_path, train_excel_path, transform_train, train=True, size=args.input_size)
        # valset = BronzeWare_Dataset(data_path, xml_path, val_excel_path, transform_test, train=False, size=args.input_size)
        # testset = BronzeWare_Dataset(data_path, xml_path, test_excel_path, transform_test, train=False, size=args.input_size)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=16, drop_last = True, collate_fn=trainset.collate_fn)
        # valloader = torch.utils.data.DataLoader(valset, batch_size=args.BATCH_SIZE, shuffle=False, num_workers=16, drop_last = False, collate_fn=valset.collate_fn)
        # testloader = torch.utils.data.DataLoader(testset, batch_size=args.BATCH_SIZE, shuffle=False, num_workers=16, drop_last = False, collate_fn=testset.collate_fn)

        # GPU
        device = torch.device("cuda:" + args.device[0])

        
        # backbone = models.resnet50(pretrained=False)
        # backbone.load_state_dict(torch.load(Modelset+'/resnet50.pth'))

        # net = AKG(args.dataset, backbone, 1024)
        # net.to(device)
        save_name = args.dataset+'_'+str(args.epoch)+'_'+str(args.BATCH_SIZE)+'_'+str(args.img_size)+'_'+str(args.proportion)+'_ResNet-50_'+'_'+args.lr_adjt

        net = torch.load("/home/zhourixin/Bronze_proj/Run_ding_and_gui/EXP/nc_exp/1_Ours_nc_82_cvpr23_new_tree_setting_exp4/bronze_pt/fold1_model_bronze_128_64_420_1.0_ResNet-50__Cos.pt")
        net.to(device)
        net = net.eval()



        # CELoss = nn.CrossEntropyLoss()
        # GRAPH = GraphLoss(trees, total_nodes, levels, device, args)

    
        

        """evaluate"""
        npy_path = os.path.join(exp_ROOT, "Metric_data")
        # if os.path.exists(npy_path) is False:
        #     os.mkdir(npy_path)
        # record_file = os.path.join(exp_ROOT, "Metric_data/final_record.csv")
        # EVAL_recall(npy_path, record_file)
        # EVAL_precision(npy_path, record_file)
        # EVAL_recall_independent(npy_path, record_file)

        """deal ding gui xv"""
        # ding_path = DATASET_ROOT+"/分析盨的数据/方鼎/all"
        # gui_path = DATASET_ROOT+"/分析盨的数据/方簋及方座簋/all"
        # xv_path = DATASET_ROOT+"/分析盨的数据/盨/all"
        
        # Dingset = DingGuiXv_Dataset(ding_path, transform_train, size=args.input_size)
        # Guiset = DingGuiXv_Dataset(gui_path, transform_test, size=args.input_size)
        # Xvset = DingGuiXv_Dataset(xv_path, transform_test, size=args.input_size)

        # Dingloader = torch.utils.data.DataLoader(Dingset, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=16, drop_last = False)
        # Guiloader = torch.utils.data.DataLoader(Guiset, batch_size=args.BATCH_SIZE, shuffle=False, num_workers=16, drop_last = False)
        # Xvloader = torch.utils.data.DataLoader(Xvset, batch_size=args.BATCH_SIZE, shuffle=False, num_workers=16, drop_last = False)

        # # save_ding_feature_for_analyse(net, Dingloader, device, exp_ROOT, "analyse_ding")
        # # save_ding_feature_for_analyse(net, Guiloader, device, exp_ROOT, "analyse_gui")
        # # save_ding_feature_for_analyse(net, Xvloader, device, exp_ROOT, "analyse_xv")
        
        # cal_distance(npy_path)

        """draw ding gui xv feature"""
        # save_feature_ding_gui_xv_reduce_dim(npy_path, "PCA")
        # draw_feature_scatter_ding_gui_xv(npy_path, "PCA")

        # save_feature_ding_gui_xv_reduce_dim(npy_path, "TSNE")
        # draw_feature_scatter_ding_gui_xv(npy_path, "TSNE")

        # save_feature_ding_gui_xv_reduce_dim(npy_path, "pacmap")
        # draw_feature_scatter_ding_gui_xv(npy_path, "pacmap")



        """deal ding gui xv noBG"""
        # ding_path = DATASET_ROOT+"/分析盨的数据/方鼎/all"
        # gui_path = DATASET_ROOT+"/分析盨的数据/方簋及方座簋/all_noBG"
        # xv_path = DATASET_ROOT+"/分析盨的数据/盨/all_noBG"
        
        # Dingset = DingGuiXv_Dataset(ding_path, transform_train, size=args.input_size)
        # Guiset = DingGuiXv_Dataset(gui_path, transform_test, size=args.input_size)
        # Xvset = DingGuiXv_Dataset(xv_path, transform_test, size=args.input_size)

        # Dingloader = torch.utils.data.DataLoader(Dingset, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=16, drop_last = False)
        # Guiloader = torch.utils.data.DataLoader(Guiset, batch_size=args.BATCH_SIZE, shuffle=False, num_workers=16, drop_last = False)
        # Xvloader = torch.utils.data.DataLoader(Xvset, batch_size=args.BATCH_SIZE, shuffle=False, num_workers=16, drop_last = False)


        # save_ding_feature_for_analyse(net, Dingloader, device, exp_ROOT, "analyse_ding_noBG")
        # save_ding_feature_for_analyse(net, Guiloader, device, exp_ROOT, "analyse_gui_noBG")
        # save_ding_feature_for_analyse(net, Xvloader, device, exp_ROOT, "analyse_xv_noBG")
        
        # cal_distance_noBG(npy_path)

        



        """save_feature"""
        save_feature(net, trainloader, device, args, exp_ROOT)
        save_PaCMAP3_feature(npy_path)
        draw_feature_scatter(npy_path, "ding_train_feature_pacmap_3D", "gui_train_feature_pacmap_3D","PaCMAP3_train_feature")

        save_PCA3_feature(npy_path)
        draw_feature_scatter(npy_path, "ding_train_feature_PCA_3D", "gui_train_feature_PCA_3D","PCA3_train_feature")

        save_TSNE3_feature(npy_path)
        draw_feature_scatter(npy_path, "ding_train_feature_TSNE_3D", "gui_train_feature_TSNE_3D","TSNE_train_feature")
