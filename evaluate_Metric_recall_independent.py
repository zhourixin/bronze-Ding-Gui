from unittest import result
import torch.hub
from sklearn.metrics import confusion_matrix, average_precision_score
import os
import random
import numpy as np
import csv

#这堆代码用来固定随机种子，保证结果可复现
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
ROOT = os.path.dirname(os.path.abspath(__file__))
seed = 2022
np.random.seed(seed)
torch.manual_seed(seed) #CPU随机种子确定
torch.cuda.manual_seed(seed) #GPU随机种子确定
torch.cuda.manual_seed_all(seed) #所有的GPU设置种子
torch.backends.cudnn.benchmark = False #模型卷积层预先优化关闭
torch.backends.cudnn.deterministic = True #确定为默认卷积算法
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)



def get_onehot(targets, classnum=11):
    target_list = []
    for i in range(targets.size(0)):
        state = [0]*classnum        
        level_label = int(targets[i])
        state[level_label] = 1
        state = torch.from_numpy(np.asarray(state).astype('int64'))
        state = state.unsqueeze(0)
        target_list.append(state)
    target_list = torch.cat(target_list, dim=0)

    return target_list


def format_print_confusion_matrix(confusion_matrix, type_name=None, placeholder_length=7):
	if type_name != None:
		type_name.insert(0, 'P \ T')    # 头部插入一个元素补齐
		for tn in type_name:
			fm = '%'+str(placeholder_length)+'s'
			print(fm%tn,end='')    # 不换行输出每一列表头
		print('\n')

	for i,cm in enumerate(confusion_matrix):
		if type_name != None:
			fm = '%'+str(placeholder_length)+'s'
			print(fm%type_name[i+1],end='')    # 不换行输出每一行表头
		
		for c in cm:
			fm = '%'+str(placeholder_length)+'s'
			print(fm%c,end='')    # 不换行输出每一行元素
		print('\n')


def OA_per_class(targets, predict, classnum=11):
    # Mix_metric = np.zeros((14,12), dtype=int)
    result_list = np.zeros((classnum,1), dtype=np.int32)
    correct = np.zeros_like(result_list)
    total = np.zeros_like(result_list)
    for i in range(0,len(targets)):
        t_label = targets[i]
        p_label = predict[i]
        # Mix_metric[p_label][t_label] = Mix_metric[p_label][t_label]+1
        if p_label == t_label:
            correct[p_label] += 1
        total[t_label] += 1
    
    # result_average = correct.sum().item()/total.sum().item()
    result = correct/total

    return result

def EVAL_recall_independent(pth_path, saved_excel_path):
    # pth_path = "/home/zrx/lab_disk1/zhourixin/zhouriixn/EXP_results/Rode2/Road_2_1/arc3_baseline_add_LeverInter_add_shape_add_illegal/Metric_data"
    ################
    MODEL_name = pth_path.split('/')[-2]
    eval_model = 3
    excel_path = saved_excel_path
    if eval_model == 1:
        
        classes_number = 20
        OA_labels = torch.load(os.path.join(pth_path, "label2.pth")) #read cpu label Tensor 
        model_predict = torch.load(os.path.join(pth_path, "predict2.pth"))#read cpu model output Tensor 
        
        
        onehot_label = get_onehot(OA_labels, classnum=classes_number) #produce one-hot label
        _, OA_predict = torch.max(model_predict.data, 1)
        PRC_predict = torch.softmax(model_predict, dim=1).data


        # 要计算的三个指标
        score = average_precision_score(onehot_label, PRC_predict, average='micro')
        OA = 100.* OA_predict.eq(OA_labels.data).cpu().sum().item()/OA_labels.size(0)
        OA_Per_Class = OA_per_class(OA_labels, OA_predict, classnum=classes_number)
        print("OA=%f, score=%f " % (OA, score))
        print("per_class_acc = ", end="")
        for i in OA_Per_Class:
            print(i, end="")
        
        record_list = []
        for i in OA_Per_Class:
            record_list.append(float(i))
        with open(excel_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([MODEL_name])
            writer.writerow(["OA=", "score= "])
            writer.writerow([OA, score])
            writer.writerow(record_list)

    
    elif eval_model == 2:
        
        """Level 2"""
        class1 = 11
        class2 = 9
        classes_number = 20
        OA_labels = torch.load(os.path.join(pth_path, "label2.pth")) #read cpu label Tensor 
        model_predict = torch.load(os.path.join(pth_path, "predict2.pth"))#read cpu model output Tensor 


        onehot_label = get_onehot(OA_labels, classnum=classes_number) #produce one-hot label 
        _, OA_predict = torch.max(model_predict.data, 1)
        PRC_predict = torch.softmax(model_predict, dim=1).data

        """独立计算鼎和簋的OA"""
        class1_index = torch.where(OA_labels < class1)
        class2_index = torch.where((OA_labels >= class1) & (OA_labels < class1+class2))
        
        class1_labels = OA_labels[class1_index]
        class2_labels = OA_labels[class2_index] - 11
        class1_predict = OA_predict[class1_index]
        class2_predict = OA_predict[class2_index] - 11

        OA_class1 = 100.* class1_predict.eq(class1_labels.data).cpu().sum().item()/class1_labels.size(0)
        OA_class2 = 100.* class2_predict.eq(class2_labels.data).cpu().sum().item()/class2_labels.size(0)

        print("Level2 OA_class1=%f, OA_class2=%f " % (OA_class1, OA_class2))
        print("\n", end="")

        with open(excel_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([["="*25]])
            writer.writerow([MODEL_name])
            writer.writerow(["level2 independent OA"])
            writer.writerow(["OA_class1=", "OA_class2= "])
            writer.writerow([OA_class1, OA_class2])
            # writer.writerow(record_list)
        
        """Level 1"""
        class1 = 4
        class2 = 4
        classes_number = 8
        OA_labels = torch.load(os.path.join(pth_path, "label1.pth")) #read cpu label Tensor 
        model_predict = torch.load(os.path.join(pth_path, "predict1.pth"))#read cpu model output Tensor 
        
        
        onehot_label = get_onehot(OA_labels, classnum=classes_number) #produce one-hot label
        _, OA_predict = torch.max(model_predict.data, 1)
        PRC_predict = torch.softmax(model_predict, dim=1).data

        """独立计算鼎和簋的OA"""
        class1_index = torch.where(OA_labels < class1)
        class2_index = torch.where((OA_labels >= class1) & (OA_labels < class1+class2))
        
        class1_labels = OA_labels[class1_index]
        class2_labels = OA_labels[class2_index] - 11
        class1_predict = OA_predict[class1_index]
        class2_predict = OA_predict[class2_index] - 11

        OA_class1 = 100.* class1_predict.eq(class1_labels.data).cpu().sum().item()/class1_labels.size(0)
        OA_class2 = 100.* class2_predict.eq(class2_labels.data).cpu().sum().item()/class2_labels.size(0)


        print("Level1 OA_class1=%f, OA_class2=%f " % (OA_class1, OA_class2))
        print("\n", end="")

        with open(excel_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(["level1 independent OA"])
            writer.writerow(["OA_class1=", "OA_class2= "])
            writer.writerow([OA_class1, OA_class2])

    elif eval_model == 3:
        
        
        cat_labels = torch.load(os.path.join(pth_path, "label0.pth")) #read cpu label Tensor 
        cat_predict = torch.load(os.path.join(pth_path, "predict0.pth"))#read cpu model output Tensor 
        _, cat_predict = torch.max(cat_predict.data, 1)
        
        
        
        """Level 2"""
        class1 = 11
        class2 = 9
        classes_number = 11
        OA_labels = torch.load(os.path.join(pth_path, "label2.pth")) #read cpu label Tensor 
        model_predict = torch.load(os.path.join(pth_path, "predict2.pth"))#read cpu model output Tensor 

        # # 筛掉簋 战国早中晚
        # leaf_labels = torch.nonzero(OA_labels < 999, as_tuple=False)
        # OA_labels = torch.index_select(OA_labels, 0, leaf_labels.squeeze())
        # model_predict = torch.index_select(model_predict, 0, leaf_labels.squeeze())
      


        # onehot_label = get_onehot(OA_labels, classnum=classes_number) #produce one-hot label 
        _, OA_predict = torch.max(model_predict.data, 1)
        # PRC_predict = torch.softmax(model_predict, dim=1).data

        """独立计算鼎和簋的OA"""
        class1_index = torch.where((cat_labels == 0) & (cat_predict == 0))
        class2_index = torch.where((cat_labels == 1) & (cat_predict == 1))
        # 筛掉簋 战国早中晚
        leaf_labels = torch.where(OA_labels < 999)
        class1_isin = torch.isin(class1_index[0], leaf_labels[0])
        class2_isin = torch.isin(class2_index[0], leaf_labels[0])
        class1_index = class1_index[0][class1_isin]
        class2_index = class2_index[0][class2_isin]


        class1_labels = OA_labels[class1_index]
        class2_labels = OA_labels[class2_index]
        class1_predict = OA_predict[class1_index]
        class2_predict = OA_predict[class2_index]

        # class1_index = torch.where(OA_labels < class1)
        # class2_index = torch.where((OA_labels >= class1) & (OA_labels < class1+class2))
        # class1_labels = OA_labels[class1_index]
        # class2_labels = OA_labels[class2_index] - 11
        # class1_predict = OA_predict[class1_index]
        # class2_predict = OA_predict[class2_index] - 11

        OA_class1 = 100.* class1_predict.eq(class1_labels.data).cpu().sum().item()/class1_labels.size(0)
        OA_class2 = 100.* class2_predict.eq(class2_labels.data).cpu().sum().item()/class2_labels.size(0)

        print("Level2 OA_class1=%f, OA_class2=%f " % (OA_class1, OA_class2))
        print("\n", end="")

        with open(excel_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([["="*25]])
            writer.writerow([MODEL_name])
            writer.writerow(["level2 independent OA"])
            writer.writerow(["OA_class1=", "OA_class2= "])
            writer.writerow([OA_class1, OA_class2])
            # writer.writerow(record_list)
        
        """Level 1"""
        class1 = 4
        class2 = 4
        classes_number = 4
        OA_labels = torch.load(os.path.join(pth_path, "label1.pth")) #read cpu label Tensor 
        model_predict = torch.load(os.path.join(pth_path, "predict1.pth"))#read cpu model output Tensor 
        
        
        # onehot_label = get_onehot(OA_labels, classnum=classes_number) #produce one-hot label
        _, OA_predict = torch.max(model_predict.data, 1)
        # PRC_predict = torch.softmax(model_predict, dim=1).data

        """独立计算鼎和簋的OA"""
        class1_index = torch.where((cat_labels == 0) & (cat_predict == 0))
        class2_index = torch.where((cat_labels == 1) & (cat_predict == 1))
        class1_labels = OA_labels[class1_index]
        class2_labels = OA_labels[class2_index]
        class1_predict = OA_predict[class1_index]
        class2_predict = OA_predict[class2_index]


        # class1_index = torch.where(OA_labels < class1)
        # class2_index = torch.where((OA_labels >= class1) & (OA_labels < class1+class2))
        # class1_labels = OA_labels[class1_index]
        # class2_labels = OA_labels[class2_index] - 11
        # class1_predict = OA_predict[class1_index]
        # class2_predict = OA_predict[class2_index] - 11

        OA_class1 = 100.* class1_predict.eq(class1_labels.data).cpu().sum().item()/class1_labels.size(0)
        OA_class2 = 100.* class2_predict.eq(class2_labels.data).cpu().sum().item()/class2_labels.size(0)


        print("Level1 OA_class1=%f, OA_class2=%f " % (OA_class1, OA_class2))
        print("\n", end="")

        with open(excel_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(["level1 independent OA"])
            writer.writerow(["OA_class1=", "OA_class2= "])
            writer.writerow([OA_class1, OA_class2])
