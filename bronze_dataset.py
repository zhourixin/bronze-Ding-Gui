import torch
import torch.utils.data as data
import torch.nn as nn
from PIL import Image
import math
import os
import numpy as np
from lxml import etree
import pandas as pd
import sys
import time
import math
from torchvision import transforms
import re

# dating_tree_dict = {
#            '0': [0,0],
#            '1': [0,1],
#            '2': [1,2],
#            '3': [1,3],
#            '4': [1,4],
#            '5': [2,5],
#            '6': [2,6],
#            '7': [2,7],
#            '8': [3,8],
#            '9': [3,9],
#            '10': [3,10],
#            '11': [4,11],
#            '12': [4,12],
#            '13': [5,13],
#            '14': [5,14],
#            '15': [5,15],
#            '16': [6,16],
#            '17': [6,17],
#            '18': [6,18],
#            '19': [7,19],

# }
dating_tree_dict = {
           '00': [0,0,0],
           '01': [0,0,1],
           '02': [0,1,2],
           '03': [0,1,3],
           '04': [0,1,4],
           '05': [0,2,5],
           '06': [0,2,6],
           '07': [0,2,7],
           '08': [0,3,8],
           '09': [0,3,9],
           '010': [0,3,10],

           '10': [1,0,0],
           '11': [1,0,1],
           '12': [1,1,2],
           '13': [1,1,3],
           '14': [1,1,4],
           '15': [1,2,5],
           '16': [1,2,6],
           '17': [1,2,7],
           '18': [1,3,8],
           '19': [1,3,9],
           '110': [1,3,10],

}

ding_age_idx = {
           '商代早期': 0,
           '商代晚期': 1,
           '西周早期': 2,
           '西周早期前段': 2,
           '西周早期后段': 2,
           '西周早期後段': 2,
           '西周中期': 3,
           '西周中期前段': 3,
           '西周中期後段': 3,
           '西周中期晚段': 3,
           '西周中期早期': 3,
           '西周中期前期': 3,
           '西周中期后期': 3,
           '西周中期后段': 3,
           '西周晚期': 4,
           '春秋早期': 5,
           '春秋中期': 6,
           '春秋晚期': 7,
           '戰國早期': 8,
           '战国早期': 8,
           '戰國中期': 9,
           '战国中期': 9,
           '戰國晚期': 10,
           '战国晚期': 10,     
}

# gui_age_idx = {
#            '商代早期': 11,
#            '商代晚期': 12,
#            '西周早期': 13,
#            '西周早期前段': 13,
#            '西周早期后段': 13,
#            '西周早期後段': 13,
#            '西周中期': 14,
#            '西周中期前段': 14,
#            '西周中期後段': 14,
#            '西周中期晚段': 14,
#            '西周中期早期': 14,
#            '西周中期前期': 14,
#            '西周中期后期': 14,
#            '西周中期后段': 14,
#            '西周晚期': 15,
#            '春秋早期': 16,
#            '春秋中期': 17,
#            '春秋晚期': 18,
#            '戰國早期': 19,
#            '战国早期': 19,
#            '戰國中期': 19,
#            '战国中期': 19,
#            '戰國晚期': 19,
#            '战国晚期': 19,     
# }

gui_age_idx = {
           '商代早期': 0,
           '商代晚期': 1,
           '西周早期': 2,
           '西周早期前段': 2,
           '西周早期后段': 2,
           '西周早期後段': 2,
           '西周中期': 3,
           '西周中期前段': 3,
           '西周中期後段': 3,
           '西周中期晚段': 3,
           '西周中期早期': 3,
           '西周中期前期': 3,
           '西周中期后期': 3,
           '西周中期后段': 3,
           '西周晚期': 4,
           '春秋早期': 5,
           '春秋中期': 6,
           '春秋晚期': 7,
           '戰國早期': 8,
           '战国早期': 8,
           '戰國中期': 9,
           '战国中期': 9,
           '戰國晚期': 10,
           '战国晚期': 10,     
}


shape_idx_together={
            '錐足方鼎': 0,  #2
            '矮扁球腹鼎': 1,  #135
            '半球形腹圓鼎': 2,  #293
            '扁足方鼎': 3,  #23
            '扁足圓鼎': 4,  #109
            '超半球腹或半球腹鼎': 5,  #81
            '垂腹方鼎': 6,  #49
            '垂腹圓鼎': 7,  #335
            '高蹄足圓鼎': 8,  #111
            '鬲鼎': 9,  #303
            '罐鼎': 10,  #50
            '淺鼓腹鼎': 11,  #6
            '收腹圓鼎': 12,  #8
            '束腰平底鼎': 13,  #47
            '晚期獸首蹄足鼎': 14,  #59
            '小口鼎': 15,  #24
            '匜鼎': 16,  #19
            '圓鼎': 17,  #597
            '圓錐形足圓鼎': 18,  #25
            '早期獸首蹄足圓鼎': 19,  #96
            '柱足方鼎': 20,  #392
            '半球腹形圓鼎':21,  #184
            '異形鼎':22,  #21
            '半球腹或超半球腹圓鼎':23,  #117
            '尖錐足圓鼎':24,  #27
            '晚期獸首蹄足圓鼎':25,  #25
            '越式鼎B':26,  #29
            '越式鼎A':27,  #21
            '蹄足方鼎':28,   #2
            '碗形': 29, 
            '罐形': 30, 
            '盂形': 31, 
            '豆形': 32, 
            '特殊': 33, 
            '方簋': 34        
}

attribute_idx_together = {"獸面紋帶 列旗脊雲雷紋": 0, "獸面紋帶 省變": 1, "獸面紋帶 雙體軀幹": 2, "獸面紋帶 其他或看不清的": 3, "獸面紋 尾上卷": 4, 
                          "獸面紋 尾下卷": 5, "獸面紋 獨體": 6, "獸面紋 分解": 7, "獸面紋 有火紋": 8, "獸面紋 其他": 9, "足部獸首": 10, "夔龍紋 直身": 11, 
                          "夔龍紋 低頭卷尾": 12, "夔龍紋 曲身拱背": 13, "夔龍紋 卷鼻": 14, "夔龍紋 捲曲": 15, "交龍紋": 16, "龍紋 單首雙身": 17, "卷龍紋": 18, 
                          "顧龍紋 折身": 19, "顧龍紋 其他斜身拱背雙首": 20, "顧龍紋 分體": 21, "蟠螭紋": 22, "蟠虺紋": 23, "直立鳥紋": 24, "小鳥紋": 25, 
                          "回首的小鳥紋": 26, "長尾鳥紋": 27, "分離C形尾長鳥紋": 28, "分離S形尾長鳥紋": 29, "大鳥紋昂首": 30, "大鳥紋回首": 31, "鳥首龍身紋": 32, 
                          "鳥首龍身紋回首": 33, "龍首鳥身": 34, "蛇紋": 35, "蟬紋": 36, "連續的蟬紋": 37, "蟬紋足": 38, "象紋": 39, "四瓣目紋": 40, 
                          "斜角目紋": 41, "目雲紋": 42, "重環紋": 43, "鱗紋": 44, "分離的分解獸面竊曲紋": 45, "有省變的分解獸面紋": 46, "夔龍紋演變的竊曲紋": 47, 
                          "S形竊曲紋": 48, "U形竊曲紋": 49, "G形竊曲紋": 50, "其他竊曲紋": 51, "普通雲雷紋": 52, "勾連雷紋": 53, "乳釘雷紋": 54, "斜角雲紋": 55, 
                          "菱格雷紋": 56, "圓渦紋": 57, "乳釘紋": 58, "三角紋": 59, "蕉葉紋": 60, "直棱紋": 61, "圓圈紋": 62, "山紋": 63, "弦紋": 64, 
                          "繩紋": 65, "瓦紋": 66, "獸面紋帶 分尾": 67, "散螭紋": 68, "虎紋": 69, "魚紋": 70, "其他紋飾": 71, "托盘": 72, "门窗型炉灶": 73, 
                          "普通炉灶": 74, "蓋": 75, "曲尺紐": 76, "三环钮": 77, "平直扉棱": 78, "F形扉棱": 79, "其他扉棱": 80, "立耳": 81, "立耳外撇": 82, 
                          "附耳": 83, "附耳彎曲": 84, "附耳S形": 85, "环耳": 86, "夔龍形扁足": 87, "鳥形扁足": 88, "錐足": 89, "柱足": 90, "蹄足": 91, 
                          "蹄足短粗": 92, "蹄足外侈": 93, "蹄足细长": 94, "柱足上粗下细": 95, "蓋圈狀捉手": 96, "虎形足": 97, "環耳象首": 98, "方座普通": 99, 
                          "環耳": 100, "環耳獸首羊角": 101, "環耳中部的鳥翼紋": 102, "方形垂珥有鳥形": 103, "環耳獸首柱狀角": 104, 
                          "鈎珥與環耳接觸部分加寬": 105, "小鈎珥": 106, "著地的象首形垂珥": 107, "象首鼻足": 108, "環耳鳥形耳": 109, "環耳獸首獸角垂直聳立": 110, 
                          "方形垂珥其他": 111, "象鼻形扁足": 112, "獸首銜環耳": 113, "環耳獸首單體獸耳B型分叉角吐舌": 114, "閉合的象鼻形垂珥": 115, 
                          "L形扁足": 116, "環耳獸首兔": 117, "環耳獸首回形角": 118, "加寬鈎珥閉合成的多邊形鈎珥": 119, "環耳獸首單體獸耳A": 120, 
                          "環耳上的浮雕鳥飾": 121, "象鼻形鈎珥": 122, "團龍紋": 123, "環耳獸首獸角分別聳立": 124, "環耳獸首復古的鼓包狀獸首": 125, 
                          "直扁足": 126, "反向鈎珥": 127, "象鼻形垂珥": 128, "方座鏤孔": 129, "蓋蓮花形捉手": 130, "環耳獸首屏風耳獸首": 131, 
                          "獸腿形足": 132, "環耳龍形": 133, "環耳獸首單體獸耳B型螺角": 134, "環耳一期": 135, "方座有缺口": 136, "環耳獸首牛": 137, 
                          "環耳S形單體獸耳式": 138, "環耳獸首小型象首": 139, "環耳獸首單體獸耳B型豎立尖耳": 140, "環耳扁體龍形": 141, "長鼻形足": 142, 
                          "垂珥位置的獸腿形足": 143, "環耳獸首單體獸耳B型豎立柱狀角": 144, "龍形耳鋬": 145, "著地的象鼻形垂珥": 146, "鳥爪形鈎珥": 147, 
                          "垂珥位置的柱足": 148

}

ware_category = {
    "鼎":0,
    "簋":1,
}

class BronzeWare_Dataset(data.Dataset):
    def __init__(self, img_dir, xml_dir, excel_dir, input_transform=None, train=None, size=None):
        self.root_dir = img_dir
        self.annotations_root=xml_dir
        self.input_transform = input_transform

        ware_img_name_for_3, _, ware_age_for3, _, ware_shape_for3, _, _, ding_gui_cat, _ = self.load_xlsx_table(excel_dir)

        age_list = []
        shape_list = []
        ware_cat_list = []
        for i in range(len(ding_gui_cat)):
            ware_cat = ding_gui_cat[i]
            if ware_category[ware_cat] == 0:
                era_dict = ding_age_idx
            elif ware_category[ware_cat] == 1:
                era_dict = gui_age_idx
            ware_age = era_dict[ware_age_for3[i]]
            ware_shape = shape_idx_together[ware_shape_for3[i]]
            age_list.append(ware_age)
            shape_list.append(ware_shape)
            ware_cat_list.append(ware_category[ware_cat])
            

        self.ware_img_name = ware_img_name_for_3
        self.ware_age = age_list
        self.ware_shape= shape_list
        # self.ding_gui_cat = ding_gui_cat
        self.ding_gui_cat = ware_cat_list
        self.ware_img = []
        self.ware_xml=[]
        self.input_size = size
        self.train = train

        self.front_img = []
        self.back_img = []

        for img_name in self.ware_img_name:
            png_name = img_name + '.png' 
            xml_name = img_name + '.xml'
            png_name = os.path.join(self.root_dir, png_name)
            xml_name = os.path.join(self.annotations_root, xml_name)
            self.ware_img.append(png_name)
            self.ware_xml.append(xml_name)

    
    
    def __len__(self):
        return len(self.ware_img_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        xml_path=self.ware_xml[idx]
        with open(xml_path,encoding='utf-8') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str.encode('utf-8'))
        xml_data = self.parse_xml_to_dict(xml)["annotation"]

        labels_level0 = []
        labels_level1 = []
        labels_level2 = []
        # iscrowd = []
        attributes = [0]*149
        shape_label = []

        level_0 = self.ding_gui_cat[idx]
        level_1 = dating_tree_dict[str(self.ding_gui_cat[idx])+str(self.ware_age[idx])][1]
        level_2 = dating_tree_dict[str(self.ding_gui_cat[idx])+str(self.ware_age[idx])][2]
        if level_0==1 and level_2 in [8,9,10]:
            level_2 = 999

        labels_level0.append(level_0)
        labels_level1.append(level_1)
        labels_level2.append(level_2)
        shape_label.append(float(self.ware_shape[idx]))
        if 'object' in xml_data:
            for obj in xml_data["object"]:
                att_name = re.sub(r'[a-z0-9\t]|[^\w\s]', '', obj["name"])
                if att_name in attribute_idx_together:
                    attribute_id = attribute_idx_together[att_name]
                    attributes[attribute_id] = 1
        image = Image.open(self.ware_img[idx]).convert('RGB')

        labels_level0 = torch.from_numpy(np.asarray(labels_level0).astype('int64'))
        labels_level1 = torch.from_numpy(np.asarray(labels_level1).astype('int64'))
        labels_level2 = torch.from_numpy(np.asarray(labels_level2).astype('int64'))
        shape_label = torch.from_numpy(np.asarray(shape_label).astype('int64'))
        attributes = torch.from_numpy(np.asarray(attributes).astype('int64'))
        # iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])

        target = {}
        target["image_id"] = image_id
        target["labels_level0"] = labels_level0
        target["labels_level1"] = labels_level1
        target["labels_level2"] = labels_level2
        target["attributes"] = attributes
        target["shape_label"] = shape_label

        if self.input_transform is not None:
            image = self.input_transform(image)
        if int(labels_level0)>1:
            print(1)

        


        return image, target["labels_level1"][0], target["labels_level0"][0], attributes, target["labels_level2"][0], target["shape_label"][0]


    def get_height_and_width(self, idx):
        # read xml
        xml_path = self.ware_xml[idx]
        with open(xml_path,encoding='utf-8') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str.encode('utf-8'))
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def load_xlsx_table(self, file_path):
        age_table = pd.read_excel(file_path, engine='openpyxl')#璇诲叆age.xlsx
        ware_id = np.asarray(age_table.iloc[:, 1],dtype=np.str)
        ware_name = np.asarray(age_table.iloc[:, 2])
        ware_age = np.asarray(age_table.iloc[:, 3])
        ware_book = np.asarray(age_table.iloc[:, 4])
        ware_shape = np.asarray(age_table.iloc[:, 5])
        now_location = np.asarray(age_table.iloc[:, 6])
        out_location = np.asarray(age_table.iloc[:, 7])
        category = np.asarray(age_table.iloc[:, 8])
        full_shape_name = np.asarray(age_table.iloc[:, 9],dtype=np.str)

        return ware_id, ware_name, ware_age, ware_book, ware_shape, now_location, out_location, category, full_shape_name




    def parse_xml_to_dict(self, xml):
        """
        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree
        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result: 
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}


    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        imgs = [x[0] for x in batch]
        level1_label = [x[1] for x in batch]
        cat_label = [x[2] for x in batch]
        att_label = [x[3] for x in batch]
        level2_label = [x[4] for x in batch]
        shape_label = [x[5] for x in batch]
        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)

        for i in range(num_imgs):
            inputs[i] = imgs[i]
        return inputs, torch.stack(level1_label), torch.stack(cat_label), torch.stack(att_label), torch.stack(level2_label), torch.stack(shape_label)



class DingGuiXv_Dataset(data.Dataset):
    def __init__(self, img_dir, input_transform=None, train=None, size=None):
        self.root_dir = img_dir
        # self.annotations_root=xml_dir
        self.input_transform = input_transform

        # ware_img_name_for_3, _, ware_age_for3, _, ware_shape_for3, _, _, ding_gui_cat, _ = self.load_xlsx_table(excel_dir)

        # age_list = []
        # shape_list = []
        # ware_cat_list = []
        # for i in range(len(ding_gui_cat)):
        #     ware_cat = ding_gui_cat[i]
        #     if ware_category[ware_cat] == 0:
        #         era_dict = ding_age_idx
        #     elif ware_category[ware_cat] == 1:
        #         era_dict = gui_age_idx
        #     ware_age = era_dict[ware_age_for3[i]]
        #     ware_shape = shape_idx_together[ware_shape_for3[i]]
        #     age_list.append(ware_age)
        #     shape_list.append(ware_shape)
        #     ware_cat_list.append(ware_category[ware_cat])
            

        # self.ware_img_name = ware_img_name_for_3
        # self.ware_age = age_list
        # self.ware_shape= shape_list
        # # self.ding_gui_cat = ding_gui_cat
        # self.ding_gui_cat = ware_cat_list
        self.ware_img = []
        # self.ware_xml=[]
        self.input_size = size
        # self.train = train

        # self.front_img = []
        # self.back_img = []

        self.ware_img_name = os.listdir(img_dir)

        for png_name in self.ware_img_name:
            # png_name = img_name + '.png' 
            # xml_name = img_name + '.xml'
            png_name = os.path.join(self.root_dir, png_name)
            # xml_name = os.path.join(self.annotations_root, xml_name)
            self.ware_img.append(png_name)
            # self.ware_xml.append(xml_name)

    
    
    def __len__(self):
        return len(self.ware_img_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # xml_path=self.ware_xml[idx]
        # with open(xml_path,encoding='utf-8') as fid:
        #     xml_str = fid.read()
        # xml = etree.fromstring(xml_str.encode('utf-8'))
        # xml_data = self.parse_xml_to_dict(xml)["annotation"]

        # labels_level0 = []
        # labels_level1 = []
        # labels_level2 = []
        # # iscrowd = []
        # attributes = [0]*149
        # shape_label = []

        # level_0 = self.ding_gui_cat[idx]
        # level_1 = dating_tree_dict[str(self.ding_gui_cat[idx])+str(self.ware_age[idx])][1]
        # level_2 = dating_tree_dict[str(self.ding_gui_cat[idx])+str(self.ware_age[idx])][2]
        # if level_0==1 and level_2 in [8,9,10]:
        #     level_2 = 999

        # labels_level0.append(level_0)
        # labels_level1.append(level_1)
        # labels_level2.append(level_2)
        # shape_label.append(float(self.ware_shape[idx]))
        # if 'object' in xml_data:
        #     for obj in xml_data["object"]:
        #         att_name = re.sub(r'[a-z0-9\t]|[^\w\s]', '', obj["name"])
        #         if att_name in attribute_idx_together:
        #             attribute_id = attribute_idx_together[att_name]
        #             attributes[attribute_id] = 1
        image = Image.open(self.ware_img[idx]).convert('RGB')

        # labels_level0 = torch.from_numpy(np.asarray(labels_level0).astype('int64'))
        # labels_level1 = torch.from_numpy(np.asarray(labels_level1).astype('int64'))
        # labels_level2 = torch.from_numpy(np.asarray(labels_level2).astype('int64'))
        # shape_label = torch.from_numpy(np.asarray(shape_label).astype('int64'))
        # attributes = torch.from_numpy(np.asarray(attributes).astype('int64'))
        # # iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        # image_id = torch.tensor([idx])

        # target = {}
        # target["image_id"] = image_id
        # target["labels_level0"] = labels_level0
        # target["labels_level1"] = labels_level1
        # target["labels_level2"] = labels_level2
        # target["attributes"] = attributes
        # target["shape_label"] = shape_label

        if self.input_transform is not None:
            image = self.input_transform(image)
        # if int(labels_level0)>1:
        #     print(1)

        

        return image
        # return image, target["labels_level1"][0], target["labels_level0"][0], attributes, target["labels_level2"][0], target["shape_label"][0]


    def get_height_and_width(self, idx):
        # read xml
        xml_path = self.ware_xml[idx]
        with open(xml_path,encoding='utf-8') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str.encode('utf-8'))
        data = self.parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def load_xlsx_table(self, file_path):
        age_table = pd.read_excel(file_path, engine='openpyxl')#璇诲叆age.xlsx
        ware_id = np.asarray(age_table.iloc[:, 1],dtype=np.str)
        ware_name = np.asarray(age_table.iloc[:, 2])
        ware_age = np.asarray(age_table.iloc[:, 3])
        ware_book = np.asarray(age_table.iloc[:, 4])
        ware_shape = np.asarray(age_table.iloc[:, 5])
        now_location = np.asarray(age_table.iloc[:, 6])
        out_location = np.asarray(age_table.iloc[:, 7])
        category = np.asarray(age_table.iloc[:, 8])
        full_shape_name = np.asarray(age_table.iloc[:, 9],dtype=np.str)

        return ware_id, ware_name, ware_age, ware_book, ware_shape, now_location, out_location, category, full_shape_name




    def parse_xml_to_dict(self, xml):
        """
        Args:
            xml: xml tree obtained by parsing XML file contents using lxml.etree
        Returns:
            Python dictionary holding XML contents.
        """

        if len(xml) == 0:  
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result: 
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}


    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        imgs = [x[0] for x in batch]
        level1_label = [x[1] for x in batch]
        cat_label = [x[2] for x in batch]
        att_label = [x[3] for x in batch]
        level2_label = [x[4] for x in batch]
        shape_label = [x[5] for x in batch]
        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)

        for i in range(num_imgs):
            inputs[i] = imgs[i]
        return inputs, torch.stack(level1_label), torch.stack(cat_label), torch.stack(att_label), torch.stack(level2_label), torch.stack(shape_label)







if __name__ == '__main__':
    DATASET_ROOT = "/home/zrx/lab_disk1/zhourixin/zhouriixn/DingAndGui/Ding_and_Gui_Dataset"
    data_path = DATASET_ROOT+"/image"
    xml_path = DATASET_ROOT+"/xml"
    train_excel_path = DATASET_ROOT+"/excel_origin_information/ding_and_gui_excel.xlsx"

    img_size = 450
    input_size = 400
    BATCH_SIZE = 10

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = BronzeWare_Dataset(data_path, xml_path, train_excel_path, transform, train=True, size=input_size)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, drop_last = True, collate_fn=trainset.collate_fn)

    for batch_idx, (inputs, _,_,attribute_label,targets,shape_label) in enumerate(trainloader):
        print(1)
