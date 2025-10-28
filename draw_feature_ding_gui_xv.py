import torch
import os
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pacmap
import matplotlib.pyplot as plt


def save_feature_ding_gui_xv_reduce_dim(pth_path, reduce_model):

    ding_feature_list = torch.load(os.path.join(pth_path, "analyse_ding_noBG_feature_list.pth"))
    gui_feature_list = torch.load(os.path.join(pth_path, "analyse_gui_noBG_feature_list.pth")) 
    xv_feature_list = torch.load(os.path.join(pth_path, "analyse_xv_noBG_feature_list.pth")) 
    ding_feature = np.array(ding_feature_list)
    gui_feature = np.array(gui_feature_list)
    xv_feature = np.array(xv_feature_list) 

    ding_len = ding_feature.shape[0]
    gui_len = gui_feature.shape[0]
    xv_len = xv_feature.shape[0]

    all_feature = np.concatenate((ding_feature, gui_feature, xv_feature), axis=0)
    
    if reduce_model == "PCA":

        # 创建PCA对象，n_components设置为3
        pca = PCA(n_components=3)
        all_feature_rd = pca.fit_transform(all_feature)
        ding_feature_rd = all_feature_rd[:ding_len, :]
        gui_feature_rd = all_feature_rd[ding_len:ding_len+gui_len, :]
        xv_feature_rd = all_feature_rd[ding_len+gui_len:, :]
        
        # ding_feature_rd = pca.fit_transform(ding_feature)
        # gui_feature_rd = pca.fit_transform(gui_feature)
        # xv_feature_rd = pca.fit_transform(xv_feature)
    elif reduce_model == "TSNE":

        # 创建PCA对象，n_components设置为3
        tsne = TSNE(n_components=3) 
        all_feature_rd = tsne.fit_transform(all_feature)
        ding_feature_rd = all_feature_rd[:ding_len, :]
        gui_feature_rd = all_feature_rd[ding_len:ding_len+gui_len, :]
        xv_feature_rd = all_feature_rd[ding_len+gui_len:, :]

        # ding_feature_rd = tsne.fit_transform(ding_feature)
        # gui_feature_rd = tsne.fit_transform(gui_feature)
        # xv_feature_rd = tsne.fit_transform(xv_feature)

    elif reduce_model == "pacmap":

        # 创建PCA对象，n_components设置为3
        PcAMAP = pacmap.PaCMAP(n_components=3, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0) 
        all_feature_rd = PcAMAP.fit_transform(all_feature, init="pca")
        ding_feature_rd = all_feature_rd[:ding_len, :]
        gui_feature_rd = all_feature_rd[ding_len:ding_len+gui_len, :]
        xv_feature_rd = all_feature_rd[ding_len+gui_len:, :]

        # ding_feature_rd = PcAMAP.fit_transform(ding_feature, init="pca")
        # PcAMAP = pacmap.PaCMAP(n_components=3, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0) 
        # gui_feature_rd = PcAMAP.fit_transform(gui_feature, init="pca")
        # PcAMAP = pacmap.PaCMAP(n_components=3, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0) 
        # xv_feature_rd = PcAMAP.fit_transform(xv_feature, init="pca")


    torch.save(ding_feature_rd, os.path.join(pth_path, "ding_feature_{}_3D_analyse.pth".format(reduce_model)))
    torch.save(gui_feature_rd, os.path.join(pth_path, "gui_feature_{}_3D_analyse.pth".format(reduce_model)))
    torch.save(xv_feature_rd, os.path.join(pth_path, "xv_feature_{}_3D_analyse.pth".format(reduce_model)))

    print(1)

def draw_feature_scatter_ding_gui_xv(pth_path, reduce_model):
    ding_feature = torch.load(os.path.join(pth_path, "ding_feature_{}_3D_analyse.pth".format(reduce_model))) 
    gui_feature = torch.load(os.path.join(pth_path, "gui_feature_{}_3D_analyse.pth".format(reduce_model)))
    xv_feature = torch.load(os.path.join(pth_path, "xv_feature_{}_3D_analyse.pth".format(reduce_model)))

    # reduced_features = pca.fit_transform(feature_list)
    # 创建一个3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 定义一个颜色列表，包含11种颜色
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#7FFFD4', '#9ACD32', '#FFA500', '#FFC0CB']

    ax.scatter(ding_feature[:, 0], ding_feature[:, 1], ding_feature[:, 2], color=colors[0], marker="o", label="ding")
    ax.scatter(gui_feature[:, 0], gui_feature[:, 1], gui_feature[:, 2], color=colors[1], marker="o", label="gui")
    ax.scatter(xv_feature[:, 0], xv_feature[:, 1], xv_feature[:, 2], color=colors[2], marker="o", label="xv")


    #draw ding
    # for i, group in enumerate(ding_feature):
    #     # group = tsne.fit_transform(group)
    #     ax.scatter(group[:, 0], group[:, 1], group[:, 2], color=colors[i], marker="o", label=str(i+1))

    # #draw gui
    # for i, group in enumerate(gui_feature):
    #     # group = tsne.fit_transform(group)
    #     ax.scatter(group[:, 0], group[:, 1], group[:, 2], color=colors[i], marker="^", label=str(i+1))



    ax.legend(title='ware color', bbox_to_anchor=(1.05, 1), loc='upper left')

    # 调整布局以确保图例完全显示在图片范围内
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)


    file_path = os.path.join(pth_path, "ding_gui_xv_{}.png".format(reduce_model))
    plt.savefig(file_path)

    # 显示图形
    # plt.show()
    # print(1)
    print(1)