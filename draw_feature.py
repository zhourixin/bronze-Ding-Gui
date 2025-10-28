import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import os
from sklearn.manifold import TSNE
import pacmap
import scipy.stats

def cal_distance(npy_path):
    # 计算KL散度
    def KL_divergence(p, q):
        return scipy.stats.entropy(p, q, base=2)
    
    # 计算JS散度
    def JS_divergence(p, q):
        M = (p + q) / 2
        return 0.5 * scipy.stats.entropy(p, M, base=2) + 0.5 * scipy.stats.entropy(q, M, base=2)


    ding_feature_path = npy_path+"/analyse_ding_feature_list.pth"
    gui_feature_path = npy_path+"/analyse_gui_feature_list.pth"
    xv_feature_path = npy_path+"/analyse_xv_feature_list.pth"

    ding_feature = torch.load(ding_feature_path).numpy()
    gui_feature = torch.load(gui_feature_path).numpy()
    xv_feature = torch.load(xv_feature_path).numpy()

    kl_distance1 = KL_divergence(ding_feature.mean(axis=0), xv_feature.mean(axis=0))
    print(f"KL散度距离：{kl_distance1:.6f}")

    kl_distance2 = KL_divergence(gui_feature.mean(axis=0), xv_feature.mean(axis=0))
    print(f"KL散度距离：{kl_distance2:.6f}")

    




    js_distance1 = JS_divergence(ding_feature.mean(axis=0), xv_feature.mean(axis=0))
    print(f"JS散度距离：{js_distance1:.6f}")

    js_distance2 = JS_divergence(gui_feature.mean(axis=0), xv_feature.mean(axis=0))
    print(f"JS散度距离：{js_distance2:.6f}")



def seaect_feature(feature_list, cat_labels, fine_age_labels):
    
    ding_index = np.where(cat_labels==0)
    gui_index = np.where(cat_labels==1)
    ding_feature = []
    gui_feature = []

    # ding
    for i in range(11):
        age_index = np.where(fine_age_labels==i)
        ding_age_index = np.intersect1d(age_index, ding_index)
        ding_age_index = ding_age_index[:min(10,ding_age_index.shape[0])]
        ding_feature.append(feature_list[ding_age_index])
    
    # gui
    for i in range(8):
        age_index = np.where(fine_age_labels==i)
        gui_age_index = np.intersect1d(age_index, gui_index)
        gui_age_index = gui_age_index[:min(10,gui_age_index.shape[0])]
        gui_feature.append(feature_list[ding_age_index])
    age_index = np.where(fine_age_labels>7)
    gui_age_index = np.intersect1d(age_index, gui_index)
    gui_age_index = gui_age_index[:gui_age_index.shape[0]]
    gui_feature.append(feature_list[ding_age_index])

    

    return ding_feature, gui_feature
def search_feature_PCA(feature_list, cat_labels, fine_age_labels, pca):
    
    ding_index = np.where(cat_labels==0)
    gui_index = np.where(cat_labels==1)
    feature_list = pca.fit_transform(feature_list)
    ding_feature_list = feature_list[ding_index]
    gui_feature_list = feature_list[gui_index]
    ding_fine_labels = fine_age_labels[ding_index]
    gui_fine_labels = fine_age_labels[gui_index]
    ding_feature = []
    gui_feature = []

    

    # ding
    for i in range(11):
        age_index = np.where(ding_fine_labels==i)[0]
        # ding_age_index = np.intersect1d(age_index, ding_index)
        age_index = age_index[:min(10,age_index.shape[0])]
        ding_feature.append(ding_feature_list[age_index])
    
    # gui
    for i in range(8):
        age_index = np.where(gui_fine_labels==i)[0]
        # gui_age_index = np.intersect1d(age_index, gui_index)
        age_index = age_index[:min(10,age_index.shape[0])]
        gui_feature.append(gui_feature_list[age_index])
    age_index = np.where(gui_fine_labels>7)[0]
    # gui_age_index = np.intersect1d(age_index, gui_index)
    # gui_age_index = gui_age_index[:gui_age_index.shape[0]]
    gui_feature.append(gui_feature_list[age_index])

    

    return ding_feature, gui_feature


def search_feature_TSNE(feature_list, cat_labels, fine_age_labels, tsne):
    
    ding_index = np.where(cat_labels==0)
    gui_index = np.where(cat_labels==1)
    feature_list = tsne.fit_transform(feature_list)
    ding_feature_list = feature_list[ding_index]
    gui_feature_list = feature_list[gui_index]
    ding_fine_labels = fine_age_labels[ding_index]
    gui_fine_labels = fine_age_labels[gui_index]
    ding_feature = []
    gui_feature = []

    

    # ding
    for i in range(11):
        age_index = np.where(ding_fine_labels==i)[0]
        # ding_age_index = np.intersect1d(age_index, ding_index)
        age_index = age_index[:min(10,age_index.shape[0])]
        ding_feature.append(ding_feature_list[age_index])
    
    # gui
    for i in range(8):
        age_index = np.where(gui_fine_labels==i)[0]
        # gui_age_index = np.intersect1d(age_index, gui_index)
        age_index = age_index[:min(10,age_index.shape[0])]
        gui_feature.append(gui_feature_list[age_index])
    age_index = np.where(gui_fine_labels>7)[0]
    # gui_age_index = np.intersect1d(age_index, gui_index)
    # gui_age_index = gui_age_index[:gui_age_index.shape[0]]
    gui_feature.append(gui_feature_list[age_index])

    

    return ding_feature, gui_feature

def search_feature_PcAMAP(feature_list, cat_labels, fine_age_labels, PcAMAP):
    
    ding_index = np.where(cat_labels==0)
    gui_index = np.where(cat_labels==1)
    feature_list = PcAMAP.fit_transform(feature_list, init="pca")
    ding_feature_list = feature_list[ding_index]
    gui_feature_list = feature_list[gui_index]
    ding_fine_labels = fine_age_labels[ding_index]
    gui_fine_labels = fine_age_labels[gui_index]
    ding_feature = []
    gui_feature = []

    

    # ding
    for i in range(11):
        age_index = np.where(ding_fine_labels==i)[0]
        # ding_age_index = np.intersect1d(age_index, ding_index)
        age_index = age_index[:min(10,age_index.shape[0])]
        ding_feature.append(ding_feature_list[age_index])
    
    # gui
    for i in range(8):
        age_index = np.where(gui_fine_labels==i)[0]
        # gui_age_index = np.intersect1d(age_index, gui_index)
        age_index = age_index[:min(10,age_index.shape[0])]
        gui_feature.append(gui_feature_list[age_index])
    age_index = np.where(gui_fine_labels>7)[0]
    # gui_age_index = np.intersect1d(age_index, gui_index)
    # gui_age_index = gui_age_index[:gui_age_index.shape[0]]
    gui_feature.append(gui_feature_list[age_index])

    

    return ding_feature, gui_feature

def search_feature_PcAMAP3(feature_list, cat_labels, fine_age_labels, PcAMAP):
    
    ding_index = np.where(cat_labels==0)
    gui_index = np.where(cat_labels==1)
    feature_list = PcAMAP.fit_transform(feature_list, init="pca")
    ding_feature_list = feature_list[ding_index]
    gui_feature_list = feature_list[gui_index]
    ding_fine_labels = fine_age_labels[ding_index]
    gui_fine_labels = fine_age_labels[gui_index]
    ding_feature = []
    gui_feature = []

    

    # ding
    for i in range(11):
        age_index = np.where(ding_fine_labels==i)[0]
        # ding_age_index = np.intersect1d(age_index, ding_index)
        # age_index = age_index[:min(10,age_index.shape[0])]
        ding_feature.append(ding_feature_list[age_index])
    
    # gui
    for i in range(8):
        age_index = np.where(gui_fine_labels==i)[0]
        # gui_age_index = np.intersect1d(age_index, gui_index)
        # age_index = age_index[:min(10,age_index.shape[0])]
        gui_feature.append(gui_feature_list[age_index])
    # age_index = np.where(gui_fine_labels>7)[0]
    # gui_age_index = np.intersect1d(age_index, gui_index)
    # gui_age_index = gui_age_index[:gui_age_index.shape[0]]
    # gui_feature.append(gui_feature_list[age_index])

    

    return ding_feature, gui_feature

def search_feature_PCA3(feature_list, cat_labels, fine_age_labels, pca):
    
    ding_index = np.where(cat_labels==0)
    gui_index = np.where(cat_labels==1)
    feature_list = pca.fit_transform(feature_list)
    ding_feature_list = feature_list[ding_index]
    gui_feature_list = feature_list[gui_index]
    ding_fine_labels = fine_age_labels[ding_index]
    gui_fine_labels = fine_age_labels[gui_index]
    ding_feature = []
    gui_feature = []



    # ding
    for i in range(11):
        age_index = np.where(ding_fine_labels==i)[0]
        # ding_age_index = np.intersect1d(age_index, ding_index)
        # age_index = age_index[:min(10,age_index.shape[0])]
        ding_feature.append(ding_feature_list[age_index])
    
    # gui
    for i in range(8):
        age_index = np.where(gui_fine_labels==i)[0]
        # gui_age_index = np.intersect1d(age_index, gui_index)
        # age_index = age_index[:min(10,age_index.shape[0])]
        gui_feature.append(gui_feature_list[age_index])
    # age_index = np.where(gui_fine_labels>7)[0]
    # gui_age_index = np.intersect1d(age_index, gui_index)
    # gui_age_index = gui_age_index[:gui_age_index.shape[0]]
    # gui_feature.append(gui_feature_list[age_index])

    

    return ding_feature, gui_feature


def search_feature_TSNE3(feature_list, cat_labels, fine_age_labels, tsne):
    
    ding_index = np.where(cat_labels==0)
    gui_index = np.where(cat_labels==1)
    feature_list = tsne.fit_transform(feature_list)
    ding_feature_list = feature_list[ding_index]
    gui_feature_list = feature_list[gui_index]
    ding_fine_labels = fine_age_labels[ding_index]
    gui_fine_labels = fine_age_labels[gui_index]
    ding_feature = []
    gui_feature = []



    # ding
    for i in range(11):
        age_index = np.where(ding_fine_labels==i)[0]
        # ding_age_index = np.intersect1d(age_index, ding_index)
        # age_index = age_index[:min(10,age_index.shape[0])]
        ding_feature.append(ding_feature_list[age_index])
    
    # gui
    for i in range(8):
        age_index = np.where(gui_fine_labels==i)[0]
        # gui_age_index = np.intersect1d(age_index, gui_index)
        # age_index = age_index[:min(10,age_index.shape[0])]
        gui_feature.append(gui_feature_list[age_index])
    # age_index = np.where(gui_fine_labels>7)[0]
    # gui_age_index = np.intersect1d(age_index, gui_index)
    # gui_age_index = gui_age_index[:gui_age_index.shape[0]]
    # gui_feature.append(gui_feature_list[age_index])

    

    return ding_feature, gui_feature


def draw_PCA(pth_path):
    # 假设你的数据存储在名为features的NumPy数组中
    # features = np.random.rand(100, 128)  # 仅作为示例
    feature_list = torch.load(os.path.join(pth_path, "feature_list.pth")) 
    cat_labels = torch.load(os.path.join(pth_path, "cat_labels.pth"))
    coarse_age_labels = torch.load(os.path.join(pth_path, "coarse_age_labels.pth")) 
    fine_age_labels = torch.load(os.path.join(pth_path, "fine_age_labels.pth"))

    feature_list = np.array(feature_list)
    cat_labels = np.array(cat_labels)
    coarse_age_labels = np.array(coarse_age_labels)
    fine_age_labels = np.array(fine_age_labels)

    ding_feature, gui_feature = seaect_feature(feature_list, cat_labels, fine_age_labels)

    # 创建PCA对象，n_components设置为3
    pca = PCA(n_components=3)

    # 对特征进行PCA降维
    # reduced_features = pca.fit_transform(feature_list)
    # 创建一个3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 定义一个颜色列表，包含11种颜色
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#7FFFD4', '#9ACD32', '#FFA500', '#FFC0CB']


    #draw ding
    for i, group in enumerate(ding_feature):
        group = pca.fit_transform(group)
        ax.scatter(group[:, 0], group[:, 1], group[:, 2], color=colors[i], marker="o", label=str(i+1))

    #draw gui
    for i, group in enumerate(gui_feature):
        group = pca.fit_transform(group)
        ax.scatter(group[:, 0], group[:, 1], group[:, 2], color=colors[i], marker="^", label=str(i+1))



    ax.legend(title='age color', bbox_to_anchor=(1.05, 1), loc='upper left')

    # 调整布局以确保图例完全显示在图片范围内
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)


    file_path = os.path.join(pth_path, "PCA.png")
    plt.savefig(file_path)

def draw_PCA2(pth_path):
    # 假设你的数据存储在名为features的NumPy数组中
    # features = np.random.rand(100, 128)  # 仅作为示例
    feature_list = torch.load(os.path.join(pth_path, "feature_list.pth")) 
    cat_labels = torch.load(os.path.join(pth_path, "cat_labels.pth"))
    coarse_age_labels = torch.load(os.path.join(pth_path, "coarse_age_labels.pth")) 
    fine_age_labels = torch.load(os.path.join(pth_path, "fine_age_labels.pth"))

    feature_list = np.array(feature_list)
    cat_labels = np.array(cat_labels)
    coarse_age_labels = np.array(coarse_age_labels)
    fine_age_labels = np.array(fine_age_labels)

    # 创建PCA对象，n_components设置为3
    pca = PCA(n_components=3)
    # feature_list = pca.fit_transform(feature_list)
    ding_feature, gui_feature = search_feature_PCA(feature_list, cat_labels, fine_age_labels, pca)

    

    # 对特征进行PCA降维
    # reduced_features = pca.fit_transform(feature_list)
    # 创建一个3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 定义一个颜色列表，包含11种颜色
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#7FFFD4', '#9ACD32', '#FFA500', '#FFC0CB']


    #draw ding
    for i, group in enumerate(ding_feature):
        # group = pca.fit_transform(group)
        ax.scatter(group[:, 0], group[:, 1], group[:, 2], color=colors[i], marker="o", label=str(i+1))

    #draw gui
    for i, group in enumerate(gui_feature):
        # group = pca.fit_transform(group)
        ax.scatter(group[:, 0], group[:, 1], group[:, 2], color=colors[i], marker="^", label=str(i+1))



    ax.legend(title='age color', bbox_to_anchor=(1.05, 1), loc='upper left')

    # 调整布局以确保图例完全显示在图片范围内
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)


    file_path = os.path.join(pth_path, "PCA.png")
    plt.savefig(file_path)


def draw_TSNE(pth_path):
    # 假设你的数据存储在名为features的NumPy数组中
    # features = np.random.rand(100, 128)  # 仅作为示例
    feature_list = torch.load(os.path.join(pth_path, "feature_list.pth")) 
    cat_labels = torch.load(os.path.join(pth_path, "cat_labels.pth"))
    coarse_age_labels = torch.load(os.path.join(pth_path, "coarse_age_labels.pth")) 
    fine_age_labels = torch.load(os.path.join(pth_path, "fine_age_labels.pth"))

    feature_list = np.array(feature_list)
    cat_labels = np.array(cat_labels)
    coarse_age_labels = np.array(coarse_age_labels)
    fine_age_labels = np.array(fine_age_labels)

    ding_feature, gui_feature = seaect_feature(feature_list, cat_labels, fine_age_labels)

    # 创建PCA对象，n_components设置为3
    tsne = TSNE(n_components=3) 

    # 对特征进行PCA降维
    # reduced_features = pca.fit_transform(feature_list)
    # 创建一个3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 定义一个颜色列表，包含11种颜色
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#7FFFD4', '#9ACD32', '#FFA500', '#FFC0CB']


    #draw ding
    for i, group in enumerate(ding_feature):
        group = tsne.fit_transform(group)
        ax.scatter(group[:, 0], group[:, 1], group[:, 2], color=colors[i], marker="o", label=str(i+1))

    #draw gui
    for i, group in enumerate(gui_feature):
        group = tsne.fit_transform(group)
        ax.scatter(group[:, 0], group[:, 1], group[:, 2], color=colors[i], marker="^", label=str(i+1))



    ax.legend(title='age color', bbox_to_anchor=(1.05, 1), loc='upper left')

    # 调整布局以确保图例完全显示在图片范围内
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)


    file_path = os.path.join(pth_path, "TSNE.png")
    plt.savefig(file_path)

def draw_TSNE2(pth_path):
    # 假设你的数据存储在名为features的NumPy数组中
    # features = np.random.rand(100, 128)  # 仅作为示例
    feature_list = torch.load(os.path.join(pth_path, "feature_list.pth")) 
    cat_labels = torch.load(os.path.join(pth_path, "cat_labels.pth"))
    coarse_age_labels = torch.load(os.path.join(pth_path, "coarse_age_labels.pth")) 
    fine_age_labels = torch.load(os.path.join(pth_path, "fine_age_labels.pth"))

    feature_list = np.array(feature_list)
    cat_labels = np.array(cat_labels)
    coarse_age_labels = np.array(coarse_age_labels)
    fine_age_labels = np.array(fine_age_labels)

    tsne = TSNE(n_components=3) 
    ding_feature, gui_feature = search_feature_TSNE(feature_list, cat_labels, fine_age_labels,tsne)


    

    # 对特征进行PCA降维
    # reduced_features = pca.fit_transform(feature_list)
    # 创建一个3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 定义一个颜色列表，包含11种颜色
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#7FFFD4', '#9ACD32', '#FFA500', '#FFC0CB']


    #draw ding
    for i, group in enumerate(ding_feature):
        # group = tsne.fit_transform(group)
        ax.scatter(group[:, 0], group[:, 1], group[:, 2], color=colors[i], marker="o", label=str(i+1))

    #draw gui
    for i, group in enumerate(gui_feature):
        # group = tsne.fit_transform(group)
        ax.scatter(group[:, 0], group[:, 1], group[:, 2], color=colors[i], marker="^", label=str(i+1))



    ax.legend(title='age color', bbox_to_anchor=(1.05, 1), loc='upper left')

    # 调整布局以确保图例完全显示在图片范围内
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)


    file_path = os.path.join(pth_path, "TSNE.png")
    plt.savefig(file_path)

def draw_PaCMAP2(pth_path):
    # 假设你的数据存储在名为features的NumPy数组中
    # features = np.random.rand(100, 128)  # 仅作为示例
    feature_list = torch.load(os.path.join(pth_path, "feature_list.pth")) 
    cat_labels = torch.load(os.path.join(pth_path, "cat_labels.pth"))
    coarse_age_labels = torch.load(os.path.join(pth_path, "coarse_age_labels.pth")) 
    fine_age_labels = torch.load(os.path.join(pth_path, "fine_age_labels.pth"))

    feature_list = np.array(feature_list)
    cat_labels = np.array(cat_labels)
    coarse_age_labels = np.array(coarse_age_labels)
    fine_age_labels = np.array(fine_age_labels)

    embedding = pacmap.PaCMAP(n_components=3, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0) 
    
    ding_feature, gui_feature = search_feature_TSNE(feature_list, cat_labels, fine_age_labels,embedding)


    

    # 对特征进行PCA降维
    # reduced_features = pca.fit_transform(feature_list)
    # 创建一个3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 定义一个颜色列表，包含11种颜色
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#7FFFD4', '#9ACD32', '#FFA500', '#FFC0CB']


    #draw ding
    for i, group in enumerate(ding_feature):
        # group = tsne.fit_transform(group)
        ax.scatter(group[:, 0], group[:, 1], group[:, 2], color=colors[i], marker="o", label=str(i+1))

    #draw gui
    for i, group in enumerate(gui_feature):
        # group = tsne.fit_transform(group)
        ax.scatter(group[:, 0], group[:, 1], group[:, 2], color=colors[i], marker="^", label=str(i+1))



    ax.legend(title='age color', bbox_to_anchor=(1.05, 1), loc='upper left')

    # 调整布局以确保图例完全显示在图片范围内
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)


    file_path = os.path.join(pth_path, "PaCMAP.png")
    plt.savefig(file_path)





def save_PaCMAP3_feature(pth_path):

    # plt.ion()  # 开启交互式模式
    # 假设你的数据存储在名为features的NumPy数组中
    # features = np.random.rand(100, 128)  # 仅作为示例
    feature_list = torch.load(os.path.join(pth_path, "feature_list.pth")) 
    cat_labels = torch.load(os.path.join(pth_path, "cat_labels.pth"))
    coarse_age_labels = torch.load(os.path.join(pth_path, "coarse_age_labels.pth")) 
    fine_age_labels = torch.load(os.path.join(pth_path, "fine_age_labels.pth"))


    # 筛掉簋 战国早中晚
    leaf_labels = torch.nonzero(fine_age_labels < 999, as_tuple=False)
    feature_list = torch.index_select(feature_list, 0, leaf_labels.squeeze())
    cat_labels = torch.index_select(cat_labels, 0, leaf_labels.squeeze())
    coarse_age_labels = torch.index_select(coarse_age_labels, 0, leaf_labels.squeeze())
    fine_age_labels = torch.index_select(fine_age_labels, 0, leaf_labels.squeeze())



    feature_list = np.array(feature_list)
    cat_labels = np.array(cat_labels)
    coarse_age_labels = np.array(coarse_age_labels)
    fine_age_labels = np.array(fine_age_labels)

    embedding = pacmap.PaCMAP(n_components=3, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0) 
    
    ding_feature, gui_feature = search_feature_PcAMAP3(feature_list, cat_labels, fine_age_labels,embedding)


    torch.save(ding_feature, os.path.join(pth_path, "ding_train_feature_pacmap_3D.pth"))
    torch.save(gui_feature, os.path.join(pth_path, "gui_train_feature_pacmap_3D.pth"))

def save_PCA3_feature(pth_path):

    # plt.ion()  # 开启交互式模式
    # 假设你的数据存储在名为features的NumPy数组中
    # features = np.random.rand(100, 128)  # 仅作为示例
    feature_list = torch.load(os.path.join(pth_path, "feature_list.pth")) 
    cat_labels = torch.load(os.path.join(pth_path, "cat_labels.pth"))
    coarse_age_labels = torch.load(os.path.join(pth_path, "coarse_age_labels.pth")) 
    fine_age_labels = torch.load(os.path.join(pth_path, "fine_age_labels.pth"))


    # 筛掉簋 战国早中晚
    leaf_labels = torch.nonzero(fine_age_labels < 999, as_tuple=False)
    feature_list = torch.index_select(feature_list, 0, leaf_labels.squeeze())
    cat_labels = torch.index_select(cat_labels, 0, leaf_labels.squeeze())
    coarse_age_labels = torch.index_select(coarse_age_labels, 0, leaf_labels.squeeze())
    fine_age_labels = torch.index_select(fine_age_labels, 0, leaf_labels.squeeze())



    feature_list = np.array(feature_list)
    cat_labels = np.array(cat_labels)
    coarse_age_labels = np.array(coarse_age_labels)
    fine_age_labels = np.array(fine_age_labels)

    # 创建PCA对象，n_components设置为3
    pca = PCA(n_components=3)
    ding_feature, gui_feature = search_feature_PCA3(feature_list, cat_labels, fine_age_labels,pca)


    torch.save(ding_feature, os.path.join(pth_path, "ding_train_feature_PCA_3D.pth"))
    torch.save(gui_feature, os.path.join(pth_path, "gui_train_feature_PCA_3D.pth"))

def save_TSNE3_feature(pth_path):

    # plt.ion()  # 开启交互式模式
    # 假设你的数据存储在名为features的NumPy数组中
    # features = np.random.rand(100, 128)  # 仅作为示例
    feature_list = torch.load(os.path.join(pth_path, "feature_list.pth")) 
    cat_labels = torch.load(os.path.join(pth_path, "cat_labels.pth"))
    coarse_age_labels = torch.load(os.path.join(pth_path, "coarse_age_labels.pth")) 
    fine_age_labels = torch.load(os.path.join(pth_path, "fine_age_labels.pth"))


    # 筛掉簋 战国早中晚
    leaf_labels = torch.nonzero(fine_age_labels < 999, as_tuple=False)
    feature_list = torch.index_select(feature_list, 0, leaf_labels.squeeze())
    cat_labels = torch.index_select(cat_labels, 0, leaf_labels.squeeze())
    coarse_age_labels = torch.index_select(coarse_age_labels, 0, leaf_labels.squeeze())
    fine_age_labels = torch.index_select(fine_age_labels, 0, leaf_labels.squeeze())



    feature_list = np.array(feature_list)
    cat_labels = np.array(cat_labels)
    coarse_age_labels = np.array(coarse_age_labels)
    fine_age_labels = np.array(fine_age_labels)

    # 创建PCA对象，n_components设置为3
    tsne = TSNE(n_components=3) 
    ding_feature, gui_feature = search_feature_TSNE3(feature_list, cat_labels, fine_age_labels,tsne)


    torch.save(ding_feature, os.path.join(pth_path, "ding_train_feature_TSNE_3D.pth"))
    torch.save(gui_feature, os.path.join(pth_path, "gui_train_feature_TSNE_3D.pth"))



    
def draw_feature_scatter(pth_path, ding_feature_name, gui_feature_name, save_image_name):


    ding_feature = torch.load(os.path.join(pth_path, ding_feature_name+".pth")) 
    gui_feature = torch.load(os.path.join(pth_path, gui_feature_name+".pth"))
    # 对特征进行PCA降维
    # reduced_features = pca.fit_transform(feature_list)
    # 创建一个3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 定义一个颜色列表，包含11种颜色
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '#7FFFD4', '#9ACD32', '#FFA500', '#FFC0CB']


    #draw ding
    for i, group in enumerate(ding_feature):
        # group = tsne.fit_transform(group)
        ax.scatter(group[:, 0], group[:, 1], group[:, 2], color=colors[i], marker="o", label=str(i+1))

    #draw gui
    for i, group in enumerate(gui_feature):
        # group = tsne.fit_transform(group)
        ax.scatter(group[:, 0], group[:, 1], group[:, 2], color=colors[i], marker="^", label=str(i+1))



    ax.legend(title='age color', bbox_to_anchor=(1.05, 1), loc='upper left')

    # 调整布局以确保图例完全显示在图片范围内
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)


    file_path = os.path.join(pth_path, save_image_name+".png")
    plt.savefig(file_path)

    # 显示图形
    plt.show()
    print(1)

