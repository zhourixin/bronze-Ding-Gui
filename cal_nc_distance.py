
import numpy as np
from scipy.stats import wasserstein_distance, entropy
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import MinMaxScaler, minmax_scale
import torch
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde
from scipy.special import kl_div
from sklearn.preprocessing import StandardScaler

def calculate_wasserstein_distance(feature1, feature2):
    return wasserstein_distance(feature1, feature2)

def calculate_kl_divergence(feature1, feature2):
    return entropy(feature1, feature2)

def calculate_mmd(feature1, feature2, kernel=rbf_kernel):
    # feature1 = feature1.reshape(-1, 1)
    # feature2 = feature2.reshape(-1, 1)
    Kxx = kernel(feature1, feature1)
    Kyy = kernel(feature2, feature2)
    Kxy = kernel(feature1, feature2)
    return np.mean(Kxx) + np.mean(Kyy) - 2 * np.mean(Kxy)

def calculate_js_divergence(feature1, feature2):
    return jensenshannon(feature1, feature2)

def calculate_bhattacharyya_distance(feature1, feature2):
    return -np.log(np.sum(np.sqrt(feature1 * feature2)))

def cal_distance(npy_path):


    # features_A = np.random.rand(100, 10)
    # features_B = np.random.rand(200, 10)

    ding_feature_path = npy_path+"/analyse_ding_feature_list.pth"
    gui_feature_path = npy_path+"/analyse_gui_feature_list.pth"
    xv_feature_path = npy_path+"/analyse_xv_feature_list.pth"

    ding_feature_origin = torch.load(ding_feature_path).numpy()
    gui_feature_origin = torch.load(gui_feature_path).numpy()
    xv_feature_origin = torch.load(xv_feature_path).numpy()

    ding_feature = ding_feature_origin.mean(axis=0)
    gui_feature = gui_feature_origin.mean(axis=0)
    xv_feature = xv_feature_origin.mean(axis=0)

    # ding_feature = torch.load(ding_feature_path).numpy()[:230, :]
    # gui_feature = torch.load(gui_feature_path).numpy()[:230, :]
    # xv_feature = torch.load(xv_feature_path).numpy()[:230, :]

    # 归一化特征
    # scaler = MinMaxScaler()
    ding_feature_minmax = minmax_scale(ding_feature, feature_range=(0, 1))
    gui_feature_minmax =minmax_scale(gui_feature, feature_range=(0, 1))
    xv_feature_minmax = minmax_scale(xv_feature, feature_range=(0, 1))

    ding_normalized = ding_feature_minmax / np.sum(ding_feature_minmax)
    gui_normalized = gui_feature_minmax / np.sum(gui_feature_minmax)
    xv_normalized = xv_feature_minmax / np.sum(xv_feature_minmax)




    # 计算Wasserstein距离
    wasserstein_distance1 = calculate_wasserstein_distance(ding_feature, xv_feature)
    print(f'ding-xv Wasserstein距离: {wasserstein_distance1}')
    wasserstein_distance2 = calculate_wasserstein_distance(gui_feature, xv_feature)
    print(f'gui-xv Wasserstein距离: {wasserstein_distance2}')

    # 计算KL散度
    kl_divergence1 = calculate_kl_divergence(ding_normalized, xv_normalized)
    print(f'ding-xv KL散度: {kl_divergence1}')
    kl_divergence2 = calculate_kl_divergence(gui_normalized, xv_normalized)
    print(f'gui-xv KL散度: {kl_divergence2}')

    # 计算最大均值差异度量（MMD）
    mmd1 = calculate_mmd(ding_feature_origin, xv_feature_origin)
    print(f'ding-xv MMD: {mmd1}')
    mmd2= calculate_mmd(gui_feature_origin, xv_feature_origin)
    print(f'gui-xv MMD: {mmd2}')

    # 计算JS散度
    js_divergence1 = calculate_js_divergence(ding_normalized, xv_normalized)
    print(f'ding-xv JS散度: {js_divergence1}')
    js_divergence2 = calculate_js_divergence(gui_normalized, xv_normalized)
    print(f'gui-xv JS散度: {js_divergence2}')

    bs_distance1 = calculate_bhattacharyya_distance(ding_normalized, xv_normalized)
    bs_distance2 = calculate_bhattacharyya_distance(gui_normalized, xv_normalized)
    print(f'ding-xv 巴氏距离: {bs_distance1}')
    print(f'gui-xv 巴氏距离: {bs_distance2}')


def cal_distance_noBG(npy_path):


    # features_A = np.random.rand(100, 10)
    # features_B = np.random.rand(200, 10)

    ding_feature_path = npy_path+"/analyse_ding_noBG_feature_list.pth"
    gui_feature_path = npy_path+"/analyse_gui_noBG_feature_list.pth"
    xv_feature_path = npy_path+"/analyse_xv_noBG_feature_list.pth"

    ding_feature_origin = torch.load(ding_feature_path).numpy()
    gui_feature_origin = torch.load(gui_feature_path).numpy()
    xv_feature_origin = torch.load(xv_feature_path).numpy()

    # 先做一下标准化试试
    # scaler = StandardScaler()
    # ding_feature_norm = scaler.fit_transform(ding_feature_origin)
    # gui_feature_norm = scaler.fit_transform(gui_feature_origin)
    # xv_feature_norm = scaler.fit_transform(xv_feature_origin)

    ding_feature_norm = ding_feature_origin
    gui_feature_norm = gui_feature_origin
    xv_feature_norm = xv_feature_origin
    

    # # 使用高斯核密度估计来估计每个特征的概率密度函数
    # kde_ding = gaussian_kde(ding_feature_norm)
    # kde_gui = gaussian_kde(gui_feature_norm)
    # kde_xv = gaussian_kde(xv_feature_norm)

    # 使用直方图方法来进行密度估计
    bins = np.histogram_bin_edges(np.concatenate((ding_feature_norm, gui_feature_norm, xv_feature_norm)), bins='auto')
    
    hist1, bin_edges1 = np.histogram(ding_feature_norm, bins=bins, density=True)
    hist2, bin_edges2 = np.histogram(gui_feature_norm, bins=bins, density=True)
    hist3, bin_edges3 = np.histogram(xv_feature_norm, bins=bins, density=True)

    # 计算KL散度
    # kl_divergence = kl_div(kde_ding.pdf(ding_feature_norm), kde_xv.pdf(xv_feature_norm))
    kl_divergence1 = entropy(hist1+1e-10, hist3+1e-10)
    kl_divergence2 = entropy(hist2+1e-10, hist3+1e-10)

    # 计算JS散度
    # js_divergence = jensenshannon(kde_ding.pdf(ding_feature_norm), kde_xv.pdf(xv_feature_norm))
    js_divergence1 = jensenshannon(hist1, hist3)
    js_divergence2 = jensenshannon(hist2, hist3)

    # 计算Wasserstein距离
    wasserstein = wasserstein_distance(features1.flatten(), features2.flatten())

    # 计算最大均值差异度量（MMD）
    mmd = np.mean(rbf_kernel(features1, features1)) + np.mean(rbf_kernel(features2, features2)) - 2 * np.mean(rbf_kernel(features1, features2))

    # 计算巴氏距离
    # 首先，我们需要计算协方差矩阵和其逆矩阵
    covariance_matrix = np.cov(np.concatenate((features1, features2), axis=0).T)
    inverse_covariance_matrix = np.linalg.inv(covariance_matrix)

    # 然后，我们计算两个特征的均值
    mean1 = np.mean(features1, axis=0)
    mean2 = np.mean(features2, axis=0)

    # 最后，我们计算巴氏距离
    mahalanobis_distance = mahalanobis(mean1, mean2, inverse_covariance_matrix)



    ding_feature = ding_feature_origin.mean(axis=0)
    gui_feature = gui_feature_origin.mean(axis=0)
    xv_feature = xv_feature_origin.mean(axis=0)

    # ding_feature = torch.load(ding_feature_path).numpy()[:230, :]
    # gui_feature = torch.load(gui_feature_path).numpy()[:230, :]
    # xv_feature = torch.load(xv_feature_path).numpy()[:230, :]

    # 归一化特征
    # scaler = MinMaxScaler()
    ding_feature_minmax = minmax_scale(ding_feature, feature_range=(0, 1))
    gui_feature_minmax =minmax_scale(gui_feature, feature_range=(0, 1))
    xv_feature_minmax = minmax_scale(xv_feature, feature_range=(0, 1))

    ding_normalized = ding_feature_minmax / np.sum(ding_feature_minmax)
    gui_normalized = gui_feature_minmax / np.sum(gui_feature_minmax)
    xv_normalized = xv_feature_minmax / np.sum(xv_feature_minmax)




    # 计算Wasserstein距离
    wasserstein_distance1 = calculate_wasserstein_distance(ding_feature, xv_feature)
    print(f'ding-xv Wasserstein距离: {wasserstein_distance1}')
    wasserstein_distance2 = calculate_wasserstein_distance(gui_feature, xv_feature)
    print(f'gui-xv Wasserstein距离: {wasserstein_distance2}')

    # 计算KL散度
    kl_divergence1 = calculate_kl_divergence(ding_normalized, xv_normalized)
    print(f'ding-xv KL散度: {kl_divergence1}')
    kl_divergence2 = calculate_kl_divergence(gui_normalized, xv_normalized)
    print(f'gui-xv KL散度: {kl_divergence2}')

    # 计算最大均值差异度量（MMD）
    mmd1 = calculate_mmd(ding_feature_origin, xv_feature_origin)
    print(f'ding-xv MMD: {mmd1}')
    mmd2= calculate_mmd(gui_feature_origin, xv_feature_origin)
    print(f'gui-xv MMD: {mmd2}')

    # 计算JS散度
    js_divergence1 = calculate_js_divergence(ding_normalized, xv_normalized)
    print(f'ding-xv JS散度: {js_divergence1}')
    js_divergence2 = calculate_js_divergence(gui_normalized, xv_normalized)
    print(f'gui-xv JS散度: {js_divergence2}')

    bs_distance1 = calculate_bhattacharyya_distance(ding_normalized, xv_normalized)
    bs_distance2 = calculate_bhattacharyya_distance(gui_normalized, xv_normalized)
    print(f'ding-xv 巴氏距离: {bs_distance1}')
    print(f'gui-xv 巴氏距离: {bs_distance2}')




