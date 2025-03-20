# __encoding__="utf-8"
"""
@Describe: 通用函数模块
@Author: ZhuQian
@DateTime: 2022/8/25 13:47
@SoftWare: PyCharm
"""
import os
import shutil

import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib as mpl
from matplotlib import pyplot as plt
# %matplotlib inline  # notebook中图片输出使用

np.set_printoptions(threshold=np.inf)           # np.inf表示正无穷
sbn.set(style="whitegrid", color_codes=True)    # 设置绘图风格
mpl.rcParams['font.family'] = ['sans-serif']
mpl.rcParams['font.sans-serif'] = ['SimHei']    # 正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12.0, 10.0)   # 设置图形大小
plt.rcParams['savefig.dpi'] = 300.              # 图片像素
plt.rcParams['figure.dpi'] = 300.               # 分辨率

# code类特征，类型转换
def float_to_int(df, convert_cols):
    """
    code类特征由float64转成str。如果某列含有NaN值，则默认是float64
    :param df: DataFrame, 源数据
    :param convert_cols: 待转换特征列表
    :return:
    """
    for col in convert_cols:
        df[col] = df[col].apply(np.int32)
    return df

# 图片是否保存
def whether_figure_save(save_path, figname):
    """
    是否保存图片
    :param save_path: str, 图片保存路径
    :param figname: str, 保存图片的名称，默认figname + '.jpg'
    :return:
    """
    if save_path is not None and figname is not None:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path)
        file_path = os.path.join(save_path, figname + ".jpg")
        plt.savefig(file_path, dpi=400)
    else:
        pass

# 单离散特征-直方图
def bar_plot(xname, yname, df, orient_type, save_path=None, figname=None, whether_text_annotate=False):
    """
    单离散特征-直方图, 支持水平直方图、垂直直方图。
    :param xname: str, x轴名称
    :param yname: str, y轴名称
    :param df: DataFrame, 数据源
    :param orient_type: str, 'h' -> 水平直方图, 'v' -> 垂直直方图
    :param save_path: str, 图片保存路径
    :param figname: str, 保存图片的名称
    :param font_size: int, 坐标轴字体大小
    :param whether_text_annotate: 是否标注文字，默认False
    :return:
    """
    fig = sbn.barplot(xname, yname, data=df, orient=orient_type)
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.title('{}-{} 直方图'.format(xname, yname))
    if whether_text_annotate:
        for p in fig.patches:
            if p.get_width() > 0:
                fig.annotate(
                    text=f"{p.get_width(): 0.3f}",  # 文字内容
                    xy=(p.get_width(), p.get_y() + p.get_height() / 2.),  # 文字的位置
                    xycoords='data', ha='center', va='center', fontsize=10., color='black',
                    xytext=(0, 0),  # 文字的偏移量
                    textcoords='offset points', clip_on=True)
        plt.tight_layout()
    plt.show()
    whether_figure_save(save_path, figname)

# 核心贡献分析 (帕累托分析 Pareto Analysis)
def pareto_analysis_plot(xname, yname1, yname2, df, line_value, save_path=None, figname=None):
    """
    核心贡献分析 (帕累托分析)
    :param xname: str, X轴名称
    :param yname1: str, Y1名称
    :param yname2: str, Y2名称
    :param df: DataFrame, 数据源
    :param line_value: float, 水平参考线的值
    :param save_path: str, 图片保存路径
    :param figname: str, 图片保存名称, xxxx.jpg格式
    :return:
    """
    fig, ax1 = plt.subplots()
    plt.xticks(rotation=90)    # X轴轴标签逆时针旋转90度
    ax1.bar(xname, yname1, data=df)
    ax1.set_xlabel(xname)      # X轴标签
    ax1.set_ylabel(yname1)     # Y轴标签
    #plt.axvline(x='福建区', ls='-.', color='steelblue')
    ax2 = ax1.twinx()
    ax2.plot(xname, yname2, data=df, marker='o', mec='r', mfc='w', color="red")
    ax2.set_ylabel(yname2)     # 另一Y轴标签
    fig.legend((yname1, yname2), loc="upper center", bbox_to_anchor=(1/2, 1), bbox_transform=ax1.transAxes)   # loc和bbox_to_anchor共同控制着图例的位置
    if line_value:
        plt.axhline(y=line_value, ls='--', color='pink')
    plt.show()
    whether_figure_save(save_path, figname)

# 双离散特征-同Y轴双折线图
def double_lineplot(xname, yname1, yname2, df,  y_value=None, line_label=None, whether_line_text=False, save_path=None, figname=None, xlabel=None, ylabel=None):
    """
    双离散特征-同Y轴双折线图
    :param xname: str, X轴名称
    :param yname1: str, Y1名称
    :param yname2: str, Y2名称
    :param df: DataFrame, 数据源
    :param y_value: float, 水平参考线
    :param line_label: str, 水平参考线的名称
    :param whether_line_text: Bool, 是否对水平参考线进行文字标注
    :param save_path: str, 图片保存路径
    :param figname: str, 图片保存名称, xxxx.jpg格式
    :param xlabel: str, X轴标签名称
    :param ylabel: str, Y轴标签名称
    :return:
    """
    plt.plot(xname, yname1, data=df, marker='o', mec='r', mfc='w', label='当日总签收量占比(%)')
    plt.plot(xname, yname2, data=df, marker='*', ms=10, label='虚假签收量占比(%)')
    if whether_line_text:
        plt.axhline(y=y_value, color='g', ls='--', label=line_label)
        plt.text(-3, y_value, str(y_value), fontdict={'size': '10', 'color': 'g'})
    plt.legend()             # 让图例生效
    if xlabel is not None:
        plt.xlabel(xlabel)
    else:
        plt.xlabel(xname)    # X轴标签
    plt.xticks(rotation=90)  # 旋转45度
    plt.ylabel(ylabel)       # Y轴标签
    plt.show()
    whether_figure_save(save_path, figname)

# 双离散特征-双Y轴折线图
def doubleYaxis_lineplot(xname, yname1, yname2, df, save_path=None, figname=None):
    """
    双离散特征-双Y轴折线图
    :param xname: str, X轴名称
    :param yname1: str, Y1名称
    :param yname2: str, Y2名称
    :param df: DataFrame, 数据源
    :param save_path: str, 图片保存路径
    :param figname: str, 图片保存名称, xxxx.jpg格式
    :param xlabel: str, X轴标签名称
    :param ylabel: str, Y轴标签名称
    :return:
    """
    fig, ax1 = plt.subplots()
    plt.xticks(rotation=90)     # X轴轴标签逆时针旋转90度
    ax1.plot(xname, yname1, data=df, marker='^', mec='r', mfc='w', color="blue", label=yname1)
    ax1.set_xlabel(xname)   # X轴标签
    ax1.set_ylabel(yname1)  # Y轴标签
    ax2 = ax1.twinx()
    ax2.plot(xname, yname2, data=df, marker='o', mec='r', mfc='w', color="red", label=yname2)
    ax2.set_ylabel(yname2)  # 另一Y轴标签
    fig.legend(loc="upper center", bbox_to_anchor=(1/2, 1), bbox_transform=ax1.transAxes)  # loc和bbox_to_anchor共同控制着图例的位置
    plt.show()
    whether_figure_save(save_path, figname)

# 单连续特征-直方图和小提琴图(支持分组)
def continuous_feature_plot(df, hist_feature_list, n_bins=50, fontsize=14, target=None, save_path=None):
    """
    连续特征的直方图和核密度图。若target不为空，同时展示分组直方图和分组箱线图.
    :param hist_feature_list: list, 连续特征列表.
    :param n_bins: int, 直方图分多少箱, 默认50箱.
    :param fontsize: int, 字体大小，默认为14.
    :param target: str, 目标变量，当前固定为2个(0: 好用户，1：坏用户).
     :param save_path: str, 图片保存目录
    """
    for col in hist_feature_list:
        print("连续特征：", col)
        try:
            if target is not None:
                fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(16, 8), facecolor="white")
                # 直方图
                plt.subplot(221)
                plt.tight_layout()
                sbn.distplot(df[col])
                plt.xlabel(col, fontdict={'weight': 'normal', 'size': fontsize})
                plt.title("{}--直方图".format(col), fontdict={'weight': 'normal', 'size': fontsize})  # 改变标题文字大小
                # 小提琴图(分位数)
                plt.subplot(222)
                plt.tight_layout()
                sbn.violinplot(x=col, data=df, palette="Set2", split=True, scale="area", inner="quartile")
                plt.xlabel(col, fontdict={'weight': 'normal', 'size': fontsize})
                plt.title("{}--小提琴图".format(col), fontdict={'weight': 'normal', 'size': fontsize})
   
                print("进行分组可视化......")
                unique_vals = df[target].unique().tolist()
                unique_val0 = df[df[target] == unique_vals[0]]
                unique_val1 = df[df[target] == unique_vals[1]]
                # unique_val2 = df[df[target] == unique_vals[2]]
                # 分组直方图
                plt.subplot(223)
                plt.tight_layout()
                sbn.distplot(unique_val0[col], bins=n_bins, kde=False, norm_hist=True, color='steelblue',
                             label=str(unique_vals[0]))
                sbn.distplot(unique_val1[col], bins=n_bins, kde=False, norm_hist=True, color='purple',
                             label=str(unique_vals[1]))
                # sns.distplot(unique_val2[col], bins=n_bins, kde=False, norm_hist=True, color='pink', label=str(unique_vals[2]))
                plt.xlabel(col, fontdict={'weight': 'normal', 'size': fontsize})
                plt.legend()
                plt.title("{}--分组直方图".format(col), fontdict={'weight': 'normal', 'size': fontsize})
                # 分组核密度图
                plt.subplot(224)
                plt.tight_layout()
                sbn.distplot(unique_val0[col], hist=False, kde_kws={"color": "red", "linestyle": "-"}, norm_hist=True,
                             label=str(unique_vals[0]))
                sbn.distplot(unique_val1[col], hist=False, kde_kws={"color": "black", "linestyle": "--"}, norm_hist=True,
                             label=str(unique_vals[1]))
                # sns.distplot(unique_val2[col], hist=False, kde_kws={"color":"green", "linestyle":"-."}, norm_hist=True, label=str(unique_vals[2]))
                plt.xlabel(col, fontdict={'weight': 'normal', 'size': fontsize})
                plt.legend()
                plt.title("{}--分组核密度图".format(col), fontdict={'weight': 'normal', 'size': fontsize})
                """
                分组箱线图    
                """
                # plt.subplot(222)
                # plt.tight_layout()
                # sns.boxplot(x=[unique_val0[col], unique_val1[col]],  labels=[unique_vals[0], unique_vals[1]])
                # plt.xlabel(col, fontdict={'weight':'normal', 'size': fontsize})
                # plt.title("{}特征的分组箱线图".format(col), fontdict={'weight':'normal', 'size': fontsize})
            else:
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8), facecolor="white")
                # 直方图
                plt.subplot(121)
                plt.tight_layout()
                sbn.distplot(df[col])
                plt.xlabel(col, fontdict={'weight': 'normal', 'size': fontsize})
                plt.title("{}--直方图".format(col), fontdict={'weight': 'normal', 'size': fontsize})  # 改变标题文字大小
                # 小提琴图(分位数)
                plt.subplot(122)
                plt.tight_layout()
                sbn.violinplot(x=col, data=df, palette="Set2", split=True, scale="area", inner="quartile")
                plt.xlabel(col, fontdict={'weight': 'normal', 'size': fontsize})
                plt.title("{}--小提琴图".format(col), fontdict={'weight': 'normal', 'size': fontsize})
    
            fig_name = "{}--直方图&箱线图.jpg".format(col)
            whether_figure_save(save_path, fig_name)
            plt.show()
        except Exception as e:
            print('{}--error info:\n{}'.format(col, e))
        print(df[col].describe())

# 连续型特征分布探查
def continuous_visualize(df, n_bins=50, fontsize=14, hist_feature_list=None, target=None, exclude_cols=None, save_path=None):
    """
    连续型特征可视化，默认直方图和核密度图。若target不空，同时展示分组直方图和分组箱线图.
    :param df: DataFrame, 数据集
    :param hist_feature_list: list, 连续型特征列表
    :param n_bins: int, 分箱数
    :param fontsize: int, 字体大小，默认10
    :param target: str, 模型label列名称
    :param exclude_cols: list, 剔除的特征列表
	:param save_path: str, 保存路径
    :return:
    """
    if hist_feature_list is None:
        hist_feature_list = list(df.columns)
    for col in hist_feature_list:
        if (col != target) & (col not in exclude_cols):
            mid_data = df[df[col] != -999]
            continuous_feature_plot(mid_data, [col], n_bins=n_bins, fontsize=fontsize, target=target, save_path=save_path)

# 相关系数热力图(左下角矩阵)
def corr_plot(df, corr_type='pearson', save_path=None, figname=None):
    """
    特征相关性分析
    :param df: DataFrame, 数据源
    :param corr_type: str, 默认pearson系数，支持spearman, kendall
    :param save_path: str, 图片保存路径
    :param figname: str, 图片保存名称
    :return:
    """
    if corr_type in ['spearman', 'kendall']:
        corr_coef = df.corr(method=corr_type)
    else:
        corr_coef = df.corr()
    mask = np.zeros_like(corr_coef)
    # 将mask的对角线及以上设置为True, 对应要被遮掉的部分
    mask[np.triu_indices_from(mask)] = True
    sbn.heatmap(corr_coef, mask=mask, fmt='.3f', cmap='Set1', center=0, annot=True, annot_kws={'size': 12, 'weight': 'normal', 'color': '#253D24'},)
    plt.title("特征相关性分析")
    plt.show()
    whether_figure_save(save_path, figname)

# 异常值处理：盖帽法
def deal_outliers(x, lower=0.05, upper=0.95):
    """
    利用盖帽法处理异常值
    :param x: Series对象
    :param lower: 分位数下限，默认0.05
    :param upper: 分位数上限，默认0.95
    :return:
    """
    ql, qu = x.quantile(lower), x.quantile(upper)
    out = x.mask(x < ql, ql)
    out = out.mask(out > qu, qu)
    return out













