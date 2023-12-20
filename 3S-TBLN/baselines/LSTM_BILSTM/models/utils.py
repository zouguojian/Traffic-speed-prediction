# -- coding: utf-8 --
from models.inits import *
import seaborn as sns

def construct_feed_dict(xs, xs_all, label_s, d_of_week, day, hour, minute, mask=[],placeholders=None, site =207, is_training=True):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['position']: np.array([[i for i in range(site)]],dtype=np.int32)})
    feed_dict.update({placeholders['labels']: label_s})
    feed_dict.update({placeholders['week']: d_of_week})
    feed_dict.update({placeholders['day']: day})
    feed_dict.update({placeholders['hour']: hour})
    feed_dict.update({placeholders['minute']: minute})
    feed_dict.update({placeholders['features']: xs})
    feed_dict.update({placeholders['features_all']: xs_all})
    feed_dict.update({placeholders['random_mask']: mask})
    feed_dict.update({placeholders['is_training']: is_training})
    return feed_dict

import matplotlib.pyplot as plt
def describe(label, predict):
    '''
    :param label:
    :param predict:
    :param prediction_size:
    :return:
    '''
    plt.figure()
    # Label is observed value,Blue
    plt.plot(label[0:], 'b', label=u'actual value')
    # Predict is predicted value，Red
    plt.plot(predict[0:], 'r', label=u'predicted value')
    # use the legend
    plt.legend()
    # plt.xlabel("time(hours)", fontsize=17)
    # plt.ylabel("pm$_{2.5}$ (ug/m$^3$)", fontsize=17)
    # plt.title("the prediction of pm$_{2.5}", fontsize=17)
    plt.show()

def metric(pred, label):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label.astype(np.float32))
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
        cor = np.mean(np.multiply((label - np.mean(label)),
                                  (pred - np.mean(pred)))) / (np.std(pred) * np.std(label))
        sse = np.sum((label - pred) ** 2)
        sst = np.sum((label - np.mean(label)) ** 2)
        r2 = 1 - sse / sst  # r2_score(y_actual, y_predicted, multioutput='raw_values')
    return mae, rmse, mape


def seaborn(x =None, len=12, heads=4):
    '''
    :param x:
    :return:
    '''
    """
    document: https://seaborn.pydata.org/generated/seaborn.heatmap.html#seaborn.heatmap
    根据data传入的值画出热力图，一般是二维矩阵
    vmin设置最小值, vmax设置最大值
    cmap换用不同的颜色
    center设置中心值
    annot 是否在方格上写上对应的数字
    fmt 写入热力图的数据类型，默认为科学计数，d表示整数，.1f表示保留一位小数
    linewidths 设置方格之间的间隔
    xticklabels，yticklabels填到横纵坐标的值。可以是bool，填或者不填。可以是int，以什么间隔填，可以是list
    color: Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, 
    Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, 
    PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, 
    PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, 
    RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, 
    Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, 
    binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, 
    copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, 
    gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, 
    gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, icefire, icefire_r, inferno, inferno_r, 
    jet, jet_r, magma, magma_r, mako, mako_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, 
    prism, prism_r, rainbow, rainbow_r, rocket, rocket_r, seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, 
    tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, twilight, twilight_r, twilight_shifted, twilight_shifted_r, 
    viridis, viridis_r, vlag, vlag_r, winter, winter_r

    """
    i=5
    f, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=heads,ncols=3)
    sns.heatmap(x[i,:,:108], annot=False, ax=ax1[0],cbar=True,cmap='Blues')
    sns.heatmap(x[i,:,108:216], annot=False, ax=ax1[1],cbar=True, cmap='Greens')
    sns.heatmap(x[i,:,216:], annot=False, ax=ax1[2],cbar=True,cmap='Greys')

    sns.heatmap(x[i+12,:,:108], annot=False, ax=ax2[0],cbar=True,cmap='Blues')
    sns.heatmap(x[i+12,:,108:216], annot=False, ax=ax2[1],cbar=True, cmap='Greens')
    sns.heatmap(x[i+12,:,216:], annot=False, ax=ax2[2],cbar=True,cmap='Greys')

    sns.heatmap(x[i+24,:,:108], annot=False, ax=ax3[0],cbar=True,cmap='Blues')
    sns.heatmap(x[i+24,:,108:216], annot=False, ax=ax3[1],cbar=True, cmap='Greens')
    sns.heatmap(x[i+24,:,216:], annot=False, ax=ax3[2],cbar=True,cmap='Greys')

    sns.heatmap(x[i+36,:,:108], annot=False, ax=ax4[0],cbar=True,cmap='Blues')
    sns.heatmap(x[i+36,:,108:216], annot=False, ax=ax4[1],cbar=True, cmap='Greens')
    sns.heatmap(x[i+36,:,216:], annot=False, ax=ax4[2],cbar=True,cmap='Greys')

    plt.show()


def mae_los(pred, label):
    mask = tf.not_equal(label, 0)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.compat.v2.where(
        condition = tf.math.is_nan(mask), x = 0., y = mask)
    loss = tf.abs(tf.subtract(pred, label))
    loss *= mask
    loss = tf.compat.v2.where(
        condition = tf.math.is_nan(loss), x = 0., y = loss)
    loss = tf.reduce_mean(loss)
    return loss

def mse_los(pred, label):
    ''''''
    mask = tf.not_equal(label, 0)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.compat.v2.where(
        condition=tf.math.is_nan(mask), x=0., y=mask)

    loss = tf.square(tf.subtract(pred, label))
    loss *= mask
    loss = tf.compat.v2.where(
        condition=tf.math.is_nan(loss), x=0., y=loss)
    loss = tf.reduce_mean(loss)
    return loss


def huber_los(pred, label):
    mask = tf.not_equal(label, 0)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.compat.v2.where(
        condition=tf.math.is_nan(mask), x=0., y=mask)
    loss = tf.square(tf.subtract(pred, label))
    loss *= mask
    loss = tf.compat.v2.where(
        condition=tf.math.is_nan(loss), x=0., y=loss)
    loss = tf.reduce_mean(loss)
    tf.sqrt(tf.reduce_mean(tf.square(pred - label), axis=-1))
    return loss

# huber loss
def huber(true, pred, delta):
    loss = np.where(np.abs(true-pred) < delta , 0.5*((true-pred)**2), delta*np.abs(true - pred) - 0.5*(delta**2))
    return np.sum(loss)


# log cosh loss
def logcosh(true, pred):
    loss = np.log(np.cosh(pred - true))
    return np.sum(loss)