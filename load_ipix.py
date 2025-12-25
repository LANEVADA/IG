'''
Loading the IPIX dataset, Convert from the netCDF to the numpy array.
'''
import os
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift, ifft, fft2
import seaborn as sns
from numpy import std, subtract, polyfit, sqrt, log
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import signal
from scipy.fftpack import fftfreq
from scipy.stats import norm
from scipy import io
from tftb.generators import fmlin
from tftb.processing import WignerVilleDistribution,PseudoWignerVilleDistribution

def ipixinfo():
    '''
    List the information of the ipix radar 1993_clutter_datasets in the netCDF file format.
    implementation the ipixinfo.m in http://soma.ece.mcmaster.ca/ipix/dartmouth/mfiles/ipixinfo.m
    :return:
    '''
def ipixazm(nc):
    # % function [azm,outlier]=ipixazm(nc)
    # % In the Dartmouth database there are a few  files in which the azimuth angle
    # % time-series is corrupt, in particular data sets 8, 16, 23, 24, 28, 110, 254, 276.
    # % This function fixes this bug.
    # % Inputs:
    # %   nc       - pointer to netCDF file
    # %
    # % Outputs:
    # %   azm      - corrected azimuth angle time series.
    # %   outliers - indices of azimuth angles considered outliers
    # %
    # % Rembrandt Bakker, McMaster University, 2001-2002
    # %
    #
    # function [azm,outlier]=ipixazm(nc)
    #
    azm =np.unwrap(nc['azimuth_angle'][:]*np.pi/180)
    dazm=np.diff(azm)
    meddazm=np.median(dazm)
    outlier= np.abs(dazm)>2*np.abs(meddazm)
    dazm[outlier]=meddazm
    newdazm = np.concatenate([[azm[0]], dazm])
    azm=np.cumsum(newdazm)*180/np.pi;
    return azm
def ipixload(nc, pol, rangebin, mode):
    # % function [I,Q,meanIQ,stdIQ,inbal]=ipixload(nc,pol,rangebin,mode);
    # %
    # % Loads I and Q data from IPIX radar cdf file.
    # % Inputs:
    # %   nc       - pointer to netCDF file
    # %   pol      - Transmit/receive polarization ('vv','hh','vh', or 'hv')
    # %   rangebin - Range bin(s) to load
    # %   mode     - Pre-processing mode:
    # %              'raw' does not apply any corrections to the data;
    # %              'auto' applies automatic corrections, assumes that radar
    # %                 does not look at still objects, such as land;
    # %              'dartmouth' first removes land, using knowledge of the geometry
    # %                 of the Dartmouth site. Then same as 'auto'.
    # %
    # % Outputs:
    # %   I,Q      - Pre-processed in-phase and quadrature component of data
    # %   meanIQ   - Mean of I and Q used in pre-processing
    # %   stdIQ    - Standard deviation of I and Q used in pre-processing
    # %   inbal    - Phase inbalance [degrees] used in pre-processing
    # %
    # % Rembrandt Bakker, McMaster University, 2001-2002
    # % Yi ZHOU, Dalian Maritime University, 01-07-2020

    #% % check inputs
    nrange = len(nc.variables['range'][:])
    if rangebin < 0 | rangebin >= nrange:
        print('Warning: rangebin %d not found in file %s ' %(rangebin, nc.NetCDF_file_name[:]))
        return
    # #% % in some cdf files, the unsigned flag is not set correctly % %
    adc_data = nc.variables['adc_data']
    #
    H_txpol = 0 #1
    V_txpol = 1 #2
    Like_adc_I  = 0#nc.variables['adc_like_I'][0]
    Like_adc_Q  = 1#nc.variables['adc_like_Q'][0]
    Cross_adc_I = 2#nc.variables['adc_cross_I'][0]
    Cross_adc_Q = 3#nc.variables['adc_cross_Q'][0]
    #
    # % % extract correct polarization from cdffile % %
    pol = str.lower(pol)
    if len(adc_data.shape) == 3:
        # % read global attribute TX_polarization,
        # there is no 'ntxpol' in the adc_data dimensions.
        txpol = nc.TX_polarization[0]
        if pol[0]!=str.lower(txpol):
            fname = nc.NetCDF_file_name[:]+'\0'+''
            print('Warning: file '+ fname+' does not contain '
                  +txpol[0]+ ' transmit polarization.')
        if pol in ['hh', 'vv']:
            xiq = adc_data[:, rangebin, [Like_adc_I,  Like_adc_Q]]
        if pol in ['hv', 'vh']:
            xiq = adc_data[:, rangebin, [Cross_adc_I, Cross_adc_Q]]
        I=xiq[:,0]
        Q=xiq[:,1]
    else:
        if pol == 'hh':
            xiq = adc_data[:, H_txpol, rangebin, [Like_adc_I, Like_adc_Q]]
        if pol == 'hv':
            xiq = adc_data[:, H_txpol, rangebin, [Cross_adc_I, Cross_adc_Q]]
        if pol == 'vv':
            xiq = adc_data[:, V_txpol, rangebin, [Like_adc_I, Like_adc_Q]]
        if pol == 'vh':
            xiq = adc_data[:, V_txpol, rangebin, [Cross_adc_I, Cross_adc_Q]]

        # Ihh=adc_data[:, 0, rangebin, Like_adc_I]
        # Qhh=adc_data[:, 0, rangebin, Like_adc_Q]
        # Ihh_sum = np.sum(Ihh)
        # Qhh_sum = np.sum(Qhh)
        #
        # Ihv=adc_data[:, 0, rangebin, Cross_adc_I]
        # Qhv=adc_data[:, 0, rangebin, Cross_adc_Q]
        # Ihv_sum = np.sum(Ihv)
        # Qhv_sum = np.sum(Qhv)
        #
        # Ivv=adc_data[:, 1, rangebin, Like_adc_I]
        # Qvv=adc_data[:, 1, rangebin, Like_adc_Q]
        # Ivv_sum = np.sum(Ivv)
        # Qvv_sum = np.sum(Qvv)
        #
        # Ivh=adc_data[:, 1, rangebin, Cross_adc_I]
        # Qvh=adc_data[:, 1, rangebin, Cross_adc_Q]
        # Ivh_sum = np.sum(Ivh)
        # Qvh_sum = np.sum(Qvh)

        I = xiq[:, 0]
        Q = xiq[:, 1]
        # check the value from the matlab.
        # pmat = io.loadmat('/Users/yizhou/code/Matlab/IPIX/Ihh.mat')
        # mat_I=pmat['I'].ravel()

    if adc_data.dtype == 'int8':
        # negI = I<0
        # negQ = Q<0
        I = I.astype('float')
        Q = Q.astype('float')
        I[I<0]+=256
        Q[Q<0]+=256

    if mode ==  'raw':
      meanI, meanQ=(0,0)
      stdI,  stdQ =(1,1)
      inbal=0
    if mode == 'auto':
#       % Pre-processing
        meanI = np.mean(I)
        meanQ = np.mean(Q)
        stdI  = np.std(I)
        stdQ  = np.std(Q)
        I     =(I-meanI)/stdI
        Q     =(Q-meanQ)/stdQ
        sin_inbal=np.mean(I[:]*Q[:])
        inbal=np.arcsin(sin_inbal)*180/np.pi
        I=(I-Q*sin_inbal)/np.sqrt(1-sin_inbal**2)
#     if mode == 'dartmouth':
# #       % Define rectangular patches of land in Dartmouth campaign.
# #       % Format: [azmStart azmEnd  rangeStart rangeEnd]
#       landcoord=[ [0,  70,     0,  600],
#                   [305,360,    0,  600],
#                   [30,  55,    0, 8000],
#                   [210, 305,   0, 4700],
#                   [320, 325,  2200, 2700]]
# #       % Exclude land from data used to estimate pre-processing parameters
#       azm=mod(ipixazm(nc),360)
#       range=nc['range'][rangebin-1]
#       nbin=len(rangebin)
#       ok=np.ones_like(I)
#
#       for tlbr in landcoord:
#           mask =  ((tlbr[0] <= azm<= tlbr[1]) & (tlbr[2]<=range<=tlbr[3]))
#           ok[mask] = 0
#       # for i=1:size(landcoord,1),
#       #   for r=1:nbin,
#       #     if range(r)>=landcoord(i,3) & range(r)<=landcoord(i,4),
#       #       ok(find(azm>=landcoord(i,1) & azm<=landcoord(i,2)),r)=0;
#       #     end
#       #   end
#       # end
#       ok=find(ok);
#       if len(ok)<100:
#         print(['Warning: not enough sweeps for land-free pre-processing.'])
#         ok=np.ones_like(I)
#
#       #% Pre-processing
#       meanI=np.mean(I[ok])
#       meanQ=np.mean(Q[ok])
#       stdI =np.std(I[ok])
#       stdQ =np.std(Q[ok])
#       I=(I-meanI)/stdI
#       Q=(Q-meanQ)/stdQ
#       sin_inbal=np.mean(I[ok]*Q[ok])
#       inbal    =np.asin(sin_inbal)*180/np.pi
#       I=(I-Q*sin_inbal)/sqrt(1-sin_inbal^2)
    meanIQ = [meanI, meanQ]
    stdIQ  = [stdI,  stdQ]
    return I,Q, meanIQ, stdIQ, inbal

# 一维幅度序列
def ipixrange(data):
    # 读取data某一行的数据
    data_abs= np.abs(data)[5]
    # 归一化data某一行
    data_abs = data_abs/np.max(data_abs)
    plt.title('1993_17_6_range')
    plt.plot(data_abs)
    plt.show()

# 幅度分布
def cal_amplitute(data):

    data_abs = np.abs(data)
    plt.hist(data_abs, bins=100, density=True, range=(0, 6), stacked=True)
    #拟合曲线
    # mu, sigma = norm.fit(data)
    # x = np.linspace(0,6,1000)
    # y = norm.pdf(x,mu, sigma)
    # plt.plot(x, y ,'r-', label='Fit')
    plt.title('1993_17_7_amplitute')
    plt.savefig('1993_17_7_amplitute.png')
    plt.show()

# 功率谱
def cal_power_spectrum(x):
    clutter = x[0:500, 1]
    target = x[800:1300, 8]
    
    # 幅值
    # plt.subplot(2,1,1)
    # plt.title('clutter')
    # plt.plot(10*np.log10(np.abs(clutter)))

    # plt.subplot(2,1,2)
    # plt.title('target')
    # plt.plot(10*np.log10(np.abs(target)))
    # plt.show()

    fs = 1000 #采样频率
    N = len(clutter) #数据点数
    n = np.arange(0, N, 1)
    # print(n)
    #快速傅里叶变换
    y1 = fft(x, N)
    y2 = fftshift(y1)

    mag1 = np.abs(y1)
    mag2 = np.abs(y2)

    f1 = n*fs/N #频率序列
    f2 = n*fs/N-fs/2
    print(10*np.log10(np.abs(fftshift(fft(clutter, N)))))
    print(10*np.log10(np.abs(fftshift(fft(target, N)))))
    plt.subplot(2,1,1)
    plt.title('clutter')
    plt.plot(f2,10*np.log10(np.abs(fftshift(fft(clutter, N)))))

    plt.subplot(2,1,2)
    plt.title('target')
    plt.plot(f2,10*np.log10(np.abs(fftshift(fft(target, N)))))
    plt.show()


    # # 直接快速傅里叶变换
    # plt.subplot(5,1,1)
    # plt.title('usual FFT')
    # plt.plot(20*np.log10(mag1))

    # # 没有fftshift的
    # plt.subplot(5,1,2)
    # plt.title('FFT without fftshift')
    # plt.plot(20 * np.log10(mag1))

    # # fftshift之后的
    # plt.subplot(5,1,3)
    # plt.title('FFT after fftshift')
    # plt.plot(20 * np.log10(mag2))

    # #功率谱，直接法
    # ps = y2**2/N
    # plt.subplot(5,1,4)
    # plt.title('power spectrum direct method')
    # plt.plot(20*np.log10(ps))

    # #相关功率谱，间接法
    # cor_x = np.correlate(data, data, 'same')
    # cor_X = fftshift(fft(cor_x, N))
    # ps_cor = np.abs(cor_X)
    # ps_cor = ps_cor / np.max(ps_cor)
    # plt.subplot(5, 1, 5)
    # plt.title('power spectrum indirect method')
    # plt.plot(20 * np.log10(ps_cor))
    # plt.suptitle('19980204_163113_24_spectrum')
    # # plt.savefig('19980204_163113_24_spectrum.png')
    # plt.tight_layout()
    # plt.show()

# 距离-多普勒图
def range_fft(data):
    data = data[:, :]
    fs = 1000 #采样频率
    N = len(data) #数据点数
    n = np.arange(0, N, 1)
    f1 = n*fs/N #频率序列
    f2 = n*fs/N-fs/2
    # 对数据的每一行做fft，然后取对数，画距离-频谱图
    r_doppler = np.zeros((N,14), dtype=float)
    for i in range(14):
        r_doppler[:, i] = 10*np.log10(np.abs(fftshift(fft(data[:, i], N))))
        # print(r_doppler[:10, i])
    # print(r_doppler.shape)
    # print(r_doppler)
    # 归一化
    # r_doppler = r_doppler/np.max(r_doppler)
    # print(r_doppler.shape)
    plt.figure(figsize=(5, 5))  # 创建一个新的图像，设置图像的大小为10x5
    plt.imshow(r_doppler, cmap='jet', aspect='auto', extent=[0,14,-500,500])  # 添加 aspect='auto' 参数，让图像的宽度和高度自动调整
    plt.gca().invert_yaxis()
    # 图像的颜色范围设置为-500到500
    # plt.clim(0, 1)
    plt.xlabel('Index')  # 添加横轴标签
    plt.ylabel('Value')  # 添加纵轴标签
    plt.xticks(np.arange(0, 14, 1))  # 设置横轴刻度
    plt.yticks(np.arange(-500,500, 100))  # 设置纵轴刻度
    # plt.yticks(f2)  # 设置纵轴刻度
    plt.clim(r_doppler.min(), r_doppler.max())
    plt.colorbar()
    plt.show()
    return r_doppler

# 有点问题，和上面的函数相似但不一样
def test_range_fft(data):
    fs = 1000 #采样频率
    N = 1024 #数据点数
    n = np.arange(0, N, 1)
    f1 = n*fs/N #频率序列
    f2 = n*fs/N-fs/2
    # 对数据的每一行做fft，然后取对数，画距离-频谱图
    rd = np.abs(fftshift(fft2(data)))
    print(rd.shape)
    print(rd)
    # 画图,图像的宽度和高度自动调整
    plt.figure(figsize=(10, 5))
    plt.imshow(10*np.log10(rd), cmap='jet', aspect='auto')
    plt.clim(0, 50)
    plt.colorbar()  # 显示颜色条
    plt.show()
    return rd

#自相关系数
def correlation(data):
    # N = len(data)
    # variance = np.var(data)
    # data = data - np.mean(data)
    # r = np.correlate(data, data, 'same')
    # print(r.size)
    # print('r:',r)
    # result = r / (variance*(np.arange(N,0,-1)))
    # plt.plot(result)

    # plot_acf(data, lags=30)
    # plt.show()

    lags=30
    n = len(data)
    x = np.array(data)
    result = [np.correlate(x[i:] - x[i:].mean(), x[:n - i] - x[:n - i].mean())[0] \
              / (x[i:].std() * x[:n - i].std() * (n - i)) for i in range(1, lags + 1)]
    plt.plot(result)
    plt.show()

# 距离像
def amplitude_spatial_temporal(data):
    # 先绝对值
    data_abs = np.abs(data)
    # 再对数
    data_log = 10 * np.log10(data_abs + np.spacing(1))
    # 再归一化
    data_norm = (data_log - np.min(data_log))*63 / (np.max(data_log) - np.min(data_log))

    plt.figure(figsize=(6, 4))
    plt.imshow(data_norm.T, cmap='jet', aspect='auto')
    plt.gca().invert_yaxis()
    plt.axis('off')
    plt.tight_layout()
    # plt.clim(0, 60)  # 设置颜色范围
    # plt.colorbar()
    # 保存到桌面
    # plt.savefig('/Users/jjw/Desktop/#17.png', dpi=300)
    plt.show()

# 求平均信杂比
def SCR_siglech(data):

    # ascr = 10*np.log10((1/N*目标单元回波序列的模值平方-估算的海杂波平均功率)/估算的海杂波平均功率)
    # 估算的海杂波平均功率pc就是在众多纯海杂波单元中选一个概率比较高的数值
    data_amplitude = np.sum(np.square(np.round(np.abs(data),2)), axis=0)
    print(data_amplitude.shape)
    # plt.hist(np.square(np.round(np.abs(data),2)), bins=100, density=True, range=(0, 0.05), stacked=True)
    # plt.show()
    pc = 0.03 #hh极化
    target = data_amplitude[8]/(data.shape[0])
    print(pc)
    print(target)
    print("snr:",10*np.log10((target-pc)/pc))

# hurst指数特征，还有点问题
def hurst(data):
    # 2^16脉冲？？？1993_17_0=0.78
    ts = np.abs(data[:, 1])
    print(ts.size)
    ts = list(ts)
    N = len(ts)
    if N < 20:
        raise ValueError("Time series is too short! input series ought to have at least 20 samples!")

    max_k = int(np.floor(N / 2))
    R_S_dict = []
    for k in range(10, max_k + 1):
        R, S = 0, 0
        # split ts into subsets
        subset_list = [ts[i:i + k] for i in range(0, N, k)]
        if np.mod(N, k) > 0:
            subset_list.pop()
            # tail = subset_list.pop()
            # subset_list[-1].extend(tail)
        # calc mean of every subset
        mean_list = [np.mean(x) for x in subset_list]
        for i in range(len(subset_list)):
            cumsum_list = pd.Series(subset_list[i] - mean_list[i]).cumsum()
            R += max(cumsum_list) - min(cumsum_list)
            S += np.std(subset_list[i])
        R_S_dict.append({"R": R / len(subset_list), "S": S / len(subset_list), "n": k})

    log_R_S = []
    log_n = []
    print(R_S_dict)
    for i in range(len(R_S_dict)):
        R_S = (R_S_dict[i]["R"] + np.spacing(1)) / (R_S_dict[i]["S"] + np.spacing(1))
        log_R_S.append(np.log(R_S))
        log_n.append(np.log(R_S_dict[i]["n"]))

    Hurst_exponent = np.polyfit(log_n, log_R_S, 1)[0]
    print(Hurst_exponent)
    return Hurst_exponent

def hurst2(data):
    """Returns the Hurst Exponent of the time series vector ts"""

    # create the range of lag values
    ts = np.abs(data[:, 0])
    i = len(ts) // 2
    lags = range(2, i)
    # Calculate the array of the variances of the lagged differences
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)

    # Return the Hurst Exponent from the polyfit output
    print(poly[0] * 2.0)

# 将数据划分成长度为N的序列的数据集，如果需要划分是否有目标，需要单独将目标单元择出来操作
def seg_datasets(data):
    datasets = []
    N = 4096 #采样样本长度
    M = 0    #采样间隔
    for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,26,27]:
        for j in range(len(data)//(M+N)):
            print('i={}, j={}'.format(i, j))
            datasets.append(np.abs(data[M*(j-1)+1:M*(j-1)+N,i]))
            print(len(datasets[j]))
    datasets = np.array(datasets)
    print(datasets)
    df = pd.DataFrame(datasets)
    df.to_csv('scratch/1998_clutter_datasets/1998_202225_clutter.csv', index=False, header=False)
    # 不要头文件，不要列索引
 
 # 计算相位
def cal_phase(data):
    # 读取data某一行的数据
    data_phase= np.angle(data)
    # 归一化data
    data_phase = data_phase/np.max(data_phase)
    # plt.title('1993_17_6_phase')
    plt.figure(figsize=(10, 5))
    plt.imshow(data_phase[:,:], cmap='jet', aspect='auto')
    # y轴反过来
    plt.gca().invert_yaxis()
    plt.clim(0, 1)
    plt.colorbar()

    plt.show()

    return data_phase

def SPWVD(data):
    # 取data的一行数据，
    data = data[:1000,7]
    print(data.shape)
    wvd = PseudoWignerVilleDistribution(data)
    wvd.run()
    wvd.tfr = np.fft.fftshift(wvd.tfr, axes=0)
    wvd.tfr = (wvd.tfr - wvd.tfr.mean())/wvd.tfr.std()
    plt.figure(figsize=(8,8))

    plt.imshow(np.abs(wvd.tfr), origin='lower', aspect='auto',extent=[0,1000,-500,500], cmap='gray')
    # plt.title('SPWVD of Sea Clutter')
    # plt.colorbar()
    
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.xlabel('时间(毫秒)', fontdict= {'size': 16, 'color': 'black'})
    plt.ylabel('频率(赫兹)', fontdict= {'size': 16, 'color': 'black'})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    # plt.savefig('/Users/jjw/Desktop/mac/Documents/study/项目/专利/20250116修改/1993_17_7_SPWVD.png', dpi=300)
    plt.show()
    
    one_matrix = np.zeros_like(wvd.tfr)
    # 沿着纵轴找到最大值的索引
    one_max_indices = np.argsort(wvd.tfr, axis=0)[-5:]
    # 在最大值处设置值
    for i in range(one_matrix.shape[1]):
        # print("target:",matrix[one_max_indices[i], i])
        # one_matrix[one_max_indices[:,i], i] = wvd.tfr[one_max_indices[:,i], i]
        one_matrix[one_max_indices[:,i], i] = 1

    plt.figure(figsize=(8,8))

    plt.imshow(np.abs(one_matrix), origin='lower', aspect='auto',extent=[0,1000,-500,500], cmap='gray')
    # plt.title('A Binary Thresholded NTFD of Sea Clutter')
    # plt.colorbar()
    # x轴加标题
    # 显示为紧凑布局

    plt.xlabel('时间(毫秒)', fontdict= {'size': 16, 'color': 'black'})
    plt.ylabel('频率(赫兹)', fontdict= {'size': 16, 'color': 'black'})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    # plt.savefig('/Users/jjw/Desktop/mac/Documents/study/项目/专利/20250116修改/1993_17_7_SPWVD_bi.png', dpi=300)
    plt.show()




if __name__=='__main__':
    fileprefix_1993 = r'./data/'
    for file_index in [17,18,19,25,26,30,31,40,54,280,283,310,311,320]:
    # file_index='17'
        nc_file_1993= fileprefix_1993 + f'{file_index}.cdf'
        fh_1993 = Dataset(nc_file_1993, mode='r')

        # fileprefix_1998 = r'/Users/jjw/Desktop/mac/Documents/study/data/ipix/1998/'
        # nc_file_1998 = fileprefix_1998 + '19980205_171437_ANTSTEP.cdf'
        # fh_1998 = Dataset(nc_file_1998, mode='r')

        # 可以改变数据文件，极化方式和数据处理方式
        nc=fh_1993
        pol=('hh')
        mode='auto'
        # 1993年数据为131072,14，1998年数据为60000,28
        for k in nc.variables:
            print(k)
        [nc.variables[k].set_auto_mask(False) for k in nc.variables]
        I = np.zeros((131072,14), dtype=float)
        Q = np.zeros((131072,14), dtype=float)
        for rangebin in range(14):
            [I[:, rangebin], Q[:, rangebin], meanIQ, stdIQ, inbal]=ipixload(nc, pol, rangebin, mode); #提取海杂波的I路和Q路数据
            I[:, rangebin] = I[:, rangebin] * np.sqrt(stdIQ[0])
            Q[:, rangebin] = Q[:, rangebin] * np.sqrt(stdIQ[1])
            data = I + 1j * Q
        
        # from matplotlib.font_manager import FontManager
        # fm = FontManager()
        # mat_fonts = set(f.name for f in fm.ttflist)
        # print(mat_fonts)



        # 保存data到csv文件
        df = pd.DataFrame(data,dtype=complex)
        df.to_csv(f'./scratch/{file_index}.csv', index=False, header=False)

        # print(data)
        # print(type(data))
        # print(data.shape)
        # print("data:", data)
        # SPWVD(data)
        # ipixrange(data) # 一维距离像
        # cal_amplitute(data) # 幅度分布
        # cal_power_spectrum(data) # 功率谱
        # range_fft(data) # 距离-多普勒图
        # test_range_fft(data)
        # correlation(data) # 自相关函数
        # amplitude_spatial_temporal(data)  #距离像
        # SCR_siglech(data) # 平均信杂比
        # hurst2(data) #hurst指数
        # seg_datasets(data)
        # cal_phase(data) #相位
