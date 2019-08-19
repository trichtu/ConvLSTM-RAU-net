#!/usr/bin/python
# -*- coding: utf-8 -*-
# created by Ma Liang
# contact with liang.ma@nlpr.ia.ac.cn

import numpy as np 
import datetime
import os
import pandas as pd
import random
import time
import threading
import multiprocessing

def check_file(tt,datelist,hour):
    ''' 
    chech file at the time of 'tt' and its continuous 24 hours and histort hours
    pretime 25 times include time.now()
    history time include 'hour' times 
    return if file is ready at the time of 'tt'
    '''
    ruitufile = '/data/output/ruitu_data/{}/{}.npy'.format(tt.strftime('%Y%m'),tt.strftime('%Y%m%d%H'))
    sign = os.path.exists(ruitufile)
    if sign:
        pass
#         shape0 = np.load(ruitufile).shape[0]
#         sign = sign and shape0==25
#         if not shape0==25:
#             print(ruitufile)
#             os.remove(ruitufile)
    else:
        return False
    pretimelist = [ tt+datetime.timedelta(seconds=3600*i) for i in range(25)]
    pretimelist = pretimelist+ [ tt-datetime.timedelta(seconds=3600*i)  for i in range(hour)]
    for pretime in pretimelist:
        # gaughDir = '/data/output/guance_data/{}/{}.npy'.format(pretime)
        timestring = pretime.strftime("%Y%m%d%H%M")
        sign =  (timestring in datelist ) and sign
        if sign==False :
#             print(timestring,os.path.exists(ruitufile),timestring in datelist)
            break
    return sign


def file_dataset(hour ):
    '''write a data-ready file list'''
    print('creating the dataset with history ', hour, ' hours')
    file_dict = pd.read_csv('/data/output/all_guance_data_name_list/all_gc_filename_list.csv',index_col=0)
    datelist = [str(line).split('_')[1] for line in  file_dict.values]
    file_dict.index = datelist
    start_time, end_time = datetime.datetime(2016,10,1,0),datetime.datetime(2019,4,1,0)
    pretimelist=[]
    pretime= start_time
    while pretime<=end_time:
        if check_file(pretime,datelist,hour):
            pretimelist.append(pretime)
        pretime += datetime.timedelta(seconds=3600*3)
    pretimelist = np.array(pretimelist)
    np.save('/data/code/ml/pretimelist_{}.npy'.format(hour),pretimelist)
    print('finishing creating dataset with history ', hour, ' hours')
    return None

def my_test_dataset( batch, history_hour, season=None ):
    '''return list shape [number , batch]'''
    file_dict = pd.read_csv('/data/output/all_guance_data_name_list/2019_04_07_gc_filename_list.csv', index_col=0)
    datelist = [str(line).split('_')[1] for line in file_dict.values]
    file_dict.index = datelist
    target = '/data/code/ml/pretimelist_test_{}.npy'.format(history_hour)
    if not os.path.exists(target):
        file_test_dataset( history_hour )
    pretimelist = np.load(target, allow_pickle=True)
    
    if season=='summer':
        tmp = []
        for pretime in pretimelist:
            if pretime.month in [4,5,6,7,8,9]:
                tmp.append(pretime)
        pretimelist = np.array(tmp)
    print('dataset lenght',len(pretimelist))
    pretimelist = pretimelist[:len(pretimelist)//batch*batch]
    pretimelist = np.array(pretimelist).reshape(-1, batch)
    return pretimelist, file_dict

def file_test_dataset(hour ):
    '''write a data-ready file list'''
    print('creating the dataset with history ', hour, ' hours')
    file_dict = pd.read_csv('/data/output/all_guance_data_name_list/2019_04_07_gc_filename_list.csv',index_col=0)
    datelist = [str(line).split('_')[1] for line in  file_dict.values]
    file_dict.index = datelist
    start_time, end_time = datetime.datetime(2019,4,1,0),datetime.datetime(2019,7,31,21)
    pretimelist=[]
    pretime= start_time
    while pretime<=end_time:
        if check_file(pretime,datelist,hour):
            pretimelist.append(pretime)
        pretime += datetime.timedelta(seconds=3600*3)
    pretimelist = np.array(pretimelist)
    np.save('/data/code/ml/pretimelist_test_{}.npy'.format(hour),pretimelist)
    print('finishing creating dataset with history ', hour, ' hours')
    return None


def my_dataset( batch, history_hour, season=None ):
    '''return list shape [number , batch]'''
    file_dict = pd.read_csv('/data/output/all_guance_data_name_list/all_gc_filename_list.csv', index_col=0)
    datelist = [str(line).split('_')[1] for line in file_dict.values]
    file_dict.index = datelist
    target = '/data/code/ml/pretimelist_{}.npy'.format(history_hour)
    if not os.path.exists(target):
        file_dataset( history_hour )
    pretimelist = np.load(target, allow_pickle=True)
    
    if season=='summer':
        tmp = []
        for pretime in pretimelist:
            if pretime.month in [6,7,8,9]:
                tmp.append(pretime)
        pretimelist = np.array(tmp)
    print('dataset lenght',len(pretimelist))
    pretimelist = pretimelist[:len(pretimelist)//batch*batch]
    random.shuffle(pretimelist)
    pretimelist = np.array(pretimelist).reshape(-1, batch)
    return pretimelist, file_dict



def conbime_thread(batch_list, batch_time):
    '''
    parallization the thread to read the data
    '''
#     print("Sub-process(es) begin.")
    ruitulist, gaugelist, histgaugelist, jobresults = [], [], [], []
    pool = multiprocessing.Pool(processes=12) # 创建4个进程
    for filelist, pretime in zip(batch_list, batch_time):
        jobresults.append(pool.apply_async(read_one, (filelist, pretime)))
    for res in jobresults:
        ruituFile, gaugeFile, histgaugeFile = res.get()
        ruitulist.append(ruituFile)
        gaugelist.append(gaugeFile)
        histgaugelist.append(histgaugeFile)
    pool.close() # 关闭进程池，表示不能在往进程池中添加进程
    pool.join() # 等待进程池中的所有进程执行完毕，必须在close()之后调用
#     print("Sub-process(es) done.")
    gaugelist, ruitulist, histgaugelist = np.array(gaugelist), np.array(ruitulist), np.array(histgaugelist)
#     print(gaugelist.shape, ruitulist.shape, histgaugelist.shape)
    return ruitulist, gaugelist, histgaugelist


def read_one(filelist, pretime):
    '''read single data in training data with preprocessing  '''
#     tt = time.time()
    ruituFile = np.load(filelist[0])[:,:,:80,:84]
#     print('processing',pretime)
    gaugeFile = np.array([np.load(file) for file in filelist[1:25]])[:,4:5,:80,:84]
    histgaugeFile = np.array([np.load(file) for file in filelist[25:]])[:,:,:80,:84]
    ruituFile, gaugeFile, histgaugeFile = norm_preprocess(ruituFile, gaugeFile, histgaugeFile, pretime)
#     print(time.time()-tt)
    return ruituFile, gaugeFile, histgaugeFile


def norm_preprocess(ruituFile, gaugeFile, histgaugeFile, pretime):
    '''
    processing with abnormal values, time , geography values, normalized values.
    '''
#     print(ruituFile.shape, gaugeFile.shape, histgaugeFile.shape)
    #remoev the abnormal value
    assert ruituFile.shape[0]==25,print(pretime,'without full prediction')
    if (np.abs(ruituFile) > 10000).any():
        meantmp = ruituFile.mean(axis=(0,2,3))
        for i in range(ruituFile.shape[1]):
            ruituFile[:,i,:,:][np.abs(ruituFile[:,i,:,:])>10000] = meantmp[i]
            
    histgaugeFile[np.isnan(histgaugeFile)]=200000
    if (np.abs(histgaugeFile) > 10000).any():
        meantmp = histgaugeFile.mean(axis=(0,2,3))
        for i in range(histgaugeFile.shape[1]):
            histgaugeFile[:,i,:,:][np.abs(histgaugeFile[:,i,:,:])>10000] = meantmp[i]        
    #normal the value
    ruituInfo = pd.read_csv('/data/output/ruitu_info.csv')
    ruitu_mean, ruitu_std = np.ones_like(ruituFile),np.ones_like(ruituFile)
    for i in range(len(ruituInfo)):
        ruitu_mean[:,i,:,:] *= ruituInfo['mean'].iloc[i]
        ruitu_std[:,i,:,:] *= ruituInfo['std'].iloc[i]
    ruituFile = (ruituFile-ruitu_mean)/ruitu_std

    gaugeInfo = pd.read_csv('/data/output/gauge_info.csv')
    gauge_mean, gauge_std = np.ones_like(histgaugeFile),np.ones_like(histgaugeFile)
    for i in range(len(gaugeInfo)):
        gauge_mean[:,i,:,:] *= gaugeInfo['mean'].iloc[i]
        gauge_std[:,i,:,:] *= gaugeInfo['std'].iloc[i]
    histgaugeFile = (histgaugeFile-gauge_mean)/gauge_std
    
    #add time and geo info
    geoinfo = np.load('/data/output/height_norm.npy')
    hist_hour = histgaugeFile.shape[0]
    pretimelist = [pretime+datetime.timedelta(seconds=i*3600) for i in range(-hist_hour+1, 25)]
    yearvariancelist = [ np.sin(2*np.pi*(tt.toordinal()-730180)/365.25) for tt in pretimelist]
    dayvariancelist = [ np.sin(2*np.pi*(tt.hour-3)/24) for tt in pretimelist]
    ruituFile[1:25, 32:35, :, :] = ruituFile[1:25, 32:35, :, :] - ruituFile[0:24,32:35,:,:]
    ruituFile_new = ruituFile[1:].copy()
    histgaugeFile[:,7,:,:] = np.array([geoinfo]*histgaugeFile.shape[0])
    histgaugeFile[:,10,:,:] = np.array([sli*yvar for sli, yvar in zip(np.ones([hist_hour,80,84]),yearvariancelist[:hist_hour])])
    histgaugeFile[:,11,:,:] = np.array([sli*dvar for sli, dvar in zip(np.ones([hist_hour,80,84]),dayvariancelist[:hist_hour])])
    tmpyear = np.expand_dims([sli*yvar for sli, yvar in zip(np.ones([24,80,84]),yearvariancelist[hist_hour:])], axis=1)
    tmpday = np.expand_dims([sli*dvar for sli, dvar in zip(np.ones([24,80,84]),dayvariancelist[hist_hour:])], axis=1)
    tmpgeo = np.expand_dims(np.array([geoinfo]*ruituFile_new.shape[0]),axis=1)
    ruituFile_new = np.concatenate((ruituFile_new, tmpyear, tmpday, tmpgeo),axis=1)
#     print(ruituFile_new.shape, gaugeFile.shape, histgaugeFile.shape)
    return ruituFile_new, gaugeFile, histgaugeFile


def load_data2(pretimelist, file_dict, history_hour, binary=False):
    '''
    load batch data in parallized way, more faster.
    input args: load_data2(pretimelist, file_dict, history_hour, binary=False)
    return args: ruitudata, gaugedata, histgaugedata 
        shape: [batch ,24, channels_1, height, width],[batch ,24 , 1, height, width],[batch , historyhour, channels_2, height, width]
    if binary is True, the gaugedata will return in shape [batch ,time, 2, height, width]
    '''
    pretimelist = list(pretimelist)
    batchfile = []
    for batch_time in pretimelist:
        ruituFile = ['/data/output/ruitu_data/{}/{}.npy'.format(batch_time.strftime('%Y%m'),batch_time.strftime('%Y%m%d%H'))]
        time24h = [ batch_time+datetime.timedelta(seconds=3600*i) for i in range(1,25)]
        gaugeFile = ['/data/output/guance_data/{}/{}'.format(tt.strftime('%Y%m'),file_dict.loc[tt.strftime('%Y%m%d%H%M')].values[0]) for tt in time24h]
        timehist = [ batch_time-datetime.timedelta(seconds=3600*i) for i in range(history_hour)]
        histgaugeFile = ['/data/output/guance_data/{}/{}'.format(tt.strftime('%Y%m'),file_dict.loc[tt.strftime('%Y%m%d%H%M')].values[0]) for tt in timehist]
        singlefile = ruituFile+gaugeFile+histgaugeFile
        batchfile.append(singlefile)

    ruitudata, gaugedata, histgaugedata = conbime_thread(batchfile, pretimelist)    

    if binary:
#         gaugedata = (gaugedata>=0.1).astype('int')
        gaugebinary = np.concatenate((gaugedata>=0.1, gaugedata<0.1),axis=2).astype('int')
    
    gaugedata[ gaugedata<0.1]=0
    histgaugedata = np.concatenate((histgaugedata, np.zeros_like(histgaugedata)), axis=1)
    return np.array(ruitudata)[:,:,:,:80,:80], np.array(gaugebinary)[:,:,:,:80,:80], np.array(gaugedata[:,:,:,:80,:80]), np.array(histgaugedata[:,:,:,:80,:80])



# def load_data(pretimelist,file_dict):
#     '''pretimelist is a batch timelist at once
#        output shape = [batch, 24, channel, 80, 84],[batch, 24, channel, 80, 84]
#     '''
#     print('old')
#     t1 = time.time()
#     pretimelist = list(pretimelist)
#     gaugedata = []
#     ruitudata = []
#     for batch_time in pretimelist:
#         ruitutmp = np.load('/data/output/ruitu_data/{}/{}.npy'.format(batch_time.strftime('%Y%m'),batch_time.strftime('%Y%m%d%H')))[:24,:,:80,:84]
#         time24h = [ batch_time+datetime.timedelta(seconds=3600) for i in range(24)]
#         guagetmp = np.array([np.load('/data/output/guance_data/{}/{}'.format(tt.strftime('%Y%m'),file_dict.loc[tt.strftime('%Y%m%d%H%M')].values[0])) for tt in time24h])[:,4:5,:80,:84]
#         gaugedata.append(guagetmp)
#         ruitudata.append(ruitutmp)
#     print('total:',time.time()-t1)
#     return np.array(gaugedata), np.array(ruitudata)


if __name__=='__main__':
    batch = 8
    historyhour = 24
    batch_filelist, file_dict = my_dataset( batch, historyhour,season='summer')
    
    split_num=0.7
    train_num = int(len(batch_filelist)*split_num)
    mydataset = {'train':batch_filelist[:train_num], 'test': batch_filelist[train_num:]}
    
    for filelist in mydataset['train']:
        tt = time.time()
        ruitudata, gaugedata, histgaugedata = load_data2(filelist,file_dict,historyhour, binary=True)
        print(gaugedata.shape, ruitudata.shape, histgaugedata.shape, 'finished time cost:',time.time()-tt)
#         print(gaugedata.mean(axis=(0,1,3,4)),gaugedata.std(axis=(0,1,3,4)))
#         print(ruitudata.mean(axis=(0,1,3,4)),ruitudata.std(axis=(0,1,3,4)))
#         print(histgaugedata.mean(axis=(0,1,3,4)),histgaugedata.std(axis=(0,1,3,4)))