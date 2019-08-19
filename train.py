import numpy as np
import matplotlib.pyplot as plt
import os
import time
import torch
from torch import nn,device
from torch.autograd import Variable
import pandas as pd
from dataset import *
from model import *
import datetime
import pandas as pd
import random
import threading
import multiprocessing
import matplotlib.pyplot as plt



def rain_compare_gc_rt_pre(ruitudata,guancedata,pre,vmax=5):
    '''
    输入：
    ruitudata: 睿图数据，只包含降水信息
    guancedata: 观测数据，同只包含降水信息
    pre:       模型预测数据，只包含降水信息
    
    输出： 预报时刻的对应的睿图、观测和模型预测降水图
    '''
    import matplotlib as mpl
    vmax=vmax
    
    ruitudata=ruitudata*10
    ruitudata[ruitudata>vmax]=vmax
    
    guancedata=guancedata*10
    guancedata[guancedata>vmax]=vmax
    
    pre=pre*10
    pre[pre>vmax]=vmax
    
    #确定预测时长，对比每个时次的降水信息
    time_length=ruitudata.shape[1]  
    for i in np.arange(time_length): 
        
        fig=plt.figure(i,figsize=(20,5))
        
        plt.subplot(1,3,1)
        norm=mpl.colors.Normalize(vmin=0,vmax=5)
        plt.imshow(ruitudata[0,i,:,:],norm=norm) #显示热力图，范围正则化
#         plt.colorbar(ruitu_f)
        plt.title('ruitu:'+str(i))
        #plt.tight_layout() #貌似会造成一个colorbar消失
   
        plt.subplot(1,3,2)
        plt.imshow(guancedata[0,i,:,:],norm=norm)
#         plt.colorbar(guance_f)
        plt.title('guance:'+str(i))
        #plt.tight_layout()
        
        plt.subplot(1,3,3)
        plt.imshow(pre[0,i,:,:],norm=norm)
#         plt.colorbar(pre_f)
        plt.title('prediction:'+str(i))
        
        plt.savefig('/data/code/ml/encoder_decoder/vis_test/'+str(i)+'_'+str(np.round(np.random.random(),4))+'.jpg')   #保存所有时次图
        
#         plt.show()


def train_test_by_batch(model,epochs,Batch_size,historyhour=24,season='summer',fut_time_steps=6 ,k=10,binary=False,rain_threshold=10):
    
    '''
    输入：
    model:待训练模型
    epochs：训练的次数
    Batch_size: 每个Batch中的数据样本数
    historyhour=24:表示选取历史24小时数据
    season：对夏季数据(6`9月)进行训练
    k：表示每隔k次batch训练进行1次测试
    binary：表示是否进行二分类，
            如果为False，load_data2中的输出的guancedata为[batch,timesteps,1,widh,high],降水没有进行归一化
            如果为True, 表示对降水进行了one-hot编码，guancedata维度为[batch,timesteps,2,widh,high]
    rain_threshold:如果binary为False，则需要进行回归，回归时
    '''
    #调用模型，编译模型,当模型为2分类时，使用交叉熵，当为回归时，使用MSE
    # model=build_forecaster_model(binary=binary)
    
    #创建列表来保存所有train，和test 的loss,acc
    total_loss_train=[]
    total_acc_train=[]
    
    total_loss_test=[]
    total_acc_test=[]
    
    #定义优化方法和损失函数
    loss_func=torch.nn.MSELoss() #MSE损失
    # loss_func=torch.nn.L1Loss()  #MAE损失
    # loss_func=torch.nn.SmoothL1Loss() #计算平滑L1损失，属于 Huber Loss中的一种(因为参数δ固定为1了)
    
    # ssim_loss=SSIM()
    opt=torch.optim.Adam(model.parameters(),lr=0.001)
    # scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5) 
    
    # mult_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[EPOCH//2, EPOCH//4*3], gamma=0.1)
                    
    k=k
    #每训练k个batch进行一次测试
    
    #将数据集划分为训练和测试集
    batch_filelist,file_dict,train_dataset,test_dataset=load_all_batch_filelist(Batch_size,historyhour,season=season,split_num=0.8)
    
    #对train_dataset进行遍历训练epoch次
    for i in range(epochs):
        
        #定义模型训练
        model.train() 
        
        #每个epoch记录一次loss
        loss_train=[]

        #每次epoch之前，对训练数据集进行shuffle
        train_dataset= np.array(train_dataset).reshape(-1)
        random.shuffle(train_dataset)
        train_dataset = train_dataset.reshape(-1,Batch_size)
        
        #确定在训练数据集上有多少个batch
        train_batch_num=len(train_dataset) 
        
        #将每个epoch创建一个记录损失和准确率的文件
        train_log_file='/data/code/fzl/encoder_forecaster_1_terminal/inference_model/train_log_file/log_epoch_'+str(i)+'.txt'
        
        
        with open(train_log_file,'w') as f:
            
            # for j in range(train_batch_num):
            for j in range(train_batch_num):   

                #获取一次batch数据,并进行训练
                ruitudata, guancedata, histguancedata = load_data2(train_dataset[j], file_dict,history_hour=historyhour, binary=binary) 
                
                #获取合适宽度范围
                histguancedata=histguancedata[:,:,:,0:80,0:80]
                ruitudata=ruitudata[:,:,:,0:80,0:80]           
                guancedata=guancedata[:,:,:,0:80,0:80]
                
                guancedata[guancedata<=0.1]=0  #卡阈值
                
                if fut_time_steps<24:
                    guancedata=guancedata[:,0:fut_time_steps,:,:,:]
                    ruitudata=ruitudata[:,0:fut_time_steps,:,:,:]
                   
                    if fut_time_steps==1:
                        guancedata=np.expand_dims(guancedata,axis=1)
                        ruitudata=np.expand_dims(ruitudata,axis=1)
                
                
                #只取睿图降水信息
                if ruitu_features==1:
                    ruitudata=ruitudata[:,:,34,:,:]
                    ruitudata=np.expand_dims(ruitudata,axis=2)
                
                #取出历史观测的最后一个时刻的降水信息，并将其乘以10
                hist_gc_0=histguancedata[:,-1,4,:,:]*0.6713073720808679+0.052805550785578505
                hist_gc_0=np.expand_dims(hist_gc_0,axis=1)
                hist_gc_0=np.expand_dims(hist_gc_0,axis=1)
            
                
                #在训练的时候，forecaster的输入为 滞后一个小时的观测和睿图信息
                forecaster_input_gc= np.concatenate((hist_gc_0,guancedata[:,0:-1,:,:,:]+ ruitudata[:,0:-1,:,:,:]),axis=1)
                print('hist_gc_0:',hist_gc_0.shape)
                print(ruitudata.shape)
                print(guancedata.shape)
                print(forecaster_input_gc.shape)
                # #只取历史观测降水信息
                # if guance_features==1:
                #     histguancedata=histguancedata[:,:,4,:,:]
                #     histguancedata=np.expand_dims(histguancedata,axis=2)
                
               
                # ruitudata=torch.from_numpy(ruitudata).type(torch.FloatTensor).cuda().to(device)
                guancedata=torch.from_numpy(guancedata).type(torch.FloatTensor).cuda().to(device)
                histguancedata=torch.from_numpy(histguancedata).type(torch.FloatTensor).cuda().to(device)
                forecaster_input_gc=torch.from_numpy(forecaster_input_gc).type(torch.FloatTensor).cuda().to(device)

                #如果binary为False，需要对guance降水进行归一化
                if binary==False:
                    guancedata=guancedata/rain_threshold
                    forecaster_input_gc=forecaster_input_gc/rain_threshold
                
                # scheduler.step()
                
                pred=model(histguancedata,forecaster_input_gc)
                pred=pred[0] #不获取状态
                
                B, S, C, H, W = guancedata.size()
                
                
                #每隔20次batch画一次图，看看效果
                if  j%10==0:
                    rain_compare_gc_rt_pre(ruitudata[:,:,0,:,:],guancedata[:,:,0,:,:].cpu().detach().numpy(),pred[:,:,0,:,:].cpu().detach().numpy())
                
                pred=pred.view(-1,C,H,W)
                guancedata=guancedata.view(-1,C,H,W)
                
                # print(type(pred))
                loss=loss_func(pred,guancedata)
                
                # loss=loss_func(pred,guancedata)
                
                #将每个batch的训练信息保存下来，h[0]为loss,h[1]为acc
                loss_train.append(loss.item())
                
                #输出每次bathc训练的损失和准确率,输出到文件中 
                print('Epoch {}/{} : train_batch {}/{}: -----train_loss:{:.4f}  \n'.format(i,epochs-1,j,train_batch_num-1,loss.item()))
                f.write('Epoch {}/{} : train_batch {}/{}: -----train_loss:{:.4f} \n'.format(i,epochs-1,j,train_batch_num-1,loss.item()))
                f.write('\n')
                      
                # loss.backward()
                
                opt.zero_grad()
                loss.backward()
                opt.step()


            #每个epoch画出训练和测试的损失信息和准确率信息,将其保存到对应文件中去
            plt.figure(figsize=(10,6))
            plt.plot(np.arange(len(loss_train)),loss_train,'-r',label='loss-train')
            plt.title('epoch_'+str(i))
            plt.legend()
            plt.savefig('/data/code/fzl/encoder_forecaster_1_terminal/inference_model/'+'epoch_'+str(i)+'.png')
            plt.show()
             
            total_loss_train.append(loss_train)
            f.close()
        
            #每个epoch保存一次模型
            # torch.save(model,'pytorch_encoder_forecaster_model_epoch_'+str(i)+'.pkl')
            torch.save({'state_dict': model.state_dict()}, '/data/code/fzl/encoder_forecaster_1_terminal/inference_model/epoch_'+str(i)+'_checkpoint.pth.tar')
            # torch.save()
        
    return model,total_loss_train



def inference(k=5,rain_threshold=10):
    Batch_size=8

    epochs = 10

    fut_time_steps=6

    filter_1=16
    filter_2=32
    filter_3=64
    filter_4=128
    filter_5=128

    #ruitu_features=46
    width=80
    height=80
    histgc_feature=41
    fut_guance_feature=1
    ruitu_features=1
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model=combined_net(histgc_feature,fut_guance_feature,width,height).to(device)
    # model = encoder_forecaster_net()epoch_5_checkpoint.pth.tar
    checkpoint = torch.load('/data/code/fzl/encoder_forecaster_1_terminal/inference_model/epoch_5_checkpoint.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    historyhour=24
    binary=False
    batch_filelist,file_dict,train_dataset,test_dataset=load_all_batch_filelist(Batch_size,historyhour,season='summer',split_num=0.8)
    #确定在训练数据集上有多少个batch
    train_batch_num=len(train_dataset) 
    for j in range(train_batch_num):   

        #获取一次batch数据,并进行训练
        ruitudata, guancedata, histguancedata = load_data2(train_dataset[j], file_dict,history_hour=historyhour, binary=binary) 
        
        #获取合适宽度范围
        histguancedata=histguancedata[:,:,:,0:80,0:80]
        ruitudata=ruitudata[:,:,:,0:80,0:80]           
        guancedata=guancedata[:,:,:,0:80,0:80]
        
        guancedata[guancedata<=0.1]=0  #卡阈值
        
        if fut_time_steps<24:
            guancedata=guancedata[:,0:fut_time_steps,:,:,:]
            ruitudata=ruitudata[:,0:fut_time_steps,:,:,:]
            
            if fut_time_steps==1:
                guancedata=np.expand_dims(guancedata,axis=1)
                ruitudata=np.expand_dims(ruitudata,axis=1)
        
        
        #只取睿图降水信息
        if ruitu_features==1:
            ruitudata=ruitudata[:,:,34,:,:]
            ruitudata=np.expand_dims(ruitudata,axis=2)
        
        #取出历史观测的最后一个时刻的降水信息，并将其乘以10
        hist_gc_0=histguancedata[:,-1,4,:,:]*10 
        hist_gc_0=np.expand_dims(hist_gc_0,axis=1)
        hist_gc_0=np.expand_dims(hist_gc_0,axis=1)

        
        #在训练的时候，forecaster的输入为 滞后一个小时的观测和睿图信息
        forecaster_input_gc= np.concatenate((hist_gc_0,guancedata[:,0:-1,:,:,:]+ ruitudata[:,0:-1,:,:,:]),axis=1)
        print('hist_gc_0:',hist_gc_0.shape)
        print(ruitudata.shape)
        print(guancedata.shape)
        print(forecaster_input_gc.shape)
        # #只取历史观测降水信息
        # if guance_features==1:
        #     histguancedata=histguancedata[:,:,4,:,:]
        #     histguancedata=np.expand_dims(histguancedata,axis=2)
        
        
        # ruitudata=torch.from_numpy(ruitudata).type(torch.FloatTensor).cuda().to(device)
        guancedata=torch.from_numpy(guancedata).type(torch.FloatTensor).cuda().to(device)
        histguancedata=torch.from_numpy(histguancedata).type(torch.FloatTensor).cuda().to(device)
        forecaster_input_gc=torch.from_numpy(forecaster_input_gc).type(torch.FloatTensor).cuda().to(device)

        #如果binary为False，需要对guance降水进行归一化
        if binary==False:
            guancedata=guancedata/rain_threshold
            forecaster_input_gc=forecaster_input_gc/rain_threshold
        
        # scheduler.step()
        
        pred=model(histguancedata,forecaster_input_gc)
        pred=pred[0] #不获取状态
        
        B, S, C, H, W = guancedata.size()
        
        
        #每隔20次batch画一次图，看看效果
        if  j%5==0:
            print('peeking in picture')
            rain_compare_gc_rt_pre(ruitudata[:,:,0,:,:],guancedata[:,:,0,:,:].cpu().detach().numpy(),pred[:,:,0,:,:].cpu().detach().numpy())
        




if __name__=='__main__':
    inference()
#     Batch_size=8

#     epochs = 10

#     fut_time_steps=1

#     filter_1=16
#     filter_2=32
#     filter_3=64
#     filter_4=128
#     filter_5=128

#     #ruitu_features=46
#     width=80
#     height=80
#     histgc_feature=41
#     fut_guance_feature=1
#     ruitu_features=1

#     device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
#     model=combined_net(histgc_feature,fut_guance_feature,width,height).to(device)
#     T_model,total_loss_train=train_test_by_batch(model,Batch_size=Batch_size,epochs=epochs,k=5,rain_threshold=10)
    
