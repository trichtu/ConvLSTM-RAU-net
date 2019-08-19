import numpy as np
import matplotlib.pyplot as plt
import os
import time
import torch
from torch import nn,device
from torch.autograd import Variable
import pandas as pd

import datetime
import pandas as pd
import random
import threading
import multiprocessing
import matplotlib.pyplot as plt



class ConvLSTMCell(nn.Module):
    
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        
        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        
        h_cur, c_cur = cur_state
        
        # print(input_tensor.shape)
        # print(cur_state.shape)
        
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
   
        # i = torch.sigmoid(cc_i)
        # f = torch.sigmoid(cc_f)
        # o = torch.sigmoid(cc_o)
        # g = torch.relu(cc_g)

        # c_next = f * c_cur + i * g
        # h_next = o * torch.relu(c_next)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).to(device),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).to(device))


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        
        Parameters
        ----------
        input_tensor: todo 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful (num_layers, 2, batch, filter, h, w)
            
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        print('hidden_state_shape before:',type( hidden_state))
        # Implement stateful ConvLSTM
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))
        print('hidden_state_shape after:',len(hidden_state), hidden_state[0].shape)
        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1) # 读取sequence length
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx] # (b,c,h,w)
            output_inner = []
            for t in range(seq_len):

                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list   = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class encoder_model(torch.nn.Module):
    
    def __init__(self, histgc_feature, width,height):
        
        super(encoder_model,self).__init__()
        self.histgc_feature=histgc_feature
        self.width=width
        self.height=height
        filter_1=16
        filter_2=32
        filter_3=64
        filter_4=128
        filter_5=128
        self.conv1=torch.nn.Conv2d(in_channels=histgc_feature,out_channels=filter_1,
                                   padding=1,kernel_size=(3,3))
        self.BN_1=torch.nn.BatchNorm2d(num_features=filter_1)
        self.convlstm1=ConvLSTM(input_size=[width,height],input_dim=filter_1,
                                hidden_dim=filter_1, num_layers=1,batch_first=True,
                                return_all_layers=True,kernel_size=(3,3))
        
        self.maxpool2=torch.nn.MaxPool2d(2)  
        self.conv2_1=torch.nn.Conv2d(in_channels=filter_1,out_channels=filter_2,
                                   padding=1,kernel_size=(3,3))
        self.BN_2_1=torch.nn.BatchNorm2d(num_features=filter_2)
        self.convlstm2=ConvLSTM(input_size=[width//2,height//2],input_dim=filter_2,
                                hidden_dim=filter_2, num_layers=1,batch_first=True,
                                return_all_layers=True,kernel_size=(3,3))
        self.conv2_2=torch.nn.Conv2d(in_channels=filter_2,out_channels=filter_2,
                                   padding=1,kernel_size=(3,3))
        self.BN_2_2=torch.nn.BatchNorm2d(num_features=filter_2)
        
        
        self.maxpool3=torch.nn.MaxPool2d(2)  
        self.conv3_1=torch.nn.Conv2d(in_channels=filter_2,out_channels=filter_3,
                                   padding=1,kernel_size=(3,3))
        self.BN_3_1=torch.nn.BatchNorm2d(num_features=filter_3)
        self.convlstm3=ConvLSTM(input_size=[width//4,height//4],input_dim=filter_3,
                                hidden_dim=filter_3, num_layers=1,batch_first=True,
                                return_all_layers=True,kernel_size=(3,3))
        self.conv3_2=torch.nn.Conv2d(in_channels=filter_3,out_channels=filter_3,
                                   padding=1,kernel_size=(3,3))
        self.BN_3_2=torch.nn.BatchNorm2d(num_features=filter_3)
        
        
        self.maxpool4=torch.nn.MaxPool2d(2)  
        self.conv4_1=torch.nn.Conv2d(in_channels=filter_3,out_channels=filter_4,
                                   padding=1,kernel_size=(3,3))
        self.BN_4_1=torch.nn.BatchNorm2d(num_features=filter_4)
        self.convlstm4=ConvLSTM(input_size=[width//8,height//8],input_dim=filter_4,
                                hidden_dim=filter_4, num_layers=1,batch_first=True,
                                return_all_layers=True,kernel_size=(3,3))
        self.conv4_2=torch.nn.Conv2d(in_channels=filter_4,out_channels=filter_4,
                                     padding=1,kernel_size=(3,3))
        self.BN_4_2=torch.nn.BatchNorm2d(num_features=filter_4)
                                     
                                   
        self.maxpool5=torch.nn.MaxPool2d(2)  
        self.conv5_1=torch.nn.Conv2d(in_channels=filter_4,out_channels=filter_5,
                                   padding=1,kernel_size=(3,3))
        self.BN_5_1=torch.nn.BatchNorm2d(num_features=filter_5)
        self.convlstm5=ConvLSTM(input_size=[width//16,height//16],input_dim=filter_5,
                                hidden_dim=filter_5, num_layers=1,batch_first=True,
                                return_all_layers=True,kernel_size=(3,3))
        
    
    def forward(self, histgc_inputs):
        
        B,T,C,W,H=histgc_inputs.size()
        
        x1=self.conv1(torch.reshape(histgc_inputs, (-1, C, W, H)))
        x1=self.BN_1(x1)
        x1=torch.relu(x1)
        x1 = torch.reshape(x1, (B, T, x1.size(1), x1.size(2), x1.size(3)))
        
        x1, state1 = self.convlstm1(x1,None)
        x1=x1[0]
        
        #第1次下采样
        B,T,C,W,H=x1.size()
        x2=self.maxpool2(torch.reshape(x1,(-1,C,W,H)))
        x2=self.conv2_1(x2)
        x2=self.BN_2_1(x2)
        x2=torch.relu(x2)
        x2 = torch.reshape(x2, (B, T, x2.size(1), x2.size(2), x2.size(3)))
        
        x2, state2=self.convlstm2(x2,None)
        x2=x2[0]
        
        B,T,C,W,H=x2.size()
        x2=self.conv2_2(torch.reshape(x2,(-1,C,W,H)))
        x2=self.BN_2_2(x2)
        x2=torch.relu(x2)
        x2 = torch.reshape(x2, (B, T, x2.size(1), x2.size(2), x2.size(3)))
        
        #第2次下采样
        B,T,C,W,H=x2.size()
        x3=self.maxpool3(torch.reshape(x2,(-1,C,W,H)))
        x3=self.conv3_1(x3)
        x3=self.BN_3_1(x3)
        x3=torch.relu(x3)
        x3 = torch.reshape(x3, (B, T, x3.size(1), x3.size(2), x3.size(3)))
        
        x3, state3=self.convlstm3(x3,None)
        x3=x3[0]
        
        B,T,C,W,H=x3.size()
        x3=self.conv3_2(torch.reshape(x3,(-1,C,W,H)))
        x3=self.BN_3_2(x3)
        x3=torch.relu(x3)
        x3 = torch.reshape(x3, (B, T, x3.size(1), x3.size(2), x3.size(3)))
        
        #第3次下采样
        B,T,C,W,H=x3.size()
        x4=self.maxpool4(torch.reshape(x3,(-1,C,W,H)))
        x4=self.conv4_1(x4)
        x4=self.BN_4_1(x4)
        x4=torch.relu(x4)
        x4 = torch.reshape(x4, (B, T, x4.size(1), x4.size(2), x4.size(3)))
        
        x4, state4=self.convlstm4(x4,None)
        x4=x4[0]
        
        B,T,C,W,H=x4.size()
        x4=self.conv4_2(torch.reshape(x4,(-1,C,W,H)))
        x4=self.BN_4_2(x4)
        x4=torch.relu(x4)
        x4 = torch.reshape(x4, (B, T, x4.size(1), x4.size(2), x4.size(3)))
        
        #第4次下采样
        B,T,C,W,H=x4.size()
        x5=self.maxpool5(torch.reshape(x4,(-1,C,W,H)))
        x5=self.conv5_1(x5)
        x5=self.BN_5_1(x5)
        x5=torch.relu(x5)
        x5 = torch.reshape(x5, (B, T, x5.size(1), x5.size(2), x5.size(3)))
        
        x5, state5=self.convlstm5(x5,None)
        x5=x5[0]
        
        state=state1+state2+state3+state4+state5
        
        return state

class forecaster_model(torch.nn.Module):

    
    def __init__(self, fut_guance_feature ,width,height):
        
        super(forecaster_model,self).__init__()
        
        self.fut_guance_feature=fut_guance_feature
        self.width=width
        self.height=height
        filter_1=16
        filter_2=32
        filter_3=64
        filter_4=128
        filter_5=128
        self.convd_1=torch.nn.Conv2d(in_channels=fut_guance_feature, out_channels=filter_1,
                                     padding=1,kernel_size=(3,3))
        self.BN_d_1=torch.nn.BatchNorm2d(num_features=filter_1)
        
        self.maxpoold_1=torch.nn.MaxPool2d(2) 
        self.convd_2=torch.nn.Conv2d(in_channels=filter_1, out_channels=filter_2,
                                     padding=1,kernel_size=(3,3))
        self.BN_d_2=torch.nn.BatchNorm2d(num_features=filter_2)
        
        self.maxpoold_2=torch.nn.MaxPool2d(2) 
        self.convd_3=torch.nn.Conv2d(in_channels=filter_2, out_channels=filter_3,
                                     padding=1,kernel_size=(3,3))
        self.BN_d_3=torch.nn.BatchNorm2d(num_features=filter_3)
        
        self.maxpoold_3=torch.nn.MaxPool2d(2) 
        self.convd_4=torch.nn.Conv2d(in_channels=filter_3, out_channels=filter_4,
                                     padding=1,kernel_size=(3,3))
        self.BN_d_4=torch.nn.BatchNorm2d(num_features=filter_4)
        
        self.maxpoold_4=torch.nn.MaxPool2d(2) 
        self.convd_5=torch.nn.Conv2d(in_channels=filter_4, out_channels=filter_5,
                             padding=1,kernel_size=(3,3))
        self.BN_d_5=torch.nn.BatchNorm2d(num_features=filter_5)
        
        
        self.convlstm_u_5=ConvLSTM(input_size=[width//16,height//16],input_dim=filter_5,
                                  hidden_dim=filter_5, num_layers=1,batch_first=True,
                                  return_all_layers=True, kernel_size=(3,3))
        
        self.convu_5_1=torch.nn.Conv2d(in_channels=filter_5,out_channels=filter_4,
                                       padding=1, kernel_size=(3,3))
        self.BN_u_5_1=torch.nn.BatchNorm2d(num_features=filter_4)
        
        self.upsample_4=torch.nn.Upsample(scale_factor=2)
        self.convu_4_1=torch.nn.Conv2d(in_channels=filter_4,out_channels=filter_4,
                                       padding=1, kernel_size=(3,3))
        self.BN_u_4_1=torch.nn.BatchNorm2d(num_features=filter_4)
        self.convlstm_u_4=ConvLSTM(input_size=[width//8,height//8],input_dim=filter_4,
                                  hidden_dim=filter_4, num_layers=1,batch_first=True,
                                  return_all_layers=True, kernel_size=(3,3))
        self.convu_4_2=torch.nn.Conv2d(in_channels=filter_4,out_channels=filter_3,
                                       padding=1, kernel_size=(3,3))
        self.BN_u_4_2=torch.nn.BatchNorm2d(num_features=filter_3)
        
        self.upsample_3=torch.nn.Upsample(scale_factor=2)
        self.convu_3_1=torch.nn.Conv2d(in_channels=filter_3,out_channels=filter_3,
                                       padding=1, kernel_size=(3,3))
        self.BN_u_3_1=torch.nn.BatchNorm2d(num_features=filter_3)
        self.convlstm_u_3=ConvLSTM(input_size=[width//4,height//4],input_dim=filter_3,
                                  hidden_dim=filter_3, num_layers=1,batch_first=True,
                                  return_all_layers=True, kernel_size=(3,3))
        self.convu_3_2=torch.nn.Conv2d(in_channels=filter_3,out_channels=filter_2,
                                       padding=1, kernel_size=(3,3))
        self.BN_u_3_2=torch.nn.BatchNorm2d(num_features=filter_2)
        
        
        self.upsample_2=torch.nn.Upsample(scale_factor=2)
        self.convu_2_1=torch.nn.Conv2d(in_channels=filter_2,out_channels=filter_2,
                                       padding=1, kernel_size=(3,3))
        self.BN_u_2_1=torch.nn.BatchNorm2d(num_features=filter_2)
        self.convlstm_u_2=ConvLSTM(input_size=[width//2,height//2],input_dim=filter_2,
                                  hidden_dim=filter_2, num_layers=1,batch_first=True,
                                  return_all_layers=True, kernel_size=(3,3))
        self.convu_2_2=torch.nn.Conv2d(in_channels=filter_2,out_channels=filter_1,
                                       padding=1, kernel_size=(3,3))
        self.BN_u_2_2=torch.nn.BatchNorm2d(num_features=filter_1)
        
        self.upsample_1=torch.nn.Upsample(scale_factor=2)
        self.convu_1_1=torch.nn.Conv2d(in_channels=filter_1,out_channels=filter_1,
                                       padding=1, kernel_size=(3,3))
        self.BN_u_1_1=torch.nn.BatchNorm2d(num_features=filter_1)
        self.convlstm_u_1=ConvLSTM(input_size=[width,height],input_dim=filter_1,
                          hidden_dim=filter_1, num_layers=1,batch_first=True,
                          return_all_layers=True, kernel_size=(3,3))
        
        self.convu_1_2=torch.nn.Conv2d(in_channels=filter_1,out_channels=1,
                                       padding=1, kernel_size=(3,3))
        
          
    def forward(self, guance_data,encoder_states):
        
        state_1=encoder_states[0]
        state_2=encoder_states[1]
        state_3=encoder_states[2]
        state_4=encoder_states[3]
        state_5=encoder_states[4]
        
        #变为filter1
        B,T,C,W,H=guance_data.size()
        print('guance data', guance_data.shape)
        x_gc_1=self.convd_1(torch.reshape(guance_data,(-1,C,W,H)))
        x_gc_1=self.BN_d_1(x_gc_1)
        x_gc_1=torch.relu(x_gc_1)
        
        #第1次下采样
        x_gc_2=self.maxpoold_1(x_gc_1)
        #变为filter2
        x_gc_2=self.convd_2(x_gc_2)
        x_gc_2=self.BN_d_2(x_gc_2)
        x_gc_2=torch.relu(x_gc_2)
        
        #第2次下采样
        x_gc_3=self.maxpoold_2(x_gc_2)
        #变为filter3
        x_gc_3=self.convd_3(x_gc_3)
        x_gc_3=self.BN_d_3(x_gc_3)
        x_gc_3=torch.relu(x_gc_3)
        
        #第3次下采样
        x_gc_4=self.maxpoold_3(x_gc_3)
        #变为filter4
        x_gc_4=self.convd_4(x_gc_4)
        x_gc_4=self.BN_d_4(x_gc_4)
        x_gc_4=torch.relu(x_gc_4)
        
        #第4次下采样
        x_gc_5=self.maxpoold_4(x_gc_4)
        #变为filter5
        x_gc_5=self.convd_5(x_gc_5)
        x_gc_5=self.BN_d_5(x_gc_5)
        x_gc_5=torch.relu(x_gc_5)
        
        x_gc_1 = torch.reshape(x_gc_1, (B,T,-1, W, H))          #filter1
        x_gc_2 = torch.reshape(x_gc_2, (B,T,-1, W//2, H//2))    #filter2
        x_gc_3 = torch.reshape(x_gc_3, (B,T,-1, W//4, H//4))    #filter3
        x_gc_4 = torch.reshape(x_gc_4, (B,T,-1, W//8, H//8))    #filter4
        x_gc_5 = torch.reshape(x_gc_5, (B,T,-1, W//16, H//16))  #filter5
        
        #第1次convlstm
        # print(x_gc_5.size())
        # print(state_5[0].size())
        
        x5, state_new_5 = self.convlstm_u_5(x_gc_5,[state_5])
        x5=x5[0]
        x5=x5+x_gc_5  
        
        #变为filter4
        B,T,C,W,H=x5.size()
        x5 = self.convu_5_1(torch.reshape(x5, (-1, C,W,H)))
        x5=self.BN_u_5_1(x5)
        x5=torch.relu(x5)
        
        #第1次上采样
        x4=self.upsample_4(x5)
        x4=self.convu_4_1(x4)
        x4=self.BN_u_4_1(x4)
        x4=torch.relu(x4)
        x4=torch.reshape(x4,(B,T, x4.size(1),x4.size(2),x4.size(3)))
        
        x4, state_new_4 = self.convlstm_u_4(x4,[state_4])
        x4=x4[0]
        
        x4=x4+x_gc_4
        
        #变为filter3
        B,T,C,W,H=x4.size()
        x4 = self.convu_4_2(torch.reshape(x4, (-1, C,W,H)))
        x4=self.BN_u_4_2(x4)
        x4=torch.relu(x4)
        
        #第2次上采样
        x3=self.upsample_3(x4)
        x3=self.convu_3_1(x3)
        x3=self.BN_u_3_1(x3)
        x3=torch.relu(x3)
        x3=torch.reshape(x3,(B,T, x3.size(1),x3.size(2),x3.size(3)))
        
        x3, state_new_3 =self.convlstm_u_3(x3,[state_3])
        x3=x3[0]
        
        x3=x3+x_gc_3
        
        #变为filter2
        B,T,C,W,H=x3.size()
        x3 = self.convu_3_2(torch.reshape(x3, (-1, C,W,H)))
        x3=self.BN_u_3_2(x3)
        x3=torch.relu(x3)
        
        #第3次上采样
        x2=self.upsample_2(x3)
        x2=self.convu_2_1(x2)
        x2=self.BN_u_2_1(x2)
        x2=torch.relu(x2)
        x2=torch.reshape(x2,(B,T, x2.size(1),x2.size(2),x2.size(3)))
        
        
        print(x2.size())
        print(state_2[0].size())
        x2, state_new_2 =self.convlstm_u_2(x2,[state_2])
        x2=x2[0]
        
        x2=x2+x_gc_2
        
        #变为filter1
        B,T,C,W,H=x2.size()
        x2 = self.convu_2_2(torch.reshape(x2, (-1, C,W,H)))
        x2=self.BN_u_2_2(x2)
        x2=torch.relu(x2)
        
        #第4次上采样
        x1=self.upsample_1(x2)
        x1=self.convu_1_1(x1)
        x1=self.BN_u_1_1(x1)
        x1=torch.relu(x1)
        x1=torch.reshape(x1,(B,T, x1.size(1),x1.size(2),x1.size(3)))
        
        x1, state_new_1 =self.convlstm_u_1(x1,[state_1])
        x1=x1[0]
        
        x1=x1+x_gc_1
        
        #变为通道1
        B,T,C,W,H=x1.size()
        x1 = self.convu_1_2(torch.reshape(x1, (-1, C,W,H)))
        x1=torch.reshape(x1,(B,T, x1.size(1),x1.size(2),x1.size(3)))
        
        new_state=state_new_1+state_new_2+state_new_3+state_new_4+state_new_5
        
        return [x1,new_state] 


class combined_net(torch.nn.Module):
    
    def __init__(self,histgc_feature,fut_guance_feature,width,height):
        
        super(combined_net,self).__init__()
        
        self.histgc_feature=histgc_feature
        self.fut_guance_feature=fut_guance_feature
        self.width=width
        self.height=height
        self.encoder=encoder_model(histgc_feature,width,height)
        self.forecaster=forecaster_model(fut_guance_feature,width,height)
        
        
#     def forward(self,hist_gc_inputs,fut_gc_inputs):
            
#         encoder_states=self.encoder(hist_gc_inputs)
#         inputdata = fut_gc_inputs[:,0:1,:,:]
#         y = torch.zeros(fut_gc_inputs.size(), device = fut_gc_inputs.device)

#         state = []
#         for i in range(fut_gc_inputs.size(1)):
#             print('time',i)
#             x = self.forecaster(inputdata, encoder_states)
            
# #             if i>3:
# #                 inputdata=x[0]
# #             else:
#             inputdata=fut_gc_inputs[:,i:i+1,:,:]
#             encoder_states=x[1]
#             y[:,i:i+1,:,:]=x[0]
#             state.append(x[1])

#         return [y,state] 
    def forward(self,hist_gc_inputs,fut_gc_inputs):
        
        encoder_states=self.encoder(hist_gc_inputs)

        x = self.forecaster(fut_gc_inputs,encoder_states)

        y=x[0]
        state=x[1]

        return [y,state] 