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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
#         print('hidden_state_shape before:',type( hidden_state))
        # Implement stateful ConvLSTM
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))
#         print('hidden_state_shape after:',len(hidden_state), hidden_state[0].shape)
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




class convLSTM_model(torch.nn.Module):
    
    def __init__(self, histgc_feature, width, height):
        
        super(convLSTM_model,self).__init__()
        self.histgc_feature=histgc_feature
        self.width=width
        self.height=height
        filter_1 = 16
        filter_2 = 32
        filter_3 = 64
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


        self.upsample_2=torch.nn.Upsample(scale_factor=2)
        self.convu_2_1=torch.nn.Conv2d(in_channels=filter_3,out_channels=filter_2,
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
        

    
    def forward(self, histgc_inputs):
        '''
        histgc_input should [batch, time, c, w, h]
        '''
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

        #第1次上采样
        x2_u=self.upsample_2(x3)
        x2_u=self.convu_2_1(x2_u)
        x2_u=self.BN_u_2_1(x2_u)
        x2_u=torch.relu(x2_u)
        x2_u=torch.reshape(x2_u,(B,T, x2_u.size(1),x2_u.size(2),x2_u.size(3)))
        

        x2_u, state_new_2 =self.convlstm_u_2(x2_u,None)
        x2_u=x2_u[0]
        
        
        #变为filter1
        B,T,C,W,H=x2_u.size()
        x2_u = self.convu_2_2(torch.reshape(x2_u, (-1, C,W,H)))
        x2_u=self.BN_u_2_2(x2_u)
        x2_u=torch.relu(x2_u)
        
        #第2次上采样
        x1_u=self.upsample_1(x2_u)
        x1_u=self.convu_1_1(x1_u)
        x1_u=self.BN_u_1_1(x1_u)
        x1_u=torch.relu(x1_u)
        x1_u=torch.reshape(x1_u,(B,T, x1_u.size(1),x1_u.size(2),x1_u.size(3)))
        
        x1_u, state_new_1 =self.convlstm_u_1(x1_u,None)
        x1_u=x1_u[0]
        
        
        #变为通道1
        B,T,C,W,H=x1_u.size()
        x1_u = self.convu_1_2(torch.reshape(x1_u, (-1, C,W,H)))
        x1_u=torch.reshape(x1_u,(B,T, x1_u.size(1),x1_u.size(2),x1_u.size(3)))
                
        return x1_u