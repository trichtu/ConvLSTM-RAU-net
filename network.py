import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from convLSTM_network import convLSTM_model

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

    
    
class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x



class recurr_conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(recurr_conv_block,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True))


    def forward(self,x,y):
        state = self.conv1(x)
        if not (type(y)==type(x)):
            x = self.conv2(state)
        else:
            x = self.conv2(state+y)
        state = self.conv3(x)    
        return x, state

    
    

class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class Recurr_Com_Att_U_Net(nn.Module):
    def __init__(self,img_ch=47,output_ch=2):
        super(Recurr_Com_Att_U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        print('img_ch',img_ch)
        self.Conv1 = conv_block(ch_in=img_ch,ch_out=16)
        self.Conv2 = recurr_conv_block(ch_in=16,ch_out=32)
        self.Conv3 = recurr_conv_block(ch_in=32,ch_out=64)
        self.Conv4 = recurr_conv_block(ch_in=64,ch_out=128)
        self.Conv5 = recurr_conv_block(ch_in=128,ch_out=256)

        self.Up5 = up_conv(ch_in=256,ch_out=128)
        self.Att5 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv5 = conv_block(ch_in=256, ch_out=128)

        self.Up4 = up_conv(ch_in=128,ch_out=64)
        self.Att4 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv4 = conv_block(ch_in=128, ch_out=64)
        
        self.Up3 = up_conv(ch_in=64,ch_out=32)
        self.Att3 = Attention_block(F_g=32,F_l=32,F_int=16)
        self.Up_conv3 = conv_block(ch_in=64, ch_out=32)
        
        self.Up2 = up_conv(ch_in=32,ch_out=16)
        self.Att2 = Attention_block(F_g=16,F_l=16,F_int=8)
        self.Up_conv2 = conv_block(ch_in=32, ch_out=16)

        self.Conv_1x1 = nn.Conv2d(20,output_ch,kernel_size=1,stride=1,padding=0)
        self.Conv_rain = nn.Conv2d(20,1,kernel_size=1,stride=1,padding=0)

    def forward(self,x, hist_rain, state2_pre=None,state3_pre=None,state4_pre=None,state5_pre=None):
        # encoding path
        x_x = torch.cat((x,hist_rain), dim=1)
#         print('xx',x_x.shape)
        x1 = self.Conv1(x_x)

        x2 = self.Maxpool(x1)

        x2,state2 = self.Conv2(x2,state2_pre)
        
        x3 = self.Maxpool(x2)

        x3,state3 = self.Conv3(x3,state3_pre)

        x4 = self.Maxpool(x3)
        x4,state4 = self.Conv4(x4,state4_pre)

        x5 = self.Maxpool(x4)
        x5,state5 = self.Conv5(x5,state5_pre)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)
        d2 = torch.cat((d2, hist_rain, x[:,32:35,:,:] ), dim=1)
        
        d1 = self.Conv_1x1(d2)
        out = F.softmax(d1, dim=1)
        rain = self.Conv_rain(d2)*out[:,0:1,:,:]+hist_rain

        return out, rain, state2, state3, state4, state5


class entire_model(nn.Module):
    def __init__(self,img_ch=47,output_ch=2):
        super(entire_model,self).__init__()
        self.convLSTM_layer = convLSTM_model(41,80,80)
        self.TAUnet = Recurr_Com_Att_U_Net(img_ch,output_ch)
    
    def forward(self,input, hist_rain):
        out = torch.zeros([input.size()[0],input.size()[1], 2, input.size()[3],input.size()[4]], device = input.device)
        rain = torch.zeros([input.size()[0],input.size()[1], 1, input.size()[3],input.size()[4]], device = input.device)
        self.prerain48 = self.convLSTM_layer(hist_rain)
        self.prerain24 = self.prerain48[:,24:,:,:,:]
        for i in range(input.size()[1]):
            if i==0:
                out[:,i,:,:,:],rain[:,i,:,:,:],state2,state3,state4,state5 = self.TAUnet(input[:,i,:,:,:], self.prerain24[:,i,:,:,:])
            else:
                out[:,i,:,:,:],rain[:,i,:,:,:],state2,state3,state4,state5 = self.TAUnet(input[:,i,:,:,:], self.prerain24[:,i,:,:,:], state2, state3, state4, state5)

        return out, rain, self.prerain24
    
    
# def recurrent_model(LSTMmodel,unetmodel, input, histrain):
#     '''
#     model: model to iterate
#     input shape : [Batch, Time, Filter, W, H ]
#     '''
# #     LSTM_model
#     out = torch.zeros([input.size()[0],input.size()[1], 2, input.size()[3],input.size()[4]], device = input.device)
#     rain = torch.zeros([input.size()[0],input.size()[1], 1, input.size()[3],input.size()[4]], device = input.device)
# #     histrain = torch.zeros([input.size()[0],input.size()[1], 1, input.size()[3],input.size()[4]], device = input.device)
#     prerain48 = LSTMmodel(histrain)
#     prerain24 = prerain48[:,24:,:,:,:]
#     for i in range(input.size()[1]):
#         if i==0:
#             out[:,i,:,:,:],rain[:,i,:,:,:],state2,state3,state4,state5 = model(input[:,i,:,:,:], prerain24[:,i,:,:,:])
#         else:
#             out[:,i,:,:,:],rain[:,i,:,:,:],state2,state3,state4,state5 = model(input[:,i,:,:,:], prerain24[:,i,:,:,:], state2, state3, state4, state5)

#     return out, rain, prerain24

