import os
import numpy as np
import time
import datetime
import torch
import torchvision
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from evaluation import *
from network import Recurr_Com_Att_U_Net,recurrent_model, entire_model
import csv
import pandas as pd
from dataset import load_data2
import random
from dataset import my_dataset, load_data2, my_test_dataset
from convLSTM_network import convLSTM_model

class Solver(object):
	def __init__(self, config, train_loader, valid_loader, test_loader):

		# Data loader
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader = test_loader

		# Models
		self.unet = None
		self.optimizer = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch
		self.criterion = torch.nn.BCELoss()
		self.criterion2 = SoftIoULoss(2)
		self.criterion3 = FocalLoss()
		self.criterion4 = torch.nn.MSELoss()
		self.augmentation_prob = config.augmentation_prob

		# Hyper-parameters
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2

		# Training settings
		self.num_epochs = config.num_epochs
		self.num_epochs_decay = config.num_epochs_decay
		self.batch_size = config.batch_size

		# Step size
		self.log_step = config.log_step
		self.val_step = config.val_step
		self.test_only = config.test_only
		# Path
		self.model_path = config.model_path
		self.result_path = config.result_path
		self.mode = config.mode

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model_type = config.model_type
		print('model:',self.model_type, 'batch_size:',self.batch_size)
		self.t = config.t
		self.build_model()

	def build_model(self):
		"""Build generator and discriminator."""
		if self.model_type =='RCA_U_Net':
			self.unet = entire_model(img_ch=self.img_ch, output_ch=self.output_ch)

		self.best_threshold = 0.5
		self.optimizer = optim.Adam(list(self.unet.parameters()),
									  self.lr, [self.beta1, self.beta2])
		self.unet.to(self.device)

		# self.print_network(self.unet, self.model_type)

	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))

	def to_data(self, x):
		"""Convert variable to tensor."""
		if torch.cuda.is_available():
			x = x.cpu()
		return x.data

	def update_lr(self, g_lr, d_lr):
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = lr

	def reset_grad(self):
		"""Zero the gradient buffers."""
		self.unet.zero_grad()

	def compute_accuracy(self,SR,GT):
		SR_flat = SR.view(-1)
		GT_flat = GT.view(-1)

		acc = GT_flat.data.cpu()==(SR_flat.data.cpu()>0.5)

	def tensor2img(self,x):
		img = (x[:,0,:,:]>x[:,1,:,:]).float()
		img = img*255
		return img


	def train(self):
		"""Train encoder, generator and discriminator."""

		#====================================== Training ===========================================#
		#===========================================================================================#

		unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' %(self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob))

		# U-Net Train
		if False : #os.path.isfile(unet_path):
			# Load the pretrained Encoder
			self.unet.load_state_dict(torch.load(unet_path))
			print('%s is Successfully Loaded from %s'%(self.model_type,unet_path))
		else:
			if os.path.isfile(unet_path):
				# Load the pretrained Encoder
				self.unet.load_state_dict(torch.load(unet_path))
				print('%s is Successfully Loaded from %s'%(self.model_type,unet_path))
				print(self.best_threshold)                
			# Train for Encoder
			lr = self.lr
			best_unet_score = 0.
			best_threshold = 0.
			best_unet_loss =100000.
			file_dict = pd.read_csv('/data/output/all_guance_data_name_list/all_gc_filename_list.csv',index_col=0)
			datelist = [str(line).split('_')[1] for line in  file_dict.values]
			file_dict.index = datelist
			historyhour = 24
			if self.test_only:
				self.num_epochs=0
			for epoch in range(self.num_epochs):
				self.unet.train(True)
				epoch_loss = 0
				tt_threshold = 0
				acc = 0.	# Accuracy
				SE = 0.		# Sensitivity (Recall)
				SP = 0.		# Specificity
				PC = 0. 	# Precision
				F1 = 0.		# F1 Score
				JS = 0.		# Jaccard Similarity
				DC = 0.		# Dice Coefficient
				epoch_loss=0             
				length = 0
				trainlist = self.train_loader
				trainlist = np.array(trainlist).reshape(-1)
				random.shuffle(trainlist)
				trainlist = trainlist.reshape(-1,self.batch_size)
				for i, batchlist in enumerate(trainlist):                  
					# GT : Ground Truth
					tt = time.time()
					images, GT, rain_true, histrain =load_data2(batchlist, file_dict, 24, binary=True)               
# 					images, GT, rain_true = load_processed_data(batchlist)
					print(time.time()-tt)
					histrain = torch.FloatTensor(histrain).to(self.device) 
					rain_true = torch.FloatTensor(rain_true).to(self.device)      
					images = torch.FloatTensor(images).to(self.device)
					GT = torch.FloatTensor(GT).to(self.device)
					# SR : Segmentation Result
					SR_probs, rain_pred, prerain24 = self.unet( images, histrain)
					# SR_probs, rain_pred = self.unet(images)
					SR_flat = SR_probs.view(SR_probs.size(0),-1)              
					GT_flat = GT.view(GT.size(0),-1)
					rpred_flat = rain_pred.view(rain_pred.size(0),-1)
					rtrue_flat = rain_true.view(rain_true.size(0),-1)
					rpred24_flat = rain_pred.view(prerain24.size(0),-1)

# 					loss = self.criterion2(SR_flat,GT_flat)+self.criterion3(SR_flat,GT_flat)+self.criterion4(rpred_flat,rtrue_flat)
					loss = self.criterion(SR_flat,GT_flat)+4*self.criterion2(SR_flat,GT_flat)+self.criterion3(SR_flat,GT_flat)+self.criterion4(rpred_flat,rtrue_flat)+4*self.criterion4(rpred24_flat,rtrue_flat)
					epoch_loss += loss.item()
					self.reset_grad()
					loss.backward()
					self.optimizer.step()
					SR_probs = SR_probs.view(-1,2,80,80)              
					GT = GT.view(-1,2,80,80)  
# 					tmp_threshold = get_best_threshold(SR_probs, GT)
					tmp = get_accuracy(SR_probs, GT)
					print('epoch: ',epoch,'batch number: ',i,'/',len(trainlist),'training loss:',loss.item(),'acc',tmp)   
					# Backprop + optimize
					epoch_loss += loss.item()
					acc += get_accuracy(SR_probs,GT)
					SE += get_sensitivity(SR_probs,GT)
					SP += get_specificity(SR_probs,GT)
					PC += get_precision(SR_probs,GT)
					F1 += get_F1(SR_probs[:,0,:,:],GT[:,0,:,:])
					JS += get_JS(SR_probs[:,0,:,:],GT[:,0,:,:])
					DC += get_DC(SR_probs[:,0,:,:],GT[:,0,:,:])
				length = len(trainlist)            
				acc = acc/length
				SE = SE/length
				SP = SP/length
				PC = PC/length
				F1 = F1/length
				JS = JS/length
				DC = DC/length
				epoch_loss = epoch_loss/length
				# Print the log info
				print('Epoch [%d/%d], Loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
					  epoch+1, self.num_epochs, \
					  epoch_loss,\
					  acc,SE,SP,PC,F1,JS,DC))



				# Decay learning rate
				if (epoch+1) > (self.num_epochs - self.num_epochs_decay):
					lr -= (self.lr / float(self.num_epochs_decay))
					for param_group in self.optimizer.param_groups:
						param_group['lr'] = lr
					print ('Decay learning rate to lr: {}.'.format(lr))
				
                
				if best_unet_loss >epoch_loss:
					best_unet_loss = epoch_loss
					best_epoch = epoch
					best_unet = self.unet.state_dict()
					print('Best %s model loss : %.4f'%(self.model_type,best_unet_loss))
					torch.save(best_unet,unet_path)
				# ===================================== Validation ====================================#
				self.unet.train(False)
				self.unet.eval()

				acc = 0.	# Accuracy
				SE = 0.		# Sensitivity (Recall)
				SP = 0.		# Specificity
				PC = 0. 	# Precision
				F1 = 0.		# F1 Score
				JS = 0.		# Jaccard Similarity
				DC = 0.		# Dice Coefficient
				length=0
# 				print(self.valid_loader)                
				for i, batchlist in enumerate(self.valid_loader):
					# GT : Ground Truth
					images, GT, rain_true, histrain =load_data2(batchlist, file_dict, 24, binary=True)
					histrain = torch.FloatTensor(histrain).to(self.device) 
					rain_true = torch.FloatTensor(rain_true).to(self.device)
					images = torch.FloatTensor(images).to(self.device)
					GT = torch.FloatTensor(GT).to(self.device)
					# SR : Segmentation Result
					SR_probs, rain_pred, prerain24= self.unet(images, histrain)
					SR_flat = SR_probs.view(SR_probs.size(0),-1)              
					GT_flat = GT.view(GT.size(0),-1)
					rpred_flat = rain_pred.view(rain_pred.size(0),-1)
					rtrue_flat = rain_true.view(rain_true.size(0),-1)
					rpred24_flat = rain_pred.view(prerain24.size(0),-1)
					loss = self.criterion2(SR_flat,GT_flat)+ self.criterion3(rpred_flat,rtrue_flat)+self.criterion3(rpred24_flat,rtrue_flat)
					epoch_loss += loss.item()
					SR_probs = SR_probs.view(-1,2,80,80)              
					GT = GT.view(-1,2,80,80)  
					acc += get_accuracy(SR_probs,GT,best_threshold)
					SE += get_sensitivity(SR_probs,GT)
					SP += get_specificity(SR_probs,GT)
					PC += get_precision(SR_probs,GT)
					F1 += get_F1(SR_probs[:,0,:,:],GT[:,0,:,:])
					JS += get_JS(SR_probs[:,0,:,:],GT[:,0,:,:])
					DC += get_DC(SR_probs[:,0,:,:],GT[:,0,:,:])
					tmp = get_accuracy(SR_probs,GT)
					print('epoch: ',epoch,'batch number: ',i,'validation loss:',loss.item(),'acc',tmp)             
				length = len(self.valid_loader)
				acc = acc/length
				SE = SE/length
				SP = SP/length
				PC = PC/length
				F1 = F1/length
				JS = JS/length
				DC = DC/length
				epoch_loss/length
				unet_score = JS + DC
				print('epoch_loss:', epoch_loss,' best_unet_loss', best_unet_loss,'unet_score:',unet_score)
				print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f'%(acc,SE,SP,PC,F1,JS,DC))
				
				'''
				torchvision.utils.save_image(images.data.cpu(),
											os.path.join(self.result_path,
														'%s_valid_%d_image.png'%(self.model_type,epoch+1)))
				torchvision.utils.save_image(SR.data.cpu(),
											os.path.join(self.result_path,
														'%s_valid_%d_SR.png'%(self.model_type,epoch+1)))
				torchvision.utils.save_image(GT.data.cpu(),
											os.path.join(self.result_path,
														'%s_valid_%d_GT.png'%(self.model_type,epoch+1)))
				'''


				# Save Best U-Net model
				# if unet_score > best_unet_score :
				# 	best_unet_score = unet_scored
				# if best_unet_loss >epoch_loss:
				# 	best_unet_loss = epdoch_loss
				# 	best_epoch = epoch
				# 	best_unet = self.unet.state_dict()
				# 	print('Best %s model loss : %.4f'%(self.model_type,best_unet_loss))
				# 	torch.save(best_unet,unet_path)
				# print('saveing picture')                    
				# rain_compare_gc_rt_pre(images[:,:,34,:,:]*10,rain_true[:,:,0,:,:],rain_pred[:,:,0,:,:], prerain24[:,:,0,:,:],vmax=5)
			#===================================== Test ====================================#
			if not self.test_only:            
				del self.unet
				del best_unet
			self.build_model()
			self.unet.load_state_dict(torch.load(unet_path))
			
			self.unet.train(False)
			self.unet.eval()

			acc = 0.	# Accuracy
			SE = 0.		# Sensitivity (Recall)
			SP = 0.		# Specificity
			PC = 0. 	# Precision
			F1 = 0.		# F1 Score
			JS = 0.		# Jaccard Similarity
			DC = 0.		# Dice Coefficient
			length=0
			batch_test, file_dict_test = my_test_dataset( self.batch_size, historyhour, season=False)
			for i, batchlist in enumerate(self.test_loader):
				print('batch',i)     
				# GT : Ground Truth
				images, GT, rain_true, histrain =load_data2(batchlist, file_dict_test, 24, binary=True)
				histrain = torch.FloatTensor(histrain).to(self.device) 
				rain_true = torch.FloatTensor(rain_true).to(self.device)
				images = torch.FloatTensor(images).to(self.device)
				GT = torch.FloatTensor(GT).to(self.device)
				# SR : Segmentation Result
				SR, rain_pred, prerain24= self.unet(images, histrain)
				np.save('./vis/prediction_{}.npy'.format(i), SR.cpu().detach().numpy())
				np.save('./vis/ground_truth_{}.npy'.format(i),GT.cpu().detach().numpy())
				np.save('./vis/ruitu_pre_{}.npy'.format(i),images[:,:,34,:,:].cpu().detach().numpy()) 
				np.save('./vis/prerain_{}.npy'.format(i),rain_pred.cpu().detach().numpy()) 
				np.save('./vis/prerain24_{}.npy'.format(i),prerain24.cpu().detach().numpy()) 
				np.save('./vis/ground_rain_{}.npy'.format(i),rain_true.cpu().detach().numpy()) 
				del SR, GT, images,rain_pred,prerain24,rain_true   


			
