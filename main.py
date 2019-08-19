import argparse
import os
from solver import Solver
from dataset import my_dataset, load_data2, my_test_dataset
from torch.backends import cudnn
import random

def main(config):
    cudnn.benchmark = True

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path,config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    
    lr = random.random()*0.0005 + 0.0000005
    augmentation_prob= random.random()*0.7
    # lr = 0.0005
    # augmentation_prob = 0
    epoch = random.choice([100,150,200,250])
    # epoch = 100
    decay_ratio = 0.8
    decay_epoch = int(epoch*decay_ratio)

    config.augmentation_prob = augmentation_prob
    config.num_epochs = epoch
    config.lr = lr
    config.num_epochs_decay = decay_epoch

    print(config)
    
    batch = config.batch_size
    historyhour = config.historyhour
    batch_filelist, file_dict = my_dataset( batch, historyhour,season='summer')

    batch_test, file_dict_test = my_test_dataset( batch, historyhour, season=False)
    split_num=0.9
    valid_num=1
    train_num = int(len(batch_filelist)*split_num)
    valid_num = int(len(batch_filelist)*valid_num)
    mydataset = {'train':batch_filelist[:train_num], 
                'valid':batch_filelist[train_num:valid_num],
                'test': batch_test}
    
    solver = Solver(config, mydataset['train'], mydataset['valid'], mydataset['test'])

    
    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')
    
    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=47)
    parser.add_argument('--output_ch', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=250)
    parser.add_argument('--num_epochs_decay', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
    parser.add_argument('--augmentation_prob', type=float, default=0.1)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)
    parser.add_argument('--historyhour', type=int, default=24)
    parser.add_argument('--test_only', type=bool, default=False)
    # misc
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_type', type=str, default='RCA_U_Net', help='U_Net/R2U_Net/AttU_Net/R2AttU_Net')
    parser.add_argument('--model_path', type=str, default='./models')
    # parser.add_argument('--train_path', type=str, default='./dataset/train/')
    # parser.add_argument('--valid_path', type=str, default='./dataset/valid/')
    # parser.add_argument('--test_path', type=str, default='./dataset/test/')
    parser.add_argument('--result_path', type=str, default='./result/')

    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()
    main(config)
