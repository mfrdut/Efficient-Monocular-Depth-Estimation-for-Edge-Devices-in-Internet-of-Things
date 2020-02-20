import argparse
import os
import shutil
import time
import sys
import csv

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import dataloader.data_utils as data_utils
from dataloader.data_utils import get_dataloader

from metrics import AverageMeter, Result

data_sets = data_utils.data_sets 

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Depth Estimation')

parser.add_argument('--data', metavar='DATA', default='nyu_data',
                    help='dataset directory: (nyudepthv2/kitti/make3d)')
parser.add_argument('-s', '--num-samples', default=0, type=int, metavar='N',
                    help='number of sparse depth samples (default: 0)')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 10)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    help='mini-batch size (default: 8)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default 0.01)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    default=True, help='use ImageNet pre-trained weights (default: True)')
parser.add_argument('--optimizer', default='sgd', type=str, required=True, help='optimizer option')
parser.add_argument('--activation', default='relu', type=str, required=True, help='activation option')
parser.add_argument('--dataset', default='nyudepth',choices=data_sets, type=str, required=True, help='datasets option')	


model_names = ['mobilenet_v2']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
fieldnames = ['mse', 'rmse', 'absrel', 'lg10', 'mae', 'delta1', 'delta2', 'delta3', 'data_time', 'gpu_time']
best_result = Result()
best_result.set_to_worst()

def evaluate(val_loader, model, epoch, write_to_file=True):
    print(model)
    average_meter = AverageMeter()
    model.eval()    
    for i, (rgb_raw, input, target, mask, h5name) in enumerate(val_loader):
        input, target = input.to(device), target.to(device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        depth_pred = model(input_var)
        result = Result()
        output1 = torch.index_select(depth_pred.data, 1, torch.cuda.LongTensor([0]))
        result.evaluate(output1, target)
        average_meter.update(result, input.size(0))
        rgb = input

    avg = average_meter.average()

    print('\n*\n'
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'Delta1={average.delta1:.3f}\n'
        'Delta2={average.delta2:.3f}\n'
        'Delta3={average.delta3:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'.format(
        average=avg))

    if write_to_file:
        with open(test_csv, 'a') as csvfile: 
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3})

def main():
    global args, best_result, output_directory, train_csv, test_csv
    args = parser.parse_args()
    if not args.data.startswith('/'):
        args.data = os.path.join('../', args.data)
    
    output_directory = os.path.join('results',
        'Dataset={}.nsample={}.lr={}.bs={}.optimizer={}'.
        format(args.dataset, args.num_samples, args.lr, args.batch_size, args.optimizer))
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    test_csv = os.path.join(output_directory, 'test.csv')

    # Data loading
    print("=> creating data loaders ...")
	
    val_on, val_loader, h, w = get_dataloader(args.dataset, args.data, args.batch_size, args.num_samples, args.workers)
    val_len = len(val_loader)
    out_size = h, w
    print(out_size)
    print("test dataloader len={}".format(val_len))
	
    print("=> data loaders created.")

    # evaluation
    if args.evaluate:
        best_model_filename = os.path.join(output_directory, 'mobilenetv2blconv7dw_0.597.pth.tar')
        if os.path.isfile(best_model_filename):
            print("=> loading best model '{}'".format(best_model_filename))
            checkpoint = torch.load(best_model_filename)# map_location={'cuda:0': 'cpu'}
            args.start_epoch = checkpoint['epoch']
            best_result = checkpoint['best_result']
            model = checkpoint['model']
            print("=> loaded model (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no best model found at '{}'".format(best_model_filename))
        evaluate(val_loader, model, checkpoint['epoch'], write_to_file=True)
        return
    else:
        print("Please specify the evaluation mode: --evaluate ")
        
    model.to(device)
    print(model)

if __name__ == '__main__':
    main()
