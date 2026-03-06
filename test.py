import sys, os
import dataloader
from model import cattleface
from tools import utils
import torch.utils.data.distributed
import torch
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from torchstat import stat
import time
import matplotlib.ticker as mticker  # 导入刻度格式化工具
sys.path.append("..")


parser = argparse.ArgumentParser(description='tester parser')
parser.add_argument('--val_list', default='', type=str,
                    help='testing list')
parser.add_argument('--device', default= , type=int, metavar='N',
                    help='GPU')
parser.add_argument('-j', '--workers', default= , type=int, metavar='N',
                    help='载入器数量')
parser.add_argument('-b', '--batch_size', default=3, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--embedding-size', default=4096, type=int,
                    help='The embedding feature size')
parser.add_argument('--input-fc-size', default=512, type=int,
                    help='The embedding feature size')
parser.add_argument('--last-fc-size', default=, type=int,
                    help='The num of last fc layers for using softmax')
parser.add_argument('--resume', default=", type=str, metavar='PATH',
                    help='model path')

args = parser.parse_args()


def main_worker(args, num):
    print('building the model...')
    model = cattleface.builder(args)
    model = model.cuda(args.device)
    args.resume = "weights/FEA_All_"+ str(num) + ".pth"
    model.load_state_dict(torch.load(args.resume, weights_only=True))

    print('building the dataloader ...')
    test_loader, len = dataloader.val_loader(args)
    model.eval()
    test_bar = tqdm(test_loader, file=sys.stdout)
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    results = []

    with torch.no_grad():
        for i, (img1, ground_face, target, path) in enumerate(test_bar):
            img1 = img1.cuda(args.device)
            target = target.cuda(args.device)
            output,x_norm, y_norm ,cos, _, _= model(img1, ground_face, target, 0)
            acc1, acc5 = utils.accuracy(args, cos, target, topk=(1, 5))
            top1.update(acc1[0], img1.size(0))
            top5.update(acc5[0], img1.size(0))
            test_bar.desc = "test acc1:{:.3f} acc5:{:.3f}".format(top1.avg, top5.avg)
    return top1.avg



if __name__ == '__main__':
    sum = 0

    for num in range(1, 11):
        
        acc = main_worker(args, num)
        sum += acc

    Accuracy = sum/10
    print(Accuracy)

