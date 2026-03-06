import sys
sys.path.append("..")
import dataloader
from model import cattleface
from tools import utils
import tqdm
import torch
import argparse
import warnings
import os
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from time import sleep


os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
TORCH_USE_CUDA_DSA = 1

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Trainer for CattleFace')
parser.add_argument('--img_list', default='', type=str,
                    help='train imgs list')
parser.add_argument('--val_list', default='', type=str,
                    help='val imgs list')
parser.add_argument('--save_path', default='weights/', type=str,
                    help='保存路径')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='训练总轮次')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='start epoch')
parser.add_argument('--device', default= , type=int, metavar='N',
                    help='GPU')
parser.add_argument('-b', '--batch-size', default=100, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--embedding-size', default=4096, type=int,
                    help='The embedding feature size')
parser.add_argument('--input-fc-size', default=512, type=int,
                    help='The embedding feature size')
parser.add_argument('--last-fc-size', default=50, type=int,
                    help='类别数量')

parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='初始学习率', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='动量的权值')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay',
                    dest='weight_decay')
parser.add_argument('--lr-drop-epoch', default=[30, 60], type=int, nargs='+',
                    help='The learning rate drop epoch')
parser.add_argument('--lr-drop-ratio', default=0.1, type=float,
                    help='The learning rate drop ratio')


args = parser.parse_args()

def main(args):

    # 调用训练函数
    for num in range(1, 11):
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()
        args.lr = 0.001
        main_worker(args, num)


def main_worker(args, num):
    #绘图参数
    Loss_list = []
    Acc_list = []

    print('building the model......')
    model = cattleface.builder(args)
    model = model.cuda(args.device)
    discriminator = cattleface.Discriminator(args.input_fc_size)
    discriminator = discriminator.cuda(args.device)
    print('building the model：√')

    print('building the optimizer......')
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    
    optimizer_D = torch.optim.Adam(discriminator.parameters(), args.lr, betas=(0.5, 0.999))

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                    milestones=[10,20,30,40], gamma=0.8)
    

    print('building the optimizer: √')

    print('building the dataloader......')
    train_loader, len_train = dataloader.img_loader(args)
    val_loader, len_val = dataloader.val_loader(args)
    print('building the dataloader: √')

    print('building the loss......')
    arcloss = cattleface.FocalLoss(gamma=2)
    genloss = cattleface.GenLoss()
    adversarial_loss = torch.nn.BCELoss()
    
    print('building the loss: √')
    print('start training：')
    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        


        global current_lr
        current_lr = utils.adjust_learning_rate(optimizer, epoch, args)

        model.train()
        discriminator.train()

        loss_id = utils.AverageMeter('Loss', ':.3f')
        loss_G = utils.AverageMeter('Loss', ':.3f')
        loss_D = utils.AverageMeter('Loss', ':.3f')
        loss_gen = utils.AverageMeter('Loss', ':.3f')
        top1 = utils.AverageMeter('Acc@1', ':6.2f')
        top5 = utils.AverageMeter('Acc@5', ':6.2f')
        
        learning_rate = utils.AverageMeter('LR', ':.4f')

        # 更新学习率
        learning_rate.update(current_lr)

        train_bar = tqdm(train_loader, file=sys.stdout)
        for i, (img, ground_face, target, img_path) in enumerate(train_bar):
            img = img.cuda(args.device)
            ground_face = ground_face.cuda(args.device)
            target = target.cuda(args.device)

            output, x_norm, y_norm, cos, ssim, second = model(img,  ground_face, target, epoch)

            
            real_loss = adversarial_loss(discriminator(y_norm), torch.ones(img.shape[0], 1).cuda(args.device))
            fake_loss = adversarial_loss(discriminator(x_norm.detach()), torch.zeros(img.shape[0], 1).cuda(args.device))
            d_loss = (real_loss + fake_loss) / 2

            optimizer_D.zero_grad()
            d_loss.backward(retain_graph=True)
            optimizer_D.step()

            loss_class= arcloss(output, target)
            loss_generate = genloss(x_norm, y_norm)
            g_loss = adversarial_loss(discriminator(x_norm), torch.ones(img.shape[0], 1).cuda(args.device))
            
          
            if loss_generate > 0 :
               loss = 2*loss_class + loss_generate + g_loss + (1-ssim)

            else  :
                loss = 2*loss_class + g_loss + (1-ssim)

            acc1, acc5 = utils.accuracy(args, cos, target, topk=(1, 5))

            loss_id.update(loss_class.item(), img.size(0))
            # loss_G.update(g_loss.item(),img.size(0))
            # loss_D.update(d_loss.item(),img.size(0)) 
            loss_gen.update(loss_generate,img.size(0)) 
            top1.update(acc1[0], img.size(0))
            top5.update(acc5[0], img.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_bar.desc = "train epoch[{}/{}] loss_id:{:.3f} loss_gen:{:.3f} acc1:{:.3f} acc5:{:.3f}".format(epoch + 1, args.epochs,
                                                                                             loss_id.avg, loss_gen.avg, 
                                                                                             top1.avg, top5.avg)
        Loss_list.append(f'{loss_id.avg:.3f}')
        Acc_list.append(f'{top1.avg:.3f}')


        model.eval()
        top1_val = utils.AverageMeter('Acc@1', ':6.2f')
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_img, val_ground, val_labels, path = val_data
                val_img = val_img.cuda(args.device)
                val_labels = val_labels.cuda(args.device)
                output, x_norm, y_norm ,cos_val,_ , _= model(val_img, val_ground, val_labels, epoch)
                acc_val = utils.accuracy(args, cos_val, val_labels, topk=(1, ))
                top1_val.update(acc_val[0], val_img.size(0))

        val_acc = top1_val.avg

        print("val acc:{}".format(val_acc))
        if val_acc >= best_acc:
            best_acc = val_acc
            print("saving the parameters of epoch {},val_acc={}.".format(epoch + 1, best_acc))
            model_save_path = "FEAFace"+ str(num) + ".pth"
            model_save_path = os.path.join(args.save_path, model_save_path)
            torch.save(model.state_dict(), model_save_path)


if __name__ == '__main__':
    main(args)
