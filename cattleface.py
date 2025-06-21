import sys
sys.path.append("..")
from model import iresnet, mobilefacenet
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
import math
import torch
import torch.nn as nn
from skimage import io, color
from skimage.metrics import structural_similarity as ssim

def builder(args):
    model = SoftmaxBuilder(args)
    return model



class SoftmaxBuilder(nn.Module):
    def __init__(self, args):
        super(SoftmaxBuilder, self).__init__()
        self.device = args.device
        # self.features = iresnet.iresnet(num_classes=args.embedding_size)
        self.features = mobilefacenet.MobileFaceNet()
        self.fc = ArcFace(args.input_fc_size, args.last_fc_size, 64, 0.1)
        self.generate = GenerateNet()
        self.sa = SpatialSENet(in_channels=512)
        self.reconstruction = Reconstruction(in_channels=512, out_height=100, out_width=100)



    def forward(self, x, y, target, epoch):

        X , mapx = self.features(x)
        
        if torch.is_tensor(y) == True :
            ssimx = 0

            # attentionx = self.ca(mapx)
            attentionx = self.sa(mapx)
            reconstructionx = self.reconstruction(attentionx)
            Y  , mapy = self.features(y)
            ssimx = calculate_ssim(y, reconstructionx)

            Y = self.generate(Y)
            y_norm = F.normalize(Y)
        else :
            y = 0
            y_norm = 0
            ssimx = 0
            
        # 为X加上少量噪声
        
        noise = np.random.normal(0, 0.005, size=X.shape)
        noise = torch.tensor(noise, dtype=X.dtype).to(X.device)
        X = X + noise

        X = self.generate(X)
        x_norm = F.normalize(X)

        logits, cosine, sec = self.fc(X, target, epoch)


        return logits, x_norm, y_norm, cosine, ssimx, mapx


class ArcFace(nn.Module):

    def __init__(self, in_features, out_features, s, m, easy_margin=True):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # nn.init.xavier_uniform_(self.weight)
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m



    def forward(self, input, label, epoch):
        cosine = torch.mm(F.normalize(input), F.normalize(self.weight))
        cosine = cosine.clamp(-1,1)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        n = epoch/100
        cos_n = math.cos(n)
        sin_n = math.sin(n)
        phj = cosine * cos_n + sine * sin_n
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
            phj = torch.where(cosine > 0, phj, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=cosine.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # output = output * self.s
        
        if epoch > 1:
            indices = torch.argsort(cosine, descending=True)
            second_largest_index = indices[:, 1]
            second_largest_one_hot = torch.zeros(cosine.size(), device=cosine.device)
            second_largest_one_hot.scatter_(1, second_largest_index.view(-1, 1).long(), 1)
            output = (one_hot * phi) + (second_largest_one_hot * phj) + ((1.0 - second_largest_one_hot - one_hot) * cosine)
        else:
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)    
        output = output * self.s


        cosine = cosine*self.s
        second_largest = 0

        return output, cosine, second_largest


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class GenerateNet(torch.nn.Module):
    def __init__(self, output_size=512):
        super(GenerateNet, self).__init__()
        self.output_size = output_size
        self.generate_features = nn.Sequential(
            #deer  
            # nn.Linear(512, 512),
            # nn.ReLU(True),
            # nn.Linear(512, 512),
            # nn.ReLU(True)

            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True)

            #cattle res
            # nn.Linear(4096, 4096),
            # nn.ReLU(True),
            # nn.Linear(4096, 2048),
            # nn.ReLU(True),
            # nn.Linear(2048, 2048),
            # nn.ReLU(True),
            # nn.Linear(2048, 1024),
            # nn.ReLU(True),
            # nn.Linear(1024, 512),
            # nn.ReLU(True)
        )

    def forward(self, input):
        x = self.generate_features(input)
        return x


class GenLoss(torch.nn.Module):

    def __init__(self):
        super(GenLoss, self).__init__()
       
    def All_Loss(self, x_norm, y_norm):
        # 欧氏距离
        diff = x_norm - y_norm
        squared_diff = diff ** 2
        sum_squared_diff = torch.sum(squared_diff)
        loss_ed = torch.sqrt(sum_squared_diff)   
        #余弦距离

        loss_cos = F.cosine_similarity(x_norm, y_norm)
        
        return loss_ed, loss_cos

    def forward(self, x_norm, y_norm):
        loss_ed, loss_cos = self.All_Loss(x_norm, y_norm)
        loss1 = 1 - loss_cos.mean() 
        loss2 = loss_ed.mean()
        loss = loss1 + loss2 
        return loss


class Discriminator(torch.nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        self.discriminate = nn.Sequential(
            nn.Linear(img_shape, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        

    def forward(self, input):
        
        validity = self.discriminate(input)
        return validity


class SEBlock(nn.Module):
    def __init__(self, channel, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction_ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ChannelSENet(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelSENet, self).__init__()
        self.se_block = SEBlock(in_channels, reduction_ratio=reduction_ratio)

    def forward(self, x):
        return self.se_block(x)

class SpatialSENet(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SpatialSENet, self).__init__()
        self.channel_attention = SEBlock(in_channels, reduction_ratio=reduction_ratio)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_out = self.channel_attention(x)
        spatial_out = self.spatial_attention(x)
        return channel_out * spatial_out




class Reconstruction(nn.Module):
    def __init__(self, in_channels, out_height=100, out_width=100):
        super(Reconstruction, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 3, 3, padding=1)
        self.out_height = out_height
        self.out_width = out_width

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        x = F.interpolate(x, size=(self.out_height, self.out_width), mode='bilinear', align_corners=False)
        return x


    


def MSE(image1, image2):

    image1 = image1.float()
    image2 = image2.float()

    ssim_value = F.mse_loss(image1, image2)

    return ssim_value.item()


def calculate_ssim(tensor1, tensor2):

    if tensor1.is_cuda:
        tensor1 = tensor1.cpu()
    if tensor2.is_cuda:
        tensor2 = tensor2.cpu()
    

    tensor1 = tensor1.detach().numpy()
    tensor2 = tensor2.detach().numpy()
    

    if tensor1.shape != tensor2.shape:
        raise ValueError("？")

    ssim_results = []

    for i in range(tensor1.shape[0]):
        img1 = tensor1[i]
        img2 = tensor2[i]

        if len(img1.shape) == 3 and img1.shape[2] in [3, 4]:
            img1 = color.rgb2gray(img1)
        if len(img2.shape) == 3 and img2.shape[2] in [3, 4]:
            img2 = color.rgb2gray(img2)

        ssim_value = ssim(img1, img2, channel_axis=0, data_range=1)
        ssim_results.append(ssim_value)

    return np.mean(ssim_results)