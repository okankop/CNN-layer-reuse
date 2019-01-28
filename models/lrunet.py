'''LruNet in PyTorch.

See the paper "Convolutional Neural Networks with Layer Reuse" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // self.groups
        x = torch.cat([x[:,num_channels//2:,:,:], x[:,:num_channels//2,:,:]], 1) # Exchanging halves
        # reshape
        x = x.view(batchsize, self.groups, 
            channels_per_group, height, width)
        #permute
        x = x.permute(0,2,1,3,4).contiguous()
        # flatten
        x = x.view(batchsize, num_channels, height, width)
        return x


class Block(nn.Module):

    def __init__(self, planes, expansion, n_iter):
        super(Block, self).__init__()
        self.n_iter        = n_iter
        self.shuffle       = ShuffleBlock(groups=2)
        self.relu          = nn.ReLU(inplace=True)
        self.dw            = nn.Conv2d(planes, planes*expansion, kernel_size=3, stride=1, padding=1, groups=planes, bias=False)
        self.pw            = nn.Conv2d(planes*expansion, planes, kernel_size=1, stride=1, padding=0, groups=8, bias=False)

        self.dw_bn_list = []
        self.pw_bn_list = []
        for i in range(self.n_iter):
            self.dw_bn_list.append(nn.BatchNorm2d(planes*expansion))
            self.pw_bn_list.append(nn.BatchNorm2d(planes))
        self.dw_bn_list = nn.ModuleList(self.dw_bn_list)
        self.pw_bn_list = nn.ModuleList(self.pw_bn_list)


    def forward(self, x):
        for i in range(self.n_iter):
            if  i == 0:
                res = x
                out = self.dw_bn_list[i](self.dw(x))
            else:
                res = out
                out = self.dw_bn_list[i](self.dw(out))
            out = self.pw_bn_list[i](self.pw(out))
            out += res
            out = self.relu(out)

            if (i != self.n_iter - 1):
                out = self.shuffle(out)

        return out


class LruNet(nn.Module):

    def __init__(self,
                 num_classes=10,
                 width_mult=1,
                 layer_reuse=8,
                 drop=0.5,
                 init_ch=3):
        super(LruNet, self).__init__()

        self.drop        = drop
        self.num_classes = num_classes
        self.n_lru       = layer_reuse
        self.width_mult  = width_mult
        self.last_size   = 2

        cfg = [64, 128, 256, 512, 256]
        cfg = [int(i * self.width_mult) for i in cfg]
        
        self.features = nn.Sequential(
            nn.Conv2d(init_ch, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            )
        self.stage1   = Block(cfg[0], 2, self.n_lru)
        self.stage2   = Block(cfg[1], 2, self.n_lru)
        self.stage3   = Block(cfg[2], 2, self.n_lru)
        self.stage4   = Block(cfg[3], 2, self.n_lru)

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(cfg[3], cfg[4], kernel_size=1, stride=1, padding=0, groups=8, bias=False),
            nn.BatchNorm2d(cfg[4]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.drop),
            nn.Conv2d(cfg[4], self.num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((self.last_size, self.last_size), stride=1)
        )

        self.mpool  = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        out = self.features(x)
        if (self.width_mult == 2):
            out = torch.cat([out, out], 1)

        out = self.stage1(out)
        out = self.mpool(out)
        out = torch.cat([out, out], 1)

        out = self.stage2(out)
        out = self.mpool(out)
        out = torch.cat([out, out], 1)      

        out = self.stage3(out)
        out = torch.cat([out, out], 1)
        
        out = self.stage4(out)
        out = self.mpool(out)

        out = self.classifier(out)

        return out.view(out.size(0), self.num_classes)


if __name__ == '__main__':
    net = LruNet(num_classes=10, width_mult=1, layer_reuse=8, drop=0.5, init_ch=3)
    x   = torch.randn(8, 3, 32, 32)
    out = net(x)
    print(out.shape)