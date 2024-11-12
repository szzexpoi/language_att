import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from resnet_multi import resnet50
from vision_transformer import Trans_encoder

class TranSalNet(nn.Module):
    """
    Reimplementation of TranSalNet (the originally one is hard to converge)
    """
    def __init__(self, ):
        super(TranSalNet, self).__init__()
        self.backbone = resnet50(pretrained=True)
        self.trans_1 = Trans_encoder(2048, project_dim=768,num_layers=2,
                                num_heads=12,num_patches=80,dropout=0) # 80
        self.trans_2 = Trans_encoder(1024, project_dim=768,num_layers=2,
                                num_heads=12,num_patches=300,dropout=0) # 300
        self.trans_3 = Trans_encoder(512, project_dim=512,num_layers=2,
                                num_heads=8,num_patches=1200,dropout=0) # 1200

        self.sal_decoder_1 = nn.Sequential(nn.Conv2d(768, 768, kernel_size=3, stride=1, padding='same'),
                                            nn.BatchNorm2d(768),
                                            nn.ReLU(),
                                            nn.Upsample(scale_factor=(2, 2)))

        self.sal_decoder_2 = nn.Sequential(nn.Conv2d(768, 512, kernel_size=3, stride=1, padding='same'),
                                            nn.BatchNorm2d(512),
                                            nn.ReLU(),
                                            nn.Upsample(scale_factor=(2, 2)))

        self.sal_decoder_3 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same'),
                                            nn.BatchNorm2d(512),
                                            nn.ReLU(),
                                            nn.Upsample(scale_factor=(2, 2)))

        self.sal_decoder_4 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, stride=1, padding='same'),
                                            nn.BatchNorm2d(128),
                                            nn.ReLU(),
                                            nn.Upsample(scale_factor=(2, 2)),
                                            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
                                            nn.BatchNorm2d(128),
                                            nn.ReLU()
                                            )

        self.sal_cls = nn.Conv2d(128, 1, kernel_size=3,
                                padding='same', stride=1, bias=False)


    def forward(self, image, probe=False):
        # multi-scale features
        _, _, x_8, x_16, x_32 = self.backbone(image)

        # saliency decoders with Transformers
        b, c, h, w = x_32.shape
        x_32 = x_32.view(b, c, h*w).transpose(1,2)
        x = self.trans_1(x_32)
        x = x.transpose(1, 2).view(b, -1, h, w)
        x = self.sal_decoder_1(x)
        x = F.interpolate(x, (15, 20))

        b, c, h, w = x_16.shape
        x_16 = x_16.view(b, c, h*w).transpose(1,2)
        x_ = self.trans_2(x_16)
        x_ = x_.transpose(1, 2).view(b, -1, h, w)
        x = x*x_
        x = torch.relu(x)
        x = self.sal_decoder_2(x)

        b, c, h, w = x_8.shape
        x_8 = x_8.view(b, c, h*w).transpose(1,2)
        x_ = self.trans_3(x_8)
        x_ = x_.transpose(1, 2).view(b, -1, h, w)
        x = x*x_
        x = torch.relu(x)
        x = self.sal_decoder_3(x)
        x = self.sal_decoder_4(x)

        x = self.sal_cls(x)
        x = F.interpolate(x, (240, 320))
        x = torch.sigmoid(x).squeeze(1)

        return x