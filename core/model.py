import torch
import torch.nn as nn
from torchvision import models

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block with FC support"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()

        # For both Conv and FC cases
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
        
        # For Conv layers
        self.avg_pool_2d = nn.AdaptiveAvgPool2d(1)
        # For FC layers
        self.avg_pool_1d = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # Determine input type
        if x.dim() == 4:  # Conv layer input [B, C, H, W]
            b, c, h, w = x.size()
            y = self.avg_pool_2d(x).view(b, c)
        elif x.dim() == 2:  # FC layer input [B, C]
            b, c = x.size()
            y = self.avg_pool_1d(x.unsqueeze(-1)).view(b, c)
        else:
            raise ValueError(f"Unexpected input dimensions: {x.dim()}")
            
        # Compute attention weights
        y = self.fc(y)
        
        # Reshape for multiplication
        if x.dim() == 4:
            y = y.view(b, c, 1, 1)
            return x * y.expand_as(x)
        else:  # FC case
            return x * y 

class HybridResNet(nn.Module):
    def __init__(self, num_classes, use_se=True, use_input_conv=True, use_modified_fc=True, unfreeze_layers=True):
        super(HybridResNet, self).__init__()
        
        # Initial projection layer for single-channel input
        if use_input_conv:
            self.input_conv = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 3, kernel_size=3, padding=1),
                nn.BatchNorm2d(3),
                nn.ReLU()
            )
        else:
            self.input_conv = None # Will use channel replication instead

        
        # Pretrained ResNet50 backbone
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        if use_se:
            self._add_se_blocks()
        
        # Freeze or unfreeze layers
        if not unfreeze_layers:
            for param in self.resnet.parameters():
                param.requires_grad = False
        else:
            self._unfreeze_layers()
        
        if use_modified_fc:
            self.resnet.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(2048, 512),
                nn.ReLU(),
                SEBlock(512) if use_se else nn.Identity(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        else:
            self.resnet.fc = nn.Linear(2048, num_classes)
            if not unfreeze_layers:
                for param in self.resnet.fc.parameters():
                    param.requires_grad = True
        
    def _add_se_blocks(self):
        """Add SE blocks to ResNet's bottleneck layers"""
        for layer in [self.resnet.layer1, self.resnet.layer2, 
                     self.resnet.layer3, self.resnet.layer4]:
            for bottleneck in layer:
                bottleneck.add_module('se_block', SEBlock(bottleneck.conv3.out_channels))

    def _unfreeze_layers(self):
        """Unfreeze last two residual layers and SE blocks"""
        # Freeze all parameters first
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # Unfreeze SE blocks and last two layers
        for layer in [self.resnet.layer3, self.resnet.layer4]:
            for param in layer.parameters():
                param.requires_grad = True
        for m in self.modules():
            if isinstance(m, SEBlock):
                for param in m.parameters():
                    param.requires_grad = True

    def forward(self, x):
        if self.input_conv is not None:
            x = self.input_conv(x)
        else:
            x = x.repeat(1, 3, 1, 1)  # Replicate single channel to 3 channels
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x



