import torch
import torch.nn as nn

# Define 3x3 convolution.
def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
    
class BasicBlock(nn.Module):
    expension = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.downsample = downsample
        # self.bn3 = nn.BatchNorm2d(planes)
        
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        out = self.conv2(out)
        if self.downsample:
            out += self.downsample(x)
        else:
            out += x
        out = self.bn2(out)
        out = self.relu(out)

        return out
    
class BottleNeck(nn.Module):
    expension = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        planes /= self.expension
        planes = int(planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.conv3 = nn.Conv2d(planes, planes*self.expension, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expension)
        
        self.downsample = downsample
        
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.conv3(out)
        
        if self.downsample:
            out += self.downsample(x)
        else:
            out += x
        out = self.bn3(out)
        out = self.relu(out)

        return out
        
class ResNet(nn.Module):
    def __init__(self, block, layers, nums, num_classes, type) -> None:
        super(ResNet,self).__init__()
        self.arch = type
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=layers[0], kernel_size=7, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(layers[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.in_channels = layers[0]
        self.layers = torch.nn.Sequential(
            self._make_layers(block, layers[0], nums[0]),
            self._make_layers(block, layers[1], nums[1], stride=2),
            self._make_layers(block, layers[2], nums[2], stride=2),
            self._make_layers(block, layers[3], nums[3], stride=2)
        )
        self.size = layers
        self.avg = nn.AvgPool2d(kernel_size=7)
        self.linear = nn.Linear(layers[3]*block.expension, num_classes)
        self.relu = nn.ReLU(inplace=True)
    
    def _make_layers(self, block, out_channels, blocks, stride=1):
        downsample = None
        out_channels *= block.expension
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = torch.nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                torch.nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return torch.nn.Sequential(*layers)

        
    def forward(self,x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layers(out)
        out = self.avg(out).reshape(-1, self.in_channels)
        out = self.linear(out)
        out = self.relu(out)
        
        return out
        



