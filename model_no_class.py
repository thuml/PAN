import torch
import torch.nn as nn
import torchvision
from torchvision import models

# convnet without the last layer
class AlexnetFc(nn.Module):
  def __init__(self):
    super(AlexnetFc, self).__init__()
    model_alexnet = models.alexnet(pretrained=True)
    self.features = model_alexnet.features
    self.classifier = nn.Sequential()
    for i in xrange(6):
      self.classifier.add_module("classifier"+str(i), model_alexnet.classifier[i])
    self.__in_features = model_alexnet.classifier[6].in_features


  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), 256*6*6)
    x = self.classifier(x)
    return x

  def output_num(self):
    return self.__in_features

class Resnet18Fc(nn.Module):
  def __init__(self):
    super(Resnet18Fc, self).__init__()
    model_resnet18 = models.resnet18(pretrained=True)
    self.conv1 = model_resnet18.conv1
    self.bn1 = model_resnet18.bn1
    self.relu = model_resnet18.relu
    self.maxpool = model_resnet18.maxpool
    self.layer1 = model_resnet18.layer1
    self.layer2 = model_resnet18.layer2
    self.layer3 = model_resnet18.layer3
    self.layer4 = model_resnet18.layer4
    self.avgpool = model_resnet18.avgpool
    self.__in_features = model_resnet18.fc.in_features

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return x

  def output_num(self):
    return self.__in_features

class Resnet34Fc(nn.Module):
  def __init__(self):
    super(Resnet34Fc, self).__init__()
    model_resnet34 = models.resnet34(pretrained=True)
    self.conv1 = model_resnet34.conv1
    self.bn1 = model_resnet34.bn1
    self.relu = model_resnet34.relu
    self.maxpool = model_resnet34.maxpool
    self.layer1 = model_resnet34.layer1
    self.layer2 = model_resnet34.layer2
    self.layer3 = model_resnet34.layer3
    self.layer4 = model_resnet34.layer4
    self.avgpool = model_resnet34.avgpool
    self.__in_features = model_resnet34.fc.in_features

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return x

  def output_num(self):
    return self.__in_features

class Resnet50Fc(nn.Module):
  def __init__(self):
    super(Resnet50Fc, self).__init__()
    model_resnet50 = models.resnet50(pretrained=True)
    self.conv1 = model_resnet50.conv1
    self.bn1 = model_resnet50.bn1
    self.relu = model_resnet50.relu
    self.maxpool = model_resnet50.maxpool
    self.layer1 = model_resnet50.layer1
    self.layer2 = model_resnet50.layer2
    self.layer3 = model_resnet50.layer3
    self.layer4 = model_resnet50.layer4
    self.avgpool = model_resnet50.avgpool
    self.__in_features = model_resnet50.fc.in_features

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return x

  def output_num(self):
    return self.__in_features

class Resnet101Fc(nn.Module):
  def __init__(self):
    super(Resnet101Fc, self).__init__()
    model_resnet101 = models.resnet101(pretrained=True)
    self.conv1 = model_resnet101.conv1
    self.bn1 = model_resnet101.bn1
    self.relu = model_resnet101.relu
    self.maxpool = model_resnet101.maxpool
    self.layer1 = model_resnet101.layer1
    self.layer2 = model_resnet101.layer2
    self.layer3 = model_resnet101.layer3
    self.layer4 = model_resnet101.layer4
    self.avgpool = model_resnet101.avgpool
    self.__in_features = model_resnet101.fc.in_features

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return x

  def output_num(self):
    return self.__in_features


class Resnet152Fc(nn.Module):
  def __init__(self):
    super(Resnet152Fc, self).__init__()
    model_resnet152 = models.resnet152(pretrained=True)
    self.conv1 = model_resnet152.conv1
    self.bn1 = model_resnet152.bn1
    self.relu = model_resnet152.relu
    self.maxpool = model_resnet152.maxpool
    self.layer1 = model_resnet152.layer1
    self.layer2 = model_resnet152.layer2
    self.layer3 = model_resnet152.layer3
    self.layer4 = model_resnet152.layer4
    self.avgpool = model_resnet152.avgpool
    self.__in_features = model_resnet152.fc.in_features

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return x

  def output_num(self):
    return self.__in_features

class Inceptionv3Fc(nn.Module):
  def __init__(self):
    super(Inceptionv3Fc, self).__init__()
    model_Inceptionv3 = models.inception_v3(pretrained=True)

    self.Conv2d_1a_3x3 = model_Inceptionv3.Conv2d_1a_3x3
    self.Conv2d_2a_3x3 = model_Inceptionv3.Conv2d_2a_3x3
    self.Conv2d_2b_3x3 = model_Inceptionv3.Conv2d_2b_3x3
    self.Conv2d_3b_1x1 = model_Inceptionv3.Conv2d_3b_1x1
    self.Conv2d_4a_3x3 = model_Inceptionv3.Conv2d_4a_3x3
    self.Mixed_5b = model_Inceptionv3.Mixed_5b
    self.Mixed_5c = model_Inceptionv3.Mixed_5c
    self.Mixed_5d = model_Inceptionv3.Mixed_5d
    self.Mixed_6a = model_Inceptionv3.Mixed_6a
    self.Mixed_6b = model_Inceptionv3.Mixed_6b
    self.Mixed_6c = model_Inceptionv3.Mixed_6c
    self.Mixed_6d = model_Inceptionv3.Mixed_6d
    self.Mixed_6e = model_Inceptionv3.Mixed_6e
    self.Mixed_7a = model_Inceptionv3.Mixed_7a
    self.Mixed_7b = model_Inceptionv3.Mixed_7b
    self.Mixed_7c = model_Inceptionv3.Mixed_7c
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
    self.avgpool = nn.AvgPool2d(kernel_size=5)
    self.__in_features = model_Inceptionv3.fc.in_features

  def forward(self, x):
    # if self.transform_input:
    #   x = x.clone()
    #   x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
    #   x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
    #   x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
    # 299 x 299 x 3
    x = self.Conv2d_1a_3x3(x)
    # 149 x 149 x 32
    x = self.Conv2d_2a_3x3(x)
    # 147 x 147 x 32
    x = self.Conv2d_2b_3x3(x)
    # 147 x 147 x 64
    x = self.maxpool(x)
    # 73 x 73 x 64
    x = self.Conv2d_3b_1x1(x)
    # 73 x 73 x 80
    x = self.Conv2d_4a_3x3(x)
    # 71 x 71 x 192
    x = self.maxpool(x)
    # 35 x 35 x 192
    x = self.Mixed_5b(x)
    # 35 x 35 x 256
    x = self.Mixed_5c(x)
    # 35 x 35 x 288
    x = self.Mixed_5d(x)
    # 35 x 35 x 288
    x = self.Mixed_6a(x)
    # 17 x 17 x 768
    x = self.Mixed_6b(x)
    # 17 x 17 x 768
    x = self.Mixed_6c(x)
    # 17 x 17 x 768
    x = self.Mixed_6d(x)
    # 17 x 17 x 768
    x = self.Mixed_6e(x)
    # 17 x 17 x 768
    x = self.Mixed_7a(x)
    # 8 x 8 x 1280
    x = self.Mixed_7b(x)
    # 8 x 8 x 2048
    x = self.Mixed_7c(x)
    # 8 x 8 x 2048
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return x

  def output_num(self):
    return self.__in_features