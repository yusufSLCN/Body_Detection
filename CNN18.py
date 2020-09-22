from torch import nn
import torch

class Resnet18(nn.Module):
    def __init__(self, image_channels, num_classes): 
        super(Resnet18,self).__init__()
        
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride = 2, padding = 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        #conv2_x
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride = 1, padding = 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride = 1, padding = 1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride = 1, padding = 1)
        self.bn5 = nn.BatchNorm2d(64)
        
        #conv3_x
        self.conv6 = nn.Conv2d(64, 128, kernel_size=3, stride = 2, padding = 1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride = 1, padding = 1)
        self.bn7 = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 128, kernel_size=3, stride = 1, padding = 1)
        self.bn8 = nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=3, stride = 1, padding = 1)
        self.bn9 = nn.BatchNorm2d(128)
        
        #conv4_x
        self.conv10 = nn.Conv2d(128, 256, kernel_size=3, stride = 2, padding = 1)
        self.bn10 = nn.BatchNorm2d(256)
        self.conv11 = nn.Conv2d(256, 256, kernel_size=3, stride = 1, padding = 1)
        self.bn11 = nn.BatchNorm2d(256)
        self.conv12 = nn.Conv2d(256, 256, kernel_size=3, stride = 1, padding = 1)
        self.bn12 = nn.BatchNorm2d(256)
        self.conv13 = nn.Conv2d(256, 256, kernel_size=3, stride = 1, padding = 1)
        self.bn13 = nn.BatchNorm2d(256)
        
        #conv5_x
        self.conv14 = nn.Conv2d(256, 512, kernel_size=3, stride = 2, padding = 1)
        self.bn14 = nn.BatchNorm2d(512)
        self.conv15 = nn.Conv2d(512, 512, kernel_size=3, stride = 1, padding = 1)
        self.bn15 = nn.BatchNorm2d(512)
        self.conv16 = nn.Conv2d(512, 512, kernel_size=3, stride = 1, padding = 1)
        self.bn16 = nn.BatchNorm2d(512)
        self.conv17 = nn.Conv2d(512, 512, kernel_size=3, stride = 1, padding = 1)
        self.bn17 = nn.BatchNorm2d(512)
        
        #avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        #fc
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        
        x = self.conv7(x)
        x = self.bn7(x)
        x = self.relu(x)
        
        x = self.conv8(x)
        x = self.bn8(x)
        x = self.relu(x)
        
        x = self.conv9(x)
        x = self.bn9(x)
        x = self.relu(x)
    
        x = self.conv10(x)
        x = self.bn10(x)
        x = self.relu(x)
        
        x = self.conv11(x)
        x = self.bn11(x)
        x = self.relu(x)
        
        x = self.conv12(x)
        x = self.bn12(x)
        x = self.relu(x)
        
        x = self.conv13(x)
        x = self.bn13(x)
        x = self.relu(x)
        
        x = self.conv14(x)
        x = self.bn14(x)
        x = self.relu(x)
        
        x = self.conv15(x)
        x = self.bn15(x)
        x = self.relu(x)
        
        x = self.conv16(x)
        x = self.bn16(x)
        x = self.relu(x)
        
        x = self.conv17(x)
        x = self.bn17(x)
        x = self.relu(x)
        
        x = self.avgpool(x)
        #reshape
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x
        
def model(img_channels = 3, num_classes = 4):
    return Resnet18(img_channels, num_classes)

def test():
    net = model()
    x = torch.randn(2,3,224,224)
    y = net(x).to('cuda')
    print("TEST")
    print(y.shape)
