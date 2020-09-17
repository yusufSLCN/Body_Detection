from torchvision import models
from bodyDataset import ImageDataset
from torch import nn
import pickle
from torch.utils.data import DataLoader
import torch
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open('dataPtr', 'rb') as f:
    dataPtr = pickle.load(f)
with open('label', 'rb') as f:
    labelBox = pickle.load(f)

ind = list(range(len(dataPtr)))
random.seed(100)
random.shuffle(ind)
bound = int(len(dataPtr) * 0.90)
dataPtr_tr,label_tr = dataPtr[:bound], labelBox[:bound]
dataPtr_val, label_val = dataPtr[bound:], labelBox[bound:]
dataset_tr   = ImageDataset(root_dir = dataPtr_val, label = label_tr)
dataset_val   = ImageDataset(root_dir = dataPtr_val, label = label_val)

batchSize = 50
trainDataLoad = DataLoader(dataset_tr, batch_size= batchSize, shuffle=True)
validDataLoad = DataLoader(dataset_val, batch_size= 1, shuffle=True)

model = models.resnet18(pretrained = True).to(device)
fc_inputSize = model.fc.in_features
model.fc = nn.Linear(fc_inputSize,4).to(device)
criterion = nn.SmoothL1Loss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


num_epochs = 3
for epoch in range(num_epochs):
    for i, data in enumerate(trainDataLoad):
        model.train()
        optimizer.zero_grad()

        imgData = data['image'].to(device)
        target = data['box'].float().to(device)

        outputs = model(imgData)
        loss = criterion(outputs, target)
        if i % 10 == 0:
            print('Batch: ', i , '/', len(dataset_tr) / batchSize, ' Loss: ', loss.item())
        loss.backward()
        optimizer.step()

torch.save(model.cpu(), r"C:\Users\user\Desktop\CS464 Machine Learning\Project\trained model\detect_body")
