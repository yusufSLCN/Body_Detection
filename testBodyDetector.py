from bodyDataset import ImageDataset
import pickle
from torch.utils.data import DataLoader
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import random
from skimage import io

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

validDataLoad = DataLoader(dataset_val, batch_size= 1, shuffle=True)

model = torch.load(r"C:\Users\user\Desktop\CS464 Machine Learning\Project\trained model\detect_body")
#for i, data in enumerate(validDataLoad):
#    model.eval()
#    with torch.no_grad():
#        imgData = data['image']
#        valOutput = model(imgData).cpu()
#        target = np.array(data['box'][0]).astype(int)
#        
#    img = np.array(imgData)[0].transpose(1,2,0)
#    pred = np.array(valOutput)[0].astype(int)
#    cv2.rectangle(img,(target[0],target[1]),(target[2],target[3]), (255, 0, 0), 2)
#    #cv2.rectangle(img,(pred[0],pred[1]),(pred[2],pred[3]), (0, 255, 0), 1)
#    plt.imshow(img)
#    if i == 0:
#        break

#data = next(iter(validDataLoad))
myphoto = ImageDataset(root_dir= [r"C:\Users\user\Desktop\CS464 Machine Learning\Project\deneme.jpg"],label = [[0,0,0,0]])
data = myphoto[0]
#random.seed()
#i = random.randint(0, len(dataset_val))
#data = dataset_val[i]

with torch.no_grad():
    model.eval()
    imgData = data['image'].unsqueeze_(0)
    valOutput = model(imgData).cpu()
    target = np.array(data['box']).astype(int).reshape((4))

img = np.array(imgData)[0].transpose(1,2,0)

pred = np.array(valOutput)[0].astype(int)
cv2.rectangle(img,(target[0],target[1]),(target[2],target[3]), (255, 0, 0), 2)
cv2.rectangle(img,(pred[0],pred[1]),(pred[2],pred[3]), (0, 255, 0), 1)
plt.imshow(img)
