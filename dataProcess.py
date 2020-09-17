from PIL import Image
import numpy as np
from os import path
import matplotlib.pyplot as plt
import cv2
import pickle

f = open("C:/Users/user/Desktop/CS464 Machine Learning/Project/MPHB-label-txt/train+val.txt", "r")
imgID = []
dataPtr = []
for imgNo in f:
  imageDir = "C:/Users/user/Desktop/CS464 Machine Learning/Project/Human Body Image/" + imgNo[:-1] + ".jpg"
  if path.isfile(imageDir):
      dataPtr.append(imageDir)
      imgID.append(imgNo)
f.close()

labelF = open("C:/Users/user/Desktop/CS464 Machine Learning/Project/MPHB-label-txt/MPHB-label.txt", "r")
trainImgNo = 0
label = []
for i, line in enumerate(labelF):
  if line == ("idx: " + imgID[trainImgNo]):
    bodyPos = []
    labelF.readline()
    while True:
      posCandidate = labelF.readline().rstrip().split(" ")
      
      if posCandidate[0][0] == "s": 
        break
      else:
        pos = [int(float(borders)) for borders in posCandidate]
        bodyPos.append(pos)
        
    label.append(bodyPos) 
    if trainImgNo < len(imgID) - 1:
        trainImgNo += 1 
        
labelF.close()

#take only images with single object
singleLabel = []
singlePtr = []
for (l, ptr) in zip(label, dataPtr):
    if len(l) == 1:
        singleLabel.append(l)
        singlePtr.append(ptr)
        
#show example sample
#j = 25
#img = np.array(Image.open(dataPtr[j]))
#for body in label[j]:
#    cv2.rectangle(img,(body[0],body[1]),(body[2],body[3]), (255, 0, 0), 2)
#plt.imshow(img)

#ones = 0
#for j in label:
#    if len(j) == 1:
#        ones +=1
with open("dataPtr", "wb") as f:   
    pickle.dump(singlePtr, f)

with open("label", "wb") as f:   
    pickle.dump(singleLabel, f)