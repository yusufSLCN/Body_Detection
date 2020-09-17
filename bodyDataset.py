from skimage import io,transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2

from torchvision.transforms import functional as TF


class Rescale(object):
    """Rescale the image in a sample to a given size.
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, box = sample['image'], sample['box']

        h,w = image.shape[:2]

        new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = np.array(transform.resize(image, (new_h, new_w)), dtype='float32')

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        box = (box * [new_w / w, new_h / h, new_w / w, new_h / h]).astype(int).reshape(4,1)
        img = TF.to_tensor(img)
        box = TF.to_tensor(box).view(4)


        return {'image': img, 'box': box}

class ImageDataset(Dataset):
    def __init__(self, root_dir, label, transform = Rescale((244,244))):
        self.root  = root_dir
        self.label   = label
        self.transform = transform



    def __getitem__(self, index):
        try:
            image = io.imread(self.root[index])
            boundingBox = np.array(self.label[index][0])
    #        image = transform.resize(image, (244, 244))

            sample = {'image': image, 'box': boundingBox}
            if self.transform:
                sample = self.transform(sample)

            return sample
        except:
            print(self.root[index])
            print(index)

            

    def __len__(self):
        return len(self.root)



#with open('dataPtr', 'rb') as f:
#    dataPtr = pickle.load(f)
#with open('label', 'rb') as f:
#    labelBox = pickle.load(f)
#
#
#dataset_tr   = ImageDataset(root_dir = dataPtr, label = labelBox)
#
#trainDataLoad = DataLoader(dataset_tr, batch_size=2, shuffle=True)
#
##for i, data in enumerate(trainDataLoad):
##    print(i)
#
#for i in range(len(dataset_tr)):
#    sample = dataset_tr[i]
#    img = np.array(sample['image'])
#    box =  np.array(sample['box'])
#    img = img.transpose(1,2,0)
#    cv2.rectangle(img,(box[0],box[1]),(box[2],box[3]), (255, 0, 0), 2)
#    plt.imshow(img)
#    if i == 0:
#        break
