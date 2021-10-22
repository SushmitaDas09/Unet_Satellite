import torch
import os, cv2, numpy as np
from skimage import io
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
from skimage.transform import resize


# data augmentation
# changin to gpu
# training

class cmu_mini_dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, inp_path, gt_path):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """        
        self.transformer = transforms.ToTensor()
        self.path1 = inp_path
        self.path2 = gt_path
        self.names2 = os.listdir(self.path2)
        self.names1 = []
        for name in self.names2:
        	self.names1.append(name.split('_')[1]+'.jpg')
        # print(self.names)
    
    def trans(self, img):
        img = resize(img, (128, 128), anti_aliasing=True)
        img = self.transformer(img).float()
        return img



    def __len__(self):
        return len(self.names2)

    def __getitem__(self, idx):
        #img_name1 = os.path.join(self.path1, self.names[idx])
        img = os.path.join(self.path1, self.names1[idx])
        gt_img = os.path.join(self.path2, self.names2[idx])           
        #img_name2 = os.path.join(self.path2, '_groundtruth_(1)_Inp'+self.names[idx][12:])
        image1 = cv2.imread(img)                                                
        image2 = cv2.imread(gt_img)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        im_tensor1 = self.trans(image1)
        im_tensor2 = torch.FloatTensor(resize(image2,(128,128),anti_aliasing=True))
        return im_tensor1,im_tensor2


if __name__ == '__main__':
    my_dataset = cmu_mini_dataset("cmu_mini_data_1","results")
    dataloader = DataLoader(my_dataset, batch_size=1, shuffle=True, num_workers=1)
    for epoch in range(1):
        for inp, gt in dataloader:
            print(inp.shape, gt.shape)
    print(my_dataset.__len__())
    #plt.imshow(my_dataset.__getitem__(0))
    #plt.show()
    
