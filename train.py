# import standard modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import io
import unet
import sys  
import os
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import cv2

# import defined modules
from dataset import cmu_mini_dataset
from common_funcs import diceLoss
from common_funcs import iouLoss

class trainer():
    def __init__(self,
                 version = 1,
                 train_frac = 0.7, batch_size=1, shuffle=True , num_workers=1, use_cuda = True, cont = False):
        self.version, self.use_cuda = version, use_cuda

        if version == '31':
            inp_path = 'Data_augmentation/Augmented_Edge_{}/Inp/output'.format(image_num) 
            gt_path = 'Data_augmentation/Augmented_Edge_{}/GT/output'.format(image_num)
        else:
            inp_path = 'input_multiclass'
            gt_path = 'output_multiclass'

        # Create the dataset and split to training/validation
        my_dataset = cmu_mini_dataset(inp_path, gt_path)
        train_size = int(train_frac * len(my_dataset))
        valid_size = len(my_dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(my_dataset, [train_size, valid_size])
        trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle , num_workers=4)
        validLoader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False , num_workers=4)
        self.loader = {'Train': trainLoader, 'Valid': validLoader}
        for inp, gt in self.loader['Train']:
            print('input shape', inp.shape, 'gt shape', gt.shape)
            shape = inp.shape
            h,w = shape[2], shape[3]
            break

        # UNet
        if self.version == 1:
            self.net = unet.UNet128((3, h, w))
        # ResUNet or Edge UNet
        elif self.version == 2 or version == 31:
            print('version', self.version)
            self.net = unet.ResUNet128((3, h, w))
        # wrcnet training the seg unet
        elif self.version == 32:
            self.net = unet.WNet()
            print('Loading pretrained Edge UNet')
            checkpoint = torch.load('Saved_Net/trained_version_2.pt')
            self.net.edgeUNet.load_state_dict(checkpoint['model_state_dict'])
            for name, param in self.net.named_parameters():
                if name.startswith('edgeUNet'):
                    param.requires_grad = False
        else:
            print('Invalid Network Version')
            return

        if self.use_cuda:
            self.net = self.net.cuda()

        if cont:
            checkpoint = torch.load('trained_version_'+str(self.version)+'.pt')
            self.net.load_state_dict(checkpoint['model_state_dict'])


        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, epochs=12):
        bestValidLoss = np.inf
        bestValidEpoch = -1
        lossData = np.zeros((epochs, 2), dtype = np.float)
        for epoch in range(epochs):
            print('\nEpoch: {}'.format(epoch))
            for m, mode in enumerate(['Train', 'Valid']):
                if mode == 'Train':
                    self.net.train()
                else:
                    self.net.eval()
                lossArr = []
                with torch.set_grad_enabled(mode == 'Train'):
                    for i, (inp, gt) in enumerate(self.loader[mode]):
                        if self.use_cuda:
                            inp, gt = inp.cuda(), gt.cuda()

                        self.optimizer.zero_grad()   # zero the gradient buffers
                        out = self.net(inp)
                        print('Shapes', inp.shape, gt.shape, out.shape)
                        loss = self.criterion(out, gt)
                        lossArr.append(loss.data)
                        if mode == 'Train':
                            loss.backward()
                            self.optimizer.step()    # Does the update
                    avgLoss = sum(lossArr)/i
                    lossData[epoch, m] = avgLoss
                    print('Mode: {}, Loss: {}'.format(mode, avgLoss))
                if mode == 'Valid' and avgLoss < bestValidLoss:
                    bestValidLoss = avgLoss
                    bestValidEpoch = epoch
                    print('Saving after epoch {}'.format(epoch))
                    if not os.path.exists('Saved_Net'):
                        os.makedirs('Saved_Net')
                    torch.save({
                                'epoch': bestValidEpoch,
                                'model_state_dict': self.net.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'loss': bestValidLoss
                                }, 'Saved_Net/trained_version_IOU_Res_'+str(self.version)+'.pt')
            print('Best Valid Epoch: {}, Best Valid Loss: {}'.format(bestValidEpoch, bestValidLoss))
        np.savetxt('training_data_IOU_Res.txt', lossData)
        
if __name__ == '__main__':
    my_trainer = trainer(version = 2, batch_size=10, num_workers=10, cont = False)
    my_trainer.train(epochs=300)
