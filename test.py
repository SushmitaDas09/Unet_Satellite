import matplotlib.pyplot as plt
#import cv2
import torch, cv2, numpy as np
from torchvision import transforms
from PIL import Image
transformer = transforms.ToTensor()
#import defined modules
import unet
#test code
import os
from common_funcs import diceLoss


# path = "cmu_mini_data_1"
# files = os.listdir(path)
# new_image = np.zeros((256,256))
# for file in files:
#     if file.endswith('marker_mask.png'):
#         img_path = os.path.join(path, file)
#         image = cv2.imread(img_path)
#         image = color.rgb2gray(image)
#     #image = io.imread('image.png', as_gray=True)
#         for i in range(256):
#             for j in range(256):
#                 if image[i,j] !=0:
#                     new_image[i,j] = 255
#                 else:
#                     new_image[i,j] = 0
#         cv2.imwrite(f'results/new_{file}', new_image)

path = "dataset_train"
path1 = "Test/test_results"
files = os.listdir(path)
i=0
for file in files:
        img_path = os.path.join(path, file)
        im = cv2.imread(img_path)
        file1 = ('new_'+ (file.split(".png"))[0] + '_marker_mask.png')
        img_path1 = os.path.join(path1, file)
        gt = cv2.imread('img_path1')
        [h,w,c] = im.shape
        print('Input image size:', [w,h])

        net = unet.UNet128((3, 128, 128)).cuda()

        checkpoint = torch.load('Saved_Net/trained_version_1.pt')
        print('Best valid loss was: ', checkpoint['loss'])
        net.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
         
        net.eval()
        # net.eval()
        im_tensor = transformer(im).unsqueeze(0).cuda()
        #print(im_tensor.shape)
        out = net(im_tensor).reshape(w,h).detach().cpu().numpy().astype(float)*255
        cv2.imwrite('Output_result/Output'+str(i)+'.png',out)
        i = i+1
        out[out<10], out[out>=10] = 0, 1
        print(out.min(), out.max())
        loss = diceLoss(out, gt)
        lossArr.append(loss.data)
np.savetxt('training_data.txt', lossArr)


# for i in range(len(files)):
#     im = cv2.imread('Test')
#     gt = cv2.imread('Test/test_results')
#     [h,w,c] = im.shape
#     print('Input image size:', [w,h])

#     net = unet.ResUNet128((3, 128, 128)).cuda()

#     checkpoint = torch.load('Saved_Net/trained_resUnet_2.pt')
#     print('Best valid loss was: ', checkpoint['loss'])
#     net.load_state_dict(checkpoint['model_state_dict'])
#     # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     epoch = checkpoint['epoch']
#     loss = checkpoint['loss']

#     net.eval()
#     # net.eval()

#     im_tensor = transformer(im).unsqueeze(0).cuda()
#     #print(im_tensor.shape)
#     out = net(im_tensor).reshape(w,h).detach().cpu().numpy().astype(float)*255
#     cv2.imwrite('Data_augmentation/Augmented_2/'+str(i)+'.png',out)
#     out[out<10], out[out>=10] = 0, 1
#     print(out.min(), out.max())
#     # cv2.imshow('window', out)
#     # cv2.waitKey(0)
