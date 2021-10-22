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
from skimage.transform import resize


path = "dataset_train"
path1 = "results"
files = os.listdir(path)
i=0
for file in files:
        img_path = os.path.join(path, file)
        im = cv2.imread(img_path)
        file1 = ('new_'+ (file.split(".jpg"))[0] + '_marker_mask.png')
        img_path1 = os.path.join(path1, file1)
        gt = cv2.imread(img_path1)
        [h,w,c] = im.shape
        print('Input image size:', [w,h])
        [h,w,c] = gt.shape
        print('GT image size:', [w,h])

        net = unet.UNet128((3, 128, 128)).cuda()

        checkpoint = torch.load('Saved_Net/trained_version_1.pt')
        print('Best valid loss was: ', checkpoint['loss'])
        net.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
         
        net.eval()
        # net.eval()
        #im_tensor = transformer(im).unsqueeze(0).cuda()
        #img = resize(im, (128, 128), anti_aliasing=True)
        im_tensor = transformer(im).unsqueeze(0).cuda()
        #gt_img = resize(gt, (128, 128), anti_aliasing=True)
        gt_tensor = transformer(gt).unsqueeze(0).cuda()
        with torch.no_grad():
            out = net(im_tensor)
            loss = diceLoss(out,gt_tensor)
            out = out.reshape(w,h).detach().cpu().numpy().astype(float)*255
        cv2.imwrite('Output_result/Output'+str(i)+'.png',out)
        lossArr.append(loss.data)

np.savetxt('testingg_data.txt', lossArr)
