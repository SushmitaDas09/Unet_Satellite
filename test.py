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
from common_funcs import iouLoss
from skimage.transform import resize


path = "input_multiclass"
path1 = "output_multiclass"
path2 = "results_multiclass"
files = os.listdir(path)
lossArr = []
for file in files:
        img_path = os.path.join(path, file)
        im = cv2.imread(img_path)
        #file1 = ('new_'+ (file.split(".jpg"))[0] + '_marker_mask.png')
        img_path1 = os.path.join(path1, file)
        gt = cv2.imread(img_path1)
        [h,w,c] = im.shape
        print('Input image size:', [w,h])
        [h,w,c] = gt.shape
        print('GT image size:', [w,h])

        net = unet.UNet128((3, 128, 128)).cuda()

        checkpoint = torch.load('Saved_Net/trained_version_IOU_Res_1.pt')
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
            print(out.shape)
            #loss = iouLoss(out,gt_tensor)
            out = out.reshape(4,w,h).detach().cpu().numpy().astype(float)
            out = np.argmax(out, axis=0)*64    
        cv2.imwrite(os.path.join(path2,file),out)
        lossArr.append(loss.data)

np.savetxt('testing_data_IOU_Res.txt', lossArr)
