import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
BN_EPS = 1e-4


class ConvBnRelu2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True,
				 is_relu=True):
		super(ConvBnRelu2d, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride,
							  dilation=dilation, groups=groups, bias=False)
		self.bn = nn.BatchNorm2d(out_channels, eps=BN_EPS)
		self.relu = nn.ReLU(inplace=True)
		if is_bn is False: self.bn = None
		if is_relu is False: self.relu = None

	def forward(self, x):
		x = self.conv(x)
		if self.bn is not None:
			x = self.bn(x)
		if self.relu is not None:
			x = self.relu(x)
		return x


## original 3x3 stack filters used in UNet
class StackEncoderOrg(nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=3):
        super(StackEncoderOrg, self).__init__()
        padding = (kernel_size - 1) // 2
        self.encode = nn.Sequential(
            ConvBnRelu2d(x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1,
                         groups=1),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1,
                         groups=1),
        )

    def forward(self, x):
        y = self.encode(x)
        y_small = F.max_pool2d(y, kernel_size=2, stride=2)
        return y, y_small


class StackDecoderOrg(nn.Module):
    def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3):
        super(StackDecoderOrg, self).__init__()
        padding = (kernel_size - 1) // 2

        self.decode = nn.Sequential(
            ConvBnRelu2d(x_big_channels + x_channels, y_channels, kernel_size=kernel_size, padding=padding,
                         dilation=1, stride=1, groups=1),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1,
                         groups=1),
            ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1,
                         groups=1),
        )
    def forward(self, x_big, x):
        N, C, H, W = x_big.size()
        y = F.upsample(x, size=(H, W), mode='bilinear')
        y = torch.cat([y, x_big], 1)
        y = self.decode(y)
        return y


## New 3x3 stack filters used in UNet
class StackEncoder(nn.Module):
	def __init__(self, x_channels, y_channels, kernel_size=3):
		super(StackEncoder, self).__init__()
		padding = (kernel_size - 1) // 2
		self.encode = nn.Sequential(
			ConvBnRelu2d(x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1,
						 groups=1),
			ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1,
						 groups=1),
		)
		self.identity = nn.Conv2d(x_channels, y_channels, kernel_size=1, padding=0, dilation=1, stride=1, groups=1)

	def forward(self, x):
		y = self.encode(x)
		x_s = self.identity(x)		# mapping x to match the output dimension / #Channels
		y = y + x_s
		y_small = F.max_pool2d(y, kernel_size=2, stride=2)
		return y, y_small


class StackDecoder(nn.Module):
	def __init__(self, x_big_channels, x_channels, y_channels, kernel_size=3):
		super(StackDecoder, self).__init__()
		padding = (kernel_size - 1) // 2

		self.decode = nn.Sequential(
			ConvBnRelu2d(x_big_channels + x_channels, y_channels, kernel_size=kernel_size, padding=padding,
						 dilation=1, stride=1, groups=1),
			ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1,
						 groups=1),
			ConvBnRelu2d(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1,
						 groups=1),
		)
		self.identity = nn.Conv2d(x_channels, y_channels, kernel_size=1, padding=0, dilation=1, stride=1, groups=1)


	def forward(self, x_big, x):
		N, C, H, W = x_big.size()
		x_s = F.upsample(x, size=(H, W), mode='bilinear')
		y = torch.cat([x_s, x_big], 1)
		y = self.decode(y)
		x_s = self.identity(x_s)		# mapping x to match the output dimension / #Channels
		y = y + x_s
		return y

# 128x128
class UNet128(nn.Module):
    def __init__(self, in_shape):
        super(UNet128, self).__init__()
        C, H, W = in_shape
        # assert(C==3)

        # 128
        self.down3 = StackEncoderOrg(C, 128, kernel_size=3)  # 64
        self.down4 = StackEncoderOrg(128, 256, kernel_size=3)  # 32
        self.down5 = StackEncoderOrg(256, 512, kernel_size=3)  # 16
        self.down6 = StackEncoderOrg(512, 1024, kernel_size=3)  # 8

        self.center = nn.Sequential(
            ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoderOrg(1024, 1024, 512, kernel_size=3)  # 16
        self.up5 = StackDecoderOrg(512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoderOrg(256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoderOrg(128, 128, 64, kernel_size=3)  # 128
        self.classify = nn.Conv2d(64, 1, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        out = x  # ;print('x    ',x.size())
        down3, out = self.down3(out)  # ;print('down3',down3.size())  #64
        down4, out = self.down4(out)  # ;print('down4',down4.size())  #32
        down5, out = self.down5(out)  # ;print('down5',down5.size())  #16
        down6, out = self.down6(out)  # ;print('down6',down6.size())  #8
        pass  # ;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        #out = torch.sigmoid(out)
        return out


# 128x128
# Unet 128 along with residual units
class ResUNet128(nn.Module):
	def __init__(self, in_shape):
		super(ResUNet128, self).__init__()
		C, H, W = in_shape
		# assert(C==3)

		# 128
		self.down3 = StackEncoder(C, 128, kernel_size=3)  # 64
		self.down4 = StackEncoder(128, 256, kernel_size=3)  # 32
		self.down5 = StackEncoder(256, 512, kernel_size=3)  # 16
		self.down6 = StackEncoder(512, 1024, kernel_size=3)  # 8

		self.center = nn.Sequential(
			ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1),
		)

		# 8
		# x_big_channels, x_channels, y_channels
		self.up6 = StackDecoder(1024, 1024, 512, kernel_size=3)  # 16
		self.up5 = StackDecoder(512, 512, 256, kernel_size=3)  # 32
		self.up4 = StackDecoder(256, 256, 128, kernel_size=3)  # 64
		self.up3 = StackDecoder(128, 128, 64, kernel_size=3)  # 128
		self.classify = nn.Conv2d(64, 1, kernel_size=1, padding=0, stride=1, bias=True)

	def forward(self, x):
		out = x  # ;print('x    ',x.size())
		down3, out = self.down3(out)  # ;print('down3',down3.size())  #64
		down4, out = self.down4(out)  # ;print('down4',down4.size())  #32
		down5, out = self.down5(out)  # ;print('down5',down5.size())  #16
		down6, out = self.down6(out)  # ;print('down6',down6.size())  #8
		pass  # ;print('out  ',out.size())

		out = self.center(out) + out  # Residual layer
		out = self.up6(down6, out)
		out = self.up5(down5, out)
		out = self.up4(down4, out)
		out = self.up3(down3, out)
		out = self.classify(out)
		out = torch.squeeze(out, dim=1)
		out = torch.sigmoid(out)
		return out

# 128x128
# Unet 128 along with residual units
class ResUNet256	(nn.Module):
	def __init__(self, in_shape):
		super(ResUNet128, self).__init__()
		C, H, W = in_shape
		# assert(C==3)

		# 128
		self.down3 = StackEncoder(C, 128, kernel_size=3)  # 64
		self.down4 = StackEncoder(128, 256, kernel_size=3)  # 32
		self.down5 = StackEncoder(256, 512, kernel_size=3)  # 16
		self.down6 = StackEncoder(512, 1024, kernel_size=3)  # 8

		self.center = nn.Sequential(
			ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1),
		)

		# 8
		# x_big_channels, x_channels, y_channels
		self.up6 = StackDecoder(1024, 1024, 512, kernel_size=3)  # 16
		self.up5 = StackDecoder(512, 512, 256, kernel_size=3)  # 32
		self.up4 = StackDecoder(256, 256, 128, kernel_size=3)  # 64
		self.up3 = StackDecoder(128, 128, 64, kernel_size=3)  # 128
		self.classify = nn.Conv2d(64, 1, kernel_size=1, padding=0, stride=1, bias=True)

	def forward(self, x):
		out = x  # ;print('x    ',x.size())
		down3, out = self.down3(out)  # ;print('down3',down3.size())  #64
		down4, out = self.down4(out)  # ;print('down4',down4.size())  #32
		down5, out = self.down5(out)  # ;print('down5',down5.size())  #16
		down6, out = self.down6(out)  # ;print('down6',down6.size())  #8
		pass  # ;print('out  ',out.size())

		out = self.center(out) + out  # Residual layer
		out = self.up6(down6, out)
		out = self.up5(down5, out)
		out = self.up4(down4, out)
		out = self.up3(down3, out)
		out = self.classify(out)
		out = torch.squeeze(out, dim=1)
		out = torch.sigmoid(out)
		return out

class WNet(nn.Module):
	def __init__(self):
		super(WNet, self).__init__()
		self.edgeUNet = ResUNet128((3,128,128))
		self.segUNet = ResUNet128((4,128,128))

	def forward(self, x):
		edgeMap = self.edgeUNet(x)
		newInp = torch.cat((x, edgeMap[:,None,:,:]), dim = 1)
		segMap = self.segUNet(newInp)
		return segMap





# 128x128
# ResUNet128 with additional hand crafted feature input
class ResWtUNet128(nn.Module):
	def __init__(self, in_shape):
		super(ResUNet128, self).__init__()
		C, H, W = in_shape
		# assert(C==3)

		# 128
		self.down3 = StackEncoder(C, 128, kernel_size=3)  # 64
		self.down4 = StackEncoder(128, 256, kernel_size=3)  # 32
		self.down5 = StackEncoder(256, 512, kernel_size=3)  # 16
		self.down6 = StackEncoder(512, 1024, kernel_size=3)  # 8

		self.center = nn.Sequential(
			ConvBnRelu2d(1024, 1024, kernel_size=3, padding=1, stride=1),
		)

		# 8
		# x_big_channels, x_channels, y_channels
		self.up6 = StackDecoder(1024, 1024, 512, kernel_size=3)  # 16
		self.up5 = StackDecoder(512, 512, 256, kernel_size=3)  # 32
		self.up4 = StackDecoder(256, 256, 128, kernel_size=3)  # 64
		self.up3 = StackDecoder(128, 128, 64, kernel_size=3)  # 128
		self.classify = nn.Conv2d(64, 1, kernel_size=1, padding=0, stride=1, bias=True)

	def forward(self, x):
		out = x  # ;print('x    ',x.size())
		down3, out = self.down3(out)  # ;print('down3',down3.size())  #64
		down4, out = self.down4(out)  # ;print('down4',down4.size())  #32
		down5, out = self.down5(out)  # ;print('down5',down5.size())  #16
		down6, out = self.down6(out)  # ;print('down6',down6.size())  #8
		pass  # ;print('out  ',out.size())

		out = self.center(out) + out  # Residual layer
		out = self.up6(down6, out)
		out = self.up5(down5, out)
		out = self.up4(down4, out)
		out = self.up3(down3, out)
		out = self.classify(out)
		out = torch.squeeze(out, dim=1)
		out = torch.sigmoid(out)
		return out

