import torch, numpy as np
SMOOTH = 1e-6

def diceLoss(result, target):
	# batchSize = result.shape[2]
	# inter = torch.sum(torch.sum(result*target, dim = 2), dim = 1)
	# union = torch.sum(torch.sum(result+target, dim = 2), dim = 1)
	# dice = torch.sum((2*inter)/union)/batchSize
	inter = torch.sum(result*target)
	total = torch.sum(result+target)
	dice = (2*inter+SMOOTH)/(total+SMOOTH)
	return(1-dice)

def iouLoss(result, target):
	# batchSize = result.shape[2]
	# inter = torch.sum(torch.sum(result*target, dim = 2), dim = 1)
	# union = torch.sum(torch.sum(result+target, dim = 2), dim = 1)
	# dice = torch.sum((2*inter)/union)/batchSize
	inter = torch.sum(result*target)
	total = torch.sum(result+target)
	iou = (inter + SMOOTH)/(total-inter + SMOOTH)
	return(1-iou)

def dice(result, target):
	inter = np.sum(result*target)
	union = np.sum(result+target)
	dice = (2*inter)/union
	return(dice)

def performance(result, target):
	TP = np.sum(result*target)
	TN = np.sum((1-result)*(1-target))
	FP = np.sum(result*(1-target))
	FN = np.sum((1-result)*target)
	e = TP/(TP+FP)
	d = TP/(TP+FN)
	F = (2*e*d)/(e+d)
	return([e, d, F])
