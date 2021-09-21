import easyocr
import torch
from model import FasterRCNNVGG16
from data.mdataset import preprocess
import cv2
import numpy as np

faster_rcnn = FasterRCNNVGG16()
state_dict = torch.load("./checkpoints/best")
faster_rcnn.load_state_dict(state_dict['model'])

faster_rcnn.eval()
faster_rcnn.cuda()

img_file = "/home/beastmaster/Downloads/CV-interview-scoreboard-task-material/mdata/train/1.png"
img = cv2.imread(img_file, 0)#.reshape((3, 540, 960))
rgbArray = np.zeros((3, 540, 960), 'uint8')
rgbArray[0, ...] = img
rgbArray[1, ...] = img
rgbArray[2, ...] = img
img = rgbArray
img = preprocess(img, 600, 1000)
img = torch.from_numpy(img).cuda().float()


print(img.shape)
sizes = [[540, 960]]



print(img[:, :, :])

pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict([img], sizes)

print("============== BBOX ===============")
print(pred_bboxes_[0])
print(pred_bboxes_[0].shape)
print("============== LABELS ===============")
print(pred_labels_[0])
print(pred_labels_[0].shape)
print("============== SCORES ===============")
print(pred_scores_[0])
print(pred_scores_[0].shape)
