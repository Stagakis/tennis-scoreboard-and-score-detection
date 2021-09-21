import easyocr
import torch
from model import FasterRCNNVGG16

faster_rcnn = FasterRCNNVGG16()
faster_rcnn.load_state_dict(torch.load("best"))

faster_rcnn.eval()