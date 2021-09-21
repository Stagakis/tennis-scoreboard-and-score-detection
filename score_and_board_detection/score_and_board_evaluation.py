import easyocr
import torch
from model import FasterRCNNVGG16
from data.mdataset import preprocess
import cv2
import numpy as np
import json
import os


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def getPreprocessedImage(img):
    rgbArray = np.zeros((3, 540, 960), 'uint8')
    rgbArray[0, ...] = img.copy()
    rgbArray[1, ...] = img.copy()
    rgbArray[2, ...] = img.copy()
    rgbArray = preprocess(rgbArray, 600, 1000)
    rgbArray = torch.from_numpy(rgbArray).cuda().float()
    return rgbArray


def extractClasses(img, bboxes):
    box = bboxes[0]

    height = box[2] - box[0]
    player1 = [box[0], box[1], box[0] + height / 2, box[3]]
    player2 = [box[0] + height / 2, box[1], box[2], box[3]]

    # serving_player = bboxes[1]
    # non_serving_player = bboxes[2]
    box = img[int(box[0]):int(box[2]), int(box[1]):int(box[3])]
    serv = img[int(player1[0]):int(player1[2]), int(player1[1]):int(player1[3])]
    noserv = img[int(player2[0]):int(player2[2]),
             int(player2[1]):int(player2[3])]
    return box, serv, noserv


list_of_IoU = []
number_of_correct_name_detections = 0
number_of_correct_score_detections = 0

if __name__ == "__main__":
    reader = easyocr.Reader(['en'])
    faster_rcnn = FasterRCNNVGG16()
    state_dict = torch.load("./checkpoints/best")
    faster_rcnn.load_state_dict(state_dict['model'])
    faster_rcnn.eval()
    faster_rcnn.cuda()

    data_folder = "/home/beastmaster/Downloads/CV-interview-scoreboard-task-material/mdata/eval"
    json_file_name = "annotations.json"

    json_file_path = os.path.join(data_folder, json_file_name)
    f = open(json_file_path)
    json_data = json.load(f)

    for i in range(len(json_data)):
        img_file = os.path.join(data_folder, str(i) + ".png")
        img = cv2.imread(img_file, 0)  # .reshape((3, 540, 960))

        img_cuda = getPreprocessedImage(img)
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict([img_cuda], [[540, 960]])
        pred_bboxes_ = pred_bboxes_[0]
        gt_boxes = json_data[str(i)]["bbox"]
        gt_boxes = [gt_boxes[1], gt_boxes[0], gt_boxes[3], gt_boxes[2]]

        list_of_IoU.append(bb_intersection_over_union(gt_boxes, pred_bboxes_[0]))

        cv2.imshow("Image", img)
        score, player1, player2 = extractClasses(img, pred_bboxes_)

        score_board_ocr_result = reader.readtext(score, detail=0)
        player1_ocr_result = reader.readtext(player1, detail=0)
        player2_ocr_result = reader.readtext(player2, detail=0)


        if json_data[str(i)]["name_1"] in player1_ocr_result:
            number_of_correct_name_detections = number_of_correct_name_detections + 1
        if json_data[str(i)]["name_2"] in player2_ocr_result:
            number_of_correct_name_detections = number_of_correct_name_detections + 1


        # format scores for easy detection
        player1_score = player1_ocr_result[0:]
        player2_score = player2_ocr_result[0:]

        if json_data[str(1)]["score1"] in player1_ocr_result:
            number_of_correct_score_detections = number_of_correct_score_detections + 1
        if json_data[str(1)]["score2"] in player2_ocr_result:
            number_of_correct_score_detections = number_of_correct_score_detections + 1

        cv2.imshow("BBOX", score)
        cv2.imshow("player1", player1)
        cv2.imshow("player2", player2)

        cv2.waitKey(0)
