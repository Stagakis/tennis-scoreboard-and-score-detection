import easyocr
import torch
from model import FasterRCNNVGG16
from data.mdataset import preprocess
import cv2
import numpy as np
import json
import os
import statistics
import time

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


def get_preprocessedImage(img):
    rgbArray = np.zeros((3, 540, 960), 'uint8')
    rgbArray[0, ...] = img.copy()
    rgbArray[1, ...] = img.copy()
    rgbArray[2, ...] = img.copy()
    rgbArray = preprocess(rgbArray, 600, 1000)
    rgbArray = torch.from_numpy(rgbArray).cuda().float()
    return rgbArray


def extract_classes(img, bboxes):
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


# This function takes a list of strings (e.g. ['1 3', '2', '3']) and outputs a single string in the format
# of the scores in the annotations (e.g. '1-3-2-3') for easy comparisson
def get_score_in_dbformat(score):
    out = ""
    for num in score:
        if ' ' in num:
            out = out + get_score_in_dbformat(num.split(' ')) + '-'
        else:
            out = out + num + '-'
    return out[:-1]  # remove the last since its bound to be a '-' character


list_of_IoU = []
number_of_correct_name_detections = 0
number_of_correct_score_detections = 0
visualize = False
times = []

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

        img_cuda = get_preprocessedImage(img)

        tic = time.time()

        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict([img_cuda], [[540, 960]])
        pred_bboxes_ = pred_bboxes_[0]
        gt_boxes = json_data[str(i)]["bbox"]
        gt_boxes = [gt_boxes[1], gt_boxes[0], gt_boxes[3], gt_boxes[2]]

        list_of_IoU.append(bb_intersection_over_union(gt_boxes, pred_bboxes_[0]))


        score, player1, player2 = extract_classes(img, pred_bboxes_)

        score_board_ocr_result = reader.readtext(score, detail=0)
        player1_ocr_result = reader.readtext(player1, detail=0)
        player2_ocr_result = reader.readtext(player2, detail=0)

        if json_data[str(i)]["name_1"] == player1_ocr_result:
            number_of_correct_name_detections = number_of_correct_name_detections + 1
        if json_data[str(i)]["name_2"] == player2_ocr_result:
            number_of_correct_name_detections = number_of_correct_name_detections + 1

        # format scores for easy detection
        player1_score = get_score_in_dbformat(player1_ocr_result[1:])
        player2_score = get_score_in_dbformat(player2_ocr_result[1:])

        if json_data[str(i)]["score_1"] in player1_score:
            number_of_correct_score_detections = number_of_correct_score_detections + 1
        if json_data[str(i)]["score_2"] in player2_score:
            number_of_correct_score_detections = number_of_correct_score_detections + 1

        toc = time.time()
        times.append(toc - tic)
        if visualize:
            cv2.imshow("Image", img)
            cv2.imshow("BBOX", score)
            cv2.imshow("player1", player1)
            cv2.imshow("player2", player2)
            print(score_board_ocr_result)
            cv2.waitKey(0)

        print(i)
        print(times[-1])
    print("================== RESULTS ================")
    print("Number of samples: {}".format(len(list_of_IoU)))
    print("Execution times: ")
    print("Mean execution time: {}".format(statistics.mean(times)))

    print("IoU stdev: {} ".format(statistics.stdev(list_of_IoU)))
    print("IoU mean: {} ".format(statistics.mean(list_of_IoU)))
    print("IoU median: {} ".format(statistics.median(list_of_IoU)))
    print("Percentage of correct score detection: {} ".format(number_of_correct_score_detections/(2*len(list_of_IoU))))
    print("Percentage of correct name detection: {} ".format(number_of_correct_name_detections/(2*len(list_of_IoU))))
