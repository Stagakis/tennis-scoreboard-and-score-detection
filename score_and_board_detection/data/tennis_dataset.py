import os
import xml.etree.ElementTree as ET
import json
import numpy as np

from .util import read_image


class TennisScoreboardDataset:
    def __init__(self, data_dir, split='trainval',
                 use_difficult=False, return_difficult=False,
                 ):
        if split == 'trainval':
            self.data_dir = os.path.join(data_dir, "train")
        else:
            self.data_dir = os.path.join(data_dir, "eval")

        id_list_file = os.path.join(
            self.data_dir, 'annotations.json')

        f = open(id_list_file)
        self.json_data = json.load(f)

        self.ids = [str(i) for i in range(len(self.json_data))]

        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = TENNIS_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)

    def get_example(self, i):
        """Returns the i-th example.

        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes

        """

        id_ = self.ids[i]

        bbox = list()
        label = list()
        difficult = list()
        difficult.append(False)
        difficult.append(False)
        difficult.append(False)


        # flip bounding boxes due to coordinate differences
        temp_bbox = self.json_data[str(id_)]["bbox"]

        temp_bbox[0], temp_bbox[1] = temp_bbox[1], temp_bbox[0]
        temp_bbox[2], temp_bbox[3] = temp_bbox[3], temp_bbox[2]

        height = temp_bbox[2] - temp_bbox[0]
        width = temp_bbox[3] - temp_bbox[1]

        scoreboard_bbox = temp_bbox

        # Lets assume player 1 is the serving player
        serving_player_bbox = [temp_bbox[0], temp_bbox[1], temp_bbox[0] + height / 2, temp_bbox[3]]
        not_serving_player_bbox = [temp_bbox[0] + height / 2, temp_bbox[1], temp_bbox[2], temp_bbox[3]]

        # If this is not the case, then swap them
        if self.json_data[str(id_)]["serving_player"] == "name_2":
            not_serving_player_bbox, serving_player_bbox = serving_player_bbox, not_serving_player_bbox

        bbox.append(scoreboard_bbox)
        bbox.append(serving_player_bbox)
        bbox.append(not_serving_player_bbox)

        label.append(TENNIS_BBOX_LABEL_NAMES.index('scoreboard'))
        label.append(TENNIS_BBOX_LABEL_NAMES.index('serving'))
        label.append(TENNIS_BBOX_LABEL_NAMES.index('noserving'))

        label = np.stack(label).astype(np.int32)
        difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool
        bbox = np.stack(bbox).astype(np.float32)

        # Load a image
        img_file = os.path.join(self.data_dir, id_ + '.png')
        img = read_image(img_file, color=True)

        # if self.return_difficult:
        #     return img, bbox, label, difficult
        return img, bbox, label, difficult

    __getitem__ = get_example


TENNIS_BBOX_LABEL_NAMES = (
    'scoreboard',
    'serving',
    'noserving',
)

BBOX_LABELS = TENNIS_BBOX_LABEL_NAMES
