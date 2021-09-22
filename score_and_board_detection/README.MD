# A Simple and Fast Implementation of Faster R-CNN For tennis scoreboard detection

This repo is a clone of https://github.com/chenyuntc/simple-faster-rcnn-pytorch.
Main changes include the development of `data/mdataset.py` and `data/tennis_dataset.py` files for interfacing with the data loader in loading custom datasets.

For installation instructions, please refer to the original repo. Extra dependencies for this project include opencv-python and easyocr. To install them:

`pip install opencv-python`

`pip install easyocr`

`score_and_board_evaluation.py` uses the trained RCNN combined with OCR to detect names and scores on a tennis scoreboard.


## ^_^
Licensed under MIT, see the LICENSE for more detail.

Contribution Welcome.

If you encounter any problem, feel free to open an issue, but too busy lately.

Correct me if anything is wrong or unclear.

model structure
![img](imgs/model_all.png)
