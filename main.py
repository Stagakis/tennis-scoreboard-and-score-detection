import cv2
import json
import os


def create_output_folders(root_folder):
    try:
        os.mkdir(root_folder)
    except:
        pass  # Folder is already created
    try:
        os.mkdir(os.path.join(root_folder, "train"))
    except:
        pass  # Folder is already created
    try:
        os.mkdir(os.path.join(root_folder, "eval"))
    except:
        pass  # Folder is already created


data_folder = "C:/Users/Stagakis/Desktop/CV-interview-scoreboard-task-material/data"
video_file_name = "top-100-shots-rallies-2018-atp-season.mp4"
json_file_name = "top-100-shots-rallies-2018-atp-season-scoreboard-annotations.json"

video_file_path = os.path.join(data_folder, video_file_name)
json_file_path = os.path.join(data_folder, json_file_name)

output_root_folder = os.path.join(data_folder, "../mdata")

if __name__ == '__main__':
    create_output_folders(output_root_folder)

    cap = cv2.VideoCapture(video_file_path)

    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    f = open(json_file_path)
    json_data = json.load(f)

    train_dict = {}
    eval_dict = {}

    train_i = 0
    eval_i = 0

    index = 0
    index_switch = 29064  # the index to change from train to eval

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            try:
                frame_data = json_data[str(index)]  # check to see if there is a scoreboard in this frame
            except KeyError:
                index = index + 1
                continue
            if (index < index_switch):
                cv2.imwrite(os.path.join(output_root_folder, "train", str(train_i) + ".png"), frame)
                train_dict[str(train_i)] = json_data[str(index)]
                train_i = train_i + 1
            else:
                cv2.imwrite(os.path.join(output_root_folder, "eval", str(eval_i) + ".png"), frame)
                eval_dict[str(eval_i)] = json_data[str(index)]
                eval_i = eval_i + 1
            index = index + 1
        else:
            break

    with open(os.path.join(output_root_folder, "train", "annotations.json"), 'w') as outfile:
        json.dump(train_dict, outfile)

    with open(os.path.join(output_root_folder, "eval", "annotations.json"), 'w') as outfile:
        json.dump(eval_dict, outfile)

    cv2.destroyAllWindows()
    cap.release()
