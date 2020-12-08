import argparse
import sys
import os
from time import sleep
import imutils as imutils
import matplotlib.pyplot as plt
import cv2
import numpy as np


def capture_image():
    cam = cv2.VideoCapture(1)
    sleep(1)
    s, img = cam.read()
    cam.release()
    if s:
        cv2.imwrite("data/mydata/camface.jpg", img)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return cv2.imread('data/mydata/camface.jpg')


def save_plot(img, filename):
    images = [img]

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("Viola-Jones recognition", fontsize=16)

    for ix, image in enumerate(images):
        ax = plt.subplot(int("11" + str(ix + 1)))
        ax.imshow(image)

    plt.savefig(f'data/test_computed/{filename}')
    plt.close()


def get_bounding_boxes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        boxes.append([x, y, x + w, y + h])

    return boxes


def compute_iou(pred_box, gt_box):
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)

    inters = iw * ih

    uni = ((pred_box[2] - pred_box[0] + 1.) * (pred_box[3] - pred_box[1] + 1.) +
           (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
           inters)

    iou = inters / uni

    return iou


def find_nose(img, gray):
    nose_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_nose.xml')
    nose = nose_cascade.detectMultiScale(gray)
    for predicted_coords in nose:
        (x, y, w, h) = predicted_coords
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(img, 'Nose', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
    return nose


def viola_jones(img, gray, left=True, right=True):
    predicted = {
        "left": [],
        "right": []
    }
    if left:
        left_ear_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_leftear.xml')
        left_ears = left_ear_cascade.detectMultiScale(gray, params["left"]["scale_step"], params["left"]["size"],
                                                      minSize=(20, 20), maxSize=(100, 100))

        for predicted_coords in left_ears:
            (x, y, w, h) = predicted_coords
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            predicted["left"].append([x, y, x + w, y + h])
            cv2.putText(img, 'Left ear', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    if right:
        right_ear_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_rightear.xml')
        right_ears = right_ear_cascade.detectMultiScale(gray, params["right"]["scale_step"],
                                                        params["right"]["size"],
                                                        minSize=(20, 20), maxSize=(100, 100))
        for predicted_coords in right_ears:
            (x, y, w, h) = predicted_coords
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            predicted["right"].append([x, y, x + w, y + h])
            cv2.putText(img, 'Right ear', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

    return predicted


def read_image(pth):
    filename = pth.split("/")[-1]
    img = cv2.imread(pth)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return filename, img, gray


tp = 0
fp = 0
fn = 0
tn = 0


def compare(img, annot_coords, predicted_coords):
    global tp, fp, fn, tn
    iou_scores = []

    fn = len(annot_coords)

    for correct in annot_coords:
        (x, y, w, h) = correct
        cv2.rectangle(img, (x, y), (w, h), (0, 0, 255), 2)
        cv2.putText(img, 'Original', (x, h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        for predicted in predicted_coords:
            iou_score = compute_iou(predicted, correct)
            iou_scores.append(iou_score)
            if iou_score > 0.5:
                tp += 1
            else:
                fp += 1
        # if not predicted_coords:
        #     fp += 1

    return iou_scores


def plot_p_r(precision, recall):
    plt.title("Precision Recall Curve")
    plt.plot(precision, recall, '-', color='limegreen')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig("data/precison_recall.png")
    plt.show()


precisions = []
recalls = []
calculations = []


def run_on_miltiple(pth):
    iou_scores_combined = []
    global precisions, recalls, calculations
    for filename in os.listdir(pth):
        (filename, img, gray) = read_image("/".join([pth, filename]))

        path_annot = f"data/testannot_rect/{filename}"
        (filename_annot, img_annot, gray_annot) = read_image(path_annot)

        annot_coords = get_bounding_boxes(img_annot)

        predicted_ears = viola_jones(img, gray)
        predicted_coords = predicted_ears["right"] + predicted_ears["left"]

        iou_scores = compare(img, annot_coords, predicted_coords)
        iou_scores_combined += iou_scores
        save_plot(img, filename)
        precision = tp / (fp + tp)
        recall = tp / (fn + tp)
        print(precision, recall)
        calculations.append((precision, recall))
        precisions.append(precision)
        recalls.append(recall)

    return iou_scores_combined


def run_on_one(pth, capture=False):
    if capture:
        capture_image()
    (filename, img, gray) = read_image(pth)

    path_annot = f"data/testannot_rect/{filename}"
    (filename_annot, img_annot, gray_annot) = read_image(path_annot)

    annot_coords = get_bounding_boxes(img_annot)

    predicted_ears = viola_jones(img, gray, left=False)
    predicted_coords = predicted_ears["right"] + predicted_ears["left"]

    iou_scores = compare(img, annot_coords, predicted_coords)
    save_plot(img, filename)
    return iou_scores


def find_params(path, one=True, right=False, left=False):
    best_params = params
    best_avg = 0
    if one:
        for size in range(1, 10, 1):
            for scale in np.arange(1.005, 2.0, .005):
                if right:
                    params["right"]["scale_step"] = scale
                    params["right"]["size"] = size
                if left:
                    params["left"]["scale_step"] = scale
                    params["left"]["size"] = size
                iou_scores = run_on_one(path)
                mean = np.mean(iou_scores)
                if mean > best_avg:
                    print(params, mean)
                    best_params = params
                    best_avg = mean
    else:
        for size in range(1, 10, 1):
            for scale in np.arange(1.005, 2.0, .005):
                print(size, scale)
                if right:
                    params["right"]["scale_step"] = scale
                    params["right"]["size"] = size
                if left:
                    params["left"]["scale_step"] = scale
                    params["left"]["size"] = size
                iou_scores = run_on_miltiple(path)
                mean = np.mean(iou_scores)
                if mean > best_avg:
                    print(params, mean)
                    best_params = params
                    best_avg = mean

    return best_params


params = {
    "left": {'scale_step': 1.0149, 'size': 3},
    "right": {'scale_step': 1.0199, 'size': 1}
}

if __name__ == '__main__':
    arguments = str(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--Path", default="data/test", help="Path to destination file or directory.")
    parser.add_argument("--plot", default=False, action='store_true', help="Plot precision and recall curve.")
    args = parser.parse_args()

    if os.path.isdir(args.Path):
        run_on_miltiple(args.Path)
        if args.plot:
            calculations = sorted(calculations, key=lambda x: x[0])
            p = []
            r = []
            for (prediction, recall) in calculations:
                p.append(prediction)
                r.append(recall)

            plot_p_r(p, r)
    elif os.path.isfile(args.Path):
        run_on_one(args.Path)
    else:
        parser.error('File or directory does not exist.')
