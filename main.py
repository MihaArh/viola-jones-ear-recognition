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
    fig.suptitle("Viola-Jones ear detection", fontsize=16)

    for ix, image in enumerate(images):
        ax = plt.subplot("11" + str(ix + 1))
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


def viola_jones(img, gray):
    left_ear_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_leftear.xml')
    left_ears = left_ear_cascade.detectMultiScale(gray, params["left"]["scale_step"], params["left"]["size"],
                                                  minSize=(20, 20), maxSize=(100, 100))

    for predicted_coords in left_ears:
        (x, y, w, h) = predicted_coords
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, 'Left ear', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    right_ear_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_rightear.xml')
    right_ears = right_ear_cascade.detectMultiScale(gray, params["right"]["scale_step"],
                                                    params["right"]["size"],
                                                    minSize=(20, 20), maxSize=(100, 100))
    for predicted_coords in right_ears:
        (x, y, w, h) = predicted_coords
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, 'Right ear', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)


def read_image(path):
    filename = path.split("/")[-1]
    img = cv2.imread(path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return filename, img, gray


params = {
    "left": {
        "scale_step": 1.005,
        "size": 3
    },
    "right": {
        "scale_step": 1.005,
        "size": 3
    },
}

if __name__ == '__main__':
    path = "data/mydata/camface.jpg"
    # capture_image()
    (filename, img, gray) = read_image(path)
    viola_jones(img, gray)
    find_nose(img, gray)
    save_plot(img, filename)
