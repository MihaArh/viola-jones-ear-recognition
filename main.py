from time import sleep
import matplotlib.pyplot as plt

import cv2


def capture_image():
    cam = cv2.VideoCapture(1)
    sleep(1)
    s, img = cam.read()
    cam.release()
    if s:
        cv2.imwrite("data/mydata/camface.jpg", img)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return cv2.imread('data/mydata/camface.jpg')


if __name__ == '__main__':
    capture_image()
