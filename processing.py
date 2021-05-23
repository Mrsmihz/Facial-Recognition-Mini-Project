import cv2
import numpy as np

def defaultImg():
    img = cv2.imread("./datasets/handcraft_train/human_face/64624.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray', img)
    # cv2.imwrite('./../SLIDE/openCanny.png', img)
    cv2.waitKey(0)
    # (thresh, img) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    return img

def canny(img):
    canny_low = 15
    canny_high = 150
    img = cv2.Canny(img, canny_low, canny_high)
    return img

def open(img):
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return img

def close(img):
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img

def erosion(img):
    kernel = np.ones((5, 5), np.uint8)
    erosed = cv2.erode(img, None, kernel, iterations=1)
    return erosed

def dilation(img):
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(img, None, kernel, iterations=1)
    return dilated

def onlyCanny():
    img = defaultImg()

    #Canny
    img = canny(img)
    cv2.imshow('mask', img)
    cv2.waitKey(0)
    cv2.imwrite('./../SLIDE/OnlyCanny.png', img)

def cannyWithOpen():
    img = defaultImg()

    #Open
    img = open(img)
    cv2.imshow('Open', img)
    cv2.waitKey(0)
    cv2.imwrite('./../SLIDE/open.png', img)


    #Canny
    img = canny(img)
    cv2.imshow('Canny', img)
    cv2.waitKey(0)
    cv2.imwrite('./../SLIDE/openCanny.png', img)

def cannyWithClose():
    img = defaultImg()

    #close
    img = close(img)
    cv2.imshow('Close', img)
    cv2.waitKey(0)
    cv2.imwrite('./../SLIDE/close.png', img)


    #Canny
    img = canny(img)
    cv2.imshow('Canny', img)
    cv2.waitKey(0)
    cv2.imwrite('./../SLIDE/closeCanny.png', img)

def cannyWithDialte():
    img = defaultImg()

    #Dilate
    img = dilation(img)
    cv2.imshow('Dilate', img)
    cv2.waitKey(0)
    cv2.imwrite('./../SLIDE/dilation.png', img)


    #Canny
    img = canny(img)
    cv2.imshow('Canny', img)
    cv2.waitKey(0)
    cv2.imwrite('./../SLIDE/dilationCanny.png', img)

def cannyWithErose():
    img = defaultImg()

    #Erose
    img = erosion(img)
    cv2.imshow('Erose', img)
    cv2.waitKey(0)
    cv2.imwrite('./../SLIDE/erosion.png', img)


    #Canny
    img = canny(img)
    cv2.imshow('Canny', img)
    cv2.waitKey(0)
    cv2.imwrite('./../SLIDE/erosionCanny.png', img)

# onlyCanny()
# cannyWithOpen()
# cannyWithClose()
# cannyWithErose()
# cannyWithDialte()
# defaultImg()