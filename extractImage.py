from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
import cv2
import os

imageName = list(filter(lambda file: file[-3:].lower() == 'jpg', os.listdir()))
columnDir = 'splitColumn'
resultsDir = 'results'
for image in imageName:
    print(image)
    img = cv2.imread(image)
    (H, W) = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray,(3,3),0)
    ret, thes = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    colList = list()
    colList.append(thes)

    # Skew correction for each column
    for i, col in enumerate(colList):
        gray = cv2.bitwise_not(col)
        pts = cv2.findNonZero(gray)
        ret = cv2.minAreaRect(pts)

        (cx,cy), (w,h), angle = ret
        if angle < -45:
        	angle = -(90 + angle)
        else:
        	angle = -angle

        M = cv2.getRotationMatrix2D((cx,cy), angle, 1.0)
        rotated = cv2.warpAffine(col, M, (col.shape[1], col.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        colList[i] = rotated

    # Split line
    lineCrop = []
    colIndex = 0
    for eachCol in colList:
        dst = cv2.erode(eachCol, kernel=np.ones((1, 30)))
        hist = cv2.reduce(dst,1, cv2.REDUCE_AVG).reshape(-1)

        th = 230
        h, w = eachCol.shape[:2]
        uppers = [y for y in range(h-1) if hist[y]>th and hist[y+1]<=th]
        uppers.append(h)

        for i in range(len(uppers) - 1):
            expand = 5
            if (uppers[i+1] - uppers[i] < 15):
                continue
            crop = eachCol[uppers[i] - expand:uppers[i+1], 0:w]
            lineCrop.append(crop)

            # cv2.line(eachCol, (0,uppers[i] - expand), (w, uppers[i+1] - expand), (0,255,0), 3)

            # cv2.imshow('image', crop)
            # cv2.waitKey(0)

    # Skew correction for each line
    for i, line in enumerate(lineCrop):
        gray = cv2.bitwise_not(line)
        pts = cv2.findNonZero(gray)
        ret = cv2.minAreaRect(pts)

        (cx,cy), (w,h), ang = ret
        ang += 90
        if w>h:
            w,h = h,w
            ang -= 90

        M = cv2.getRotationMatrix2D((cx,cy), ang, 1.0)
        rotated = cv2.warpAffine(line, M, (line.shape[1], line.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        lineCrop[i] = rotated

        cv2.imshow('image', line)
        cv2.waitKey(0)


    config = ("-l vie --oem 1 --psm 7")
    outputText = []
    for crop in lineCrop:
        outputText.append(pytesseract.image_to_string(crop, config=config))
        # print(pytesseract.image_to_string(crop, config=config))
        # outputText.append(pytesseract.image_to_data(crop, config=config))
        # a = pytesseract.image_to_pdf_or_hocr(crop, extension='hocr', config=config)
        # with open("result.xml", "wb") as wf:
        #     wf.write(a)

    # for text in outputText:
    #     print(text)
    with open(image[0:-3] + 'txt', 'w', encoding='utf8') as f:
        for text in outputText:
            f.write(text + '\n')
