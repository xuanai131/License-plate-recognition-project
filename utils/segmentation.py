import cv2 
import numpy as np
from tensorflow import keras


# path = '/home/xuanai/xuanai/AI/new/yolov7'
# model = keras.models.load_model(path + '/letter_weight2.h5')

def seperate_letter(img, classnames, model, path):
    (height, width) = img.shape[:2]

    gray_plate = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray_plate, 120, 255, cv2.THRESH_BINARY)[1]   #180
    # cv2.imshow("thresh", thresh)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]


    clone_plate = img.copy()
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area > height*width/4 and area < height*width*9/10:
            mask = np.zeros(gray_plate.shape, np.uint8)
            cv2.fillPoly(mask, pts =[cnt], color=(255,255,255))
            
            mask_clone = mask.copy()
            crop = cv2.bitwise_and(img, img, mask=mask_clone)
            # print(crop.shape)

            new_gray_plate = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # new_blurred_plate = cv2.GaussianBlur(new_gray_plate,(3, 3),0)
            new_thresh = cv2.threshold(new_gray_plate,120,255,cv2.THRESH_BINARY_INV)[1]
            new_cnts = cv2.findContours(new_thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            # print('new_cnts: ', len(new_cnts))
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            bbox = list(box)
            # print(bbox)
            def sortFunc1(e):
                return e[0]+e[1]
            bbox.sort(key=sortFunc1)
            # print(bbox)
            
            if bbox[1][1]>bbox[2][1]:
                temp = bbox[1]
                bbox[1]=bbox[2]
                bbox[2]=temp
            
            temp = bbox[2]
            bbox[2]=bbox[3]
            bbox[3]=temp
            box = np.array(bbox)
            pts1 = np.array(box, np.float32)
            # print(box)

            (h_crop, w_crop) = crop.shape[:2]
            l = (h_crop+w_crop)/2

            for i in range(4):
                cv2.circle(clone_plate, box[i], 3, (255, 255, 0), 3)
            # pts2 = np.array([[20, 20], [120, 20], [120,120], [20, 120]], np.float32)
            # pts2 = np.array([[10, 10], [l-10, 10], [l-10, l-20], [10, l-20]], np.float32)
            pts2 = np.array([[0, 0], [w_crop, 0], [w_crop, h_crop], [0, h_crop]], np.float32)
            # pts2 = np.array([[120,120], [120, 20], [20, 20], [20, 120]], np.float32)
            # print(type(pts1), type(pts2))
            M = cv2.getPerspectiveTransform(pts1,pts2)
            out = cv2.warpPerspective(crop,M,(clone_plate.shape[1], clone_plate.shape[0]),flags=cv2.INTER_LINEAR)
            new_gray_plate2 = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            # new_blurred_plate2 = cv2.GaussianBlur(new_gray_plate2,(3, 3),0)
            new_thresh2 = cv2.threshold(new_gray_plate2,180,255,cv2.THRESH_BINARY)[1]
            kernel = np.ones((3,3), np.uint8)
            new_thresh2 = cv2.dilate(new_thresh2, kernel, iterations=1)
            new_thresh2 = cv2.erode(new_thresh2, kernel, iterations=1)
            # new_cnts2 = cv2.findContourSs(new_thresh2.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            # print('new_cnts2: ', len(new_cnts2))

            contours, hierarchy= cv2.findContours(new_thresh2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # clone_plate = cv2.drawContours(out, contours, -1, (0, 0, 255), 2)    
            # contours = imutils.grab_contours(contours)
            # contours,boundingBoxes = sort_contours(contours)
            # print('contours: ', len(contours))
            (w1, h1) = new_thresh2.shape
            row1=[]
            row2=[]
            for i in range(len(contours)):
                area = cv2.contourArea(contours[i])
                # print(area, w1*h1)
                # print("area: ", area, height*width)
                if area > w1*h1/100 and area < w1*h1/10:
                    print(1)
                    # print("...........\n")
                    # mask = np.zeros(gray_plate.shape, np.uint8)
                    rect = cv2.minAreaRect(contours[i])
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    if box[1][1]<h1/4:
                        row1.append(box)
                    else:
                        row2.append(box)
                    # cv2.imwrite(f'/home/xuanai/xuanai/AI/new/yolov7/image_test/{i}.jpg', out[box[2][1]:box[0][1], box[0][0]:box[2][0]])
            
            def sortFunc(e):
                return e[2][0]
            row1.sort(key=sortFunc)
            # print(row1)
            row2.sort(key=sortFunc)
            boxes = row1+row2
            # print(len(boxes))
            # print(boxes)
            
            letter_text = ""
            letters = []
            for i in range(len(boxes)):
                x=[]
                y=[]
                box = boxes[i]
                for b in box:
                    x.append(b[0])
                    y.append(b[1])
                x.sort()
                y.sort()
                letter = out[y[0]:y[-1], x[0]:x[-1]]
                letter = cv2.cvtColor(letter, cv2.COLOR_BGR2GRAY)
                letter = cv2.resize(letter, (28, 28), interpolation = cv2.INTER_AREA)
                letters.append(letter)
                cv2.imwrite(path + f'/image_test/{i}.jpg', letter)

            letters = np.array(letters)
            y_pred = model.predict(letters)
            for i in y_pred:
                n = np.argmax(i)
                letter_text += classnames[n]

            print(letter_text)
    return letter_text