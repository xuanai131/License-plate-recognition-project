import argparse
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
# from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
# from utils.segmentation import seperate_letter
from tensorflow import keras

absolute_path = '/home/xuanai/xuanai/AI/submit'
model_cnn = keras.models.load_model(absolute_path + '/letter_weight2.h5')

classnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']



def seperate_letter(img):
    letter_text = ""
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
            if len(boxes)!=0:
                for i in range(len(boxes)):
                    x=[]
                    y=[]
                    box = boxes[i]
                    for b in box:
                        x.append(b[0])
                        y.append(b[1])
                    x.sort()
                    y.sort()
                    try:
                        letter = out[y[0]:y[-1], x[0]:x[-1]]
                        letter = cv2.cvtColor(letter, cv2.COLOR_BGR2GRAY)
                        letter = cv2.resize(letter, (28, 28), interpolation = cv2.INTER_AREA)
                        letters.append(letter)
                        cv2.imwrite(absolute_path + f'/image_test/{i}.jpg', letter)
                    except:
                        print('.')

            letters = np.array(letters)
            if letters.shape[0] != 0:
                y_pred = model_cnn.predict(letters)
                for i in y_pred:
                    n = np.argmax(i)
                    letter_text += classnames[n]

            print(letter_text)
    return letter_text


i=0
def detect(save_img=False):
    absolute_path = '/home/xuanai/xuanai/AI/new/yolov7'
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    source = absolute_path + '/inference/videos/xe.mp4'
    # source = absolute_path + '/inference/images/xe1.jpg'
    # source = '1'
    weights = 'best.pt'

    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    cropped_img = None
    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    device = select_device('cpu')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        # print(pred)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # print("det: ", det)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                        cropped_img = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                        (hh, ww) = cropped_img.shape[:2]
                        # print(cropped_img.shape)
                        if hh!=0 and ww!=0:
                            i+=1
                            letter_text = seperate_letter(cropped_img)
                            if len(letter_text)>4:
                                corner = (int(xyxy[0]), int(xyxy[1]))
                                points = np.array([[corner[0], corner[1]-int(hh/6)], [corner[0]+ww, corner[1]-int(hh/6)], [corner[0]+ww, corner[1]+5], [corner[0], corner[1]+5]])
                                # print(points)
                                cv2.fillPoly(im0, pts=[points], color=(255, 255, 0))
                                cv2.putText(im0, letter_text, corner, cv2.FONT_HERSHEY_COMPLEX, 0.0053*ww, (255, 0, 0), 1)
                                cv2.imwrite(absolute_path + f'/image_test_2/{i}.jpg', cropped_img)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.35, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    #check_requirements(exclude=('pycocotools', 'thop'))
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
