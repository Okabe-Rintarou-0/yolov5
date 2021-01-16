import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, non_max_suppression, apply_classifier, scale_coords, \
    xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detectBox(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()

    # img is the imagine that is to be detected
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            boxNumber = len(det)
            if boxNumber:
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                classes = []
                possibilities = []
                box_x = []
                box_y = []
                for index in range(boxNumber):
                    classes.append(int(det[index, -1]))
                    box_x.append([int(det[index, 0]), int(det[index, 2])])
                    box_y.append([int(det[index, 1]), int(det[index, 3])])
                    possibilities.append(round(float(det[index, 4]), 2))
                # print('\nclasses:', classes)
                # print('possibilities', possibilities)
                # print('box_x', box_x)
                # print('box_y', box_y)
                # for index in range(boxNumber):
                upperImgIndex = 0
                upperY = sys.maxsize
                for index in range(boxNumber):
                    if box_y[index][0] < upperY:
                        upperY = box_y[index][0]
                        upperImgIndex = index

                pt1 = (box_x[upperImgIndex][0], box_y[upperImgIndex][0])
                pt2 = (box_x[upperImgIndex][1], box_y[upperImgIndex][1])
                cv2.rectangle(im0, pt1, pt2, [0, 0, 255], 1)
                # resized = cv2.resize(im0, (800, 800))
                cv2.imshow('img', im0)
                cv2.waitKey(50)

                print('\nDetect Results:')
                print('ClassIndex:', classes[upperImgIndex])
                print('ClassName:', names[classes[upperImgIndex]])
                print('Possibility:', possibilities[upperImgIndex])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp16/weights/best.pt',
                        help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_false', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    check_requirements()
    imgPath = r'C:\Users\92304\Documents\yolov5\data\images\img.jpg'
    img = cv2.imread(imgPath)
    # print(img.shape)
    # print(type(img))
    # print(cv2.split(img))
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detectBox()
                strip_optimizer(opt.weights)
        else:
            detectBox()
