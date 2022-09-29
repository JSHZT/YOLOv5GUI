import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (check_file, check_img_size, check_imshow, non_max_suppression, scale_coords)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from utils.augmentations import letterbox
import numpy as np

@torch.no_grad()

class PredictModel():
    def __init__(self, weights_path, mainwin):
        self.weights_path = weights_path
        self.source = ''
        self.device = select_device('')
        # self.save_img=save_img
        # self.save_path='D:/'
        self.imgsz=(640,640)  # inference size (pixels)
        # self.imgsz=[640,640],  # inference size (pixels)
        self.conf_thres=0.25  # confidence threshold
        self.iou_thres=0.45  # NMS IOU threshold
        self.max_det=1000  # maximum detections per image
        self.view_img=False  # show results
        # 保留特定的类
        self.classes=None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=False  # class-agnostic NMS
        self.augment=False  # augmented inference
        self.visualize=False  # visualize features
        self.update=False  # update all models
        self.line_thickness=3  # bounding box thickness (pixels)
        self.hide_labels=False  # hide labels
        self.hide_conf=False  # hide confidences
        # 是否使用半精度 Float16 推理 可以缩短推理时间 但是默认是False
        self.half=False  # use FP16 half-precision inference
        self.dnn = False
        # self.model_safebelt = self.modelLoad(self.weights_safebelt_path)
        # self.model_clothes_helmet = self.modelLoad(self.weights_clothes_helmet_path)
        self.mainwin = mainwin

        self.predict_info_show = ''

    def modelLoad(self, weightst_path):
        self.weights_path = weightst_path
        # print(type(device))
        # self.device = select_device(self.device) # 获取当前主机可用设备
        self.model = DetectMultiBackend(self.weights_path, device=self.device, dnn=self.dnn)
        # print("载入完毕")
        pt, engine =self.model.pt, self.model.engine
        self.half = self.device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            self.model.model.half() if self.half else self.model.model.float()
        self.model.warmup(imgsz=(1, 3, *(self.imgsz)), half=self.half)  # warmup(热身)

    def displayImg_out(self, img):
        RGBImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        img_out = QImage(RGBImg, RGBImg.shape[1], RGBImg.shape[0], QImage.Format_RGBA8888)
        img_out = QPixmap(img_out)
        # img_out = img_out.scaledToWidth(self.mainwin.labelsize[1])
        img_out = self.mainwin.resizeImg(img_out)
        self.mainwin.label_out.setPixmap(img_out)
    def displayImg_in(self, img):
        RGBImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        img_out = QImage(RGBImg, RGBImg.shape[1], RGBImg.shape[0], QImage.Format_RGBA8888)
        img_out = QPixmap(img_out)
        # img_out = img_out.scaledToWidth(self.mainwin.labelsize[1])
        img_out = self.mainwin.resizeImg(img_out)
        self.mainwin.label_in.setPixmap(img_out)

    # def display_result(self):
    #     # print(len(self.head_img))
    #     if len(self.head_img) == 0:
    #         return
    #     results = self.head_img
    #     for i, img_head in enumerate(results):
    #         self.result_info_show = self.result_info[i]
    #         cv2.imwrite(self.mainwin.violation_path + '/' + self.result_info_show.split(' ')[0] + '.jpg', img_head)
    #         RGBImg = cv2.cvtColor(img_head, cv2.COLOR_BGR2RGBA)
    #         img_out = QImage(RGBImg, RGBImg.shape[1], RGBImg.shape[0], QImage.Format_RGBA8888)
    #         img_out = QPixmap(img_out)
    #         # img_out = img_out.scaledToWidth(self.mainwin.labelsize[1])
    #         img_out = self.mainwin.resizeImg(img_out)
    #         self.mainwin.label_result.setPixmap(img_out)
    #         # self.mainwin.plainTextEdit_result.setPlainText(self.result_info[i])
    #         cv2.waitKey(1000)
    #     self.head_img = []  # 清空列表
    #     self.result_info = []
    #     self.get_head = True

    def displayInfo(self, info):
        # self.mainwin.predict_info_plainTextEdit.appendPlainText(info)
        self.predict_info_show = info

    def run(self, source, save_img, save_path):
        self.save_img = save_img
        self.save_path = save_path
        self.source = source
        self.source = str(self.source)
        is_file = Path(self.source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = self.source.isnumeric() or self.source.endswith('.txt') or (is_url and not is_file)
        time_now = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        if save_img:
            if is_file:
                # print "it's a directory"
                suf = self.source.split('.')[-1].lower()
                fname = '/' + time_now + '.' + suf
                self.save_path += fname
            else:
                fname ='/' + time_now + '.mp4'
                self.save_path += fname
                # print(fname)
            if is_url and is_file:
                self.source = check_file(self.source)  # download
        stride, names, pt, jit = self.model.stride, self.model.names, self.model.pt, self.model.jit
        # 确保用户设定的输入图片分辨率能整除32(如不能则调整为能整除并返回)
        self.imgsz = check_img_size(self.imgsz, s=stride)  # check image size

        # Dataloader
        if webcam:
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(self.source, img_size=self.imgsz, stride=stride, auto=pt and not jit)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(self.source, img_size=self.imgsz, stride=stride, auto=pt and not jit)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        dt, seen = [0.0, 0.0, 0.0], 0

        for path, im, im0s, vid_cap, s in dataset:
            # print('.......................')
            if self.mainwin.stop == 1:
                # if vid_cap != None:
                #     vid_cap.release()
                # if dataset.mode == 'stream':
                #     dataset.destory()
                break
            t1 = time_sync()
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            # 没有batch_size的话则在最前面添加一个轴
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference 向前推理
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False

            pred = self.model(im, augment=self.augment, visualize=self.visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
            dt[2] += time_sync() - t3

            # 每个batch中的每幅图
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                self.displayImg_in(im0)

                p = Path(p)  # to Path
                # save_path = str(save_dir / p.name)  # im.jpg
                s += '%gx%g ' % im.shape[2:]  # print string # 设置打印信息(图片长宽)
                annotator = Annotator(im0, line_width=self.line_thickness, example=str(names))
                if len(det):
                    # 调整预测框的坐标：基于resize+pad的图片的坐标-->基于原size图片的坐标
                    # 此时坐标格式为xyxy
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    # 打印检测到的类别数量
                    for c in det[:, -1].unique(): # unique() 删除重复元素并按序排列
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    for *xyxy, conf, cls in det:
                        # print(det[1,:])
                        # if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if self.hide_labels else (names[c] if self.hide_conf else f'{names[c]} {conf:.2f}')
                        # xmin, ymin, xmax, ymax, cls
                        annotator.box_label(xyxy, label, color=colors(c, True))

                # Stream results
                self.displayInfo(f'{s}Done. ({t3 - t2:.3f}s)')

                im0 = annotator.result()
                self.displayImg_out(im0)
                # Save results (image with detections)
                if self.save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(self.save_path, im0)
                        # print(self.save_path)
                    else:  # 'video' or 'stream'
                        # print('save video')
                        if vid_path[i] != self.save_path:  # new video
                            vid_path[i] = self.save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                # save_path += '.mp4'
                            vid_writer[i] = cv2.VideoWriter(self.save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)
                        # print(save_path)

        if self.save_img:
            # print(f"Results saved to {self.save_path}")
            self.displayInfo(f"Results saved to {self.save_path}")

    def loadimg(self, img0, img_size, stride, auto):
        img = letterbox(img0, img_size, stride, auto)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        return img

    def run_img(self, im0, model, imgsz, save_img, save_path='D:/1.jpg'):
        imgsz *= 2 if len(imgsz) == 1 else 1  # expand
        # stride: 下采样率8, 16, 32; 
        # names: 所有类的类名; 
        stride, names, pt, jit = model.stride, model.names, model.pt, model.jit
        # 确保用户设定的输入图片分辨率能整除32(如不能则调整为能整除并返回)
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        # Dataloader
        im = self.loadimg(im0, img_size=imgsz, stride=stride, auto=pt and not jit)

        dt, seen = [0.0, 0.0, 0.0], 0

        t1 = time_sync()
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        # 没有batch_size的话则在最前面添加一个轴
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference 向前推理
        visualize =  False# increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=self.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        dt[2] += time_sync() - t3

        xy_cls = torch.zeros(1, 4)   # xmin, ymin, xmax, ymax
        xy_cls_t = torch.zeros(1, 4)
        cls_name = []
        for i, det in enumerate(pred):  # per image
            seen += 1
            im0= im0.copy()
            # s += '%gx%g ' % im.shape[2:]  # print string # 设置打印信息(图片长宽)
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(names))
            if len(det):
                # 调整预测框的坐标：基于resize+pad的图片的坐标-->基于原size图片的坐标
                # 此时坐标格式为xyxy
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                # 保存预测信息: txt、im0上画框、crop_img
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if self.hide_labels else (names[c] if self.hide_conf else f'{names[c]} {conf:.2f}')
                    # xmin, ymin, xmax, ymax, cls
                    xy_cls_t[0, 0] = xyxy[0]
                    xy_cls_t[0, 1] = xyxy[1]
                    xy_cls_t[0, 2] = xyxy[2]
                    xy_cls_t[0, 3] = xyxy[3]
                    xy_cls = torch.cat([xy_cls, xy_cls_t], dim = 0)
                    cls_name.append(label)
                    # print("xmin:%d, ymin:%d, xmax:%d, ymax:%d, class:%s" % (xyxy[0], xyxy[1], xyxy[2], xyxy[3], label))
                    # if save_img or save_crop:  # Add bbox to image
                    annotator.box_label(xyxy, label, color=colors(c, True))

            im0 = annotator.result()

            # Save results (image with detections)
            if save_img:
                cv2.imwrite(save_path, im0)
        # Print results
        # t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        return xy_cls ,cls_name

    def compute_iou(self, class1, class2,p = 0.3):
        x_min = max(class1[0],class2[0])
        y_min = max(class1[1],class2[1])
        x_max = min(class1[2],class2[2])
        y_max = min(class1[3],class2[3])
        # print(x_max-x_min,y_max-y_min)
        # 负值直接为0
        if (x_max-x_min) <= 0 or (y_max-y_min) <= 0:
            return False
        
        n = (x_max-x_min) * (y_max-y_min)
        u = (class2[2] - class2[0]) * (class2[3] - class2[1]) + (class1[2] - class1[0]) * (class1[3] - class1[1]) - n
        iou = n/u
        # print(iou)
        if iou >= p:
            return True
        else:
            return False

# if __name__ == "__main__":
#     device = '0'
#     weights_clothes_helmet=r'D:\A_BiShe\yolov5-master\runs\train\test_clothes_helmet\weights\best.pt'
#     weights_safebelt=r'D:\A_BiShe\yolov5-master\runs\train\test_safebelt3\weights\best.pt'
#     source = r'D:\test.jpg'
#     # source = '0'
#     run(weights_safebelt,
#         weights_clothes_helmet,
#         device,
#         source,
#         save_img=True,
#         save_path = 'D:/',
#         view_img=False
#         )