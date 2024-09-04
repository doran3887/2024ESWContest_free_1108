import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import argparse
import time
from pathlib import Path
import cv2
import torch
import random
import numpy as np
import sys
sys.path.append('YOLOPv2')
from utils.utils import time_synchronized, select_device, increment_path, scale_coords, xyxy2xywh, non_max_suppression, split_for_trace_model, driving_area_mask, lane_line_mask, plot_one_box, show_seg_result, AverageMeter, LoadImages
sys.path.append('sort')
from sort import Sort  # 여기서 Sort 클래스를 명확히 import합니다.

# BDD100K 클래스 이름 리스트
bdd100k_classes = [
    'pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign'
]

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=r'YOLOPv2/data/weights/yolopv2.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='test3.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    return parser


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    # 객체마다 다른색깔의 박스 생성
    if color % 3 == 0:
        r = 0
        g = 0 
        b = random.randint(0, 255)
    elif color % 3 == 1:
        r = 0
        g = random.randint(0, 255) 
        b = 0
    else :
        r = random.randint(0, 255)
        g = 0 
        b = 0
    color = (r, g, b)
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # Font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # Filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def display_depth_map(color_image, depth_map, roi):
    # roi에 해당하는 부분만 depth_map을 표시
    (x1, y1), (x2, y2) = roi
    depth_map_resized = cv2.resize(depth_map, (color_image.shape[1], color_image.shape[0]))
    depth_map_resized = cv2.normalize(depth_map_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_map_colored = cv2.applyColorMap(depth_map_resized, cv2.COLORMAP_JET)
    
    mask = np.zeros_like(color_image, dtype=np.uint8)
    mask[y1:y2, x1:x2] = depth_map_colored[y1:y2, x1:x2]

    combined = cv2.addWeighted(color_image, 0.8, mask, 0.2, 0)

    # 중간 좌표 계산
    mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2

    # 중간 좌표의 depth 값을 정규화하여 표시
    depth_value = depth_map_resized[mid_y, mid_x]
    normalized_depth = (depth_value / 255.0) * 100  # 0에서 100 사이로 정규화

    # Depth 값을 박스에 표시
    label = f'Depth: {normalized_depth:.1f}'
    tf = 1  # Font thickness
    t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=tf)[0]
    label_y1 = max(mid_y - 10, 0)
    label_y2 = label_y1 - t_size[1] - 5
    label_x1 = max(mid_x - t_size[0] // 2, 0)
    label_x2 = label_x1 + t_size[0] + 5
    cv2.rectangle(combined, (label_x1, label_y1), (label_x2, label_y2), (255, 255, 255), -1)
    cv2.putText(combined, label, (label_x1, label_y1 - 2), 0, 0.5, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

    cv2.rectangle(combined, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return combined


def detect(opt):
    source, weights, imgsz = opt.source, opt.weights, opt.img_size
    view_img = True  # Always view image in real-time processing

    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()

    device = select_device(opt.device)

    # Load yolo model
    stride = 32
    model_yolo = torch.jit.load(weights)  # Load yolo model
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model_yolo = model_yolo.to(device)

    # Load metric_3D model
    model_metric = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
    model_metric = model_metric.to(device)
    model_metric.eval()

    if half:
        model_yolo.half() # to FP16
        # model_metric.half()  
    model_yolo.eval()

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Initialize SORT tracker
    mot_tracker = Sort()

    # Run inference
    if device.type != 'cpu':
        model_yolo(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model_yolo.parameters())))  # run once

    t0 = time.time()
    frame_count = 0
    for path, img, im0s, vid_cap in dataset:
        frame_count += 1
        if frame_count % 6 != 0:
            continue  # Skip every frame except every third frame
        
        color_map = img

        # Metric용 이미지 처리
        resized_frame = cv2.resize(im0s, (512, 512))
        resized_frame = resized_frame / 255.0  # Normalize to [0, 1]
        img_metric = torch.tensor(resized_frame).float().permute(2, 0, 1).unsqueeze(0).to(device)

        # YOLO와 Metric에 쓰일 이미지 공통 처리
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        # Metric 처리
        with torch.no_grad():
            pred_depth, confidence, output_dict = model_metric.inference({'input': img_metric})

        depth_map = pred_depth.squeeze().cpu().numpy()

        # YOLO용 이미지 처리 
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference YOLO
        t1 = time_synchronized()
        [pred, anchor_grid], seg, ll = model_yolo(img)
        t2 = time_synchronized()

        # Waste time: the incompatibility of torch.jit.trace causes extra time consumption in demo version
        # but this problem will not appear in official version 
        tw1 = time_synchronized()
        pred = split_for_trace_model(pred, anchor_grid)
        tw2 = time_synchronized()

        # Apply NMS
        t3 = time_synchronized()
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t4 = time_synchronized()

        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)

        # Get pixel values for lane and driving area
        lane_pixels = np.sum(ll_seg_mask)
        driving_area_pixels = np.sum(da_seg_mask)

        detections = []
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Prepare detections for SORT
                for *xyxy, conf, cls in det:
                    xyxy = torch.tensor(xyxy).cpu().numpy()  # GPU에서 CPU로 복사
                    conf = conf.cpu().numpy()  # GPU에서 CPU로 복사
                    cls = cls.cpu().numpy()  # GPU에서 CPU로 복사
                    detections.append([*xyxy, conf, cls])

                # Convert detections to numpy array
                detections = np.array(detections)
                # print(f'Detections: {detections}')  # 디버깅을 위해 추가

                # Update SORT tracker
                track_bbs_ids = mot_tracker.update(detections[:, :5])  # 클래스 ID 제외하고 트래커 업데이트
                # print(f'Track BBS IDs: {track_bbs_ids}')  # 디버깅을 위해 추가

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # print(f"Detected class {int(c)}: {bdd100k_classes[int(c)]}, Count: {n}")
                
                obj_img = im0
                depth_img = im0
                # Write results
                # 여기부분 현재 코드 짜기 쉽게 두 이미지를 보이게 함
                # 나중에 원본하고 처리하는 이미지만 봐도 될 듯
                for track_bb in track_bbs_ids:
                    xyxy = track_bb[:4].astype(int)
                    track_id = int(track_bb[4])
                    cls_index = np.where(np.all(np.isclose(detections[:, :4], xyxy, atol=100), axis=1))[0]
                    depth_roi = obj_roi(xyxy)
                    if cls_index.size > 0:
                        cls = int(detections[cls_index[0], 5])  # matching cls with tolerance
                    else:
                        cls = None
                    
                    if cls is not None and cls < len(bdd100k_classes):  # cls 값이 리스트 범위 내에 있는지 확인
                        label = f'{bdd100k_classes[cls]} {track_id}'
                        color = track_id                        
                        plot_one_box(xyxy, obj_img, color=color, label=label, line_thickness=3) 
                        plot_one_box(xyxy, depth_img, color=color, label=label, line_thickness=3)
                    else:
                        # 기본 라벨로 경계 상자를 그립니다.
                        plot_one_box(xyxy, obj_img, line_thickness=3)  
                        plot_one_box(xyxy, depth_img, line_thickness=3)                      
                    depth_img = display_depth_map(depth_img, depth_map, depth_roi) 

            # Show results
            show_seg_result(im0, (da_seg_mask, ll_seg_mask), is_demo=True)
            cv2.imshow('obj_img', obj_img)
            cv2.imshow('depth_img', depth_img)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration

        inf_time.update(t2 - t1, img.size(0))
        nms_time.update(t4 - t3, img.size(0))
        waste_time.update(tw2 - tw1, img.size(0))
    
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg, nms_time.avg))
    print(f'Done. ({time.time() - t0:.3f}s)')
    

def obj_roi(xyxy):
    depth_roi = [(int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))]
    return depth_roi

    

if __name__ == '__main__':
    opt = make_parser().parse_args()

    # print(opt)

    with torch.no_grad():
        detect(opt)
