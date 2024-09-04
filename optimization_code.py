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
from sort import Sort  

# BDD100K 클래스 이름 리스트
bdd100k_classes = [
    'pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign'
]

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=r'YOLOPv2/data/weights/yolopv2.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='test1.mp4', help='source')  # file/folder, 0 for webcam
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
    parser.add_argument('--max-objects', type=int, default=10, help='maximum number of objects to detect per frame')
    parser.add_argument('--min-move-threshold', type=int, default=10, help='minimum movement threshold for updating path')
    return parser

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # Font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # Filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def display_depth_map(color_image, depth_map, roi):
    (x1, y1), (x2, y2) = roi
    depth_map_resized = cv2.resize(depth_map, (color_image.shape[1], color_image.shape[0]))
    depth_map_resized = cv2.normalize(depth_map_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_map_colored = cv2.applyColorMap(depth_map_resized, cv2.COLORMAP_JET)
    
    mask = np.zeros_like(color_image, dtype=np.uint8)
    mask[y1:y2, x1:x2] = depth_map_colored[y1:y2, x1:x2]

    combined = cv2.addWeighted(color_image, 0.8, mask, 0.2, 0)

    mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
    depth_value = depth_map_resized[mid_y, mid_x]
    normalized_depth = (depth_value / 255.0) * 100

    label = f'Depth: {normalized_depth:.1f}'
    tf = 1
    t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=tf)[0]
    label_y1 = max(mid_y - 10, 0)
    label_y2 = label_y1 - t_size[1] - 5
    label_x1 = max(mid_x - t_size[0] // 2, 0)
    label_x2 = label_x1 + t_size[0] + 5
    cv2.rectangle(combined, (label_x1, label_y1), (label_x2, label_y2), (255, 255, 255), -1)
    cv2.putText(combined, label, (label_x1, label_y1 - 2), 0, 0.5, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

    cv2.rectangle(combined, (x1, y1), (x2, y2), (255, 0, 0), 2)

    center_bottom_x = color_image.shape[1] // 2
    center_bottom_y = color_image.shape[0] - 1
    cv2.circle(combined, (center_bottom_x, center_bottom_y), 10, (0, 255, 0), -1)

    bottom_depth_value = depth_map_resized[center_bottom_y, center_bottom_x]
    normalized_bottom_depth = (bottom_depth_value / 255.0) * 100

    bottom_label = f'Bottom Depth: {normalized_bottom_depth:.1f}'
    bottom_label_size = cv2.getTextSize(bottom_label, 0, fontScale=0.5, thickness=1)[0]
    bottom_label_x1 = center_bottom_x - bottom_label_size[0] // 2
    bottom_label_y1 = center_bottom_y - 15
    bottom_label_x2 = bottom_label_x1 + bottom_label_size[0] + 5
    bottom_label_y2 = bottom_label_y1 - bottom_label_size[1] - 5
    cv2.rectangle(combined, (bottom_label_x1, bottom_label_y1), (bottom_label_x2, bottom_label_y2), (255, 255, 255), -1)
    cv2.putText(combined, bottom_label, (bottom_label_x1, bottom_label_y1 - 2), 0, 0.5, [0, 0, 0], thickness=1, lineType=cv2.LINE_AA)

    return combined

def get_depth(color_image, depth_map, roi):
    (x1, y1), (x2, y2) = roi
    depth_map_resized = cv2.resize(depth_map, (color_image.shape[1], color_image.shape[0]))
    depth_map_resized = cv2.normalize(depth_map_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
    depth_value = depth_map_resized[mid_y, mid_x]
    normalized_depth = (depth_value / 255.0) * 100
    return normalized_depth

class TrackInfo:
    def __init__(self):
        self.tracks = {}
        self.track_statuses = {}  # 상태 변경을 위한 필드

    def update_track(self, track_id, center, current_depth, min_move_threshold):
        if track_id in self.tracks:
            last_center = self.tracks[track_id]['last_center']
            last_depth = self.tracks[track_id]['last_depth']
            movement = np.sqrt((center[0] - last_center[0]) ** 2 + (center[1] - last_center[1]) ** 2)
            if movement > min_move_threshold:
                path = self.tracks[track_id]['path']
                path.append(center)
                if len(path) > 25:
                    path.pop(0)
                smooth_center = np.mean(path[-3:], axis=0).astype(int)
                self.tracks[track_id]['smoothed_path'].append(tuple(smooth_center))
                if len(self.tracks[track_id]['smoothed_path']) > 50:
                    self.tracks[track_id]['smoothed_path'].pop(0)
            self.tracks[track_id]['last_center'] = center
            self.tracks[track_id]['last_depth'] = current_depth
        else:
            self.tracks[track_id] = {'last_center': center, 'path': [center], 'smoothed_path': [center], 'last_depth': current_depth}
            self.track_statuses[track_id] = {'True': 0, 'False': 0, 'status': False}  # 새로운 트랙 추가 시 상태 초기화

    def get_path(self, track_id):
        if track_id in self.tracks:
            return self.tracks[track_id]['smoothed_path']
        return None

    def get_last_two_points(self, track_id):
        if track_id in self.tracks and len(self.tracks[track_id]['smoothed_path']) > 1:
            return self.tracks[track_id]['smoothed_path'][-2:]
        return None

    def get_last_depth(self, track_id):
        if track_id in self.tracks:
            return self.tracks[track_id]['last_depth']
        return None

    def clear_track(self, track_id):
        if track_id in self.tracks:
            del self.tracks[track_id]
        if track_id in self.track_statuses:
            del self.track_statuses[track_id]

    def update_status(self, track_id, current_status, required_consistency=5):
        if track_id in self.track_statuses:
            status_info = self.track_statuses[track_id]
            if current_status == True:
                status_info['True'] += 1
                status_info['False'] = 0
            else:
                status_info['False'] += 1
                status_info['True'] = 0

            if status_info['True'] >= required_consistency:
                status_info['status'] = True
            elif status_info['False'] >= required_consistency:
                status_info['status'] = False

            print(status_info)

        return status_info['status']

       # return current_status


def define_obstacle(last_two_points=None, center_point=None, last_depth=None, current_depth=None):
    if last_two_points is None:
        return current_depth < last_depth
    (x1, y1), (x2, y2) = last_two_points
    direction_vector = np.array([x2 - x1, y2 - y1])
    center_vector = np.array([center_point[0] - x1, center_point[1] - y1])
    direction_magnitude = np.linalg.norm(direction_vector)
    center_magnitude = np.linalg.norm(center_vector)
    dot_product = np.dot(direction_vector, center_vector)
    cos_theta = dot_product / (direction_magnitude * center_magnitude)
    # 내적으로 판단, 따라서 각의 오차를 코사인 값으로 조정 가능 
    return cos_theta > 0.5 and current_depth < last_depth


def detect(opt):
    source, weights, imgsz = opt.source, opt.weights, opt.img_size
    view_img = True

    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()

    device = select_device(opt.device)

    stride = 32
    model_yolo = torch.jit.load(weights)
    half = device.type != 'cpu'
    model_yolo = model_yolo.to(device)

    model_metric = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
    model_metric = model_metric.to(device)
    model_metric.eval()

    if half:
        model_yolo.half()
    model_yolo.eval()

    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    mot_tracker = Sort()
    track_info = TrackInfo()

    if device.type != 'cpu':
        model_yolo(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model_yolo.parameters())))

    t0 = time.time()
    frame_count = 0
    for path, img, im0s, vid_cap in dataset:
        frame_count += 1
        if frame_count % 5 != 0:
            continue
        
        color_map = img
        resized_frame = cv2.resize(im0s, (512, 512))
        resized_frame = resized_frame / 255.0
        img_metric = torch.tensor(resized_frame).float().permute(2, 0, 1).unsqueeze(0).to(device)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0

        with torch.no_grad():
            pred_depth, confidence, output_dict = model_metric.inference({'input': img_metric})

        depth_map = pred_depth.squeeze().cpu().numpy()

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()
        [pred, anchor_grid], seg, ll = model_yolo(img)
        t2 = time_synchronized()

        tw1 = time_synchronized()
        pred = split_for_trace_model(pred, anchor_grid)
        tw2 = time_synchronized()

        t3 = time_synchronized()
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t4 = time_synchronized()

        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)

        lane_pixels = np.sum(ll_seg_mask)
        driving_area_pixels = np.sum(da_seg_mask)

        detections = []
        for i, det in enumerate(pred):
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)
            s += '%gx%g ' % img.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                if len(det) > opt.max_objects:
                    det = det[:opt.max_objects]
                for *xyxy, conf, cls in det:
                    xyxy = torch.tensor(xyxy).cpu().numpy()
                    conf = conf.cpu().numpy()
                    cls = cls.cpu().numpy()
                    detections.append([*xyxy, conf, cls])
                detections = np.array(detections)
                # 여기까지가 객체 검출

                # 여기 부터 객체 추적
                track_bbs_ids = mot_tracker.update(detections[:, :5])
                print(track_bbs_ids)

                obj_img = im0
                depth_img = im0
                true_img = np.zeros_like(im0)  # 검정색 바탕 이미지
                for track_bb in track_bbs_ids:
                    xyxy = track_bb[:4].astype(int)
                    track_id = int(track_bb[4])
                    center = ((xyxy[0] + xyxy[2]) // 2, (xyxy[1] + xyxy[3]) // 2)

                    depth_roi = obj_roi(xyxy)
                    current_depth = get_depth(depth_img, depth_map, depth_roi)
                    last_depth = track_info.get_last_depth(track_id)
                    track_info.update_track(track_id, center, current_depth, opt.min_move_threshold)
                    #여기까지 객체 추적

                    cls_index = np.where(np.all(np.isclose(detections[:, :4], xyxy, atol=100), axis=1))[0]
                    if cls_index.size > 0:
                        cls = int(detections[cls_index[0], 5])
                    else:
                        cls = None

                    # 객체 상태 정의 & 시각화
                    color = track_id
                    if last_depth is not None:
                        last_two_points = track_info.get_last_two_points(track_id)
                        center_point = (obj_img.shape[1] // 2, obj_img.shape[0] // 2)
                        current_status = define_obstacle(last_two_points, center_point, last_depth, current_depth)
                        facing_center = track_info.update_status(track_id, current_status, required_consistency=3) # 여기 required_conistency 조절해서 TF 바뀌는 거 조절 가능
                        label = f'{bdd100k_classes[cls]} {track_id} {"True" if facing_center else "False"}'
                        if facing_center:  # True 상태인 객체만 표시
                            plot_one_box(xyxy, true_img, color=color, label=label, line_thickness=3)
                    else:
                        label = f'{bdd100k_classes[cls]} {track_id}'

                    plot_one_box(xyxy, obj_img, color=color, label=label, line_thickness=3)
                    plot_one_box(xyxy, depth_img, color=color, label=label, line_thickness=3)
                    depth_img = display_depth_map(depth_img, depth_map, depth_roi)

                    #객체 추적 경로 시각화
                    path = track_info.get_path(track_id)
                    if path and len(path) > 1:
                        for j in range(1, len(path)):
                            cv2.line(obj_img, path[j-1], path[j], color, 2)
                            cv2.line(depth_img, path[j-1], path[j], color, 2)
                            if facing_center:  # True 상태인 객체만 표시
                                cv2.line(true_img, path[j-1], path[j], color, 2)
                        cv2.arrowedLine(obj_img, path[-2], path[-1], color, 2, tipLength=0.5)
                        cv2.arrowedLine(depth_img, path[-2], path[-1], color, 2, tipLength=0.5)
                        if facing_center:  # True 상태인 객체만 표시
                            cv2.arrowedLine(true_img, path[-2], path[-1], color, 2, tipLength=0.5)
                # 활성화된 추적 ID 관리
                active_track_ids = {int(track_bb[4]) for track_bb in track_bbs_ids}
                for track_id in list(track_info.tracks.keys()):
                    if track_id not in active_track_ids:
                        track_info.clear_track(track_id)

            # 창 띄우기
            show_seg_result(im0, (da_seg_mask, ll_seg_mask), is_demo=True)
            cv2.imshow('obj_img', obj_img)
            cv2.imshow('depth_img', depth_img)
            cv2.imshow('true_img', true_img)  # True 상태 객체만 표시하는 창
            if cv2.waitKey(1) == ord('q'):
                raise StopIteration

        # Detection 끝부분



        inf_time.update(t2 - t1, img.size(0))
        nms_time.update(t4 - t3, img.size(0))
        waste_time.update(tw2 - tw1, img.size(0))
    
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg, nms_time.avg))
    print(f'Done. ({time.time() - t0:.3f}s)')


def obj_roi(xyxy):
    return [(int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))]

if __name__ == '__main__':
    opt = make_parser().parse_args()
    with torch.no_grad():
        detect(opt)
