# current_status 장애물 TF, obstacle_score 장애물 점수, depth_score  

import argparse
import datetime
import time
from pathlib import Path
import cv2
import torch
import random
import numpy as np
import sys
sys.path.append('sort')
from sort import Sort  
from ultralytics import YOLOv10


# 여러 인자들 정의(주석처리된 것은 아래에서 쓴 것)
weights = 'YOLOv10_model/best.pt'   # YOLO 가중치
max_objects = 20                   # 탐지되는 최대 객체 수
min_move_threshold = 10             # 객체의 경로 만들 때 저장되는 최소 움직임 (이보다 더 커야 경로로 저장됨)
conf_threshold = 0.5                # 객체의 감지 결과 가능성
# video_path = 'test1.mp4'

names = [
    'building',
    'wall',
    'fence',
    'pole',
    'person',
    'rider',
    'car',
    'truck',
    'bus',
    'train',
    'motorcycle',
    'bicycle'
]


# SORT 알고리즘용 Class
class TrackInfo:
    obstacle_state_log = {}
    def __init__(self):
        self.tracks = {}
        self.track_statuses = {}
        self.get_acc = {}

    def update_acc(self, track_id, speed):
        if track_id in self.get_acc:
            speed_log = self.get_acc[track_id]['speed']
            speed_log.append(speed)
            if len(speed_log) > 3:
                speed_log.pop(0)

            # acc는 speed_log의 마지막 두 값의 차이입니다.
            acc = speed_log[-2] - speed_log[-1]
            self.get_acc[track_id]['speed'] = speed_log
            self.get_acc[track_id]['acceleration'].append(acc)

            # 마지막 3개의 값의 평균을 구합니다.
            smooth_acc = np.mean(self.get_acc[track_id]['acceleration'][-3:]).astype(float)
            self.get_acc[track_id]['smoothed_acc'].append(smooth_acc)
            if len(self.get_acc[track_id]['smoothed_acc']) > 3:
                self.get_acc[track_id]['smoothed_acc'].pop(0)
        else:
            # 처음에 track_id가 없을 때, 기본값을 설정합니다.
            self.get_acc[track_id] = {'speed': [speed], 'acceleration': [0.0], 'smoothed_acc': [0.0]}
            acc = 0

        return self.get_acc[track_id]['smoothed_acc'][-1], acc

    
    def update_track(self, track_id, center, current_depth, min_move_threshold):
        if track_id in self.tracks:
            last_center = self.tracks[track_id]['last_center']
            depth_log = self.tracks[track_id]['depth']
            depth_log.append(current_depth)
            if len(depth_log) > 3:
                depth_log.pop(0)
            smooth_depth = np.mean(depth_log[-3:], axis=0).astype(float)
            self.tracks[track_id]['smoothed_depth'].append(smooth_depth)
            if len(self.tracks[track_id]['smoothed_depth']) > 3:
                self.tracks[track_id]['smoothed_depth'].pop(0)
            movement = np.sqrt((center[0] - last_center[0]) ** 2 + (center[1] - last_center[1]) ** 2)
            if movement > min_move_threshold:
                path = self.tracks[track_id]['path']
                path.append(center)
                if len(path) > 15:
                    path.pop(0)
                smooth_center = np.mean(path[-3:], axis=0).astype(int)
                self.tracks[track_id]['smoothed_path'].append(tuple(smooth_center))
                if len(self.tracks[track_id]['smoothed_path']) > 50:
                    self.tracks[track_id]['smoothed_path'].pop(0)
            self.tracks[track_id]['last_center'] = center
        else:
            self.tracks[track_id] = {'last_center': center, 'path': [center], 'smoothed_path': [center], 'depth': [current_depth], 'smoothed_depth': [current_depth]}
    
    def calculate_motion_vector(self, track_id):
        if track_id in self.tracks and len(self.tracks[track_id]['smoothed_path']) >= 2:
            smoothed_path = self.tracks[track_id]['smoothed_path']

            # Get the last two points
            point1 = np.array(smoothed_path[-2])
            point2 = np.array(smoothed_path[-1])

            # Calculate the motion vector (point2 - point1)
            motion_vector = point2 - point1

            return tuple(motion_vector)
        else:
            # If there are not enough points or the track_id does not exist
            return None

    def calculate_previous_motion_vector(self, track_id, center_point):
        if track_id in self.tracks and len(self.tracks[track_id]['smoothed_path']) >= 3:
            smoothed_path = self.tracks[track_id]['smoothed_path']
            
            center_point = center_point[:2]
            # Get the previous two points
            point1 = np.array(center_point)
            point2 = np.array(smoothed_path[-2])

            # Calculate the previous motion vector (point2 - point1)
            previous_motion_vector = point1 - point2

            return tuple(previous_motion_vector)
        else:
            # If there are not enough points or the track_id does not exist
            return None

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
            return self.tracks[track_id]['smoothed_depth']
        return None

    def get_last_3D_coordinate(self, track_id):
        if track_id in self.tracks:
            last_center = self.tracks[track_id]['last_center']
            last_depth = self.tracks[track_id]['smoothed_depth'][-1]
            x1, y1 = last_center[0], last_center[1]
            last_3D_coordinate = [x1, y1, last_depth]
            return last_3D_coordinate
        return None

    def clear_track(self, track_id):
        if track_id in self.tracks:
            del self.tracks[track_id]
        if track_id in self.track_statuses:
            del self.track_statuses[track_id]

    def update_status(self, track_id, current_status, current_depth, current_roi):
        if track_id in self.track_statuses:
            status_info = self.track_statuses[track_id]
            status_info['in_roi'] = current_roi

            if current_status == True:
                status_info['True'] += 1
                status_info['False'] = 0

            elif current_status == False:
                status_info['False'] += 1
                status_info['True'] = 0
            
            if status_info['True'] >= 2:
                status_info['status'] = True
                if status_info['in_roi']:
                    if current_depth <= 20:
                        status_info['status_buffer'] = True
                    else :
                        status_info['status_buffer'] = False
                else :
                    status_info['status_buffer'] = False
            elif status_info['False'] >= 7:
                if status_info['in_roi']:
                    if status_info['status_buffer'] == True:
                        status_info['status'] = True
                    else :
                        status_info['status'] = False
                        status_info['status_buffer'] = False
                        if current_depth > 15:
                            status_info['status_buffer'] = False
                else:
                    status_info['status'] = False
                    status_info['status_buffer'] = False

            self.obstacle_state_log[track_id] = {'status' : status_info['status'], 'status_buffer' : status_info['status_buffer']}
            return status_info['status']        
        else :
            if track_id not in self.obstacle_state_log:
                self.track_statuses[track_id] = {'True': 0, 'False': 0, 'status': False, 'status_buffer': None , 'in_roi' : False}  # 새로운 트랙 추가 시 상태 초기화
            else:
                state_log = self.obstacle_state_log[track_id]
                self.track_statuses[track_id] = {'True': 0, 'False': 0, 'status': state_log['status'], 'status_buffer': state_log['status_buffer'], 'in_roi' : False}

def get_depth(color_image, depth_map, center):
    '''
    객체의 depth 값을 가져오는 함수
    Parameters :
    color_image : 원본 이미지(프레임)
    depth_map : Metric_3D로 생성한 depth_map
    centere : 객체박스의 한 가운데
    '''
    mid_x, mid_y = center
    depth_map_resized = cv2.resize(depth_map, (color_image.shape[1], color_image.shape[0]))
    depth_map_resized = cv2.normalize(depth_map_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_value = depth_map_resized[mid_y, mid_x]
    normalized_depth = (depth_value / 255.0) * 100
    return normalized_depth

def plot_one_box(xyxy, img, color=None, label=None, line_thickness=3):
    '''
    객체의 경계 상자를 이미지에 그려주는 함수

    Parameters:
    xyxy : tuple 또는 list
        객체의 경계 상자를 나타내는 좌표로, 왼쪽 위와 오른쪽 아래의 (x, y) 좌표
        예: (x1, y1, x2, y2)
    img : numpy.ndarray
        경계 상자가 그려질 이미지.
    color : tuple 또는 list, optional
        경계 상자의 색상. (B, G, R) 형식의 색상 값을 입력받음. 지정하지 않으면 무작위 색상
    label : str, optional
        경계 상자에 표시할 텍스트 라벨. 지정하지 않으면 라벨은 표시되지 않음.
    line_thickness : int, optional
        경계 상자의 두께. 기본값은 3이며, 이미지 크기에 비례하여 두께가 자동으로 조정될 수 있음.

    Returns:
    None
    '''
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    모델에서 나온 좌표를 원본 이미지로 맞추는 함수

    Parameters
    coords: 모델에서 얻은 좌표 (x1, y1, x2, y2)
    img1_shape: 모델에서 사용하는 입력 이미지 크기
    img0_shape: 원본 이미지 크기
    """
    # Gain 계산 (원본 이미지 크기와 입력 이미지 크기 간의 비율)
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, img_shape):
    """
    경계 상자를 이미지 크기에 맞게 자르는 함수.

    Parameters:
    boxes : numpy.ndarray
        경계 상자 좌표. (x1, y1, x2, y2) 형식이어야 합니다.
    img_shape : tuple
        이미지의 크기 (height, width)입니다.
    """
    boxes[:, 0] = np.clip(boxes[:, 0], 0, img_shape[1])  # x1 좌표를 0에서 이미지 너비로 제한
    boxes[:, 1] = np.clip(boxes[:, 1], 0, img_shape[0])  # y1 좌표를 0에서 이미지 높이로 제한
    boxes[:, 2] = np.clip(boxes[:, 2], 0, img_shape[1])  # x2 좌표를 0에서 이미지 너비로 제한
    boxes[:, 3] = np.clip(boxes[:, 3], 0, img_shape[0])  # y2 좌표를 0에서 이미지 높이로 제한

def initialize_video_capture(video_input):
    if video_input == 0:
        video_input = int(video_input)
        cap = cv2.VideoCapture(video_input)
    else:
        cap = cv2.VideoCapture(video_input)
    
    if not cap.isOpened():
        raise ValueError("Unable to open video source")
    
    return cap

def preprocess_frame(img):
    resized_frame = cv2.resize(img, (512, 512))
    resized_frame = resized_frame / 255.0
    resized_frame = torch.tensor(resized_frame).float().permute(2, 0, 1).unsqueeze(0).to(torch.device("cuda"))
    return resized_frame

def proceess_frame_yolo(frame, model, class_names, colors):
    results = model(frame, verbose=False)[0]
    detections = [] 
    for det in results.boxes:
        label, confidence, bbox = det.cls, det.conf, det.xyxy[0]

        # GPU에서 CPU로 데이터를 이동시키고 numpy 배열로 변환
        label = int(label.cpu().numpy())
        confidence = float(confidence.cpu().numpy())  # float으로 변환
        bbox = bbox.cpu().numpy()

        x1, y1, x2, y2 = map(int, bbox)
        class_id = label

        if confidence < conf_threshold:
            continue

        # bbox의 구조를 [x1, y1, x2, y2]로 유지
        detections.append([x1, y1, x2, y2, confidence, class_id])
    
    # 모든 detections를 numpy 배열로 변환
    detections = np.array(detections)
    if len(detections) > 15:
        detections = detections[:15]  # max_objects 수만큼 자름
    return detections

def distance_two_point(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

def is_closer_to_roi(previous_point, current_point, points):
    x_values = [x for x, y in points]
    y_values = [y for x, y in points]
    x_roi = sum(x_values)
    y_roi = sum(y_values)    
    roi_center = x_roi/4, y_roi/4

    prev_distance = distance_two_point(previous_point, roi_center)
    curr_distance = distance_two_point(current_point, roi_center)
    speed = abs(curr_distance - prev_distance)
    # 거리가 줄어들면 다가가고 있는 것으로 판단
    closer_score = min(100, speed * 5)

    return curr_distance < prev_distance, closer_score


def will_cross_roi(last_coordinate=None, current_coordinate=None, points = None):
    last_coordinate = last_coordinate[:2]
    current_coordinate = current_coordinate[:2]
    if last_coordinate is not None and current_coordinate is not None:
        T_F, closer_score = is_closer_to_roi(last_coordinate, current_coordinate, points)
    else:
        T_F = False
        closer_score = -100
    
    return T_F, closer_score

# 특정 픽셀이 ROI(테두리인 Points)안에 있는지 판별하는 함수. 0볻다 크면 내부에, 0이면 경계, 음수면 외부에
def is_point_in_polygon(polygon_points, point):
    polygon = np.array(polygon_points, dtype=np.int32)
    distance = cv2.pointPolygonTest(polygon, point, False)
    return distance >= 0

# 장애물 판별 함수 장애물일 경우 True 반환(Bool)
# - 정확한 장애물 판단 기준을 정해서 개선해야함
def define_obstacle_3D(last_coordinate=None, current_coordinate=None, center_point=None, points = None, motion_vector = None, previous_motion_vector = None, previous_motion_vector_roi = None):
    T_F = False
    depth_ = current_coordinate[2]
    if depth_ > 30:
        depth_ = 0
    x_values = [x for x, y in points]
    x_roi = sum(x_values)/2    
    y_values = [y for x, y in points]
    y_roi = sum(y_values)/3
    center_roi = x_roi, y_roi
    direction_score = 0

    las_point = (int(last_coordinate[0]), int(last_coordinate[1]))
    curren_point = (int(current_coordinate[0]), int(current_coordinate[1]))

    will_TF, closer_score_roi = will_cross_roi(las_point, curren_point, points)
    is_closer, speed = is_come_close(last_coordinate[2], current_coordinate[2])
    basic_position = 0
    if current_coordinate and last_coordinate and motion_vector and previous_motion_vector is not None :
        motion_roi_vector = compute_vector(las_point, curren_point)
        direction_TF, direction_score = get_direction_relation(previous_motion_vector, motion_vector)
        direction_ROI, direction_score_roi = get_direction_relation(motion_vector, previous_motion_vector_roi)
        if depth_ > 0:
            if is_point_in_polygon(points, curren_point):
                T_F = is_closer and direction_TF
            else:
                T_F = (will_TF and is_closer) and direction_ROI
                direction_score = direction_score_roi
        else:
            T_F = False

    if is_point_in_polygon(points, curren_point):        
        if depth_ < 10:
            T_F = True
            basic_position = 3071
        elif depth_ >= 10 and depth_ < 20:
            T_F = True
            basic_position = 2047          

    return T_F, speed, depth_

# 두 점으로 벡터 만드는 함수 - 단 두 점의 차원이 같아야 함.
def compute_vector(start_point, end_point):
    start_point = np.array(start_point)
    end_point = np.array(end_point)
    vector = end_point - start_point
    return vector

# 두 3차원 벡터가 같은 방향을 향하는지 검증하여 점술를 내는 함수- treshold가 cos값 - 디폴트값 설정해야함 
# 1이면 완전 동일한 방향 / -1이면 정 반대 방향 / 둘 다 아니면 다른방향
# 전 프레임과 center 점과의 벡터와 현재 프레임과 전 프레임 점과의 벡터를 비교할 것
# 내적 결과가 1에 가까울수록 높은 점수 -> 장애물 점수임 0부터 10
def get_direction_relation(vector1, vector2):
    """
    두 3차원 벡터가 같은 방향, 완전히 반대 방향, 또는 전혀 다른 방향인지 판단
    """
    # 입력값을 numpy 배열로 변환
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    
    # 벡터를 정규화하여 방향만 비교
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    
    if norm1 == 0 or norm2 == 0:
        return False , 0
    
    # 정규화된 벡터
    unit_vector1 = vector1 / norm1
    unit_vector2 = vector2 / norm2
    
    # 내적 계산
    dot_product = np.dot(unit_vector1, unit_vector2)
    if dot_product > 0.9:
        T_F = True
    else:
        T_F = False
    
    # 내적 값 범위 [-1, 1]을 점수 범위 [0, 10]으로 변환
    score = (dot_product + 1) * 5
    
    return T_F, score

# 가까워지는지 팓별하는 함수 - 인자 값 조절해야함
# 더 빨리 가까워질 수록 높은 점수
def is_come_close(last_depth, current_depth):
    depth_change = current_depth - last_depth
    is_closer = depth_change <= 0
    closer_score = 0
    if is_closer:
        # 깊이 감소 속도 계산
        speed = abs(depth_change)
        # 점수를 10점 만점으로 매김
        closer_score = min(20, speed)  # 속도가 클수록 점수는 더 높아지며, 최대 10점까지
    
    return is_closer, closer_score

def draw_roi(points, img):
    cv2.polylines(img, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)

# ROI 전역변수
points = []  # 좌표를 저장할 리스트
roi_defined = False  # ROI가 정의되었는지 확인하는 플래그
def set_roi(event, x, y, flags, img):
    global points, roi_defined
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        print('point ', len(points)+1)
        points.append((x, y))
        print(points)
        cv2.circle(img, (x,y), 10, (255,0,0), -1)
        if len(points) == 4:
            points = roi_sort(points)
            # 다각형 그리기
            draw_roi(points, img)
            roi_defined = True
        cv2.imshow('Image', img)

# roi를 그리기 위한 정렬 함수
def roi_sort(points):
    # 1. x값 기준으로 정렬
    sorted_points = sorted(points, key=lambda p: p[0])

    # x값이 가장 작은 두 점
    min_x_points = sorted_points[:2]
    # 나머지 두 점
    rest_points = sorted_points[2:]

    # 1번 위치에 배치할 점: x값이 가장 작은 두 점 중 y값이 더 작은 점
    min_x_points = sorted(min_x_points, key=lambda p: p[1])
    first_point = min_x_points[0]
    fourth_point = min_x_points[1]

    # 2번 위치에 배치할 점: 나머지 두 점 중 x값이 더 작은 점
    rest_points = sorted(rest_points, key=lambda p: p[0])
    second_point = rest_points[0]
    third_point = rest_points[1]

    # 결과 리스트: [1, 2, 3, 4] 위치에 맞게 점을 배치
    sorted_list = [first_point, second_point, third_point, fourth_point]

    return sorted_list

def detect(frame, model_metric, model_yolo, class_names, colors, mot_tracker, track_info, points):

    # 여러가지 처리용 이미지 생성
    color_map = frame
    img = frame.copy() # for yolo
    # 모델에 쓰기위한 이미지 처리
    input_img = preprocess_frame(img)
    # depth_map 추출
    with torch.no_grad():
        pred_depth, confidence, output_dict = model_metric.inference({'input': input_img})
    depth_map = pred_depth.squeeze().cpu().numpy()
    # 객체 탐지 수행
    detections = proceess_frame_yolo(frame, model_yolo, class_names, colors)
    # tracker.update에 전달    
    if detections.ndim == 1:
        detections = detections.reshape(-1, 6)
    track_bbs_ids = mot_tracker.update(detections[:, :5])
    img1 = frame.copy()
    draw_roi(points, img1)        
    obj_img = img1.copy()  # 원본 이미지 복사본
    depth_img = frame.copy()
    
    true_img = np.zeros_like(frame)  # 검정색 바탕 이미지
    draw_roi(points, true_img)

    obstacle_score_com = -100
    is_obstacle = False
    basic_position = 0
    score = 0

    obstacle = False
    for track_bb in track_bbs_ids:
        xyxy = track_bb[:4].astype(int)
        track_id = int(track_bb[4])
        # 객체의 현 xy (center 좌표)
        center = ((xyxy[0] + xyxy[2]) // 2, (xyxy[1] + xyxy[3]) // 2)
        mid_x, mid_y = center
        # 객체의 현 depth 좌표
        current_depth = get_depth(depth_img, depth_map, center)
        current_coordinate = [mid_x, mid_y, current_depth]
        # 객체의 지난 프레임 3차원 좌표 [x1, y1, depth]의 형식
        last_coordinate = track_info.get_last_3D_coordinate(track_id) 
        # 객체의 지난 프레임 depth 좌표 -> 이건 last_coordinate에서 해결하는 걸로 해서 삭제하기
        last_depth = track_info.get_last_depth(track_id)
        track_info.update_track(track_id, center, current_depth, min_move_threshold)
        # 객체 상태 정의 & 시각화
        color = track_id
        acc = 0
        if last_depth is not None:
            x_values = [x for x, y in points]
            x_roi = sum(x_values)/2    
            y_values = [y for x, y in points]
            y_roi = sum(y_values)/3
            center_roi = x_roi, y_roi
            last_two_points = track_info.get_last_two_points(track_id)
            # 현재 center랑 last_two_point랑 다름. last_two_point는 [(x1,y1), (x2, t2)]의 list
            center_point = [obj_img.shape[1] // 2, obj_img.shape[0] // 2, 0]
            current_point = (int(current_coordinate[0]), int(current_coordinate[1]))
            in_roi = is_point_in_polygon(points, current_point)
            motion_vector = track_info.calculate_motion_vector(track_id)
            previous_motion_vector = track_info.calculate_previous_motion_vector(track_id, center_point)
            previous_motion_vector_roi = track_info.calculate_previous_motion_vector(track_id, center_roi)
            current_status, speed, depth = define_obstacle_3D(last_coordinate, current_coordinate ,center_point, points, motion_vector, previous_motion_vector, previous_motion_vector_roi)
            obstacle = track_info.update_status(track_id, current_status, current_depth, in_roi)
            if speed is None:
                speed = 0
            acc, tlqkf = track_info.update_acc(track_id, speed)
            label = f'ID {track_id} {"True" if obstacle else "False"} {acc: .2f} {depth: .2f},{in_roi}'
                    
        else:
            label = f'ID {track_id}'
        plot_one_box(xyxy, obj_img, color=color, label=label, line_thickness=3)
        plot_one_box(xyxy, depth_img, color=color, label=label, line_thickness=3)

        if obstacle:  # True 상태인 객체만 표시
            plot_one_box(xyxy, true_img, color=color, label=label, line_thickness=3)
            is_obstacle = True
            if acc < score:
                score = acc
            if current_depth < 10:
                basic_position = 3071
            elif current_depth < 20 and current_depth > 10:
                basic_position = 2047
            else :
                basic_position = 0    
        else:
            basic_position = 0 

        path = track_info.get_path(track_id)
        if path and len(path) > 1:
            for j in range(1, len(path)):
                cv2.line(obj_img, path[j-1], path[j], color, 2)
                # cv2.line(depth_img, path[j-1], path[j], color, 2)
                if obstacle:  # True 상태인 객체만 표시
                    cv2.line(true_img, path[j-1], path[j], color, 2)
            cv2.arrowedLine(obj_img, path[-2], path[-1], color, 2, tipLength=0.5)
            if obstacle:  # True 상태인 객체만 표시
                cv2.arrowedLine(true_img, path[-2], path[-1], color, 2, tipLength=0.5)


    # 활성화된 추적 ID 관리
    active_track_ids = {int(track_bb[4]) for track_bb in track_bbs_ids}
    for track_id in list(track_info.tracks.keys()):
        if track_id not in active_track_ids:
            track_info.clear_track(track_id)


    return obj_img, true_img, is_obstacle, score, basic_position 


    