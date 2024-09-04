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
    '''
    SORT 알고리즘 용 Class
    Func name : param
    update_track() : track_id, center, current_depth, min_move_threshold
    get_path() : track_id
    get_last_two_points() : track_id
    get_last_depth() : track_id
    clear_track : track_id
    update_status : track_id, current_status, required_consistency
    '''
    obstacle_state_log = {}
    def __init__(self):
        self.tracks = {}
        self.track_statuses = {}

    def update_track(self, track_id, center, current_depth, min_move_threshold):
        if track_id in self.tracks:
            last_center = self.tracks[track_id]['last_center']
            last_depth = self.tracks[track_id]['last_depth']
            movement = np.sqrt((center[0] - last_center[0]) ** 2 + (center[1] - last_center[1]) ** 2 + (current_depth - last_depth) ** 2)
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
            self.tracks[track_id]['last_depth'] = current_depth
        else:
            self.tracks[track_id] = {'last_center': center, 'path': [center], 'smoothed_path': [center], 'last_depth': current_depth}



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

    def get_last_3D_coordinate(self, track_id):
        if track_id in self.tracks:
            last_center = self.tracks[track_id]['last_center']
            last_depth = self.tracks[track_id]['last_depth']
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
''''
# depth 표시 함수 - 현재는 2D. 3D로 개선해야함 -- 속도 때문에 안 쓸듯
def display_depth_map(color_image, depth_map, roi):
    x1, y1, x2, y2 = roi[0], roi[1], roi[2], roi[3],
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
'''

# depth 값을 띄우는 함수
def draw_depth_label(image, depth_map, roi):
    '''
    객체의 중심에 depth 값을 숫자로 표시하는 함수

    Parameters :
    image : 원본 이미지(프레임)
    depth_map : Metric_3D로 생성한 depth_map
    roi : [(x1, y1),(x2, y2)] 형식의 객체 박스의 왼쪽 위, 오른쪽 아래 좌표 리스트
    '''
    x1, y1, x2, y2 = roi[0], roi[1], roi[2], roi[3]
    
    depth_map_resized = cv2.resize(depth_map, (image.shape[1], image.shape[0]))
    depth_map_resized = cv2.normalize(depth_map_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
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
    cv2.rectangle(image, (label_x1, label_y1), (label_x2, label_y2), (255, 255, 255), -1)
    cv2.putText(image, label, (label_x1, label_y1 - 2), 0, 0.5, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

    return image

# 객체의 depth 값을 가져오는 함수 
# - 최적화를 위해 파라미터 수정 필요 - color_map에서 사이즈만 받아오기 때문.
# - display-depth_map과 비슷한 부분이 많아 개선이 필요함. TrackInfo depth update()용임
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

# 객체 박스 만드는 함수
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
    if video_input.isdigit():
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

def distance_from_point_to_line(point, center_point):
    global points
    x0, y0 = point
    if x0 < center_point[0]:    
        x1, y1 = points[3]
        x2, y2 = points[0]
    else:
        x1, y1 = points[2]
        x2, y2 = points[1]

    # 직선의 기울기와 절편을 고려한 거리 계산
    numerator = abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1))
    denominator = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    distance = numerator / denominator
    return distance

def is_closer_to_line(previous_point, current_point, center_point):
    prev_distance = distance_from_point_to_line(previous_point, center_point)
    curr_distance = distance_from_point_to_line(current_point, center_point)
    speed = abs(curr_distance - prev_distance)
    # 거리가 줄어들면 다가가고 있는 것으로 판단
    closer_score = min(10, speed * 5)

    return curr_distance < prev_distance, closer_score

def will_cross_roi(last_coordinate=None, current_coordinate=None, center_point = None):
    global points
    last_coordinate = last_coordinate[:2]
    current_coordinate = current_coordinate[:2]
    if last_coordinate is not None and current_coordinate is not None:
        T_F, closer_score = is_closer_to_line(last_coordinate, current_coordinate, center_point)
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
# - 3D로 확장하면 코드 자체를 다시 짜야함
def define_obstacle_3D(last_coordinate=None, current_coordinate=None, center_point=None):
    '''
    종합적으로 장애물을 판별하는 함수
    '''
    global points
    T_F = False
    obstacle_score = 0
    depth_score = -100
    las_point = (int(last_coordinate[0]), int(last_coordinate[1]))
    point = (int(current_coordinate[0]), int(current_coordinate[1]))
    if current_coordinate and last_coordinate is not None:
        last_vector = compute_vector(last_coordinate, center_point)
        motion_vector = compute_vector(last_coordinate, current_coordinate)
        direction_TF, direction_score = get_direction_relation(last_vector, motion_vector)
        if is_point_in_polygon(points, point):
            is_closer, closer_score, depth_score = is_come_close(last_coordinate[2], current_coordinate[2])
            obstacle_score = direction_score + closer_score

            if depth_score == -100:
                T_F = False
            else :
                T_F = is_closer and direction_TF
        else:
            T_F = will_cross_roi(last_coordinate=las_point, current_coordinate=point, center_point=center_point)
    return T_F, obstacle_score, depth_score

# 두 점으로 3차원 벡터 만드는 함수
def compute_vector(start_point, end_point):
    """
    두 3차원 점을 받아서 시작점에서 끝점으로의 벡터를 반환합니다.

    :param start_point: 시p작점 (x1, y1, z1) - numpy 배열 또는 리스트
    :param end_point: 끝점 (x2, y2, z2) - numpy 배열 또는 리스트
    :return: 시작점에서 끝점으로의 벡터 (x2-x1, y2-y1, z2-z1) - numpy 배열
    """
    # 입력값을 numpy 배열로 변환
    start_point = np.array(start_point)
    end_point = np.array(end_point)
    
    # 벡터 계산
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
def is_come_close(last_depth, current_depth, sensitivity_increase=0.1, sensitivity_decrease=0.01):
    """
    깊이 값이 감소하여 물체가 가까워지고 있는지 판별합니다.
    가까워지는 경우에는 민감하게 반응하고, 멀어지는 경우에는 덜 민감하게 반응합니다.

    :param last_depth: 이전 깊이 값
    :param current_depth: 현재 깊이 값
    :param sensitivity_increase: 깊이 값이 감소할 때 민감도 조정 값 (기본값: 0.1)
    :param sensitivity_decrease: 깊이 값이 증가할 때 민감도 조정 값 (기본값: 0.01)
    :return: 물체가 가까워지고 있는지 여부 (True/False), 점수 (0-10)
    """
    # 깊이 차이 계산
    depth_change = current_depth - last_depth

    # 물체가 가까워지고 있는지 여부
    is_closer = depth_change <= 0

    # 점수 계산
    closer_score = 0
    depth_score = 0

    if current_depth > 30:
        depth_score = -100
    else :
        depth_score = current_depth

    if is_closer:
        # 깊이 감소 속도 계산
        speed = abs(depth_change)
        # 점수를 10점 만점으로 매김
        closer_score = min(10, speed * 5)  # 속도가 클수록 점수는 더 높아지며, 최대 10점까지
    
    return is_closer, closer_score, depth_score
'''
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
'''
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


def main(video_path):
    '''
    메인 동작 코드
    원래 코드의 detect() 함수 부분이랑 유사
    
    Pramter :
    video_path : 분석할 비디오 주소 // 나중에 웹캠으로 변경하면 됨
    '''
        
    start_time = time.time()

    device = torch.device('cuda')

    # 욜로 모델 로드 - 다 하고 half() 되는지도 확인하기
    model_yolo = YOLOv10('YOLOv10_model/best.pt')
    model_yolo.to(device)

    # Metric_3D 모델 로드
    model_metric = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
    model_metric = model_metric.to(device)

    # Sort 알고리즘 로드
    mot_tracker = Sort()
    track_info = TrackInfo()

    class_names = ['building','wall','fence','pole','person','rider','car','truck','bus','train','motorcycle','bicycle']
    colors = np.random.randint(0, 255, size=(len(class_names), 3))

    # 비디오 캡처 객체 생성
    #cap = initialize_video_capture(video_path)
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    ret, first_frame = cap.read()
    
    results = []

    # 프레임 스킵을 위한 프레임 카운트
    frame_count = 0
    # 처음 시작할 때 ROI 지정하는 부분
    global roi_defined, points
    print('**************************************************')
    print("Set obstacle's roi")
    print('**************************************************')
    cv2.imshow('Image', first_frame)
    cv2.setMouseCallback('Image', lambda event, x, y, flags, param: set_roi(event, x, y, flags, first_frame))

    # ROI가 정의될 때까지 대기
    while not roi_defined:
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키를 누르면 종료
            cap.release()
            cv2.destroyAllWindows()
            return

    print(points)

    if torch.cuda.is_available():
        print(f"CUDA is available. Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU device: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is not available. Running on CPU.")

    while cap.isOpened(): 
        frame_count += 1 # 프레임 카운터 증가
        ret, frame = cap.read()

        if not ret:
            break

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
        print(detections)
        if detections.ndim == 1:
            detections = detections.reshape(-1, 6)
        track_bbs_ids = mot_tracker.update(detections[:, :5])
        print(track_bbs_ids) 

        img1 = frame.copy()

        draw_roi(points, img1)        

        obj_img = img1.copy()  # 원본 이미지 복사본
        depth_img = frame.copy()
        
        true_img = np.zeros_like(frame)  # 검정색 바탕 이미지
        draw_roi(points, true_img)
        for track_bb in track_bbs_ids:
            xyxy = track_bb[:4].astype(int)
            track_id = int(track_bb[4]) # 객체 ID를 50으로 제한
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
            if last_depth is not None:
                
                last_two_points = track_info.get_last_two_points(track_id)
                # 현재 center랑 last_two_point랑 다름. last_two_point는 [(x1,y1), (x2, t2)]의 list
                center_point = [obj_img.shape[1] // 2, obj_img.shape[0] // 2, 0]
                current_point = (int(current_coordinate[0]), int(current_coordinate[1]))
                in_roi = is_point_in_polygon(points, current_point)
                current_status, obstacle_score, depth_score = define_obstacle_3D(last_coordinate, current_coordinate ,center_point)
                obstacle = track_info.update_status(track_id, current_status, current_depth, in_roi)
                label = f'ID {track_id} {"True" if obstacle else "False"} depth: {depth_score: .2f} in_roi {in_roi}'
                if obstacle:  # True 상태인 객체만 표시
                    plot_one_box(xyxy, true_img, color=color, label=label, line_thickness=3)
            else:
                label = f'ID {track_id}'
            plot_one_box(xyxy, obj_img, color=color, label=label, line_thickness=3)
            plot_one_box(xyxy, depth_img, color=color, label=label, line_thickness=3)
            # draw_depth_label(obj_img, depth_map, xyxy)
            # depth_img = display_depth_map(depth_img, depth_map, xyxy)
            # 객체 추적 경로 시각화
            path = track_info.get_path(track_id)
            if path and len(path) > 1:
                for j in range(1, len(path)):
                    cv2.line(obj_img, path[j-1], path[j], color, 2)
                    # cv2.line(depth_img, path[j-1], path[j], color, 2)
                    if obstacle:  # True 상태인 객체만 표시
                        cv2.line(true_img, path[j-1], path[j], color, 2)
                cv2.arrowedLine(obj_img, path[-2], path[-1], color, 2, tipLength=0.5)
                # cv2.arrowedLine(depth_img, path[-2], path[-1], color, 2, tipLength=0.5)
                if obstacle:  # True 상태인 객체만 표시
                    cv2.arrowedLine(true_img, path[-2], path[-1], color, 2, tipLength=0.5)
        # 활성화된 추적 ID 관리
        active_track_ids = {int(track_bb[4]) for track_bb in track_bbs_ids}
        for track_id in list(track_info.tracks.keys()):
            if track_id not in active_track_ids:
                track_info.clear_track(track_id)


        # 창 띄우기
        cv2.imshow('obj_img', obj_img)
        # cv2.imshow('depth_img', depth_img)
        cv2.imshow('true_img', true_img)  # True 상태 객체만 표시하는 창
        if cv2.waitKey(1) == ord('q'):
            raise StopIteration


    cap.release()
    cv2.destroyAllWindows()


    # 시간 측정 종료
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total processing time: {total_time:.2f} seconds")
    

if __name__ == "__main__":#
    video_path = 'test1.mp4' # 분석할 동영상
    
    main(video_path)