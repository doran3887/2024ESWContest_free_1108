from new_YOLO_new import TrackInfo, draw_roi,roi_sort, detect, initialize_video_capture
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

def main(video_path):
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

    # 비디오 캡처 객체 생성 1번쨰가 비디오 ,2번째가 웹캠
    cap = initialize_video_capture(video_path)
    #cap = cv2.VideoCapture(0)
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
#########################################################################################
    # 전처리 끝
    while cap.isOpened(): 
        frame_count += 1 # 프레임 카운터 증가q
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % 7 != 0:
            continue
        else :
            obj_img, true_img, is_obstacle, obstacle_score = detect(frame, model_metric, model_yolo, class_names, colors, mot_tracker, track_info, points)

        print(is_obstacle, obstacle_score)

        
        # 창 띄우기
        cv2.imshow('obj_img', obj_img)
        cv2.imshow('true_img', true_img)  # True 상태 객체만 표시하는 창
        if cv2.waitKey(1) == ord('q'):
            raise StopIteration


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":#
    video_path = 'test1.mp4' # 분석할 동영상
    main(video_path)