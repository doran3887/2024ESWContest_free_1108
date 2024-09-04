import cv2
import torch
import numpy as np
import time
import serial
import pygame
import obd
from pynput import keyboard
from pathlib import Path
from new_YOLO_new import TrackInfo, draw_roi, roi_sort, detect, initialize_video_capture
from sort import Sort  
from ultralytics import YOLOv10
from GPS_func import load_data, track_speed_limit
import argparse  # Import argparse for command-line argument parsing

# ROI 전역변수
points = []
roi_defined = False
curren_velocity = 0
music_flag = False
MUSIC_END_EVENT = pygame.USEREVENT + 1

def initialize_obd_connection():
    try:
        connection = obd.OBD()  # 자동으로 포트 탐색하여 연결 시도
        if connection.is_connected():
            print("OBD-II 모듈에 성공적으로 연결되었습니다.")
            return connection
        else:
            print("OBD-II 모듈에 연결하지 못했습니다.")
            return None
    except Exception as e:
        print(f"OBD-II 연결 중 오류가 발생했습니다: {e}")
        return None

def get_velocity(connection):
    if connection is None or not connection.is_connected():
        print("OBD-II 연결이 유효하지 않습니다.")
        return 0  # 연결 실패 시 기본값 반환

    try:
        cmd = obd.commands.SPEED  # 속도 명령 설정
        response = connection.query(cmd)  # 명령 실행 및 응답 받기

        if not response.is_null():
            speed = response.value.to("kph").magnitude  # 속도를 km/h 단위로 변환
            return speed
        else:
            print("속도 데이터를 가져올 수 없습니다.")
            return 0  # 데이터 수신 실패 시 기본값 반환
    except Exception as e:
        print(f"속도 데이터를 가져오는 중 오류가 발생했습니다: {e}")
        return 0  # 예외 발생 시 기본값 반환

def set_roi(event, x, y, flags, img):
    global points, roi_defined
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        print('point ', len(points)+1) 
        points.append((x, y))
        print(points)
        cv2.circle(img, (x,y), 10, (255,0,0), -1)
        if len(points) == 4:
            points = roi_sort(points)
            draw_roi(points, img)
            roi_defined = True
        cv2.imshow('Image', img)

def map_range(value, from_min, from_max, to_min, to_max):
    from_range = from_max - from_min
    to_range = to_max - to_min
    value = max(from_min, min(value, from_max))
    return to_min + ((value - from_min) / from_range) * to_range

def vel_press(key):
    global curren_velocity
    try:
        if key.char == 'w':
            curren_velocity += 1
            print(curren_velocity)    
        elif key.char == 's':
            curren_velocity -= 1
            print(curren_velocity)
    except AttributeError:
        pass

def play_warning_sound(warning_sound_path):
    global music_flag
    pygame.mixer.music.load(warning_sound_path)  # 경고음 파일 로드
    pygame.mixer.music.play()  # 경고음 파일 재생
    music_flag = True
    print(music_flag)    

def main(video_path, server_url, com_port):
    pygame.init()
    pygame.mixer.init()
    global curren_velocity
      
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
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    ret, first_frame = cap.read()
    ser = serial.Serial(port=com_port, baudrate=9600)  # COM 포트를 인수로 받음
    results = []
    
    data = load_data('gps.csv')

    # 프레임 스킵을 위한 프레임 카운트
    frame_count = 0
    # 처음 시작할 때 ROI 지정하는 부분
    global roi_defined, points
    print('**************************************************')
    print("Set obstacle's roi")
    print('**************************************************')
    cv2.imshow('Image', first_frame)
    cv2.setMouseCallback('Image', lambda event, x, y, flags, param: set_roi(event, x, y, flags, first_frame))
    # 비동기 리스너를 사용하여 루프를 방해하지 않음
    listener = keyboard.Listener(on_press=vel_press)
    listener.start()  # 백그라운드에서 실행
    # ROI가 정의될 때까지 대기
    while not roi_defined:
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키를 누르면 종료
            cap.release()
            cv2.destroyAllWindows()
            return

    print(points)
    # 전처리 끝
    while cap.isOpened(): 
        frame_count += 1 # 프레임 카운터 증가
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % 6 == 0:
            obj_img, true_img, is_obstacle, obstacle_score, basic_position  = detect(frame, model_metric, model_yolo, class_names, colors, mot_tracker, track_info, points)
            
            speed_limit = track_speed_limit(server_url, data)
             
            speed_diff = curren_velocity - speed_limit

            if is_obstacle or speed_diff > 0:
                accel_ = abs(obstacle_score)*5
                accel_ = map_range(accel_, 0, 10, 0, 4095) + basic_position                     
                speed_diff_ = abs(speed_diff)/speed_limit * 100
                speed_diff = map_range(speed_diff, 0, 100, 0, 4095)
                if speed_diff > 0:
                    global music_flag
                    if not pygame.mixer.music.get_busy():
                        music_flag = False    
                    if not music_flag :
                        if accel_ > speed_diff:
                            accel = accel_
                        else:
                            accel = speed_diff 
                        ser.write(f'{accel}\n'.encode())
                        print(f'현재 제한속도는 {speed_limit: .1f} 입니다')
                        print(f'제한속도를 {speed_diff_:.2f}% 초과했습니다')
                        play_warning_sound('accel_warning.wav')
                    else:
                        if accel_ > speed_diff:
                            accel = accel_
                        else:
                            accel = speed_diff 
                        ser.write(f'{accel}\n'.encode())
                        print(f'현재 제한속도는 {speed_limit: .1f} 입니다')
                        print(f'제한속도를 {speed_diff_:.2f}% 초과했습니다')
                elif obstacle_score < 0:          
                    accel = accel_  
                else:
                    accel = basic_position
            else:
                accel = 0            
            
            ser.write(f'{accel}\n'.encode())
            
        else :
            continue

        # 창 띄우기
        cv2.imshow('obj_img', obj_img)
        cv2.imshow('true_img', true_img)  # True 상태 객체만 표시하는 창
        if frame_count == 0:
            time.sleep(3)

        if cv2.waitKey(1) == ord('q'):
            raise StopIteration

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Argparse to receive command line arguments with default values
    parser = argparse.ArgumentParser(description="Run the video analysis script with specified parameters.")
    parser.add_argument('--video_path', type=str, default='test1.mp4', help="Path to the video file (default: test1.mp4)")
    parser.add_argument('--server_url', type=str, default='http://192.0.0.2:8080/', help="URL of the server (default: http://192.0.0.4:8080/)")
    parser.add_argument('--com_port', type=str, default='COM11', help="COM port for serial communication (default: COM11)")

    args = parser.parse_args()

    # Run the main function with the arguments received
    main(args.video_path, args.server_url, args.com_port)
