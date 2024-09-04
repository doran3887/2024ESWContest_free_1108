import pyrealsense2 as rs
import numpy as np
import cv2
import torch

# PyTorch Hub를 통해 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
model = model.to(device)
model.eval()

# RealSense 카메라 초기화
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline_profile = pipeline.start(config)

# 카메라 설정 고정
sensor = pipeline_profile.get_device().first_color_sensor()
sensor.set_option(rs.option.enable_auto_exposure, False)
sensor.set_option(rs.option.enable_auto_white_balance, False)
sensor.set_option(rs.option.exposure, 156)  # 노출 값을 필요에 따라 조정
sensor.set_option(rs.option.white_balance, 4600)  # 화이트 밸런스 값을 필요에 따라 조정

def preprocess_frame(frame):
    frame = np.asanyarray(frame)
    resized_frame = cv2.resize(frame, (512, 512))
    resized_frame = resized_frame / 255.0  # Normalize to [0, 1]
    tensor_frame = torch.tensor(resized_frame).float().permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor_frame

def display_depth_map(color_image, depth_map):
    # depth_map의 크기를 color_image와 동일하게 조정
    depth_map_resized = cv2.resize(depth_map, (color_image.shape[1], color_image.shape[0]))
    
    # depth_map의 값을 정규화하여 8비트로 변환
    depth_map_resized = cv2.normalize(depth_map_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # depth_map을 BGR로 변환
    depth_map_colored = cv2.applyColorMap(depth_map_resized, cv2.COLORMAP_JET)

    # color_image와 depth_map_colored의 크기 및 채널 수 일치 확인
    combined = cv2.addWeighted(color_image, 0.6, depth_map_colored, 0.4, 0)

    cv2.imshow('Depth Map', combined)

def display_depth_values(depth_map):
    depth_map_text = np.zeros((depth_map.shape[0], depth_map.shape[1], 3), dtype=np.uint8)

    # 깊이 값 텍스트로 표시
    for y in range(0, depth_map.shape[0], 40):  # 40픽셀 간격으로 샘플링
        for x in range(0, depth_map.shape[1], 40):
            depth_value = depth_map[y, x]
            cv2.putText(depth_map_text, f'{depth_value:.2f}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 이미지의 총 픽셀 수를 텍스트로 표시
    total_pixels = depth_map.shape[0] * depth_map.shape[1]
    cv2.putText(depth_map_text, f'Total Pixels: {total_pixels}', (10, depth_map.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Depth Values', depth_map_text)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        input_image = preprocess_frame(color_image)

        with torch.no_grad():
            pred_depth, confidence, output_dict = model.inference({'input': input_image})

        depth_map = pred_depth.squeeze().cpu().numpy()
        display_depth_map(color_image, depth_map)
        display_depth_values(depth_map)

        # 깊이 맵의 최소값과 최대값을 출력
        min_depth = np.min(depth_map)
        max_depth = np.max(depth_map)
        print(f'Depth Map Range: Min={min_depth}, Max={max_depth}')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
