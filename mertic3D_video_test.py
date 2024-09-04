import numpy as np
import cv2
import torch

# PyTorch Hub를 통해 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True)
model = model.to(device)
model.eval()

# 동영상 파일 경로
video_path = 'test1.mp4'  # 여기에 분석할 동영상 파일의 경로를 입력하세요

# OpenCV를 사용하여 동영상 파일 열기
cap = cv2.VideoCapture(video_path)

def preprocess_frame(frame):
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

    # 20픽셀씩 묶어서 평균 깊이 값 계산 및 표시
    step = 20
    font_scale = 0.4
    font_thickness = 1
    text_color = (255, 255, 255)

    for y in range(0, depth_map.shape[0], step):
        for x in range(0, depth_map.shape[1], step):
            block = depth_map[y:y+step, x:x+step]
            avg_depth = np.mean(block)
            text_size, _ = cv2.getTextSize(f'{avg_depth:.1f}', cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            text_position = (x + (step - text_size[0]) // 2, y + (step + text_size[1]) // 2)
            cv2.putText(depth_map_text, f'{avg_depth:.1f}', text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

    # 이미지의 총 픽셀 수를 텍스트로 표시
    total_pixels = depth_map.shape[0] * depth_map.shape[1]
    cv2.putText(depth_map_text, f'Total Pixels: {total_pixels}', (10, depth_map.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    cv2.imshow('Depth Values', depth_map_text)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        color_image = frame

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
    cap.release()
    cv2.destroyAllWindows()
