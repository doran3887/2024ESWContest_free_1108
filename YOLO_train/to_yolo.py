import os
import json
import cv2

# 카테고리 이름과 YOLO 클래스 ID 매핑
category_mapping = {
    'building': 0,
    'wall': 1,
    'fence': 2,
    'pole': 3,
    'person': 4,
    'rider': 5,
    'car': 6,
    'truck': 7,
    'bus': 8,
    'train': 9,
    'motorcycle': 10,
    'bicycle': 11
}

def convert_bbox_to_yolo_format(bbox, img_width, img_height):
    """
    바운딩 박스 좌표를 YOLO 형식으로 변환합니다.
    bbox: [x1, y1, x2, y2]
    img_width: 이미지의 너비
    img_height: 이미지의 높이
    """
    x1, y1, x2, y2 = bbox

    # 이미지 경계로 클리핑 (음수 방지)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_width, x2)
    y2 = min(img_height, y2)
    
    # 중간 계산 및 정규화
    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    # 만약 정규화된 값이 잘못된 경우 (0 이하이거나, 1 이상), 로그로 출력하고 무시
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and width > 0 and height > 0):
        print(f"Invalid YOLO bbox: {x_center}, {y_center}, {width}, {height}")
        return None

    return [x_center, y_center, width, height]

def convert_json_to_yolo(json_file, output_dir, img_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)

    for frame in data['frames']:
        img_file = os.path.join(img_dir, frame['name'])
        img = cv2.imread(img_file)
        if img is None:
            print(f"Image {img_file} not found.")
            continue
        img_height, img_width, _ = img.shape

        yolo_labels = []
        for label in frame['labels']:
            category = label['category']
            if category in category_mapping:
                if 'box2d' in label:  # Box2D 데이터가 있는 경우
                    bbox = [label['box2d']['x1'], label['box2d']['y1'], label['box2d']['x2'], label['box2d']['y2']]
                    yolo_bbox = convert_bbox_to_yolo_format(bbox, img_width, img_height)
                    if yolo_bbox:  # 유효한 경우에만 추가
                        class_id = category_mapping[category]
                        yolo_labels.append(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")

        if yolo_labels:
            label_file = os.path.join(output_dir, frame['name'].replace('.jpg', '.txt'))
            with open(label_file, 'w') as f:
                f.writelines(yolo_labels)

# 경로 설정
json_file = r'C:\Users\admin\Desktop\BDD100K_new\bdd100k\labels\sem_seg\rles\sem_seg_val.json'
output_dir = r'C:\Users\admin\Desktop\BDD100K_new\val_labels'
img_dir = r'C:\Users\admin\Desktop\BDD100K\bdd10k_val_images\images'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

convert_json_to_yolo(json_file, output_dir, img_dir)
