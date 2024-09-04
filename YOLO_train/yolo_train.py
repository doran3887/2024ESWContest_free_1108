import os
import yaml
from ultralytics import YOLO
from multiprocessing import freeze_support
import cv2
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 사용자 정의 데이터셋 클래스
class CustomDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        label_path = os.path.join(self.label_dir, self.img_files[idx].replace('.jpg', '.txt').replace('.png', '.txt'))
        
        image = cv2.imread(img_path)
        with open(label_path, 'r') as f:
            labels = [list(map(float, line.strip().split())) for line in f.readlines()]
        
        if self.transform:
            transformed = self.transform(image=image, bboxes=labels)
            image = transformed['image']
            labels = transformed['bboxes']
        
        return image, labels

def check_data_loading(data_yaml):
    with open(data_yaml, 'r') as file:
        data = yaml.safe_load(file)
        print(f"Train data: {data['train']}")
        print(f"Val data: {data['val']}")
        print(f"Number of classes (nc): {data['nc']}")
        print(f"Class names: {data['names']}")
    return data

if __name__ == '__main__':
    freeze_support()

    # CUDA 런타임 오류를 동기적으로 보고
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # 데이터 로딩 확인
    data_yaml = r"C:\Users\admin\Desktop\yolo(fast)\bdd100k.yaml"
    data_config = check_data_loading(data_yaml)
    
    # 사전 학습된 모델 가중치 파일 경로
    pretrained_weights = r"C:\Users\admin\Desktop\yolo_train\yolov10l.pt"  # 절대 경로 사용

    # 데이터 디렉토리 설정 (절대 경로 사용)
    train_img_dir = r'C:\Users\admin\Desktop\BDD100K_new\bdd10k_train_images\images'
    train_label_dir = r'C:\Users\admin\Desktop\BDD100K_new\bdd10k_train_images\labels'
    val_img_dir = r'C:\Users\admin\Desktop\BDD100K_new\bdd10k_val_images\images'
    val_label_dir = r'C:\Users\admin\Desktop\BDD100K_new\bdd10k_val_images\labels'

    # 데이터 증강 파이프라인 정의
    transform = A.Compose([
        A.RandomRotate90(),
        A.Flip(),
        A.Transpose(),
        A.OneOf([
            A.HueSaturationValue(),
            A.RGBShift(),
            A.GaussianBlur(),
            A.CoarseDropout(),
        ], p=0.5),
        A.RandomBrightnessContrast(),
        A.RandomScale(),
        A.RandomCrop(width=640, height=640),
        A.Resize(width=640, height=640),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    # 사용자 정의 데이터셋 및 데이터로더 설정
    train_dataset = CustomDataset(img_dir=train_img_dir, label_dir=train_label_dir, transform=transform)
    val_dataset = CustomDataset(img_dir=val_img_dir, label_dir=val_label_dir)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=lambda x: tuple(zip(*x)), num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=lambda x: tuple(zip(*x)), num_workers=4)

    # 모델 정의 (사전 학습된 가중치 로드)
    try:
        model = YOLO(pretrained_weights)
        print("YOLO 모델이 성공적으로 로드되었습니다.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        exit(1)

    # 모델 학습
    try:
        model.train(
            data=data_yaml,
            epochs=70,  # 에포크 수 증가
            imgsz=640,
            batch=16,  # 배치 크기 유지
            lr0=0.001,  # 초기 학습률 유지
            lrf=0.0001,  # 최종 학습률 유지
            patience=10,  # 조기 종료를 위한 patience 조정
            save_period=1, # 모델 체크포인트 저장 주기
            weight_decay=5e-4,  # 가중치 감소 추가
            warmup_epochs=3,  # 초기 워밍업 에포크 수
            warmup_momentum=0.8,  # 초기 워밍업 모멘텀
            box=0.05,  # Box loss gain
            cls=0.5,  # Class loss gain
            dfl=1.0   # DFL (Distribution Focal Loss) gain
        )
        print("모델 학습이 시작되었습니다.")
    except Exception as e:
        print(f"Error during training: {e}")

