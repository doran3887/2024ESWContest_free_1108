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


def detect(opt):
    source, weights, imgsz = opt.source, opt.weights, opt.img_size
    view_img = True  # 항상 실시간 처리 중 이미지를 확인

    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()

    # 모델 로드
    stride = 32
    model = torch.jit.load(weights)  # 모델 로드
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision은 CUDA에서만 지원됨
    model = model.to(device)

    if half:
        model.half()  # FP16으로 변환
    model.eval()

    # 데이터 로더 설정
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # SORT 트래커 초기화
    mot_tracker = Sort()

    # 추론 실행
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # 한 번 실행

    t0 = time.time()
    frame_count = 0
    for path, img, im0s, vid_cap in dataset:
        frame_count += 1
        if frame_count % 4 != 0:
            continue  # 모든 프레임 중 세 번째 프레임마다 스킵

        # 이미지 데이터를 텐서로 변형하고 모델에 맞게 전처리 하는 과정
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8을 fp16/32로 변환
        img /= 255.0  # 0 - 255를 0.0 - 1.0으로 변환

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 추론
        t1 = time_synchronized()
        [pred, anchor_grid], seg, ll = model(img)
        t2 = time_synchronized()

        # 비효율적인 시간 소비: torch.jit.trace의 비호환성으로 인해 데모 버전에서 추가 시간 소비 발생
        # 그러나 이 문제는 공식 버전에서는 나타나지 않음
        tw1 = time_synchronized()
        pred = split_for_trace_model(pred, anchor_grid)
        tw2 = time_synchronized()

        # NMS 적용
        t3 = time_synchronized()
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t4 = time_synchronized()

        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)

        # 차선 및 주행 영역에 대한 픽셀 값 가져오기
        lane_pixels = np.sum(ll_seg_mask)
        driving_area_pixels = np.sum(da_seg_mask)

        detections = []
        # 탐지 결과 처리
        for i, det in enumerate(pred):  # 이미지당 탐지 결과
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # 경로로 변환
            s += '%gx%g ' % img.shape[2:]  # 출력 문자열
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 정규화 게인 whwh
            if len(det):
                # img_size에서 im0 크기로 박스 재조정
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # SORT를 위한 탐지 결과 준비
                for *xyxy, conf, cls in det:
                    xyxy = torch.tensor(xyxy).cpu().numpy()  # GPU에서 CPU로 복사
                    conf = conf.cpu().numpy()  # GPU에서 CPU로 복사
                    cls = cls.cpu().numpy()  # GPU에서 CPU로 복사
                    detections.append([*xyxy, conf, cls])

                # 탐지 결과를 numpy 배열로 변환
                detections = np.array(detections)
                print(f'Detections: {detections}')  # 디버깅을 위해 추가

                # SORT 트래커 업데이트
                track_bbs_ids = mot_tracker.update(detections[:, :5])  # 클래스 ID 제외하고 트래커 업데이트
                print(f'Track BBS IDs: {track_bbs_ids}')  # 디버깅을 위해 추가

                # 결과 출력
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # 클래스당 탐지 결과
                    print(f"Detected class {int(c)}: {bdd100k_classes[int(c)]}, Count: {n}")

                # 결과 작성
                for track_bb in track_bbs_ids:
                    xyxy = track_bb[:4].astype(int)
                    track_id = int(track_bb[4])
                    cls_index = np.where(np.all(np.isclose(detections[:, :4], xyxy, atol=100), axis=1))[0]
                    if cls_index.size > 0:
                        cls = int(detections[cls_index[0], 5])  # tolerance와 일치하는 cls 찾기
                    else:
                        cls = None

                    if cls is not None and cls < len(bdd100k_classes):  # cls 값이 리스트 범위 내에 있는지 확인
                        label = f'{bdd100k_classes[cls]} {track_id}'
                        color = track_id
                        plot_one_box(xyxy, im0, color=color, label=label, line_thickness=3)
                    else:
                        # 기본 라벨로 경계 상자를 그립니다.
                        plot_one_box(xyxy, im0, line_thickness=3)

            # 결과 표시
            show_seg_result(im0, (da_seg_mask, ll_seg_mask), is_demo=True)
            cv2.imshow(str(p), im0)
            if cv2.waitKey(1) == ord('q'):  # q를 눌러 종료
                raise StopIteration

        inf_time.update(t2 - t1, img.size(0))
        nms_time.update(t4 - t3, img.size(0))
        waste_time.update(tw2 - tw1, img.size(0))

    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg, nms_time.avg))
    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    opt = make_parser().parse_args()

    print(opt)

    with torch.no_grad():
        detect(opt)
