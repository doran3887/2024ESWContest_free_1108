# ASAP-Accelerator_Stop_Assistance_Program




**This repo is a driving assistance program to prevent incorrect operation of Excel when driving a vehicle.**



---
## Repo We used

**YOLO v10: Object Recognition for Driving** 

YOLO v10 is the latest model for object recognition while driving, offering exceptional performance in real-time scenarios.

- **🔗 Repository**: [YOLO v10 on GitHub](https://github.com/THU-MIG/yolov10)
- **🚀 Features**:
  - High-speed object recognition performance
  - Optimized for real-time response
  - High detection rates in various environments


**Metric 3D: Depth Estimation**

Metric 3D provides advanced depth estimation capabilities, essential for understanding the positioning of objects in 3D space.

- **🔗 Repository**: [Metric 3D on GitHub](https://github.com/YvanYin/Metric3D)
- **🚀 Features**:
  - Accurate depth estimation
  - Optimized for 3D spatial recognition
  - Applicable across various use cases


**Sort: Object Tracking**

Sort (Simple Online and Realtime Tracking) is a simple yet effective algorithm for object tracking, providing lightweight tracking functionality.

- **🔗 Repository**: [Sort on GitHub](https://github.com/abewley/sort?tab=readme-ov-file)
- **🚀 Features**:
  - Simple and fast tracking algorithm
  - Optimized for real-time object tracking
  - Lightweight performance suitable for various devices



---

## Setup Env

### Clone repo
```bash
mkdir [your ws name]
cd [your ws name]
git clone https://github.com/doran3887/ASAP-Accelerator_Stop_Assistance_Program-.git
```

### Clone traking repo 'Sort'
- **Repository**: [YOLO v10 on GitHub](https://github.com/THU-MIG/yolov10)


**To use Sort tracker** 
 1. Download the [2D MOT 2015 benchmark dataset](https://motchallenge.net/data/MOT15/)
 2. Create a symbolic link to the dataset - Run Command Prompt as Administrator
    ```bash
    mklink /D [symbolic_link_path] [dataset_path]
    
ex)
```bash
mklink /D "C:\Users\kyle\Desktop\project_1\dataset_link" "C:\Users\kyle\Datasets\my_dataset"
```

**When the command executes successfully, you will see a message like this:**
```bash
symbolic link created for C:\Users\kyle\Desktop\project_1\dataset_link <<===>> C:\Users\kyle\Datasets\my_dataset
```


### Set virtual env - Conda
```bash
conda env export > environment.yaml
conda env create -f environment.yaml -n [env_name]
conda activate [env_name]
```

---


mkdir ~/

git clone ~yolo  주소
pip install ~
conda ~~

## demo

train code 

test code

## hardware code
hardware 구성도 (port 구성)

.ino

## apk

apk의 역할

apk 스마트폰에 설치

apk 사용법



