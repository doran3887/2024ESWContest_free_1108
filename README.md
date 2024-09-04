🚗ASAP-Accelerator_Stop_Assistance_Program 
===========================================

<br>
<br>

**This repo is a driving assistance program to prevent incorrect operation of Excel when driving a vehicle.**  <img src = "https://img.icons8.com/?size=40&id=VF7zRdVii0QF&format=png&color=000000">

<br>

---
## Repo We used

<br>

>**YOLO v10: Object Recognition for Driving** 
>
>YOLO v10 is the latest model for object recognition while driving, offering exceptional performance in real-time scenarios.
>
>- **🔗 Repository**: [YOLO v10 on GitHub](https://github.com/THU-MIG/yolov10)
>- **🚀 Features**:
>  - High-speed object recognition performance
>  - Optimized for real-time response
> - High detection rates in various environments
>
>
>**Metric 3D: Depth Estimation**
>
>Metric 3D provides advanced depth estimation capabilities, essential for understanding the positioning of >objects in 3D space.
>
>- **🔗 Repository**: [Metric 3D on GitHub](https://github.com/YvanYin/Metric3D)
>- **🚀 Features**:
>  - Accurate depth estimation
>  - Optimized for 3D spatial recognition
>  - Applicable across various use cases
>
>
>**Sort: Object Tracking**
>
>Sort (Simple Online and Realtime Tracking) is a simple yet effective algorithm for object tracking, providing lightweight tracking functionality.
>
>- **🔗 Repository**: [Sort on GitHub](https://github.com/abewley/sort?tab=readme-ov-file)
>- **🚀 Features**:
>  - Simple and fast tracking algorithm
>  - Optimized for real-time object tracking
>  - Lightweight performance suitable for various devices
---

<br>

## Setup Env


### Clone repo
```bash
mkdir [your ws name]
cd [your ws name]
git clone https://github.com/doran3887/ASAP-Accelerator_Stop_Assistance_Program-.git
```

<br>

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

<br>

### Set virtual env - Conda
```bash
conda env export > environment.yaml
conda env create -f environment.yaml -n [env_name]
conda activate [env_name]
```

---

<br>

## Extra setting - Hardware and dataset

**To follow our repo, you should set Hardware below**
1. OBD2 - any OBD2 that uses OBD lib(python)
2. [Linear motor(MIGHTY ZAP)](https://smartstore.naver.com/irrobot/products/4937561648)
3. Add [lib](https://drive.google.com/file/d/1gnpz7gdhOqTuFVuHxabJKpLKsAKc_nWO/view) to your [Arduino IDE](https://www.arduino.cc/en/software)

**Our HardWare Desgin**

<br>

<img src="https://github.com/user-attachments/assets/2c22d3f0-f56a-42b4-bfc0-a7b036db3d70" width="400">

<img src="https://github.com/user-attachments/assets/307cecb1-7a6f-45a7-ac4f-ddaa7d1c9c73" width="400">


<br><br>

**You can get this design in foder 'hardware design'**

<br>

## Launch

<br>

### Demo - video

<br>

```bash
conda activate [env_name]
cd [to\your\clone\path]
python final_final_control.py
```

<br>

**Obj Imagw**
https://github.com/user-attachments/assets/1727dafe-2723-4974-ae4b-93eff640c0cf

**True Image**



<br>

1. Install 




## apk

apk의 역할

apk 스마트폰에 설치

apk 사용법


# CopyRight of DataSet for GPS Location and YOLO training
- [DataSet for GPS Location](https://www.data.go.kr/data/15028200/standard.do)
- [DataSet for YOLO training](https://dl.cv.ethz.ch/bdd100k/data/box_track_labels_trainval.zip)



