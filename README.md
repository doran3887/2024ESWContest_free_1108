<br><br>

🚗ASAP-Accelerator_Stop_Assistance_Program 
===========================================

<br>

## Introdution

**This project is a driving assistance program to prevent incorrect operation of Excel when driving a vehicle.**  <img src = "https://img.icons8.com/?size=40&id=VF7zRdVii0QF&format=png&color=000000">

**In recent years, sudden acceleration accidents have occurred frequently and have become quite an issue. A sudden acceleration accident is an accident caused by a mechanical defect in a vehicle. However, to get to the bottom of most sudden acceleration accidents, it was found that it was not a sudden acceleration accident caused by a mechanical defect, but an acceleration pedal misconception caused by the driver's poor driving. Although the situation has to stop due to the confusion between the acceleration pedal and the brake, accidents often occur by continuing to misstep the acceleration pedal. Therefore, a device has been created to distinguish whether it is a sudden acceleration accident such as a foot cam or a misoperation accident, and in order to prevent such accidents, a technology to prevent misoperation of the acceleration pedal is emerging by using various sensors attached to the vehicle.**

**In this society, the project aimed to design an "accelerated pedal malfunction prevention device" (ASAP) that can be installed in almost any vehicle. The currently developed "accelerated pedal malfunction prevention device" cannot be attached to aging vehicles or vehicles that do not fit into developed devices because multiple sensors must be attached to the vehicle and it is designed for a specific vehicle. Because the "accelerated pedal malfunction case" is a "misoperation case," it is especially common for older people with relatively poor cognitive abilities. Owners owned by older people are also likely to be aged, so they will not be able to benefit from the technology even if new technology is introduced.**

**'ASAP', which is to be developed in this project, is a simple form consisting of a single camera and two linear motors that do not require a special sensor and can be mounted on almost any vehicle. With the speed of driving, the speed limit of the current GPS value-based road, and vision data obtained through the camera, accidents that can be caused by incorrect manipulation of the accelerator pedal are prevented in advance.**

<br>

## Languages
![Python](https://img.shields.io/badge/Language-Python-blue?logo=python)
![C++](https://img.shields.io/badge/Language-C++-informational?logo=c%2B%2B)
![Kotlin](https://img.shields.io/badge/Language-Kotlin-orange?logo=kotlin)

## Tools
![Arduino IDE](https://img.shields.io/badge/Tool-Arduino%20IDE-00979D?logo=arduino)
![Fusion 360](https://img.shields.io/badge/Tool-Fusion%20360-DD5232?logo=autodesk)
![VSCode](https://img.shields.io/badge/Tool-VSCode-007ACC?logo=visual-studio-code)
![Android Studio](https://img.shields.io/badge/Tool-Android%20Studio-3DDC84?logo=android-studio)

## Environment
![Windows 10](https://img.shields.io/badge/Environment-Windows%2010-0078D6?logo=windows)
![Anaconda](https://img.shields.io/badge/Environment-Anaconda-44A833?logo=anaconda)



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

```bash
git clone https://github.com/abewley/sort.git
```

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
4. Install GPS.apk in your mobile phone
**Our HardWare Desgin**

<br>

<img src="https://github.com/user-attachments/assets/2c22d3f0-f56a-42b4-bfc0-a7b036db3d70" width="400">

<img src="https://github.com/user-attachments/assets/307cecb1-7a6f-45a7-ac4f-ddaa7d1c9c73" width="400">


<br><br>

**You can get this design in foder 'hardware design'**

<br>

---

<br>

## Configuration

```bash
|__README.md
|__environment.yaml
|__GPS.apk.zip
|__control.py
|__YOLO.py
   |__best.pt
   |__accel_warning.wav
   |__test1.mp4
   |__GPS_func.py
      |__gps.csv

```

---

<br>

## Launch


### Demo - video(test1.mp4)

![icons8-check-48](https://github.com/user-attachments/assets/abd20a99-c74a-4c44-9998-ef5111c85fba) **Check list**

 
![icons8-1-24](https://github.com/user-attachments/assets/d4387ef8-bade-4ee8-bb4b-df0b7a7f7bfa) You need to check port(Serial), URL(GPS.apk app) before launch


![icons8-2-24](https://github.com/user-attachments/assets/3c1e64e4-22fd-4fa5-a76a-c57b1412f61c) Your laptop or desktop should be connected to the same network as your mobile phone with the GPS app installed



![icons8-3-24](https://github.com/user-attachments/assets/75f44465-dce2-47f1-8ed2-f782427568fc) You can get URL when you launch GPS.apk



```bash
conda activate [env_name]
cd [to\your\clone\path]
python control.py
```

<br>

**1.Set ROI**

<br>

<img src = "https://github.com/user-attachments/assets/ef91ce05-6222-4fa3-a49b-b500ca718bbc" width="640">

<br><br>

**2.Obj Imagw**

<br>

<video src="https://github.com/user-attachments/assets/f4925c19-2b0e-4787-82e0-bed0aad6adc8" width="640" height="360" controls>
  Your browser does not support the video tag.
</video>

<br>

**3.True Image**

<video src="https://github.com/user-attachments/assets/168b21e0-25ea-487a-820b-792882d8ac2a" width="640" height="360" controls>
  Your browser does not support the video tag.
</video>

<br>

### If you don't have OBD2, you can speed up using keyboard

<img src="https://github.com/user-attachments/assets/fc0c124c-3e33-4d43-939d-b11b87bf796b" width="40" height="40">
  speed up 1

<img src="https://github.com/user-attachments/assets/e3e0d6e1-7803-4b54-a39e-3f7c16433404" width="40" height="40">
  speed down 1

<br>

### If you have OBD2 change code in cotrol.py
1. Tinize line 139 and 140
2. Add below code in main function before while loop
```python
connection = initialize_obd_connection

```
3. Add below code in line 159
```python
curren_velocity = get_velocity(connection)
```

---

<br>

## Args
```bash
python control.py --video_path another_video.mp4 --server_url http://example.com --com_port COM3
```

<br>

**video_path : video path or webcam(0)**

<br>



<br>





# CopyRight of DataSet for GPS Location and YOLO training
- [DataSet for GPS Location](https://www.data.go.kr/data/15028200/standard.do)
- [DataSet for YOLO training](https://dl.cv.ethz.ch/bdd100k/data/box_track_labels_trainval.zip)



