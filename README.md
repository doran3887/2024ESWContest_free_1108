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
4. Install GPS.apk in your mobile phone
**Our HardWare Desgin**

<br>

<img src="https://github.com/user-attachments/assets/2c22d3f0-f56a-42b4-bfc0-a7b036db3d70" width="400">

<img src="https://github.com/user-attachments/assets/307cecb1-7a6f-45a7-ac4f-ddaa7d1c9c73" width="400">


<br><br>

**You can get this design in foder 'hardware design'**

<br>

## Launch

<br>

### Demo - video(test1.mp4)

**You need to check port(Serial), URL(GPS.apk app) before launch**

<br>

**Your laptop or desktop should be connected to the same network as your mobile phone with the GPS app installed**

<br>

**You can get URL when you launch GPS.apk**

<br>

```bash
conda activate [env_name]
cd [to\your\clone\path]
python final_final_control.py
```

<br>

**1.Set ROI**

<br>

<img src = "https://github.com/user-attachments/assets/ef91ce05-6222-4fa3-a49b-b500ca718bbc" width="640">
<br>

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

## args
```bash
python final_final_control.py --video_path another_video.mp4 --server_url http://example.com --com_port COM3
```

<br>

**video_path : video path or webcam(0)**

<br>



<br>





# CopyRight of DataSet for GPS Location and YOLO training
- [DataSet for GPS Location](https://www.data.go.kr/data/15028200/standard.do)
- [DataSet for YOLO training](https://dl.cv.ethz.ch/bdd100k/data/box_track_labels_trainval.zip)



