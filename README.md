# Facial Recognition Mini Project
# Group: Positive Mental Attitude
## Members<br>
| ชื่อ | นามสกุล | รหัสนักศึกษา |
| --- | --- | --- |
|เดชพนต์| นุ่นเสน| 62070070|
|ณัฐวัตน์| สามสี| 62070067|
|ธีรภัทร |บุญช่วยแล้ว| 62070096|
|ชวิน |โลห์รัตนเสน่ห์ |62070045|
## Preparations
### Install Dependency
```
conda install --file requirements.txt
```
### Datasets
<strong>[Datasets](https://drive.google.com/file/d/1qy2POaMjaYG_R7tY9YMd8__fye8KhePf/view?usp=sharing)</strong>

## Project 2 Handcraft Base
### How to Run
```
python3 handcraft_based.py
```

 - train - เรียกใช้ผ่านฟังก์ชั่น ```train()```
 - test - เรียกใช้ผ่านฟังก์ชั่น ```test()```
### เปลี่ยน Datasets
![handcraft path](./assets/imgs/handcraft_path.png)
สามารเปลี่ยนได้โดยการ แก้ค่าของตัวแปร TEST_PATH และ TRAIN_PATH

## Project 3 Learning Base
### How to Run
```
python3 learning_based.py
```
 - train - เรียกใช้ผ่านฟังก์ชั่น ```main()```
### เปลี่ยน Datasets
![learning path](./assets/imgs/learning_path.png)
สามารเปลี่ยนได้โดยการ แก้ค่าของตัวแปร TEST_ROOT และ TRAIN_ROOT
