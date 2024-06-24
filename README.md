# Depth-Supervised Dynamic NeRF #  
### MixVoxels with depth ###
#### Distill static model depth to Dynamic model ####

--- 
## RESULT ##
<p align="center">
  <img src="https://github.com/kyuminKim00/Depth-Supervised-Dynamic-NeRF/assets/112574294/82530f3e-c22e-4088-93f2-3cf18658b45e" width="700" height="auto">
</p>

--- 
## PIPELINE ##
<p align="center">
  <img src="https://github.com/kyuminKim00/Depth-Supervised-Dynamic-NeRF/assets/112574294/26aa842d-1aad-42c7-b341-a0c9ae5b68d1" width="800" height="auto">
</p>

--- 
## COMPARISON ##
<p align="center">
  <img src="https://github.com/kyuminKim00/Depth-Supervised-Dynamic-NeRF/assets/112574294/f8aa01d1-5c41-4316-95b2-02c6fd5369ae" width="500" height="auto">
</p>

--- 
## 실행 방법 ##
### Static model 만들기 ###
1. 하나의 카메라로 장면 이미지 얻기(카메라 위치 당 이미지 1장)  

2. static data dir 구성
static_dir<br/>
|frames_?<br/>
&emsp;|cam00<br/>
&emsp;&emsp;|0.jpg, 1.jpg, ...<br/>
&emsp;|cam01...<br/>

3. static data 준비<br/>
'''
python prepare_static.py --static_data_path [data_path] --frame_num [n_frames]  
'''

4. static config file 만들기  

5. static model 학습  
'''
  python train --config [static_config_path] --render_path 0
'''

### Dynamic model 학습 ###
5. Dynamic data dir 구성<br/>
dynamic_dir<br/>
|frames_?<br/>
&emsp;|cam00<br/>
&emsp;&emsp;|0.jpg, 1.jpg, ...<br/>
&emsp;|cam01... (카메라 위치 당 N frames)<br/>
|depth<br/>
|poses_bounds.npy<br/>

6. dynamic config file 만들기  
  depthmap_npy_path 에 5번에서 생성한 all_depth.npy 경로 설정

7. 학습한 static model로 depth 추출  
'''
  python static_model2depth.py --config [dynamic_config_path] --ckpt [static_model_ckpt_path] --cam_num_list [cam1 cam2 ...]
  inference cam 제외(inference cam은 학습에 사용되지 않아서 depth 추출할 필요 없음)
'''

8. dynamic model 학습  
'''
  python train.py --config [dynamic_config_path] --render_path 0
'''

---  

### COLMAP depth 사용(DS-NeRF) ###
1. data/your_datadir/video 에 동영상 넣기(cam0, cam1, cam2 ,...)  
데이터중에 cam0는 testset, 나머지는 trainset, cam0 depth, rgb는 훈련에서 사용하지 않음

2. 비디오를 프레임 단위로 자르기  
'''
  python tools/prepare_video.py [data_dir]  
'''
3. 각 카메라마다 이미지 한 장씩 새로운 폴더에 넣어서 colmap 실행  
카메라 포즈, colmap depth 추출

4. colmap 실행한 폴더를 사용해서 imgs2poses.py 실행(LLFF 코드 참조) -> poses_bounds.npy 파일 생성  
'''
  python imgs2poses.py [data_dir] 
'''

5. colmap 실행한 폴더를 사용해서 colmap2depth.py 실행 -> colmap_depth.npy 파일 생성
'''
  python colmap2depth.py [data_dir]  
'''

6. 생성된 poses_bounds.npy, colmap_depth.npy 파일을 기존의 데이터 디렉토리에 이동  
디렉토리 구성  
data  
|frames  
&emsp;|cam00, cam01, cam02, ...  
|videos  
&emsp;|cam00, cam01, cam02, ...  
|images  
&emsp;|1.jpg, 2.jpg, ..., (colmap 실행할 때 사용한 이미지, 카메라 당 하나)  
|sparse  
&emsp;|0  
&emsp;&emsp;|cameras.bin, images.bin, points3D.bin  
|colmap_depth.npy  
|poses_bounds.npy  

7. TRAIN  
'''
  python train.py --config [config] render_path 0 --use_depth 1  
'''

---  
- 하나의 view에서 동영상 렌더링   
'''
  python render_one_video.py --config [config file path] --ckpt [checkpoint path] --cam_num [rendering cam number]
'''
## Reference 
1. [ICCV2023 Oral] MixVoxels: Mixed Neural Voxels for Fast Multi-view Video Synthesis   
[Project Page](https://fengres.github.io/mixvoxels/) 
2. [CVPR2022] Depth-supervised NeRF: Fewer Views and Faster Training for Free  
[Project Page](https://www.cs.cmu.edu/~dsnerf/)

Installation, Dataset 설명은 MixVoxels Git에 나와있음
