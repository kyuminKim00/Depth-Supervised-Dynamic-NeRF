# MixvVoxels With DEPTH #
### MixVoxels 논문 참고 ###

# 실행 방법 #
1. data/your_datadir/video 에 동영상 넣기(cam0, cam1, cam2 ,...)
(데이터중에 cam0는 testset, 나머지는 trainset, cam0 depth, rgb는 훈련에서 사용하지 않음)

2. python tools/prepare_video.py [data_dir]
비디오를 프레임 단위로 자르기

3. 각 카메라마다 이미지 한 장씩 새로운 폴더에 넣어서 colmap 실행(카메라 포즈, colmap depth 추출)
![alt text](etc/colmap_pointcloud.pngimage.png)

4. colmap 실행한 폴더를 사용해서 imgs2poses.py 실행(LLFF 코드 참조) -> poses_bounds.npy 파일 생성
python imgs2poses.py [data_dir] 

5. colmap 실행한 폴더를 사용해서 colmap2depth.py 실행 -> colmap_depth.npy 파일 생성

6. 생성된 poses_bounds.npy, colmap_depth.npy 파일을 기존의 데이터 디렉토리에 이동
데이터 구성
data
|frames
  |cam00, cam01, cam02, ...
|videos
  |cam00, cam01, cam02, ...
|images
  |1.jpg, 2.jpg, ..., (colmap 실행할 때 사용한 이미지, 카메라 당 하나)
|sparse
  |0
    |cameras.bin, images.bin, points3D.bin
|colmap_depth.npy
|poses_bounds.npy

7. TRAIN
python train.py --config [config] render_path 0


하나의 view에서 동영상 렌더링 
python render_one_image.py --config configs/schedule5000/flame_steak_5000.txt --ckpt log/flame_steak_5000/flame_steak_5000.th

## Reference 
1. [ICCV2023 Oral] MixVoxels: Mixed Neural Voxels for Fast Multi-view Video Synthesis ##
[Project Page](https://fengres.github.io/mixvoxels/) 
2. [CVPR2022] Depth-supervised NeRF: Fewer Views and Faster Training for Free
[Project Page](https://www.cs.cmu.edu/~dsnerf/)

installation, Dataset 설명은 MixVoxels Git에 나와있음
