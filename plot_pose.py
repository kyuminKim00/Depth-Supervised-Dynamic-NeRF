import numpy as np
from Gen_pose_bounds import *

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

basement = np.load("/data2/kkm/km/mixvoxels/data/basement/poses_bounds.npy")
steak = np.load("/data2/kkm/km/mixvoxels/data/flame_steak/poses_bounds.npy")
print(steak[0, :])
print(steak[1, :])
print(steak[2, :])
print(steak[3, :])
printdatasetINFO(0, steak)
printdatasetINFO(1, steak)
printdatasetINFO(2, steak)
printdatasetINFO(3, steak)






# 예시 카메라 포즈 리스트
camera_poses = [
    [5.017494760198344, -1.030802988650452, 1.0795405806437084, 1.59939433, 0.10142021, 2.56938885],  # 카메라 1
    [1.1230897299243194, -1.2955050901838918, -0.2920971584520613, 1.57827371, 0.07956426, 3.04846918],  # 카메라 2
    [-2.126745681859328, -1.3733993152600037, -0.11568647646977881, 1.57423859,  0.09559837, -2.84438386],# 카메라 3
    [-4.029084192199031, -1.2959633762514087, 0.8989345157417333, 1.57393309,  0.10439937, -2.59397906]   # 카메라 4
]



# 예시 좌표 리스트
x_list = [5.017494760198344, 1.1230897299243194, -2.126745681859328, -4.029084192199031]
y_list = [-1.030802988650452, -1.2955050901838918, -1.3733993152600037, -1.2959633762514087]
z_list = [1.0795405806437084, -0.2920971584520613, -0.11568647646977881, 0.8989345157417333]

#rx1 rx2 rx3
# cam1_pose = np.array([[-0.028447171015281974, 0.9944545834301041, -0.10124643181902988, 5.017494760198344], 
#                     [0.8387982982171259, 0.07884046176437519, 0.5387036258434071, -1.030802988650452], 
#                     [0.5436986052669996, -0.06960074053943865,  -0.8363896003341271, 1.0795405806437084],
#                     [0, 0, 0, 1]])

# cam2_pose = np.array([[-0.007453655218185679, 0.9968085669061868, -0.07948033697917092, 1.1230897299243194], 
#                         [0.995584044401637, 0.0148354974732204, 0.09269476008707997, -1.2955050901838918],
#                          [0.09357806130054018, -0.07843844055790646, -0.992517283217817, -0.2920971584520613],
#                          [0, 0, 0, 1]])
# cam3_pose = np.array([[-0.0034265372903273827, 0.9954280576273004, -0.09545282044307614, -2.126745681859328],
#                      [0.956248184468991, -0.024662094581487257, -0.29151533542956604, -1.3733993152600037],
#                       [-0.2925366106010608, -0.09227547441868784, -0.9517917672888588, -0.11568647646977881], 
#                       [0, 0, 0, 1]])
# cam4_pose = np.array([[-0.003119679193679128, 0.994550440778185, -0.10420982846951796, -4.029084192199031 ],
#                          [0.8539354298480775, -0.05157864466673521, -0.5178164974819959, -1.2959633762514087],
#                           [-0.5203696275263456, -0.09060388602185061,  -0.8491209493269094, 0.8989345157417333],
#                           [0, 0, 0, 1]])

#rx1 ry1 rz1 -> 이게 맞음

cam1_pose = np.array([[-2.84471710e-02,  8.38798298e-01,  5.43698605e-01,  5.01749476e+00], 
    [9.94454583e-01,  7.88404618e-02, -6.96007405e-02, -1.03080299e+00],
    [-1.01246432e-01,  5.38703626e-01,-8.36389600e-01,  1.07954058e+00],
    [0, 0, 0, 1]])
cam2_pose = np.array([[-7.45365522e-03,  9.95584044e-01,  9.35780613e-02,  1.12308973e+00],
    [9.96808567e-01,  1.48354975e-02, -7.84384406e-02,-1.29550509e+00],
    [-7.94803370e-02,  9.26947601e-02, -9.92517283e-01, -2.92097158e-01],
    [0, 0, 0, 1]])
cam3_pose = np.array([[-3.42653729e-03,  9.56248184e-01, -2.92536611e-01, -2.12674568e+00],
    [9.95428058e-01, -2.46620946e-02 ,-9.22754744e-02, -1.37339932e+00],
    [-9.54528204e-02, -2.91515335e-01,-9.51791767e-01 ,-1.15686476e-01],
    [0, 0, 0, 1]])
cam4_pose = np.array([[-3.11967919e-03,  8.53935430e-01 ,-5.20369628e-01,-4.02908419e+00],
    [9.94550441e-01, -5.15786447e-02,-9.06038860e-02, -1.29596338e+00],
    [-1.04209828e-01 ,-5.17816497e-01,-8.49120949e-01 , 8.98934516e-01],
    [0, 0, 0, 1]])

pose = [cam1_pose, cam2_pose, cam3_pose, cam4_pose]

def plot_multiple_camera_poses(poses, filename):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
   

    for pose in poses:
        # 카메라 위치
        camera_pos = pose[:3, 3]
        
        # 카메라 방향
        camera_dir = pose[:3, :3] @ np.array([[0, 0, 1]]).T
        
        ax.quiver(camera_pos[0], camera_pos[1], camera_pos[2],
                  camera_dir[0], camera_dir[1], camera_dir[2], color='r', label='Camera Direction')

        
        # 카메라의 x, y, z 축
        camera_x_axis = pose[:3, :3] @ np.array([[1, 0, 0]]).T
        camera_y_axis = pose[:3, :3] @ np.array([[0, 1, 0]]).T
        camera_z_axis = pose[:3, :3] @ np.array([[0, 0, 1]]).T
        
        ax.quiver(camera_pos[0], camera_pos[1], camera_pos[2],
                  camera_dir[0], camera_dir[1], camera_dir[2], color='r', label='Camera Direction')
        ax.quiver(camera_pos[0], camera_pos[1], camera_pos[2],
                  camera_x_axis[0], camera_x_axis[1], camera_x_axis[2], color='g', label='Camera X-Axis')
        ax.quiver(camera_pos[0], camera_pos[1], camera_pos[2],
                  camera_y_axis[0], camera_y_axis[1], camera_y_axis[2], color='b', label='Camera Y-Axis')
        ax.quiver(camera_pos[0], camera_pos[1], camera_pos[2],
                  camera_z_axis[0], camera_z_axis[1], camera_z_axis[2], color='y', label='Camera Z-Axis')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Multiple Camera Poses')
    ax.set_xlim(-10, 10)  # X 축 범위 조정
    ax.set_ylim(-10, 10)  # Y 축 범위 조정
    ax.set_zlim(-10, 10)  # Z 축 범위 조정
    plt.savefig(filename)
    plt.close()

plot_multiple_camera_poses(pose, "multiple_camera_poses_3d.png")


def plot_camera_poses_2d(poses, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.figure(figsize=(8, 6))

    for pose in poses:
        camera_pos = pose[:3, 3]
        camera_dir = pose[:3, :3] @ np.array([[0, 0, 1]]).T
        
        plt.scatter(camera_pos[0], camera_pos[1], color='b', marker='o')
        plt.quiver(camera_pos[0], camera_pos[1], camera_dir[0], camera_dir[1], color='r', scale=1, scale_units='xy', angles='xy')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Camera Poses in 2D')
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(filename)
    plt.close()
plot_camera_poses_2d(pose, 'multiple_camera_poses_2d.png')
