# chl_monodepth2

# 0. Overview
This Project is for learning Monodepth2 and Visualizing the model. Not Accuracy.  
<br>
RViz Image  
![monodepth2_rviz_2022_08_10_19_34_31](https://user-images.githubusercontent.com/58837749/184044738-99e6b878-2336-4bf4-b93e-9d71c1531b35.gif)  
<br>
Open3D Image  
![monodepth2_open3d_2022_08_10_22_33_24](https://user-images.githubusercontent.com/58837749/184044751-4c76e36e-5972-4676-b30f-104e55fb5be7.gif)  
<br>

# 1. Requirement
## 1. Developed OS
This project is developed in docker image "nvidia/cuda11.7.0-devel-ubuntu18.04"

## 2. Python
You need python version 3.8.  
All pip dependencies are set to python version 3.8.

## 3. requirements.txt
You can install pip dependencies  
```bash
$ sudo python3.8 -m pip install -r requirements.txt
```

## 4. cv_bridge for python3.8 on ubuntu18.04
You can refer this paga  
[stack overflow - how to install cv_bridge](https://stackoverflow.com/questions/49221565/unable-to-use-cv-bridge-with-ros-kinetic-and-python3)

## 5. tf_publisher_gui
For convenience to control tf in RViz, you can use this Thirdparty "tf_publisher_gui". Because the coordinate systems of sensors are different but if you don't have physical information, this is good for you.  
[install tf_publisher_gui](https://github.com/yinwu33/tf_publisher_gui.git)

## 6. Etc
You need to install other necessary things during installing.


# 2. Run
You should modify arguments in "mono2.launch" such as model_path etc.
```bash
source devel/setup.bash
roslaunch mono mono2.py
```

# 3. Limitations
## 1. Real Time  
This is not suitable for using in real time because it takes about 0.2 sec(5FPS) to process from subscribing image topic to publishing estimated depth image.

## 2. Evalutation  
If you don't have precise calibration information, it is hard to evaluate accuracy. Especially, if you have Lidar data and calibration information, you can reproject this Lidar dara to 2D image and on that 2D coordinate(x,y) you can compare and evaluate GT(Lidar Data) and estimated Depth(model). For this process, an understanding of linear algebra is essential. However, for evaluating GT and estimated depth, monodepth2 uses a mask, but it is hard to make that mask using Lidar Data.  
[monodepth2 evalutation code](https://github.com/nianticlabs/monodepth2/blob/master/evaluate_depth.py)  

## 3. Open3D
For rendering non-blocking visualization in Open3D, you should know transformation Matrix. To get that information in monodepth2, you need to use pose-estimation-model, but it takes more time if you use if. So I chose to render image frame by one.  
![Screenshot from 2022-08-10 22-07-53](https://user-images.githubusercontent.com/58837749/184052001-cfc089ef-6aac-4492-b05b-8581a9771782.png)  
(this is a rendering image of Lidar data using Open3D)

## 4. RViz
For converting Open3D data(float type) to ROS data type(binary type), it should be an integer type. So a lot of information loss occurs during the conversion process.  
Compared to Open3D rendering images, it is very sparse.

## 5. Onnx & Tensorrt
I wanted model lightening, so I tried to use onnx and tensorrt. I wrote all codes. But input a decoder is a list of tensors, but method "torch.onnx.export" needs input type "tensor". So I couldn't convert onnx(I successfully did encoder). So it is necessary to keep track of the problem.


