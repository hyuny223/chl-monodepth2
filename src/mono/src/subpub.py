import numpy as np
import time

import PIL.Image as pil
from utils import process_open3d

import rospy
from sensor_msgs.msg import PointCloud2, Image
import message_filters

import cv2, open3d
from cv_bridge import CvBridge
from cv_bridge.boost.cv_bridge_boost import getCvType

from utils import calibration, draw_open3d
from depth import process_data, inference

class SubPub_v1:
    def __init__(self, int_param, distortion, int_param_scaling,
                 original_height, original_width,
                 feed_height, feed_width,
                 device, encoder, depth_decoder,
                 K_,Tcw, Twl):

        self.int_param = int_param
        self.distortion = distortion
        self.int_param_scaling = int_param_scaling
        self.original_height = original_height
        self.original_width = original_width
        self.feed_height = feed_height
        self.feed_width = feed_width
        self.device = device
        self.encoder = encoder
        self.depth_decoder = depth_decoder
        self.K_ = np.array(K_)
        self.Tcw = Tcw
        self.Twl = np.vstack((Twl,np.array([0,0,0,1])))


        self.cv_image = None
        self.lidar_data = None
        self.bridge = CvBridge()

        self.lidar_sub = message_filters.Subscriber("/rslidar_points", PointCloud2)
        self.image_sub = message_filters.Subscriber("/pylon_camera_node/image_raw", Image)

        self.lidar_pub = rospy.Publisher("/original/points", PointCloud2, queue_size=1)
        self.image_pub = rospy.Publisher("/estimated/points", PointCloud2, queue_size=1)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.lidar_sub, self.image_sub],1, 0.02, allow_headerless=True)
        # self.ts = message_filters.TimeSynchronizer([self.lidar_sub, self.image_sub],10)
        self.ts.registerCallback(self.callback)
        

        # self.Draw = draw_open3d.Open3D()

    def callback(self, point, image):
        self.cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        self.cv_image = calibration.calibration(self.cv_image,
                                                self.int_param,
                                                self.distortion,
                                                self.int_param_scaling,
                                                self.original_height,
                                                self.original_width)

        point.header.frame_id = "my_lidar"
        self.lidar_data = point
        self.lidar_process()
        # self.image_process()
        print("subscribing")

    def publish_(self, lidar_data, ros_cloud):
        self.lidar_pub.publish(lidar_data)
        self.image_pub.publish(ros_cloud)
        print("publishing")
        print("=======================")
        
    def lidar_process(self):
        open3d_cloud = process_open3d.convertCloudFromRosToOpen3d(self.lidar_data)
        lidar_xyz = np.asarray(open3d_cloud.points).transpose()
        lidar_xyz = np.vstack((lidar_xyz, np.array([1]*lidar_xyz.shape[1])))
        lidar_xyz = self.K_.dot(self.Tcw.dot(self.Twl.dot(lidar_xyz)))

        lidar_xyz = lidar_xyz.transpose()
        black = np.zeros((1080,2040),dtype=np.uint8) # 가로 = 3.1875 세로 = 5.625배

        for i in range(0,len(lidar_xyz)):
            x,y,z = lidar_xyz[i]
            if(z > 0):
                x,y = x/z, y/z
                if(0 <= x < 2040 and 0 <= y < 1080):
                    cv2.circle(self.cv_image, (int(x),int(y)), radius=1, color=(0,0,255))
        cv2.imshow("circle",self.cv_image)
        cv2.waitKey(22)
        

    def image_process(self):
        resized_origin = cv2.resize(self.cv_image,(640,192))
        input_image = pil.fromarray(self.cv_image)

        input_image = process_data.preprocessImage(input_image, self.feed_height, self.feed_width)
        disp, pred_disp, pred_depth = inference.inference(input_image, self.encoder, self.depth_decoder, self.device, self.original_height, self.original_width)
        colormapped_im = process_data.postprocessImage(disp, self.original_height, self.original_width)
        
        # cv2.imshow("depth",colormapped_im)
        # cv2.waitKey(1)

        # frame_id = "my_image"
        # fx, fy = self.K_[0][0], self.K_[1][1]
        # cx, cy = self.K_[0][2], self.K_[1][2]

        # camera_info = [640, 192, fx, fy, cx, cy]
        # open3d_cloud = process_open3d.create_open3d_point_cloud_from_rgbd(
        #         resized_origin, pred_depth,
        #         camera_info)
        # open3d_cloud = process_open3d.create_open3d_point_cloud_from_rgbd(
        #         resized_origin, pred_disp,
        #         camera_info)

        # ros_cloud = process_open3d.convertCloudFromOpen3dToRos(open3d_cloud, frame_id)
        # self.publish_(self.lidar_data, ros_cloud)



class SubPub_v2:
    def __init__(self, int_param, distortion, int_param_scaling,
                 original_height, original_width,
                 feed_height, feed_width,
                 device, encoder, depth_decoder,
                 K_):

        self.int_param = int_param
        self.distortion = distortion
        self.int_param_scaling = int_param_scaling
        self.original_height = original_height
        self.original_width = original_width
        self.feed_height = feed_height
        self.feed_width = feed_width
        self.device = device
        self.encoder = encoder
        self.depth_decoder = depth_decoder
        self.K_ = K_

        self.cv_image = None
        self.lidar_data = None
        self.bridge = CvBridge()

        # self.lidar_sub = rospy.Subscriber("/rslidar_points", PointCloud2, self.lidar_callback)
        self.image_sub = rospy.Subscriber("/pylon_camera_node/image_raw", Image, self.image_callback)

        # self.lidar_pub = rospy.Publisher("/original/points", PointCloud2)
        self.image_pub = rospy.Publisher("/estimated/points", PointCloud2)
        # self.Draw = draw_open3d.Open3D()

    def lidar_callback(self, point):
        point.header.frame_id = "my_lidar"
        self.lidar_data = point
        print("lidar_subscribing")
        # self.Draw.drawLiDAR(point)
        # self.lidar_publish(self.lidar_data)

    def image_callback(self, image):
        self.cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        print("image_subscribing")
        self.image_process()

    def lidar_publish(self, lidar_data):
        self.lidar_pub.publish(lidar_data)
        print("lidar_publishing")

    def image_publish(self, ros_cloud):
        self.image_pub.publish(ros_cloud)
        print("image_publishing")

    def image_process(self):
        self.cv_image = calibration.calibration(self.cv_image,
                                                self.int_param,
                                                self.distortion,
                                                self.int_param_scaling,
                                                self.original_height,
                                                self.original_width)
        resized_origin = cv2.resize(self.cv_image,(640,192))
        input_image = pil.fromarray(self.cv_image)

        input_image = process_data.preprocessImage(input_image, self.feed_height, self.feed_width)
        disp, pred_disp, pred_depth = inference.inference(input_image, self.encoder, self.depth_decoder, self.device, self.original_height, self.original_width)
        # colormapped_im = process_data.postprocessImage(disp, self.original_height, self.original_width)

        frame_id = "my_image"
        fx, fy = self.K_[0][0], self.K_[1][1]
        cx, cy = self.K_[0][2], self.K_[1][2]

        camera_info = [640, 192, fx, fy, cx, cy]
        open3d_cloud = process_open3d.create_open3d_point_cloud_from_rgbd(
                resized_origin, pred_depth,
                camera_info)
        # open3d_cloud = process_open3d.create_open3d_point_cloud_from_rgbd(
        #         resized_origin, pred_disp,
        #         camera_info)

        # self.Draw.drawImage(open3d_cloud)

        ros_cloud = process_open3d.convertCloudFromOpen3dToRos(open3d_cloud, frame_id)
        self.image_publish(ros_cloud)
        # print("=========================")
