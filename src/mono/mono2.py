#!/usr/bin/env python3.8

import argparse
import numpy as np
import yaml
import time

import warnings

import rospy
import open3d

from depth import load_model
from src import subpub
from utils import load_yaml

warnings.filterwarnings("ignore", category=UserWarning)


def parse():
    args = argparse.ArgumentParser(description="estimate depth!!")
    args.add_argument("--model_path",
                    type=str,
                    help="put the path of your model")

    parser = args.parse_args(rospy.myargv()[1:])
    model_path = parser.model_path

    return model_path


if __name__ == "__main__":

    model_path = parse()
    K_, int_param, distortion, int_param_scaling, original_height, original_width, Tcw, Twl = load_yaml.loadYaml()
    encoder, depth_decoder, feed_height, feed_width, device = load_model.loadModel(model_path)

    rospy.init_node("my_node", anonymous=True)
    # Just do it
    # ROS = subpub.SubPub_v2(int_param, distortion, int_param_scaling,
    #                         original_height, original_width,
    #                         feed_height, feed_width,
    #                         device, encoder, depth_decoder,
    #                         K_)
    # rospy.spin()



    # For syncronization. But incompleted
    ROS = subpub.SubPub_v1(int_param, distortion, int_param_scaling,
                            original_height, original_width,
                            feed_height, feed_width,
                            device, encoder, depth_decoder,
                            K_, Tcw, Twl)
    rospy.spin()
    
