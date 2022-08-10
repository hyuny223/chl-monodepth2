import cv2
import numpy as np


def calibration(img, int_param, distortion, int_param_scaling, original_height, original_width):

    rectification = np.eye(3)

    mapx, mapy = cv2.initUndistortRectifyMap(
        int_param,
        distortion,
        rectification,
        int_param_scaling,
        (original_width, original_height),
        cv2.CV_32FC1,
    )

    calibrated_img = cv2.remap(
        img,
        mapx,
        mapy,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return calibrated_img
