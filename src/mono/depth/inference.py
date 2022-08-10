import sys, cv2
import numpy as np
from networks.layers import disp_to_depth


def inference(input_image, encoder, depth_decoder, device, original_height, original_width):
    pred_depth_scale_factor = 5.4
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    # PREDICTION
    input_image = input_image.to(device)
    features = encoder(input_image)
    outputs = depth_decoder(features)

    disp = outputs[("disp", 0)]

    pred_disp, _ = disp_to_depth(disp, 0.1, 100)
    pred_disp = pred_disp.squeeze().cpu().detach().numpy()
    # pred_disp = cv2.resize(pred_disp, (original_width, original_height))

    pred_depth = 1 / pred_disp
    pred_depth *= pred_depth_scale_factor

    # pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
    # pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

    return disp, pred_disp, pred_depth

def compute3DPoints(pred_disp):
    global K_
    rows ,cols = pred_disp.shape[0:]
    fx, fy = K_[0][0], K_[1][1]
    cx, cy = K_[0][2], K_[1][2]

    for y in range(rows):
        for x in range(cols):
            estimated_z = pred_disp[y][x]
            estimated_x = (estimated_z / fx) * (x - cx)
            estimated_y = (estimated_z / fy) * (y - cy)
