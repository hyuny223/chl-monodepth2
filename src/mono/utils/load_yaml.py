import numpy as np
import yaml

def loadYaml():

    path = "/root/ws/src/mono/info/calibration.yaml"
    with open(path, 'r') as f:
        infos = yaml.load(f, Loader=yaml.FullLoader)

        K_ = infos['camera']['front']['K']
        Rcw = infos['camera']['front']['R']
        tcw = infos['camera']['front']['T']

        Tcw = np.concatenate((Rcw,tcw),axis=1)

        D_ = infos['camera']['front']['D']
        P_ = infos['camera']['front']['P']

        original_height = infos['camera']['front']['size']['height']
        original_width = infos['camera']['front']['size']['width']
        
        Rwl = infos['lidar']['rs80']['R']
        twl = infos['lidar']['rs80']['T']
        Twl = np.concatenate((Rwl, twl),axis=1)

    int_param = np.array(K_)
    distortion = np.array(D_)
    int_param_scaling = np.array(P_).reshape((3, 4))[:3, :3]

    return K_, int_param, distortion, int_param_scaling, original_height, original_width, Tcw, Twl
