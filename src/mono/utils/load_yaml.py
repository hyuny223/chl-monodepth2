import numpy as np
import yaml

def loadYaml():

    path = "/root/ws/src/mono/info/calibration.yaml"
    with open(path, 'r') as f:
        infos = yaml.load(f, Loader=yaml.FullLoader)

        K_ = infos['camera']['front']['K']
        R_ = infos['camera']['front']['R']
        t_ = infos['camera']['front']['T']

        T_ = np.concatenate((R_,t_),axis=1)

        D_ = infos['camera']['front']['D']
        P_ = infos['camera']['front']['P']

        original_height = infos['camera']['front']['size']['height']
        original_width = infos['camera']['front']['size']['width']

    int_param = np.array(K_)
    distortion = np.array(D_)
    int_param_scaling = np.array(P_).reshape((3, 4))[:3, :3]

    return K_, int_param, distortion, int_param_scaling, original_height, original_width
