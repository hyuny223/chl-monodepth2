import torch
from torchvision import transforms
import PIL.Image as pil


import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm


def preprocessImage(input_image, feed_height, feed_width):
    input_image = input_image.resize((feed_width, feed_height), pil.Resampling.LANCZOS)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0) # 1, 1, 192, 640
    # input_image = input_image.repeat(1,3,1,1) # 1, 3, 192, 640 <- this is for gray scale

    return input_image

def postprocessImage(disp, original_height, original_width):
    disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

    disp_resized_np = disp_resized.squeeze().cpu().detach().numpy()

    vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
    # im = pil.fromarray(colormapped_im)

    # cv2.imshow("im",colormapped_im)
    # cv2.waitKey(1)

    return colormapped_im
