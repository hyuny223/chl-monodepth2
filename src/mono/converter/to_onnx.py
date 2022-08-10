import argparse, os
import torch
from torch.utils.data import DataLoader
import onnx
import numpy as np

import networks
from options import MonodepthOptions

parser = argparse.ArgumentParser(description="onnx for monodepth2")
parser.add_argument("--pretrained_model",
                    type=str,
                    help="put the path of pretrained model")

args = parser.parse_args()


encoder_path = os.path.join(args.pretrained_model,"encoder.pth")
encoder = networks.ResnetEncoder(18, False)

decoder_path = os.path.join(args.pretrained_model, "depth.pth")
depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

# encoder_dict = torch.load(encoder_path)
# model_dict = encoder.state_dict()
# encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

# decoder_dict = torch.load(decoder_path)
# depth_decoder.load_state_dict(decoder_dict)

# encoder.cuda()
# encoder.eval()

depth_decoder.cuda()
depth_decoder.eval()

# # model input size에 맞게 b c h w 순으로 파라미터 설정

# f = torch.rand((1,3,192,640)).cuda()
# images = encoder(f)

f0_onnx = torch.rand((1, 64, 160, 256)).cuda()
f1_onnx = torch.rand((1, 64, 80, 128)).cuda()
f2_onnx = torch.rand((1, 128, 40, 64)).cuda()
f3_onnx = torch.rand((1, 256, 20, 32)).cuda()
f4_onnx = torch.rand((1, 512, 10, 16)).cuda()

images = (f0_onnx, f1_onnx, f2_onnx, f3_onnx, f4_onnx)

export_onnx_file = "monodepth.onnx"
torch.onnx.export(depth_decoder,
				  images,
				  export_onnx_file,
				  export_params=True,
				  do_constant_folding=True,
				  opset_version=10,
				  input_names = ['encoder_output_0', 'encoder_output_1', 'encoder_output_2', 'encoder_output_3', 'encoder_output_4'],
				  output_names = ['decoder_output_0', 'decoder_output_1', 'decoder_output_2', 'decoder_output_final'],
				  dynamic_axes={'encoder_output_0' : {0 : 'batch_size'},
				  				'encoder_output_1' : {0 : 'batch_size'},
				  				'encoder_output_2' : {0 : 'batch_size'},
				  				'encoder_output_3' : {0 : 'batch_size'},
				  				'encoder_output_4' : {0 : 'batch_size'},
				  				'decoder_output_0' : {0 : 'batch_size'},
				  				'decoder_output_1' : {0 : 'batch_size'},
				  				'decoder_output_2' : {0 : 'batch_size'},
				  				'decoder_output_final' : {0 : 'batch_size'}}
				  )

# onnx_decoder = onnx.load("monodepth.onnx")
# onnx.checker.check_model(onnx_decoder)
# print("Done: converting decoder to onnx format!")
