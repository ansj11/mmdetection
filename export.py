import os
import sys
import cv2
import yaml
import numpy as np
import requests
import argparse
import torch
import torch.onnx
from torch import nn
import onnxruntime

from pdb import set_trace
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Export onnx models')
    parser.add_argument('--input',  '-i',
                        dest="input",
                        metavar='FILE',
                        help =  'path to the model file',
                        default='')
    parser.add_argument('--output',  '-o',
                        dest="output",
                        metavar='FILE',
                        help =  'output path',
                        default='')
    parser.add_argument('--verify',  '-v',
                        dest="verify",
                        default=True)
    
    args = parser.parse_args()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    device = torch.device('cpu')
    path = args.input
    x = process_img(path, device).float()

    if args.verify:
        import onnx
        
        onnx_model = onnx.load(args.output)
        try:
            onnx.checker.check_model(onnx_model)
        except Exception:
            print("Model incorrect")
            sys.exit(0)
        else:
            print("Model correct")
            onnx_model = ONNXModel(args.output)
            set_trace()
            cls, masks = onnx_model.run(*x)[0]
            print(cls.shape, cls)
    

def process_img(path, device):
    bgr = cv2.imread(path, -1)[..., :3]
    rgb = bgr # cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (1024, 576), interpolation=cv2.INTER_AREA)
    img = (rgb.astype('float32') - np.array([123, 116, 103])[None,None]) / 57   # -1~1
    img = torch.from_numpy(img).permute(0, 1, 2).to(device)
    x = img # torch.cat([img, torch.zeros_like(img[:,:2])], dim=1)
    return x

class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))
 
    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name
 
    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name
 
    def get_input_feed(self, input_name, image_tensor):
        """
        input_feed={self.input_name: image_tensor}
        :param input_name:
        :param image_tensor:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_tensor
        return input_feed
 
    def run(self, image_tensor):
        '''
        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor[np.newaxis, :]
        onnx_session.run([output_name], {input_name: x})
        :param image_tensor:
        :return:
        '''
        input_feed = self.get_input_feed(self.input_name, image_tensor.numpy())
        return self.onnx_session.run(self.output_name, input_feed=input_feed)

        
        
if __name__ == '__main__':
    main()
