# Copyright (c) OpenMMLab. All rights reserved.
"""Image Demo.

This script adopts a new infenence class, currently supports image path,
np.array and folder input formats, and will support video and webcam
in the future.

Example:
    Save visualizations and predictions results::

        python demo/image_demo.py demo/demo.jpg rtmdet-s

        python demo/image_demo.py demo/demo.jpg \
        configs/rtmdet/rtmdet_s_8xb32-300e_coco.py \
        --weights rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth

        python demo/image_demo.py demo/demo.jpg \
        glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365 --texts bench

        python demo/image_demo.py demo/demo.jpg \
        glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365 --texts 'bench . car .'

        python demo/image_demo.py demo/demo.jpg \
        glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365
        --texts 'bench . car .' -c

        python demo/image_demo.py demo/demo.jpg \
        glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365 \
        --texts 'There are a lot of cars here.'

    Visualize prediction results::

        python demo/image_demo.py demo/demo.jpg rtmdet-ins-s --show

        python demo/image_demo.py demo/demo.jpg rtmdet-ins_s_8xb32-300e_coco \
        --show
"""
import os
import mmcv
import json
from tqdm import tqdm
from glob import glob
from argparse import ArgumentParser
from mmdet.registry import VISUALIZERS
from mmengine.logging import print_log
from mmdet.utils import register_all_modules
from mmdet.apis import init_detector, inference_detector

from pdb import set_trace

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'config', type=str, help='Input config file or folder path.')
    parser.add_argument(
        'inputs', type=str, help='Input image file or folder path.')
    parser.add_argument('--weights', default=None, help='Checkpoint file')
    parser.add_argument(
        '--out-dir',
        type=str,
        default='outputs',
        help='Output directory of images or prediction results.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the image in a popup window.')


    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    # config_name = 'mask2former_r50_8xb2-fusion-50e_roi1024'
    config_name = args.config
    config_file = f"configs/mask2former/{config_name}.py"
    if args.weights:
        checkpoint_file = args.weights
    else:
        # checkpoint_file = f"work_dirs/{config_name}/iter_350000.pth"
        checkpoint_file = f"work_dirs/{config_name}/last_checkpoint"

    if not checkpoint_file.endswith('.pth'):
        with open(checkpoint_file, 'r') as f:
            checkpoint_file = f.readline().strip()

    register_all_modules()

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    paths = glob(os.path.join(args.inputs, '*.jpg'))
    paths += glob(os.path.join(args.inputs, '*.png'))
    for img_path in tqdm(paths):
        img = mmcv.imread(img_path, channel_order='bgr')    # bgr更好, -> rgb
        h, w = img.shape[:2]
        result = inference_detector(model, img)

        visualizer = VISUALIZERS.build(model.cfg.visualizer)
        visualizer.dataset_meta = model.dataset_meta
        # set_trace()
        save_path = os.path.join('result/', os.path.basename(img_path))
        # show the results
        visualizer.add_datasample(
            'result',
            img,
            data_sample=result,
            draw_gt=False,
            wait_time=0,
            pred_score_thr=0.015,
            out_file=save_path
        )
        # visualizer.show()

if __name__ == '__main__':
    main()
