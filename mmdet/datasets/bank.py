# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List

from mmengine import fileio

from mmdet.registry import DATASETS
from .base_semseg_dataset import BaseSegDataset
from .base_det_dataset import BaseDetDataset
from .coco import CocoDataset
from .coco_panoptic import CocoPanopticDataset

ADE_PALETTE = [(120, 120, 120), (180, 120, 120), (6, 230, 230), (80, 50, 50),
               (4, 200, 3)]


@DATASETS.register_module()
class BankDataset(CocoDataset):
    METAINFO = {
        'classes':
        ('floor', 'desk', 'cabin', 'door', 'other'),
        'palette':
        ADE_PALETTE
    }

@DATASETS.register_module()
class BankSegDataset(BaseSegDataset):
    METAINFO = {
        'classes':
        ('floor', 'desk', 'cabin', 'door', 'other'),
        'palette':
        ADE_PALETTE
    }

@DATASETS.register_module()
class BankDetDataset(BaseDetDataset):
    METAINFO = {
        'classes':
        ('floor', 'desk', 'cabin', 'door', 'other'),
        'palette':
        ADE_PALETTE
    }

