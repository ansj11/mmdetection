import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


pre = np.fromfile("./align/pre", dtype=np.float32).reshape(3, 576, 1024)

score = np.fromfile("./align/Mask2FormerROISeg_000001_000001_pred_scores.bin", dtype=np.float32)

mask = np.fromfile("./align/Mask2FormerROISeg_000001_000001_pred_masks.bin", dtype=np.float32)

mask = mask.reshape(-1, 576, 1024)
