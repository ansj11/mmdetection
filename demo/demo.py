from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.registry import VISUALIZERS
from mmdet.structures.det_data_sample import DetDataSample
import mmcv
import json
import os
import cv2
import torch.nn.functional as F
import numpy as np
from pdb import set_trace
from tqdm import tqdm

config_name = 'mask2former_r50_8xb2-fusion-50e_roi1024'
# config_name = 'mask2former_r50_8xb2-fusion-50e_roi'
config_file = f"configs/mask2former/{config_name}.py"
# checkpoint_file = f"work_dirs/{config_name}/iter_350000.pth"
checkpoint_file = f"work_dirs/{config_name}/last_checkpoint"

if not checkpoint_file.endswith('.pth'):
    with open(checkpoint_file, 'r') as f:
        checkpoint_file = f.readline().strip()

register_all_modules()

model = init_detector(config_file, checkpoint_file, device='cuda:0')

json_path = "testsets/test.json"

with open(json_path, 'r') as f:
    lines = f.readlines()


## 模型后处理
def postprocess(netout, thresholds=[2e-5, 0.2, 0.2, 0.2], ratio=0.5, height=1024, width=1024):
    pred_scores = netout[0].squeeze(0).cpu().numpy()[...,:-1]  # 100 x class
    pred_masks = netout[1].squeeze(0).cpu().numpy()    # 100 x H x W

    class_ids = np.argmax(pred_scores, axis=-1)
    # print(class_ids)
    # print(pred_scores[:, class_ids])
        
    num, H, W = pred_masks.shape

    results = {}
    for i in range(num):
        cls_id = class_ids[i]
        score = pred_scores[i, cls_id]
        mask = pred_masks[i]
        pix_factor = 100 * mask.sum() / np.prod(mask.shape)
        if score > thresholds[cls_id] and pix_factor > ratio:
            _, binary = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(binary.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # set_trace()
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            epsilon = 0.01 * cv2.arcLength(contours[0], True)   # calc closet length
            hull = cv2.approxPolyDP(contours[0], epsilon, True).astype('float32')

            if cls_id not in results:
                results[cls_id] = []
            
            hull[..., 0] *= float(width / W)
            hull[..., 1] *= float(height / H)
            
            results[cls_id] = [score, hull.flatten().tolist()]
            # image = cv2.imread(args.img)
            # cv2.drawContours(image, [hull], -1, (0, 255, 0), 2)
    
    return results, pred_scores


def postprocessV2(netout, thresholds=[2e-5, 0.2, 0.2, 0.2], ratio=0.5, height=1024, width=1024, topk=3, iou_thresh=0.5):
    pred_scores = netout[0].squeeze(0).cpu().numpy()[...,:-1]  # 20 x class
    pred_masks = netout[1].squeeze(0).cpu().numpy()    # 20 x H x W

    max_class_ids = np.argmax(pred_scores, axis=-1)
    max_scores = pred_scores.max(axis=-1)

    num = max_scores.shape[0]
    top_indices = max_scores.argsort()[::-1][:topk]  # topk
    top_scores = max_scores[top_indices]
    class_ids = max_class_ids[top_indices]
    mask_pred = pred_masks[top_indices]
    assert len(set(top_indices)) == len(top_indices), "%d ! = %d" % (len(set(top_indices)), len(top_indices))
    
    filtered_indices = []
    indices = [x for x in range(len(top_indices))]
    while len(indices) > 0:
        i = indices[0]
        filtered_indices.append(i)
        
        res_indices = []
        for j in range(len(indices[1:])):
            # calculate iou
            overlap = mask_pred[i] * mask_pred[indices[j+1]]
            union = mask_pred[i] + mask_pred[indices[j+1]] - overlap
            iou = overlap.sum() / (union.sum() + 1e-16) # 必须先算iou，再判断
            if class_ids[i] == class_ids[indices[j+1]] and iou > iou_thresh:
                # mask_pred[i] = union
                continue
            else:
                res_indices.append(indices[j+1])
                
        indices = res_indices
    
    class_ids, mask_pred, top_scores = class_ids[filtered_indices], mask_pred[filtered_indices], top_scores[filtered_indices]

    num, H, W = mask_pred.shape

    results = {}
    masks = []
    for i in range(num):
        cls_id = class_ids[i]
        score = top_scores[i]
        mask = mask_pred[i]
        pix_factor = 100 * mask.sum() / np.prod(mask.shape)
        if score > thresholds[cls_id] and pix_factor > ratio:
            _, binary = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(binary.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # set_trace()
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            epsilon = 0.01 * cv2.arcLength(contours[0], True)   # calc closet length
            hull = cv2.approxPolyDP(contours[0], epsilon, True).astype('float32')

            if cls_id not in results:
                results[cls_id] = []
            
            hull[..., 0] *= float(width / W)
            hull[..., 1] *= float(height / H)
            
            results[cls_id].append({'confidence': score, 'data': hull.flatten().tolist()})
            masks.append(mask)
            # image = cv2.imread(args.img)
            # cv2.drawContours(image, [hull], -1, (0, 255, 0), 2)
    
    return results, top_scores, mask_pred

categories = {560020006: 0, 560020008: 1, 560020009: 2, 560020010: 3, 0: 4}
cate2tagid = [560020006, 560020008, 560020009, 560020010, 0]
dics, results = [], []

c = [0, 0, 0, 0]
for line in tqdm(lines):
    data = json.loads(line)
    tagnameid = data['result'][0]['tagnameid']
    # if 'defee1b3-575f-44a7-8c61-8d9802302f8e.mp4_00013036.jpg' not in data['image']:
    #     continue
    img_path = os.path.join('testsets/', data['image'])
    img = mmcv.imread(img_path, channel_order='bgr')    # bgr更好, -> rgb
    h, w = img.shape[:2]
    result = inference_detector(model, img)
    
    if not isinstance(result, DetDataSample):
        # thresholds = [2e-5, 2e-5, 2e-5, 2e-5]
        thresholds = [0.00002, 0.15, 0.15, 0.15]
        # if '日/none/疑似员工拍屏/cd4e5b36-726b-4abf-94fd-114bc56a9888.jpg' in img_path:
        #     set_trace()
        ret, scores, masks = postprocessV2(result, thresholds=thresholds, ratio=0.5, height=h, width=w)
        # if len(ret) == 0 and c[categories[tagnameid]] == 1:
        #     continue
        # c[categories[tagnameid]] += 1
        dics.append(line)
        res_dict = {}
        res_dict['image'] = data['image']
        res_dict['result'] = []
        for key, val in ret.items():
            for item in val:
                tmpdict = {}
                tmpdict['tagnameid'] = cate2tagid[key]
                tmpdict['confidence'] = [float(item['confidence'])]
                tmpdict['data'] = item['data']
                res_dict['result'].append(tmpdict)
            
        results.append(json.dumps(res_dict, ensure_ascii=False))
        # print(scores.max(), scores.min(), scores.mean())
        # if scores.max() > 0.2 and len(res_dict['result']) == 0: set_trace()
        # if scores.max() < 0.2:
        #     cv2.imwrite('testsets/miss/' + os.path.basename(data['image']), img)
        os.makedirs('result/postV2', exist_ok=True)
        for i, mask in enumerate(masks):
            mask = cv2.resize(mask, (w, h), dst=mask, interpolation=cv2.INTER_NEAREST)
            img[mask > 0.5,:] = img[mask > 0.5,:]*0.5 + np.array([0, 255//(i+1), 0], dtype=np.uint8)*0.5
        cv2.imwrite('result/postV2/' + os.path.basename(data['image']), img)
        if tagnameid in [560020006, 560020008, 560020010]:
            # print(tagnameid, res_dict['result'], data['result'])
            if len(res_dict['result']) > 1:
                print(img_path, tagnameid)
                print([(x['confidence'][0], x['tagnameid']) for x in res_dict['result']], len(data['result']))
    else:
        visualizer = VISUALIZERS.build(model.cfg.visualizer)
        visualizer.dataset_meta = model.dataset_meta
        # set_trace()
        save_path = os.path.join('result/', os.path.basename(data['image']))
        # show the results
        visualizer.add_datasample(
            'result',
            img,
            data_sample=result,
            draw_gt=False,
            wait_time=0,
            out_file=save_path
        )
        # visualizer.show()
    
if len(dics):
    with open('mask2former_accuracy_contour_gts.json', 'w', encoding='utf-8') as f:
        for line in dics:
            f.write(line)
    with open('mask2former_accuracy_contour_preds.json', 'w', encoding='utf-8') as f:
        for line in results:
            f.write(line + '\n')
