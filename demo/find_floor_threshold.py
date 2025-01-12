
import os
import cv2
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pdb import set_trace


# path = "data/results.Mask2FormerROISeg.test1024_0002.json_trt_92_ampere_ezi_sorted"
path = "mask2former_accuracy_contour_preds.json"

with open(path, 'r') as f:
    lines = f.readlines()

scores_list = []
for line in tqdm(lines):
    data = json.loads(line)
    # img_path = data['image'].replace("/var/imagetests/Mask2FormerROISeg", "testsets")
    img_path = os.path.join("testsets", data['image'])
    basename = os.path.basename(img_path)
    
    img = cv2.imread(img_path, -1)
    img_height, img_width = img.shape[:2]
    
    results = data['result']
    is_floor = False
    best_score, best_contour = 0, None
    scores, contours = [], []
    for result in results:
        tagid = result['tagnameid']
        # if tagid not in [560020006]:
        #     continue
        is_floor = True
        score = result['confidence'][0]
        contour = np.array(result['data']).reshape(-1, 1, 2).astype(np.int32)
        
        # if score > 7e-6:
        # if score > 9e-6:
        if score > 2e-5:
            scores.append(score)
            contours.append(contour)
        if score > best_score:
            best_score = score
            best_contour = contour
    
    # if not is_floor:
    #     continue
    
    scores_list.append(scores)
    
    # show = cv2.drawContours(img.copy(), [best_contour], -1, (0, 255, 0), 2)
    # cv2.putText(show, f"{best_score:.6f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    
    # save_path = f"threshold/{basename}.jpg"
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # cv2.imwrite(save_path, show)
    if len(scores) < 2:
        continue
    
    show = img.copy()
    for i, (score, contour) in enumerate(zip(scores, contours)):
        show = cv2.drawContours(show, [contour], -1, (0, 255, 0), 2)
        cv2.putText(show, f"{score:.6f}", (10, 100*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    
    save_path = f"threshold3/{basename}_{i}.jpg"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, show)
    
print(len(scores_list))
len_list = [len(score) for score in scores_list]
print(len_list)
print([x for x in len_list if x > 0])

