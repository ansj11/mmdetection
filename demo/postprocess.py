import os
import cv2
import numpy as np


## 模型后处理
def postprocess(netout, threshold=0.7, ratio=0.5):
    pred_scores = netout[0].cpu().numpy()   # 100 x class
    pred_masks = netout[1].cpu().numpy()    # 100 x H x W

    class_ids = np.argmax(pred_scores, axis=-1)
    
    num = pred_scores.shape[1]


    results = {}
    for i in range(num):
        cls_id = class_ids[i]
        score = pred_scores[i, cls_id]
        mask = pred_masks[i]
        pix_factor = 100 * mask.sum() / np.prod(mask.shape)
        
        if score > threshold and pix_factor > ratio:
            threshold, binary = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(binary.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # set_trace()
            epsilon = 0.01 * cv2.arcLength(contours[0], True)   # calc closet length
            hull = cv2.approxPolyDP(contours[0], epsilon, True)

            if cls_id not in results:
                results[cls_id] = []
            
            results[cls_id].append(hull)
        
            # image = cv2.imread(args.img)
            # cv2.drawContours(image, [hull], -1, (0, 255, 0), 2)
    
    return results



## 模型后处理
def postprocessV2(netout, threshold=0.7, ratio=0.5, num_classes=4, topk=3, iou_thresh=0.5):
    pred_scores = netout[0].cpu().numpy()   # 1x20 x class
    pred_masks = netout[1].cpu().numpy()    # 1x20 x H x W

    assert pred_scores.shape[2] == num_classes, "class number not match pred_scores"
    
    num = pred_scores.shape[1]
    labels = np.array([x for x in range(num_classes)] * num)   # [0,1,2,3,0,1,2,3...]x20
    top_indices = pred_scores.flatten().argsort()[::-1][:topk]  # topk
    top_scores = pred_scores.flatten()[top_indices]
    class_ids = labels[top_indices]
    mask_ids = top_indices // num_classes
    mask_pred = pred_masks[mask_ids]
    
    filtered_indices = []
    indices = [x for x in range(len(top_indices))]
    while len(indices) > 0:
        i = indices[0]
        filtered_indices.append(i)
        
        # calculate iou
        overlaps = mask_pred[i] * mask_pred[indices[1:]]
        unions = mask_pred[i] + mask_pred[indices[1:]] - overlaps # or mask_pred[i] | mask_pred[indices[1:]]
        ious = overlaps / (unions + 1e-16)
        # idx = np.where(ious <= iou_thresh)[0]
        # indices = indices[idx + 1]  # 处理剩余的边框
        res_indices = []
        for j in len(indices[1:]):
            if ious[j] > iou_thresh:
                mask_pred[i] = unions[j]
            else:
                res_indices.append(j+1)
                
        indices = res_indices
    
    class_ids, mask_pred, top_scores = class_ids[filtered_indices], mask_pred[filtered_indices], top_scores[filtered_indices]

    results = {}
    for i in range(len(class_ids)):
        cls_id = class_ids[i]
        score = top_scores[i]
        mask = mask_pred[i]
        pix_factor = 100 * mask.sum() / np.prod(mask.shape)
        
        if score > threshold and pix_factor > ratio:
            threshold, binary = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(binary.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # set_trace()
            epsilon = 0.01 * cv2.arcLength(contours[0], True)   # calc closet length
            hull = cv2.approxPolyDP(contours[0], epsilon, True)

            if cls_id not in results:
                results[cls_id] = []
            
            results[cls_id].append(hull)
        
            # image = cv2.imread(args.img)
            # cv2.drawContours(image, [hull], -1, (0, 255, 0), 2)
    
    return results



## 判断相机是否移动
def calc_movement(frame1, frame2):
    H, W = frame1.shape[:2]
    is_moved = False
    try:
        (kpsA, featuresA) = detectAndDescribe(frame1)
        (kpsB, featuresB) = detectAndDescribe(frame2)
        movement, len1, len2 = matchKeypoints(kpsA, kpsB, featuresA, featuresB)
    except:
        print(" key points not enough, skip: ", len(kpsA), len(kpsB))
    rate = movement / W
    movement = int(movement)
    is_moved = movement > 20 or rate > 0.2
    
    return movement, is_moved

def detectAndDescribe(image):
    # detect and extract features from the image
    # descriptor = cv2.xfeatures2d.SURF_create()
    # descriptor = cv2.xfeatures2d.SIFT_create()
    descriptor = cv2.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(image, None)

    # convert the keypoints from KeyPoint objects to NumPy
    # arrays
    kps = np.float32([kp.pt for kp in kps])

    # return a tuple of keypoints and features
    return (kps, features)

def matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio=0.75, reprojThresh=4.0):
    # compute the raw matches and initialize the list of actual
    # matches
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []

    # loop over the raw matches
    for m in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx, m))
    
    average_movement = 0
    if len(matches) > 0:
        total_movement = 0
        for m in matches:
            # set_trace() # distance != p1-p0
            j, i = m[:2]
            distance = np.sqrt((kpsA[i][0] - kpsB[j][0]) ** 2 + (kpsA[i][1] - kpsB[j][1]) ** 2)
            total_movement += distance
        
        average_movement = total_movement / len(matches)
    # print(len(rawMatches), len(matches), average_movement)
    
    return average_movement, len(rawMatches), len(matches)