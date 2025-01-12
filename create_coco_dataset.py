import json
import os
import cv2
import sys
import requests
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
import skimage.io as io
import matplotlib.pyplot as plt
from pdb import set_trace
# from IPython import embed
import base64
from io import BytesIO
from PIL import Image
from copy import deepcopy

def contours_to_rle(contours,  image_width, image_height):
    # 创建空白掩码图像
    mask_image = np.zeros((image_height, image_width), dtype=np.uint8)

    # 将每个轮廓绘制在掩码图像上
    cv2.drawContours(mask_image, contours, -1, 255, -1)

    # 将二进制掩码图像转换为 RLE 格式
    rle_encoding = mask_utils.encode(np.asfortranarray(mask_image))

    rle_encoding['counts'] = rle_encoding['counts'].decode('utf-8')
    # 返回 RLE 格式的分割信息
    return rle_encoding

def save_coco_json(instance, save_path):
    import io
    #json.dump(instance, io.open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)  # indent=2 更加美观显示
    with io.open(save_path, 'w', encoding="utf-8") as outfile:
        my_json_str = json.dumps(instance, ensure_ascii=False, indent=1)
        outfile.write(my_json_str)

def create_coco_dataset(root, annotation_file, mode='train'):
    # 初始化 COCO 注释数据结构
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "person"},
                       {"id": 2, "name": "desk"},
                       {"id": 3, "name": "cabin"},
                       {"id": 4, "name": "door"}]  # 假设只有一类 "person"
    }

    image_id = 1
    annotation_id = 1
    
    path = os.path.join(root, 'images')
    img_paths = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                img_path = os.path.join(root, file)
                img_paths.append(img_path)
    
    for image_path in tqdm(img_paths):
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        mask_path = image_path.replace('images', 'annotations')[:-3] + 'png'
        if not os.path.exists(mask_path):
            continue
        mask = cv2.imread(mask_path, -1)
        if mask is None or mask.max() < 1:
            continue
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        instance = mask
        indexes = np.unique(mask)
        if mask.max() > 1: print(image_path, indexes)
        for index in indexes:
            if index == 0: continue
            cur_mask = (instance == index).astype('uint8')

            # 将二值Mask转换为多边形分割信息
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue
            # each person has one instance, but serveral contours
            all_points = np.concatenate(contours)
            x, y, w, h = cv2.boundingRect(all_points)
            bbox = [x,y,w,h]
            area = 0
            for contour in contours:
                area += cv2.contourArea(contour)
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,  # 对应 "person" 类别
                "segmentation": [np.array(contours).flatten().tolist()] if len(contours)==1 else contours_to_rle(contours, width, height),
                "area": area,  # 填入分割区域的面积
                "bbox": bbox,  # 填入边界框信息 [x, y, width, height]
                "iscrowd": 0 if len(contours)==1 else 1
            }

            coco_data["annotations"].append(annotation)
            annotation_id += 1
        # 保存图像信息到 COCO 数据
        coco_image = {
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": image_path
        }
        coco_data["images"].append(coco_image)

        image_id += 1
        # break
    print(image_id-1, annotation_id-1)
    # 将 COCO 数据保存为 JSON 文件
    save_coco_json(coco_data, annotation_file)

def demo_coco_dataset(path):
    # 初始化COCO数据集的实例
    coco = COCO(path)

    # 获取类别列表
    categories = coco.loadCats(coco.getCatIds())
    category_names = [category['name'] for category in categories]
    print("类别列表：", category_names)

    image_ids = coco.getImgIds()
    for i in tqdm(image_ids):
        image_id = image_ids[i]
        image_data = coco.loadImgs(image_id)[0]
        image_path = os.path.join('/gemini/data-1/mmdetection/data/bank/', image_data['file_name'])
        image = io.imread(image_path)
        
        annotation_ids = coco.getAnnIds(imgIds=image_id, iscrowd=None)
        annotations = coco.loadAnns(annotation_ids)
        # if len(annotations) <= 1:
        #     continue
        # coco.showAnns(anns)

        h, w = image_data['height'], image_data['width']
        mask = np.zeros((h, w), dtype=np.uint8)

        idx = 1
        iscrowd = False
        for annotation in annotations:
            if annotation['iscrowd'] == 1:
                iscrowd = True
            tmp = coco.annToMask(annotation)
            mask[tmp > 0] = idx
            idx += 1
        if iscrowd:
            image[mask>0,:] = image[mask>0,:] * 0.5 + np.array([0,255,0]) * 0.5
            save_path = os.path.join('iscrowd', '%06d.jpg' % image_id)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.imsave(save_path, image)


def process_coco(paths, annotation_file, mode='train', save_ins=True):
    image_id = 1
    annotation_id = 1
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "floor"},
                       {"id": 2, "name": "desk"},
                       {"id": 3, "name": "cabin"},
                       {"id": 4, "name": "door"}]  # 假设只有一类 "person"
    }
    dirname2id = {'metaloop_20240729165359': 3,
                  'metaloop_20240729171406': 2,
                  'metaloop_20240729172145': 4,
                  'metaloop_20240729173736': 1,
                  'metaloop_20240816180230': 1,
                  'metaloop_20240816180301': 1,
                  }
    ins_dict = {'metaloop_20240729165359': 0,
                  'metaloop_20240729171406': 0,
                  'metaloop_20240729172145': 0,
                  'metaloop_20240729173736': 0,
                  'metaloop_20240816180230': 0,
                  'metaloop_20240816180301': 0,
                  }
    if not isinstance(paths, list):
        paths = [paths]
    lines = []
    for path in paths:
        dirname = os.path.basename(path)
        filepath = os.path.join(path, 'output.json')
        with open(filepath, 'r') as f:
            tp_lines = f.readlines()
        for line in tqdm(tp_lines):
            sample = json.loads(line)

            if len(sample['result']) == 1 and sample['result'][0]['tagtype'] == 'delete':
                continue
    
            results = sample['result']
            image_path = os.path.join(path, sample['image_path'])
            filename = os.path.join(dirname, sample['image_path'])
            img = cv2.imread(image_path)
            if img is None:
                continue
            height, width, _ = img.shape
            if len(results) == 0:
                continue
        
            coco_image = {
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": filename
            }
            coco_data["images"].append(coco_image)
            instance = np.zeros_like(img[...,0])
            ins_ids = 1
            for i, result in enumerate(results):
                if result.get('tagtype') == 'delete':
                    continue
                if result.get('datatype') not in ['mask', 'polygon']:
                    continue
                if result.get('datatype') == 'mask':
                    response = requests.get(result['maskData'])
                    mask_array = np.array(bytearray(response.content), dtype='uint8')
                    mask = cv2.imdecode(mask_array, cv2.IMREAD_COLOR)
                    mask = cv2.resize(mask, (width, height))
                elif result.get('datatype') == 'polygon':
                    contours = [np.array(result['data']).astype('int').reshape(1,-1,2)]
                    mask = np.zeros((height, width), dtype=np.uint8)
                    # 将每个轮廓绘制在掩码图像上
                    cv2.drawContours(mask, contours, -1, 255, -1)
                    mask = mask.reshape(height, width, 1)
                if save_ins:
                    instance[np.max(mask, axis=2) > 1] = ins_ids
                    ins_ids += 1
                ret, thresh1 = cv2.threshold(np.max(mask, axis=2), 1, 255, cv2.THRESH_BINARY)
                contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # contours = [c for c in contours if cv2.contourArea(c)>100]    # coco will filter in dataset
                if len(contours) == 0:
                    continue
                all_points = np.concatenate(contours)
                x, y, w, h = cv2.boundingRect(all_points)
                bbox = [x,y,w,h]
                area = 0
                for c in contours:
                    area+=cv2.contourArea(c)

                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": dirname2id[dirname],  # 对应 "person" 类别
                    "segmentation": [np.array(contours).flatten().tolist()] if len(contours)==1 else contours_to_rle(contours, width, height),  # 填入多边形形式的分割信息
                    "area": area,  # 填入分割区域的面积
                    "bbox": bbox,  # 填入边界框信息 [x, y, width, height]
                    "iscrowd": 0 if len(contours)==1 else 1
                }
                coco_data["annotations"].append(annotation)
                annotation_id += 1

            if save_ins:
                ins_dict[dirname] = max(ins_dict[dirname], ins_ids-1)
                # instance_path = image_path.replace('left', 'instance')
                instance_path = image_path[:-3] + 'png'
                os.makedirs(os.path.dirname(instance_path), exist_ok=True)
                # cv2.imwrite(instance_path, instance)
                cv2.imwrite(instance_path[:-4]+'_show.jpg', instance*80)
            image_id += 1
    print(image_id-1, annotation_id-1, ins_dict)
        
    # 将 COCO 数据保存为 JSON 文件
    # set_trace()
    save_coco_json(coco_data, annotation_file)

def merge_jsons(path1, path2, annotation_file):
    with open(path1, 'r') as f:
        data_dict1 = json.load(f)
    with open(path2, 'r') as f:
        data_dict2 = json.load(f)
    
    print(len(data_dict2['images']), len(data_dict2['annotations']))
    print(len(data_dict1['images']), len(data_dict1['annotations']))
    image_id = 0
    annotation_id = 0
    for dic in data_dict2['images']:
        image_id = dic['id'] if dic['id'] > image_id else image_id
    
    for dic in data_dict2['annotations']:
        annotation_id = dic['id'] if dic['id'] > annotation_id else annotation_id
    
    id2new = {}
    for dic in data_dict1['images']:
        tmp_dic = deepcopy(dic)
        id2new[tmp_dic['id']] = tmp_dic['id'] + image_id
        tmp_dic['id'] += image_id
        data_dict2['images'].append(tmp_dic)

    for dic in data_dict1['annotations']:
        tmp_dic = deepcopy(dic)
        tmp_dic['id'] += annotation_id
        assert tmp_dic['image_id']+image_id == id2new[tmp_dic['image_id']]
        tmp_dic['image_id'] += image_id
        data_dict2['annotations'].append(tmp_dic)
    
    print(len(data_dict2['images']), len(data_dict2['annotations']))
    save_coco_json(data_dict2, annotation_file)


def split_json(path, train_path, test_path):
    with open(path, 'r') as f:
        data_dict = json.load(f)
    
    ins_dict = {'1': [],
                '2': [],
                '3': [],
                '4': []}
    
    print(len(data_dict['images']), len(data_dict['annotations']))
    
    for dic in data_dict['annotations']:
        image_id = dic['image_id']
        if 'annotations' not in data_dict['images'][image_id-1]:
            data_dict['images'][image_id-1]['annotations'] = []
            
        data_dict['images'][image_id-1]['annotations'].append(dic)
        
        if image_id not in ins_dict[str(dic['category_id'])]:
            ins_dict[str(dic['category_id'])].append(image_id)
        
    print(ins_dict)
    
    np.random.seed(20240731)
    
    train_dict = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "floor"},
                       {"id": 2, "name": "desk"},
                       {"id": 3, "name": "cabin"},
                       {"id": 4, "name": "door"}]  # 假设只有一类 "person"
    }
    test_dict = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "floor"},
                       {"id": 2, "name": "desk"},
                       {"id": 3, "name": "cabin"},
                       {"id": 4, "name": "door"}]  # 假设只有一类 "person"
    }
    test_image_id, test_annotation_id = 1, 1
    train_image_id, train_annotation_id = 1, 1
    for key, value in ins_dict.items():
        test_ids = np.random.choice(value, 300)
        for image_id in tqdm(value):
            tmp_dict = deepcopy(data_dict['images'][image_id-1])
            if image_id in test_ids:
                for annotation in tmp_dict['annotations']:
                    annotation['id'] = test_annotation_id
                    annotation['image_id'] = test_image_id
                    test_dict['annotations'].append(annotation)
                    test_annotation_id += 1
                tmp_dict['id'] = test_image_id
                tmp_dict.pop('annotations')
                test_dict['images'].append(tmp_dict)
                test_image_id += 1
            else:
                for annotation in tmp_dict['annotations']:
                    annotation['id'] = train_annotation_id
                    annotation['image_id'] = train_image_id
                    train_dict['annotations'].append(annotation)
                    train_annotation_id += 1
                tmp_dict['id'] = train_image_id
                tmp_dict.pop('annotations')
                train_dict['images'].append(tmp_dict)
                train_image_id += 1

        print(key, len(value), len(test_ids))

    save_coco_json(train_dict, train_path)
    save_coco_json(test_dict, test_path)

def create_testset(path):
    with open(path, 'r') as f:
        data_dict = json.load(f)
    
    categories = {1:{"tagnameid": 560020006, "evaltype": "mask2former_accuracy_contour"},
                  2:{"tagnameid": 560020008, "evaltype": "mask2former_accuracy_contour"},
                  3:{"tagnameid": 560020009, "evaltype": "mask2former_accuracy_contour"},
                  4:{"tagnameid": 560020010, "evaltype": "mask2former_accuracy_contour"},}
    # height, width = 768, 1344
    # height, width = 576, 1024
    height, width = data_dict['images'][0]['height'], data_dict['images'][0]['width']
    for dic in data_dict['annotations']:
        image_id = dic['image_id']
        if 'annotations' not in data_dict['images'][image_id-1]:
            data_dict['images'][image_id-1]['annotations'] = []
            
        data_dict['images'][image_id-1]['annotations'].append(dic)

    lines = []
    coco = COCO(path)
    out_dir = 'testsets'
    for dic in data_dict['images']:
        origin_path = os.path.join('/gemini/data-1/mmdetection/data/bank/', dic['file_name'])
        image_path = os.path.join('images', dic['file_name'])
        image = cv2.imread(origin_path, -1)
        image = cv2.resize(image, (width, height), cv2.INTER_AREA)
        tmp_dic = {}
        tmp_dic['image'] = image_path
        tmp_dic['result'] = []
        for annotation in dic['annotations']:
            if annotation['iscrowd'] == 1:
                continue
            category_id = annotation['category_id']
            mask = coco.annToMask(annotation)
            mask = cv2.resize(mask, (width, height))
            result = deepcopy(categories[category_id])
            
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            epsilon = 0.01 * cv2.arcLength(contours[0], True)
            hull = cv2.approxPolyDP(contours[0], epsilon, True)
            contours = [hull]
            
            result['data'] = [dic['height'], dic['width']]
            result['data'] += np.array(contours).flatten().tolist()
            tmp_dic['result'].append(result)
            
        if len(tmp_dic['result']) != 0:
            lines.append(json.dumps(tmp_dic))
            save_path = os.path.join(out_dir, image_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, image)
    
    with open('testsets/test1024.json', 'w') as f:
        for line in lines:
            f.write(line+'\n')

def update_label(path, annotation_file, save_ins=True):
    train_json = 'data/bank/annotations/trainV2.json'
    # test_json = 'data/bank/annotations/test.json'
    with open(train_json, 'r') as f:
        train_data = json.load(f)
    # with open(test_json, 'r') as f:
    #     test_data = json.load(f)

    image_ids = [dic['id'] for dic in train_data['images']]
    annotation_ids = [dic['id'] for dic in train_data['annotations']]
    
    image_id = max(image_ids) + 1
    annotation_id = max(annotation_ids) + 1
    
    class2ids = {'floor': 1, 'desk': 2, 'cabin': 3, 'door': 4}
    for dirname in os.listdir(path):
        filepath = os.path.join(path, dirname, 'output.json')
        with open(filepath, 'r') as f:
            tp_lines = f.readlines()
        for line in tqdm(tp_lines):
            sample = json.loads(line)

            if len(sample['result']) == 1 and sample['result'][0]['tagtype'] == 'delete':
                continue
    
            results = sample['result']
            image_path = os.path.join(path, dirname, sample['image_path'])
            filename = os.path.join('bubiao', dirname, sample['image_path'])
            img = cv2.imread(image_path)
            if img is None:
                continue
            height, width, _ = img.shape
            if len(results) == 0:
                continue
        
            coco_image = {
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": filename,
            }

            instance = np.zeros_like(img[...,0])
            ins_ids = 1
            for i, result in enumerate(results):
                if result.get('tagtype') == 'delete':
                    continue
                if result.get('datatype') not in ['mask', 'polygon']:
                    continue
                if result.get('datatype') == 'mask':
                    response = requests.get(result['maskData'])
                    mask_array = np.array(bytearray(response.content), dtype='uint8')
                    mask = cv2.imdecode(mask_array, cv2.IMREAD_COLOR)
                    mask = cv2.resize(mask, (width, height))
                elif result.get('datatype') == 'polygon':
                    contours = [np.array(result['data']).astype('int').reshape(1,-1,2)]
                    mask = np.zeros((height, width), dtype=np.uint8)
                    # 将每个轮廓绘制在掩码图像上
                    cv2.drawContours(mask, contours, -1, 255, -1)
                    mask = mask.reshape(height, width, 1)
                if save_ins:
                    instance[np.max(mask, axis=2) > 1] = ins_ids
                    ins_ids += 1
                ret, thresh1 = cv2.threshold(np.max(mask, axis=2), 1, 255, cv2.THRESH_BINARY)
                contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # contours = [c for c in contours if cv2.contourArea(c)>100]    # coco will filter in dataset
                if len(contours) == 0:
                    continue
                all_points = np.concatenate(contours)
                x, y, w, h = cv2.boundingRect(all_points)
                bbox = [x,y,w,h]
                area = 0
                for c in contours:
                    area+=cv2.contourArea(c)

                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class2ids[result['tagtype']],  # 对应 "person" 类别
                    "segmentation": [np.array(contours).flatten().tolist()] if len(contours)==1 else contours_to_rle(contours, width, height),  # 填入多边形形式的分割信息
                    "area": area,  # 填入分割区域的面积
                    "bbox": bbox,  # 填入边界框信息 [x, y, width, height]
                    "iscrowd": 0 if len(contours)==1 else 1
                }
                train_data["annotations"].append(annotation)
                annotation_id += 1

            if save_ins:
                # instance_path = image_path.replace('left', 'instance')
                instance_path = image_path[:-3] + 'png'
                os.makedirs(os.path.dirname(instance_path), exist_ok=True)
                # cv2.imwrite(instance_path, instance)
                cv2.imwrite(instance_path[:-4]+'_show.jpg', instance*80)
            train_data["images"].append(coco_image)
            image_id += 1
    print(image_id-1, annotation_id-1)
    
    # 将 COCO 数据保存为 JSON 文件
    save_coco_json(train_data, annotation_file)


def merge_label(path, annotation_file, save_ins=True):
    train_json = 'data/bank/annotations/trainV3.json'
    with open(train_json, 'r') as f:
        train_data = json.load(f)

    trainpath2ids = {}
    train_dict = {}
    for dic in train_data['images']:
        key = dic['file_name'][24:] if 'bubiao' not in dic['file_name'] else dic['file_name'][30:]
        if key not in trainpath2ids:
            trainpath2ids[key] = [dic['id']]
        else:
            trainpath2ids[key].append(dic['id'])
            print("repeat", key, trainpath2ids[key])
        dic['annotations'] = []
        train_dict[dic['id']] = dic
    
    for annotation in train_data['annotations']:
        if annotation['image_id'] in train_dict:
            train_dict[annotation['image_id']]['annotations'].append(annotation)
    
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "person"},
                       {"id": 2, "name": "desk"},
                       {"id": 3, "name": "cabin"},
                       {"id": 4, "name": "door"}]  # 假设只有一类 "person"
    }
    for key, values in trainpath2ids.items():
        dic = merge_mask(key, values, train_dict)
        coco_image = {
            "id": dic['id'],
            "width": dic['width'],
            "height": dic['height'],
            "file_name": dic['file_name'],
        }
        coco_data['images'].append(coco_image)
        for annotation in dic['annotations']:
            coco_data['annotations'].append(annotation)
        
        if len(values) > 1 and len(dic['annotations']) > 1:
            image = cv2.imread(os.path.join('data/bank', dic['file_name']))
            masks = annotations2masks(dic['annotations'], dic['height'], dic['width'])
            for i, mask in enumerate(masks):
                alpha = 0.5 + 0.5 * (i+1) / len(masks)
                image[mask>0,:] = image[mask>0,:] * (1-alpha) + alpha * np.array([0,255,0])
            cv2.imwrite("./debug/%s" % (key.replace('/', '_')), image)

            
    # 将 COCO 数据保存为 JSON 文件
    print(len(coco_data['images']), len(coco_data['annotations']))
    set_trace()
    save_coco_json(coco_data, annotation_file)

def merge_mask(key, values, train_dict):
    if len(values) == 1:
        return train_dict[values[0]]
    height, width = train_dict[values[0]]['height'], train_dict[values[0]]['width']
    annotations0 = train_dict[values[0]]['annotations']
    masks0 = annotations2masks(annotations0, height, width)
    if '陕西安阎良人民东路支行高柜1' in key and '疑似违规离柜/其他/2024-04-13 11-50-30_41887290.jpg' in key:
        train_dict[values[0]]['annotations'] = annotations0[1:] + train_dict[image_id]['annotations']
        return train_dict[values[0]]
    for image_id in values[1:]:
        annotations1 = train_dict[image_id]['annotations']
        masks1 = annotations2masks(annotations1, height, width)
        for j in range(len(masks1)):
            overlap = False
            for i in range(len(masks0)):
                iou = np.sum(np.logical_and(masks0[i], masks1[j])) / np.sum(np.logical_or(masks0[i], masks1[j]))
                if iou > 0.5:
                    if annotations1[j]['category_id'] != annotations0[i]['category_id']:
                        set_trace()
                    overlap = True
                    break
            if not overlap:
                annotations1[j]['image_id'] = values[0]
                annotations0.append(annotations1[j])
                masks0.append(masks1[j])
    
    return train_dict[values[0]]


def annotations2masks(annotations, height, width):
    masks = []
    for annotation in annotations:
        if annotation['iscrowd'] == 1:
            mask = mask_utils.decode(annotation['segmentation'])
        else:
            mask = np.zeros((height, width), dtype=np.uint8)
            contours = np.array(annotation['segmentation']).reshape(-1,1,2)
            cv2.drawContours(mask, [contours], -1, 1, -1)
        masks.append(mask)
        
    return masks
    
key = 'bank'
# key = 'floor_seg'
# key = 'split'
# key = 'concat'
# key = 'demo'
# key = 'testset'
# key = sys.argv[1]
key = 'update'
print(key)
if key == 'bank':
    paths = [
            # '/gemini/data-2/bank/metaloop_20240729165359',    # cabin 3940, door 4
            # '/gemini/data-2/bank/metaloop_20240729171406',    # desk: 6938
            # '/gemini/data-2/bank/metaloop_20240729172145',    # door, floor
            # '/gemini/data-2/bank/metaloop_20240729173736',    # floor without person 1000
            '/gemini/data-2/bank/metaloop_20240816180230',      # floor
            '/gemini/data-2/bank/metaloop_20240816180301'       # floor
            ]
    dirpath = os.path.dirname(paths[0])
    annotation_file = os.path.join(dirpath, 'floorV2.json')
    process_coco(paths, annotation_file)
elif key == 'floor_seg':
    path = '/gemini/data-3/mmsegmentation/data/bank/floor_seg'
    annotation_file = os.path.join(path, 'train.json')
    create_coco_dataset(path, annotation_file)
elif key == 'concat':
    # path1 = 'data/bank/floor_seg/train.json'
    # path2 = 'data/bank/train.json'
    # path = 'data/bank/annotations/all.json'
    path1 = 'data/bank/floorV2.json'
    path2 = 'data/bank/annotations/train.json'
    path = 'data/bank/annotations/trainV2.json'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    merge_jsons(path1, path2, path)
elif key == 'split':
    path = 'data/bank/annotations/all.json'
    train_path = 'data/bank/annotations/train.json'
    test_path = 'data/bank/annotations/test.json'
    split_json(path, train_path, test_path)
elif key == 'demo':
    train_path = 'data/bank/annotations/train.json'
    test_path = 'data/bank/annotations/test.json'
    for path in [train_path, test_path]:
        demo_coco_dataset(path)
elif key == 'testset':
    test_path = 'data/bank/annotations/test.json'
    create_testset(test_path)
elif key == 'update':
    path = 'data/bank/bubiao'
    annotation_file ='data/bank/annotations/trainV3.json'
    # update_label(path, annotation_file)
    annotation_file ='data/bank/annotations/trainV4.json'
    merge_label(path, annotation_file)
    
