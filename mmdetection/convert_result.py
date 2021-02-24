import pickle
import os.path as osp
import json
"""
instance_classes = {
     'person'    : 24,
     'rider'     : 25,
     'car'       : 26,
     'truck'     : 27,
     'train'     : 31,
     'motorcycle': 32,
     'bicycle'   : 33,
     'bus'       : 28
}
"""
instance_classes = [24, 25, 26, 27, 28, 31, 32, 33]

anns = json.load(open('/home/wangyu/env/mmdetection_train/mmdetection/data/foggy_cityscapes/annotations/instancesonly_filtered_gtFine_val.json'))
image = anns['images']
result = pickle.load(open('gnwm_result.pkl', 'rb'))
convert_result = []
for idx, img in enumerate(result):
    img_name = image[idx]
    name = osp.splitext(osp.split(img_name['file_name'])[-1])[0]
    for cat_idx, cat in enumerate(img) :
        for obj in cat:
            obj = [i.astype(float) for i in obj]
            det = {}
            det['image_id'] = name
            det['category_id'] = instance_classes[cat_idx]
            det['score'] = obj[-1]
            det['bbox'] = obj[0:4]
            convert_result.append(det)

with open('convert_gnwm_result.json', 'w') as f:
    json.dump(convert_result, f)
