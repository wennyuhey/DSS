import xml.dom.minidom as xmldom
import os
import json


json_out = {}
images = []
categories = [{'id' : 26, 'name': 'car'}]
annotations = []
ann_id = 0
dir = '/lustre/S/wangyu/dataset/sim10k/VOC2012/Annotations'
file_list = os.listdir(dir)

for img_id, file_name in enumerate(file_list):
    dir_name = dir + '/' + file_name
    xml_file = xmldom.parse(dir_name)
    file_name = file_name.replace('xml', 'jpg')
    img = {}
    img.update({'file_name': file_name})
    eles = xml_file.documentElement
    size = eles.getElementsByTagName('size')
    width = int(size[0].getElementsByTagName('width')[0].firstChild.data)
    height = int(size[0].getElementsByTagName('height')[0].firstChild.data)
    img.update({'width' : width, 'height': height, 'id': img_id})
    images.append(img)

    objects = eles.getElementsByTagName('object')
    for object in objects:
        ann = {'iscrowd': 1}
       
        category = object.getElementsByTagName('name')[0].firstChild.data
        for cat in categories:
            if cat['name'] == category:
                cat_id = cat['id']
        ann.update({'category_id':cat_id})
        bbox = object.getElementsByTagName('bndbox')[0]
        xmin = int(bbox.getElementsByTagName('xmin')[0].firstChild.data)
        xmax = int(bbox.getElementsByTagName('xmax')[0].firstChild.data)
        ymin = int(bbox.getElementsByTagName('ymin')[0].firstChild.data)
        ymax = int(bbox.getElementsByTagName('ymax')[0].firstChild.data)
        ann.update({'bbox': [xmin, ymin, xmax-xmin, ymax-ymin]})
        ann.update({'area': (xmax-xmin)*(ymax-ymin)})
        ann.update({'image_id':img_id})
        ann.update({'id':ann_id})
        annotations.append(ann)
        ann_id = ann_id + 1

json_out.update({'images': images, 'categories': categories, 'annotations': annotations})

with open('data.json', 'w') as f:
    json.dump(json_out, f)
