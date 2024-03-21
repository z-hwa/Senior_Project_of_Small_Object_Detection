# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import xml.etree.ElementTree as ET
import shutil
import numpy as np
from mmcv.fileio import dump, list_from_file
from mmcv import mkdir_or_exist, track_progress

# from mmdet.evaluation import voc_classes
from mmdet.core import voc_classes

label_ids = {name: i for i, name in enumerate(voc_classes())}

def parse_xml(args):
    xml_path, img_path = args
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    bboxes = []
    labels = []
    bboxes_ignore = []
    labels_ignore = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        label = label_ids[name]
        difficult = obj.find('difficult')
        difficult = 0 if difficult is None else int(difficult.text)  # 设置默认值为0
        bnd_box = obj.find('bndbox')
        bbox = [
            round(float(bnd_box.find('xmin').text)),  # 使用四舍五入取整
            round(float(bnd_box.find('ymin').text)),  # 使用四舍五入取整
            round(float(bnd_box.find('xmax').text)),  # 使用四舍五入取整
            round(float(bnd_box.find('ymax').text))   # 使用四舍五入取整
        ]
        if difficult:
            bboxes_ignore.append(bbox)
            labels_ignore.append(label)
        else:
            bboxes.append(bbox)
            labels.append(label)
    if not bboxes:
        bboxes = np.zeros((0, 4))
        labels = np.zeros((0, ))
    else:
        bboxes = np.array(bboxes, ndmin=2) - 1
        labels = np.array(labels)
    if not bboxes_ignore:
        bboxes_ignore = np.zeros((0, 4))
        labels_ignore = np.zeros((0, ))
    else:
        bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
        labels_ignore = np.array(labels_ignore)
    annotation = {
        'filename': img_path,
        'width': w,
        'height': h,
        'ann': {
            'bboxes': bboxes.astype(np.float32),
            'labels': labels.astype(np.int64),
            'bboxes_ignore': bboxes_ignore.astype(np.float32),
            'labels_ignore': labels_ignore.astype(np.int64)
        }
    }
    return annotation

def cvt_annotations(devkit_path, years, split, out_file):
    if not isinstance(years, list):
        years = [years]
    annotations = []
    for year in years:
        filelist = osp.join(devkit_path,
                            f'VOC{year}/ImageSets/Main/{split}.txt')
        if not osp.isfile(filelist):
            print(f'filelist does not exist: {filelist}, '
                  f'skip voc{year} {split}')
            return
        img_names = list_from_file(filelist)
        xml_paths = [
            osp.join(devkit_path, f'VOC{year}/Annotations/{img_name}.xml')
            for img_name in img_names
        ]
        img_paths = [
            f'VOC{year}/JPEGImages/{img_name}.jpg' for img_name in img_names
        ]
        part_annotations = track_progress(parse_xml,
                                          list(zip(xml_paths, img_paths)))
        annotations.extend(part_annotations)
    if out_file.endswith('json'):
        annotations = cvt_to_coco_json(annotations)
    dump(annotations, out_file)
    return annotations

def cvt_to_coco_json(annotations):
    image_id = 0
    annotation_id = 0
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instance'
    coco['categories'] = []
    coco['annotations'] = []
    image_set = set()

    def addAnnItem(annotation_id, image_id, category_id, bbox, difficult_flag):
        annotation_item = dict()
        annotation_item['segmentation'] = []

        seg = []
        # bbox[] is x1,y1,x2,y2
        # left_top
        seg.append(int(bbox[0]))
        seg.append(int(bbox[1]))
        # left_bottom
        seg.append(int(bbox[0]))
        seg.append(int(bbox[3]))
        # right_bottom
        seg.append(int(bbox[2]))
        seg.append(int(bbox[3]))
        # right_top
        seg.append(int(bbox[2]))
        seg.append(int(bbox[1]))

        annotation_item['segmentation'].append(seg)

        xywh = np.array(
            [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
        annotation_item['area'] = int(xywh[2] * xywh[3])
        if difficult_flag == 1:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 1
        else:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 0
        annotation_item['image_id'] = int(image_id)
        annotation_item['bbox'] = xywh.astype(int).tolist()
        annotation_item['category_id'] = int(category_id)
        annotation_item['id'] = int(annotation_id)
        coco['annotations'].append(annotation_item)
        return annotation_id + 1

    for category_id, name in enumerate(voc_classes()):
        category_item = dict()
        category_item['supercategory'] = str('none')
        category_item['id'] = int(category_id)
        category_item['name'] = str(name)
        coco['categories'].append(category_item)

    for ann_dict in annotations:
        file_name = ann_dict['filename']
        ann = ann_dict['ann']
        assert file_name not in image_set
        image_item = dict()
        image_item['id'] = int(image_id)
        image_item['file_name'] = str(file_name)
        image_item['height'] = int(ann_dict['height'])
        image_item['width'] = int(ann_dict['width'])
        coco['images'].append(image_item)
        image_set.add(file_name)

        bboxes = ann['bboxes'][:, :4]
        labels = ann['labels']
        for bbox_id in range(len(bboxes)):
            bbox = bboxes[bbox_id]
            label = labels[bbox_id]
            annotation_id = addAnnItem(
                annotation_id, image_id, label, bbox, difficult_flag=0)

        bboxes_ignore = ann['bboxes_ignore'][:, :4]
        labels_ignore = ann['labels_ignore']
        for bbox_id in range(len(bboxes_ignore)):
            bbox = bboxes_ignore[bbox_id]
            label = labels_ignore[bbox_id]
            annotation_id = addAnnItem(
                annotation_id, image_id, label, bbox, difficult_flag=1)

        image_id += 1

    return coco

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert PASCAL VOC annotations to mmdetection format')
    parser.add_argument('devkit_path', help='pascal voc devkit path')
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--out-format',
        default='coco',
        choices=('pkl', 'coco'),
        help='output format, "coco" indicates coco annotation format')
    args = parser.parse_args()
    return args

def copy_images_to_folders(devkit_path, out_dir):
    jpeg_images_dir = osp.join(devkit_path, 'JPEGImages')
    image_sets_main_dir = osp.join(devkit_path, 'ImageSets', 'Main')

    splits_and_dirs = {
        'train': 'train2017',
        'test': 'test2017',
        'val': 'val2017'
    }

    for split, new_dir_name in splits_and_dirs.items():
        txt_file = osp.join(image_sets_main_dir, f'{split}.txt')
        new_folder = osp.join(out_dir, new_dir_name)
        mkdir_or_exist(new_folder)  # Ensure the directory exists

        with open(txt_file, 'r') as f:
            img_names = [line.strip() for line in f.readlines()]

        for img_name in img_names:
            src_img_path = osp.join(jpeg_images_dir, f'{img_name}.jpg')
            dest_img_path = osp.join(new_folder, f'{img_name}.jpg')
            shutil.copy(src_img_path, dest_img_path)

import os.path as osp
import shutil

def copy_images_to_dest(src_dir, dest_dir, img_list_file):
    """复制图片到目标文件夹"""
    with open(img_list_file, 'r') as f:
        lines = f.readlines()
        img_names = [line.strip() + '.jpg' for line in lines]
        print(f"Copying {len(img_names)} images from {src_dir} to {dest_dir} ...")
        for img_name in img_names:
            src_path = osp.join(src_dir, img_name)
            dest_path = osp.join(dest_dir, img_name)
            if not osp.exists(src_path):
                print(f"Image {src_path} does not exist!")
                continue
            shutil.copy(src_path, dest_path)
            print(f"Copied {img_name} to {dest_dir}")

def main():
    args = parse_args()
    devkit_path = args.devkit_path
    out_dir = args.out_dir if args.out_dir else devkit_path
    mkdir_or_exist(out_dir)

    years = []
    if osp.isdir(osp.join(devkit_path, 'VOC2007')):
        years.append('2007')
    if osp.isdir(osp.join(devkit_path, 'VOC2012')):
        years.append('2012')
    if '2007' in years and '2012' in years:
        years.append(['2007', '2012'])

    if not years:
        raise IOError(f'The devkit path {devkit_path} contains neither '
                      '"VOC2007" nor "VOC2012" subfolder')

    out_fmt = f'.{args.out_format}'
    if args.out_format == 'coco':
        out_fmt = '.json'

    for year in years:
        if isinstance(year, list):
            multi_year = year
        else:
            multi_year = [year]

        for y in multi_year:
            for split in ['train', 'val']:
                if split == 'train':
                    dataset_name = 'instances_train2017'
                    sub_folder = 'train2017'
                elif split == 'val':
                    dataset_name = 'instances_val2017'
                    sub_folder = 'val2017'
                else:
                    continue

                print(f'processing {dataset_name} for VOC{y} ...')
                cvt_annotations(devkit_path, y, split,
                                osp.join(out_dir, dataset_name + out_fmt))

                # 创建并复制图片到对应的文件夹
                img_out_dir = osp.join(out_dir, "..", sub_folder, f'VOC{y}', 'JPEGImages')
                mkdir_or_exist(img_out_dir)
                img_list_file = osp.join(devkit_path, f'VOC{y}', 'ImageSets', 'Main', f'{split}.txt')
                src_img_dir = osp.join(devkit_path, f'VOC{y}', 'JPEGImages')
                copy_images_to_dest(src_img_dir, img_out_dir, img_list_file)

    print('Done!')

if __name__ == '__main__':
    main()