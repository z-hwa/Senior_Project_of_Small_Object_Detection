# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

import mmcv
import numpy as np
from mmcv import Config, DictAction

from mmdet.core.evaluation import eval_map
from mmdet.core.visualization import imshow_gt_det_bboxes
from mmdet.datasets import build_dataset, get_loading_pipeline
from mmdet.utils import update_data_root


def bbox_map_eval(det_result, annotation):
    """Evaluate mAP of single image det result.

    Args:
        det_result (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotation (dict): Ground truth annotations where keys of
             annotations are:

            - bboxes: numpy array of shape (n, 4)
            - labels: numpy array of shape (n, )
            - bboxes_ignore (optional): numpy array of shape (k, 4)
            - labels_ignore (optional): numpy array of shape (k, )

    Returns:
        float: mAP
    """

    # use only bbox det result
    if isinstance(det_result, tuple):
        bbox_det_result = [det_result[0]]
    else:
        bbox_det_result = [det_result]
    # mAP
    iou_thrs = np.linspace(
        .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
    mean_aps = []
    for thr in iou_thrs:
        mean_ap, _ = eval_map(
            bbox_det_result, [annotation], iou_thr=thr, logger='silent')
        mean_aps.append(mean_ap)
    return sum(mean_aps) / len(mean_aps)


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet eval image prediction result for each')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'prediction_path', help='prediction path where test pkl result')
    parser.add_argument(
        'show_dir', help='directory where painted images will be saved')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=0,
        help='the interval of show (s), 0 is block')
    parser.add_argument(
        '--topk',
        default=20,
        type=int,
        help='saved Number of the highest topk '
        'and lowest topk after index sorting')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0,
        help='score threshold (default: 0.)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    mmcv.check_file_exist(args.prediction_path)

    cfg = Config.fromfile(args.config)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.data.test.test_mode = True

    cfg.data.test.pop('samples_per_gpu', 0)
    cfg.data.test.pipeline = get_loading_pipeline(cfg.data.train.pipeline)
    dataset = build_dataset(cfg.data.test)
    outputs = mmcv.load(args.prediction_path)

    # 指定要處理的圖片索引列表
    indices_to_process = [i for i in range(0, 10)] + [i for i in range(1970, 1980)] + \
                            [i for i in range(3711, 3721)] + \
                            [i for i in range(3738, 3748)] + \
                            [i for i in range(6009, 6019)]

    for i in indices_to_process:
        if 0 <= i < len(dataset):
            data_info = dataset.prepare_train_img(i)
            result = outputs[i]  # 假設 outputs 的順序與 dataset 一致
            mAP = bbox_map_eval(result, data_info['ann_info'])

            # calc save file path
            filename = data_info['filename']
            if data_info['img_prefix'] is not None:
                filename = osp.join(data_info['img_prefix'], filename)
            fname, name = osp.splitext(osp.basename(filename))
            save_filename = fname + '_' + str(round(mAP, 3)) + name
            out_file = osp.join(args.show_dir, save_filename)

            imshow_gt_det_bboxes(
                data_info['img'],
                data_info,
                result,
                dataset.CLASSES,
                gt_bbox_color=(200, 255, 200),
                gt_text_color=(200, 200, 200),
                gt_mask_color=dataset.PALETTE,
                det_bbox_color=dataset.PALETTE,
                det_text_color=(200, 200, 200),
                det_mask_color=dataset.PALETTE,
                thickness=1,
                font_size=11,
                show=args.show,
                score_thr=args.show_score_thr,
                wait_time=args.wait_time,
                out_file=out_file)
            print(f"Processed and visualized image {i}")
        else:
            print(f"Index {i} is out of range for the dataset.")

        
if __name__ == '__main__':
    main()