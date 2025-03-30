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


class ResultVisualizer:
    """Display and save evaluation results.

    Args:
        show (bool): Whether to show the image. Default: True
        wait_time (float): Value of waitKey param. Default: 0.
        score_thr (float): Minimum score of bboxes to be shown.
           Default: 0
    """

    def __init__(self, show=False, wait_time=0, score_thr=0):
        self.show = show
        self.wait_time = wait_time
        self.score_thr = score_thr

    def evaluate_and_show(self,
                          dataset,
                          results,
                          show_dir='work_dir',
                          eval_fn=None,
                          show=False,  # 將 show 參數提到這裡
                          wait_time=0,  # 將 wait_time 參數提到這裡
                          score_thr=0):  # 將 score_thr 參數提到這裡
        """Evaluate and show results for each image.

        Args:
            dataset (Dataset): A PyTorch dataset.
            results (list): Det results from test results pkl file
            show_dir (str, optional): The filename to write the image.
                Default: 'work_dir'
            eval_fn (callable, optional): Eval function, Default: None
            show (bool): Whether to show the image. Default: False
            wait_time (float): Value of waitKey param. Default: 0.
            score_thr (float): Minimum score of bboxes to be shown.
                Default: 0
        """
        mmcv.mkdir_or_exist(show_dir)
        if eval_fn is None:
            eval_fn = bbox_map_eval
        else:
            assert callable(eval_fn)

        prog_bar = mmcv.ProgressBar(len(results))
        for i, (result,) in enumerate(zip(results)):
            data_info = dataset.prepare_train_img(i)
            mAP = eval_fn(result, data_info['ann_info'])

            # calc save file path
            filename = data_info['filename']
            if data_info['img_prefix'] is not None:
                filename = osp.join(data_info['img_prefix'], filename)
            fname, name = osp.splitext(osp.basename(filename))
            save_filename = fname + '_' + str(round(mAP, 3)) + name
            out_file = osp.join(show_dir, save_filename)

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
                show=show,  # 使用傳入的 show 參數
                score_thr=score_thr,  # 使用傳入的 score_thr 參數
                wait_time=wait_time,  # 使用傳入的 wait_time 參數
                out_file=out_file)
            prog_bar.update()


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

    result_visualizer = ResultVisualizer()  # 初始化 ResultVisualizer
    result_visualizer.evaluate_and_show(
        dataset,
        outputs,
        show_dir=args.show_dir,
        show=args.show,  # 傳遞 show 參數
        wait_time=args.wait_time,  # 傳遞 wait_time 參數
        score_thr=args.show_score_thr  # 傳遞 score_thr 參數
    )


if __name__ == '__main__':
    main()