import cv2
import numpy as np
import os

'''
載入指定的兩張圖片
並查看兩者的光流
'''

def calculate_optical_flow(prev_frame, curr_frame):
    """計算光流"""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def visualize_optical_flow(flow):
    """可視化光流場"""
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr

# 讀取圖片
path = '/home/zhwa/Document/data/MVA2025/phase_1/unzip/pub_test/'
name1 = '0001/00003.jpg'
name2 = '0001/00004.jpg'

prev_frame = cv2.imread(os.path.join(path, name1))
curr_frame = cv2.imread(os.path.join(path, name2))

# 計算光流
flow = calculate_optical_flow(prev_frame, curr_frame)

# 可視化光流
flow_visual = visualize_optical_flow(flow)

# 顯示光流可視化結果
cv2.imshow('Optical Flow', flow_visual)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存光流可視化結果
# cv2.imwrite('optical_flow.jpg', flow_visual)