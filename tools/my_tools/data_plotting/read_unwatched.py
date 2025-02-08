import matplotlib.pyplot as plt
import numpy as np
import re
import argparse
import os

'''
讀取unwatched的檔案
並繪製成散佈圖 以及 長條圖

如果帶上第二個檔案名稱
可以繪製比較

 run
 python tools/my_tools/data_plotting/read_unwatched.py
'''


# 設定預設檔案路徑
default_data_file = "tools/my_tools/swin_4stage_data/gt_bbox.txt"  # 替換為你的檔案名稱
default_data_file_2 = "tools/my_tools/swin_4stage_data/false_positive.txt"  # 替換為你的檔案名稱
# default_data_file_2 = None#"tools/my_tools/swin_4stage_data/true_negative.txt"  # 第二個檔案路徑，預設為空

# 設定命令列參數解析
parser = argparse.ArgumentParser(description="Aspect Ratio Distribution")
parser.add_argument('--custom', action='store_true', help='Enable custom path selection')
parser.add_argument('--data_file', type=str, default=default_data_file, help='Path to the data file')
parser.add_argument('--data_file_2', type=str, default=default_data_file_2, help='Path to the second data file for comparison')

# 解析命令列參數
args = parser.parse_args()

# 根據 --custom 標誌決定是否手動選擇路徑
if args.custom:
    data_file = input(f"Enter the data file path (default: {args.data_file}): ") or args.data_file
    data_file_2 = input(f"Enter the second data file path (optional): ") or args.data_file_2
else:
    data_file = args.data_file
    data_file_2 = args.data_file_2

# 讀取數據的函式
def read_data(file_path):
    data = []
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            for line in file:
                # 使用正則表達式解析每行數據
                match = re.search(r"\(([\d.]+),\s*([\d.]+),\s*([\d]+)\)", line)
                if match:
                    width, height, _ = map(float, match.groups())
                    data.append((width, height))
        return np.array(data)
    else:
        print(f"File {file_path} not found.")
        return np.array([])

# 讀取第一個檔案的數據
data1 = read_data(data_file)
widths1 = data1[:, 0] if data1.size > 0 else []
heights1 = data1[:, 1] if data1.size > 0 else []

# 讀取第二個檔案的數據
if data_file_2:
    data2 = read_data(data_file_2)
    widths2 = data2[:, 0] if data2.size > 0 else []
    heights2 = data2[:, 1] if data2.size > 0 else []
else:
    widths2 = heights2 = []

# 繪製散佈圖
plt.figure(figsize=(10, 6))

# 第一個檔案的散佈圖
if widths1.size > 0 and heights1.size > 0:
    plt.scatter(widths1, heights1, alpha=0.7, edgecolors='k', label=os.path.basename(data_file))  # 使用檔名作為圖例

# 第二個檔案的散佈圖（如果有提供）
if data_file_2 and widths2.size > 0 and heights2.size > 0:
    plt.scatter(widths2, heights2, alpha=0.7, edgecolors='r', label=os.path.basename(data_file_2))  # 使用檔名作為圖例

plt.title("Width vs Height Distribution", fontsize=14)
plt.xlabel("Width", fontsize=12)
plt.ylabel("Height", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# 設定更精細的 x 軸刻度
plt.xticks(np.arange(int(min(widths1)), int(max(widths1)) + 1, step=10), fontsize=7)  # x 軸刻度間隔設為 1

# 儲存散佈圖
plt.savefig("scatter_plot_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

# 繪製直方圖
plt.figure(figsize=(12, 6))

# 寬度直方圖
plt.subplot(1, 2, 1)
if widths1.size > 0:
    plt.hist(widths1, bins=15, color='blue', alpha=0.7, edgecolor='k', label=os.path.basename(data_file))  # 使用檔名作為圖例
if data_file_2 and widths2.size > 0:
    plt.hist(widths2, bins=15, color='red', alpha=0.7, edgecolor='k', label=os.path.basename(data_file_2))  # 使用檔名作為圖例
plt.title("Width Distribution", fontsize=14)
plt.xlabel("Width", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# 設定更精細的 x 軸刻度
plt.xticks(np.arange(int(min(widths1)), int(max(widths1)) + 1, step=10), fontsize=7)  # x 軸刻度間隔設為 1

# 高度直方圖
plt.subplot(1, 2, 2)
if heights1.size > 0:
    plt.hist(heights1, bins=15, color='green', alpha=0.7, edgecolor='k', label=os.path.basename(data_file))  # 使用檔名作為圖例
if data_file_2 and heights2.size > 0:
    plt.hist(heights2, bins=15, color='orange', alpha=0.7, edgecolor='k', label=os.path.basename(data_file_2))  # 使用檔名作為圖例
plt.title("Height Distribution", fontsize=14)
plt.xlabel("Height", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()


# 設定更精細的 x 軸刻度
plt.xticks(np.arange(int(min(widths1)), int(max(widths1)) + 1, step=10), fontsize=7)  # x 軸刻度間隔設為 1

plt.tight_layout()

# 儲存直方圖
plt.savefig("histograms_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
