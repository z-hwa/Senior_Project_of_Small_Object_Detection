import matplotlib.pyplot as plt
import numpy as np
import re
import argparse
import os

'''
讀取unwatched檔案
並繪製長寬比例長條圖

一樣可以帶上第二個檔案做比較

'''

# 設定預設檔案路徑
default_data_file = "tools/my_tools/swin_4stage_data/gt_bbox.txt"  # 替換為你的檔案名稱
default_data_file_2 = "tools/my_tools/swin_4stage_data/true_negative.txt"  # 第二個檔案路徑，預設為空

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

# 讀取數據檔案的函式
def read_data(file_path):
    data = []
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            for line in file:
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

# 計算長寬比
aspect_ratios1 = widths1 / heights1 if widths1.size > 0 else []

# 讀取第二個檔案的數據並計算長寬比（如果有提供第二個檔案）
if data_file_2:
    data2 = read_data(data_file_2)
    widths2 = data2[:, 0] if data2.size > 0 else []
    heights2 = data2[:, 1] if data2.size > 0 else []
    aspect_ratios2 = widths2 / heights2 if widths2.size > 0 else []
else:
    aspect_ratios2 = []

# 顯示長寬比的統計數據
print(f"第一個檔案的長寬比統計:")
if aspect_ratios1.size > 0:
    print(f"最大長寬比: {aspect_ratios1.max()}")
    print(f"最小長寬比: {aspect_ratios1.min()}")
    print(f"平均長寬比: {aspect_ratios1.mean()}")
    print(f"標準差: {aspect_ratios1.std()}")
else:
    print("沒有數據可供統計。")

# 顯示第二個檔案的統計數據（如果有第二個檔案）
if aspect_ratios2.size > 0:
    print(f"\n第二個檔案的長寬比統計:")
    print(f"最大長寬比: {aspect_ratios2.max()}")
    print(f"最小長寬比: {aspect_ratios2.min()}")
    print(f"平均長寬比: {aspect_ratios2.mean()}")
    print(f"標準差: {aspect_ratios2.std()}")

# 繪製長寬比的直方圖比較
plt.figure(figsize=(10, 6))

# 第一個檔案的直方圖
if aspect_ratios1.size > 0:
    plt.hist(aspect_ratios1, bins=20, color='purple', alpha=0.7, edgecolor='k', label='Dataset 1')

# 第二個檔案的直方圖
if aspect_ratios2.size > 0:
    plt.hist(aspect_ratios2, bins=20, color='orange', alpha=0.7, edgecolor='k', label='Dataset 2')

plt.title("Aspect Ratio Distribution Comparison", fontsize=14)
plt.xlabel("Aspect Ratio", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# 設定更精細的刻度
plt.xticks(np.arange(min(aspect_ratios1.min(), aspect_ratios2.min()), max(aspect_ratios1.max(), aspect_ratios2.max()) + 0.1, step=0.1), fontsize=5)
plt.yticks(fontsize=5)

# 儲存長寬比直方圖
plt.savefig("aspect_ratio_histogram_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
