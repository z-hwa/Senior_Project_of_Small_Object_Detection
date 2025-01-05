import re
from collections import defaultdict
import os

# 動態生成區間列表
def generate_intervals(start, step, end):
    intervals = []
    current = start
    while current < end:
        intervals.append((current, min(current + step - 1, end)))
        current += step
    return intervals

# 根據需求生成區間
length_intervals = generate_intervals(8, 10, 318)
width_intervals = generate_intervals(8, 10, 318)

def find_interval(value, intervals):
    """找到數值所在的區間"""
    for idx, (low, high) in enumerate(intervals):
        if low <= value <= high:
            return (idx, f"[{low}, {high}]")  # 回傳索引與區間
    return (float('inf'), "Out of Range")  # 如果不在範圍內

# 路徑
input_file = "tools/my_tools/swin_4stage/unwatched_swin_4stage"
output_file = "tools/my_tools/swin_4stage/classified_sizes_sorted.txt"

# 儲存結果
classified_data = defaultdict(list)

# 檢查檔案是否存在
if not os.path.exists(input_file):
    print(f"檔案 {input_file} 不存在，請檢查路徑是否正確。")
    exit()

with open(input_file, "r") as file:
    for line in file:
        # 解析格式：unwatch size: (長, 寬, 其他值)
        match = re.search(r"unwatch size: \((\d+\.?\d*), (\d+\.?\d*), (\d+)\)", line)
        if match:
            length = float(match.group(1))
            width = float(match.group(2))
            other = int(match.group(3))

            # 找到區間
            length_idx, length_range = find_interval(length, length_intervals)
            width_idx, width_range = find_interval(width, width_intervals)

            # 將資料分類
            key = (length_idx, length_range, width_idx, width_range)
            classified_data[key].append((length, width, other))

# 排序並輸出結果
with open(output_file, "w") as file:
    for key, items in sorted(classified_data.items(), key=lambda x: (x[0][0], x[0][2])):
        print(key[1])
        length_range = key[1]  # key[1] 是長的區間範圍
        width_range = key[3]   # key[3] 是寬的區間範圍
        file.write(f"區間: 長 {length_range}, 寬 {width_range}\n")
        for item in sorted(items):
            file.write(f"  {item}\n")
        file.write("\n")

print(f"已完成分類並根據區間排序，結果儲存到 {output_file}")
