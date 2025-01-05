import json

# 將json檔案的置信度
# 進一步調整

def load_json(file_path):
    """讀取 JSON 檔案"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, file_path):
    """儲存資料到 JSON 檔案"""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def filter_by_confidence(data, threshold):
    """過濾掉置信度低於指定閾值的項目"""
    return [item for item in data if item["score"] >= threshold]

# 主程式
input_file = "work_dirs/cascade_mask_rcnn_swin_finetune_rfla_4stage/results_005.json"  # 輸入檔案名稱
output_file = "filtered_data.json"  # 輸出檔案名稱
confidence_threshold = 0.2  # 設定置信度閾值

# 讀取資料
data = load_json(input_file)

# 過濾資料
filtered_data = filter_by_confidence(data, confidence_threshold)

# 儲存結果
save_json(filtered_data, output_file)

print(f"過濾完成，結果已儲存至 {output_file}")
