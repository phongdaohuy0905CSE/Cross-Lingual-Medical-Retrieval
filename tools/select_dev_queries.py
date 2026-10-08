import argparse
import json
import random
import re
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pyarrow.parquet as pq

def count_words(text):
    # Đếm số từ thô bằng regex
    return len(re.findall(r'\w+', text))

def categorize_intent(text):
    # Dùng heuristic từ khoá đơn giản để phân loại nhóm câu hỏi
    text_lower = text.lower()
    if any(k in text_lower for k in ["thuốc", "uống", "liều", "tác dụng phụ", "tác dụng", "vitamin"]):
        return "thuoc_men"
    if any(k in text_lower for k in ["xét nghiệm", "chỉ số", "beta", "scc", "siêu âm", "chụp", "nội soi"]):
        return "xet_nghiem"
    if any(k in text_lower for k in ["phòng tránh", "nguy cơ", "lây nhiễm", "vắc xin", "tiêm", "ăn", "thực phẩm"]):
        return "phong_ngua_loi_song"
    return "trieu_chung_benh_ly"

def main():
    # Đặt seed cố định để chạy lại vẫn ra đúng 100 câu này
    random.seed(42)
    
    print("Đọc dữ liệu từ data/query.parquet...")
    tbl = pq.read_table("data/query.parquet")
    d = tbl.to_pydict()
    
    queries = []
    for i in range(len(d["id"])):
        queries.append({
            "id": d["id"][i],
            "query": d["query"][i],
            "word_count": count_words(d["query"][i]),
            "intent": categorize_intent(d["query"][i])
        })
        
    print(f"Tổng số query: {len(queries)}")
    
    # Chia bucket theo độ dài câu hỏi
    short_qs = [q for q in queries if q["word_count"] < 15]
    med_qs = [q for q in queries if 15 <= q["word_count"] <= 40]
    long_qs = [q for q in queries if q["word_count"] > 40]
    
    # Trộn ngẫu nhiên trong mỗi bucket
    random.shuffle(short_qs)
    random.shuffle(med_qs)
    random.shuffle(long_qs)
    
    print(f"Phân bố độ dài gốc: Ngắn (<15 từ): {len(short_qs)}, Trung bình (15-40 từ): {len(med_qs)}, Dài (>40 từ): {len(long_qs)}")
    
    # Chọn 100 query: 25 ngắn, 50 trung bình, 25 dài
    selected = short_qs[:25] + med_qs[:50] + long_qs[:25]
    
    # Thống kê intent của 100 câu đã chọn
    intent_counts = {}
    for q in selected:
        intent = q["intent"]
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
    print(f"\nPhân bố Ý định y khoa (Intent) trong 100 câu:")
    for intent, count in intent_counts.items():
        print(f"  - {intent}: {count} câu")
    
    # Lưu ra JSON
    out_path = "data/dev_100_queries.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
        
    print(f"\nĐã lưu 100 câu query đại diện vào '{out_path}'. Bạn có thể mở file này để gửi cho mô hình gán nhãn.")

if __name__ == "__main__":
    main()
