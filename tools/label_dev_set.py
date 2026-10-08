import argparse
import json
import os
import sys
import pyarrow as pa
import pyarrow.parquet as pq

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Vui lòng cài đặt google-genai: pip install google-genai")
    exit(1)

prompt_template = """Bạn là một bác sĩ chuyên khoa đánh giá hệ thống tìm kiếm tài liệu y khoa.
Câu hỏi của người dùng: "{query}"

Dưới đây là {num_candidates} đoạn trích được hệ thống tìm kiếm trả về (mỗi đoạn trích có chứa thông tin văn bản):
{candidates_text}

Nhiệm vụ của bạn là xác định đoạn trích nào CHỨA THÔNG TIN TRẢ LỜI CHÍNH XÁC VÀ ĐỦ cho câu hỏi trên.
Nếu có đoạn trả lời được, hãy trả về chỉ số của đoạn đó (ví dụ: 1, 2, 3...). Ưu tiên đoạn có chứa câu trả lời trực tiếp và rõ ràng nhất.
Nếu không có đoạn nào phù hợp, trả về null.
Chỉ trả về định dạng JSON hợp lệ: {{"relevant_chunk": 1}} hoặc {{"relevant_chunk": null}}.
"""

def main():
    parser = argparse.ArgumentParser(description="Dùng Gemini API để gán nhãn tập dev candidates")
    parser.add_argument("--candidates", default="data/dev_candidates.json", help="File đầu vào chứa candidates")
    parser.add_argument("--out", default="data/qrels_dev.parquet", help="File qrels đầu ra")
    args = parser.parse_args()
    
    if not os.path.exists(args.candidates):
        print(f"Lỗi: Không tìm thấy {args.candidates}. Hãy chạy tools/build_dev_candidates.py trước.")
        return
        
    with open(args.candidates, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Lỗi: Vui lòng thiết lập biến môi trường GEMINI_API_KEY.")
        print("Windows (PowerShell): $env:GEMINI_API_KEY=\"AQ.Ab8RN6LftpD_0C5C1x7gtqiZIyFS2KnpRbd9_mqFjfguftdyNQ\"")
        return
        
    client = genai.Client(api_key=api_key)
    
    qids = []
    doc_ids = []
    chunk_texts = []
    
    print(f"Bắt đầu gán nhãn {len(data)} câu hỏi với Gemini...")
    for item in data:
        qid = item["id"]
        query = item["query"]
        cands = item.get("candidates", [])
        
        if not cands:
            continue
            
        cand_text_blocks = []
        for i, c in enumerate(cands, 1):
            cand_text_blocks.append(f"--- Chunk {i} ---\n{c['text']}")
            
        prompt = prompt_template.format(
            query=query,
            num_candidates=len(cands),
            candidates_text="\n\n".join(cand_text_blocks)
        )
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash', # Dùng 2.5 flash cho nhanh và tiết kiệm
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.0
                    ),
                )
                res_json = json.loads(response.text)
                relevant_idx = res_json.get("relevant_chunk")
                
                if relevant_idx is not None and 1 <= relevant_idx <= len(cands):
                    chosen = cands[relevant_idx - 1]
                    qids.append(qid)
                    doc_ids.append(chosen["doc_id"])
                    chunk_texts.append(chosen["text"])
                    print(f"[{qid}] Nhãn: Chunk {relevant_idx}")
                else:
                    print(f"[{qid}] Không có đoạn phù hợp.")
                break # Break out of retry loop on success
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    print(f"[{qid}] Bị giới hạn API (Rate limit), đợi 31s rồi thử lại (lần {attempt+1}/{max_retries})...")
                    import time
                    time.sleep(31)
                else:
                    print(f"[{qid}] Lỗi khi gọi API: {e}")
                    break
            
    # Ghi ra parquet
    if qids:
        table = pa.table({
            "qid": pa.array(qids, type=pa.int64()),
            "doc_id": pa.array(doc_ids, type=pa.string()),
            "chunk_text": pa.array(chunk_texts, type=pa.string())
        })
        pq.write_table(table, args.out)
        print(f"\nĐã ghi file qrels ra {args.out} ({len(qids)} nhãn).")
    else:
        print("\nKhông tìm thấy nhãn nào hợp lệ.")

if __name__ == "__main__":
    main()
