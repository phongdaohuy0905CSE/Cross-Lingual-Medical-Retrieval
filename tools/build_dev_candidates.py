import argparse
import json
import os
from pathlib import Path
import pyarrow.parquet as pq

def main():
    parser = argparse.ArgumentParser(description="Lọc Top-5 ứng viên từ file run BM25 cho tập dev_100_queries")
    parser.add_argument("--run", default="runs/structure_bm25.parquet", help="Đường dẫn file run BM25 (chạy từ mir.retrieve)")
    parser.add_argument("--queries", default="data/dev_100_queries.json", help="Đường dẫn dev_100_queries.json")
    parser.add_argument("--chunks", default="chunks/structure", help="Thư mục chunks (để lấy text đoạn)")
    parser.add_argument("--out", default="data/dev_candidates.json", help="File đầu ra chứa Top-5 ứng viên")
    parser.add_argument("--topk", type=int, default=5, help="Số ứng viên tối đa mỗi câu hỏi")
    args = parser.parse_args()
    
    # 1. Đọc danh sách 100 queries
    if not os.path.exists(args.queries):
        print(f"Lỗi: Không tìm thấy {args.queries}.")
        return
        
    with open(args.queries, "r", encoding="utf-8") as f:
        dev_queries = json.load(f)
    dev_qids = {q["id"]: q for q in dev_queries}
    
    # 2. Đọc file run
    if not os.path.exists(args.run):
        print(f"Lỗi: Không tìm thấy file run '{args.run}'. Vui lòng chạy BM25 trước:\n"
              f"  py -m mir.clean --apply --force\n"
              f"  py -m mir.chunk\n"
              f"  py -m mir.index bm25-tokenize\n"
              f"  py -m mir.index bm25-build\n"
              f"  py -m mir.retrieve run --branch bm25")
        return
        
    print(f"Đọc {args.run}...")
    tbl = pq.read_table(args.run)
    run_dict = tbl.to_pydict()
    
    # Lọc ra top-k
    candidates = {}
    for i in range(len(run_dict["qid"])):
        qid = int(run_dict["qid"][i])
        if qid in dev_qids:
            if qid not in candidates:
                candidates[qid] = []
            if len(candidates[qid]) < args.topk:
                candidates[qid].append({
                    "chunk_id": run_dict["chunk_id"][i],
                    "doc_id": run_dict["doc_id"][i],
                    "score": float(run_dict["score"][i]),
                    "rank": int(run_dict["rank"][i])
                })
                
    # 3. Đọc text từ chunks
    print(f"Đọc nội dung từ {args.chunks}...")
    chunk_texts = {}
    for p in Path(args.chunks).glob("part-*.parquet"):
        tbl_chunk = pq.read_table(p, columns=["chunk_id", "text"])
        for cid, txt in zip(tbl_chunk.column("chunk_id").to_pylist(), tbl_chunk.column("text").to_pylist()):
            chunk_texts[cid] = txt
            
    # Gắn text vào candidates
    out_data = []
    for qid, cands in candidates.items():
        q_info = dev_qids[qid].copy()
        for c in cands:
            c["text"] = chunk_texts.get(c["chunk_id"], "")
        q_info["candidates"] = cands
        out_data.append(q_info)
        
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
        
    print(f"Đã xuất {len(out_data)} câu với ứng viên ra {args.out}")

if __name__ == "__main__":
    main()
