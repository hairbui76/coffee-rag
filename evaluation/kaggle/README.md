# Chạy thí nghiệm v6 trên Kaggle

Mục tiêu: để Kaggle chạy `evaluation/run_v6.sh` (4 cấu hình × N run + soft recall) bằng GPU miễn phí và Anthropic API, máy bạn rảnh.

## Vì sao cần hai thứ tách nhau

| Thứ | Lên Kaggle bằng | Lý do |
|---|---|---|
| Code | `git clone` từ GitHub trong notebook | Repo nhẹ, luôn mới nhất |
| Dữ liệu (~240 MB: parquet, index FAISS bge-m3, dataset v6) | Kaggle Dataset upload một lần | Repo dùng git LFS, Kaggle clone LFS chậm và hay lỗi |

## Chuẩn bị (làm một lần, trên máy bạn)

1. **Commit và push code mới lên GitHub** (các file mới chưa được commit):
   ```bash
   git add -A
   git commit -m "v6 benchmark, Anthropic provider, annotation tooling, Kaggle notebook"
   git push origin main
   ```
2. **Đóng gói dữ liệu:**
   ```bash
   python -m evaluation.kaggle.pack_data --zip
   ```
   Tạo `kaggle_upload/` và `kaggle_upload.zip` (~240 MB). Script sẽ báo lỗi nếu file nào đang là git-lfs pointer, khi đó chạy `git lfs pull` trước.
3. **Tạo Kaggle Dataset:** https://www.kaggle.com/datasets → **New Dataset** → kéo `kaggle_upload.zip` vào → Title: `coffee-rag-data` → Create. Kaggle tự giải nén, kết quả mount tại `/kaggle/input/<tên-dataset>/` (ví dụ `coffee_rag_upload`, xem panel Input) với đúng cấu trúc `data/processed/...`, `data/embeddings/...`, `ragas_eval_dataset_v6.json`. Để Private.
4. **Xác minh số điện thoại** trong Kaggle Settings (bắt buộc để bật Internet và GPU).

## Tạo notebook

1. https://www.kaggle.com/code → **New Notebook** → File → **Import Notebook** → chọn `evaluation/kaggle/coffee_rag_kaggle.ipynb`.
2. Panel bên phải:
   - **Accelerator:** GPU T4 x2 (hoặc GPU P100).
   - **Internet:** On.
   - **Add Input** → tab Datasets → tìm `coffee-rag-data` của bạn → Add. Copy đường dẫn panel hiện ra vào `DATA_INPUT` (notebook cũng tự dò theo file `ragas_eval_dataset_v6.json` nếu khai sai).
3. **Add-ons → Secrets** → Add secret:
   - Label `ANTHROPIC_API_KEY`, Value: key của bạn. Bật toggle "attach to notebook".
   - Nếu repo GitHub private: thêm `GITHUB_TOKEN` (Personal Access Token, quyền `repo` read).
4. Ô cấu hình đầu notebook: chỉnh `ONLY`, `RUNS`, `WORKERS`, `JUDGE_MODEL` nếu cần. Mặc định: `ONLY="all"`, `RUNS="1 2 3"`, judge `claude-sonnet-5-5`, pipeline `claude-haiku-5-5`.

## Chạy

**Lần đầu, chạy interactive để bắt lỗi:** Run All. Ô smoke test chạy 3 case và in bảng điểm. Nếu có lỗi (sai key, rate limit, thiếu package) sẽ dừng ở đây, sửa rồi chạy lại ô đó. Khi smoke qua, dừng kernel (không cần chờ ô chạy cả bộ).

**Chạy thật:** **Save Version** (góc trên phải) → **Save & Run All (Commit)** → Save. Kaggle chạy nền trên máy chủ, tối đa 12 giờ, không cần mở trình duyệt. Theo dõi tiến độ ở trang notebook → tab **Logs** của version đang chạy.

Thời gian ước tính với `WORKERS=8` (Build tier, 5M token/phút): 1 run full ≈ 25 đến 40 phút, 12 run ≈ 6 đến 8 giờ. Nếu log có nhiều `429`, đặt `WORKERS="4"`, soft recall trên GPU vài phút mỗi run. `RUNS="1 2 3"` với `ONLY="all"` vừa trong 12 giờ. Nếu dùng `RUNS="1 2 3 4 5"`, tách làm hai version: một với `ONLY="bge"`, một với `ONLY="minilm"`.

## Khi phiên kết thúc hoặc bị cắt

Output nằm ở tab **Output** của version: thư mục `results_v6/` (CSV từng run, `*_contexts.jsonl`, `tables/`, `annotation_sample.json`), `coffee_rag_results_v6.zip`, và `embeddings_minilm/`.

- **Tải về:** Output → Download All, giải nén vào `evaluation/results/v6/` trên máy bạn.
- **Nối tiếp (resume) nếu chưa xong:** tạo version mới của notebook, **Add Input → Your Work → chọn notebook này (output version trước)**. Ô khôi phục sẽ copy mọi CSV vào chỗ cũ, `ragas_eval.py --limit 0` tự bỏ qua case đã chấm, và index MiniLM không phải build lại. Save & Run All lần nữa.

## Sau khi có kết quả

Trên máy bạn, với `evaluation/results/v6/` đã tải về:

```bash
# bảng hệ thống + bảng metric (notebook đã chạy sẵn, chạy lại nếu gộp thêm run)
bash evaluation/run_v6.sh    # chỉ phần aggregate, hoặc gọi thẳng:
python -m evaluation.aggregate_runs --run "BGE-M3" evaluation/results/v6/bge_norrf_r*.csv \
    --run "BGE-M3+RRF" evaluation/results/v6/bge_rrf_r*.csv \
    --run "MiniLM" evaluation/results/v6/minilm_norrf_r*.csv --run "MiniLM+RRF" evaluation/results/v6/minilm_rrf_r*.csv \
    --soft-file evaluation/results/v6/bge_rrf_soft_r*.csv --soft-label "BGE-M3+RRF" \
    --baseline "BGE-M3" --metric-table "BGE-M3+RRF" --out evaluation/results/v6/tables

# gán nhãn: mở evaluation/annotation/annotate.html, nạp results_v6/annotation_sample.json
# rồi:
python -m evaluation.annotation.analyze_annotations --labels "evaluation/annotation/labels_*.json" \
    --results evaluation/results/v6/bge_rrf_r1.csv --soft-results evaluation/results/v6/bge_rrf_soft_r1.csv \
    --out evaluation/annotation/report
python -m evaluation.sweep_soft_recall --contexts evaluation/results/v6/bge_rrf_r1_contexts.jsonl \
    --results evaluation/results/v6/bge_rrf_r1.csv --annotations "evaluation/annotation/labels_*.json" \
    --out evaluation/results/v6/sweep
```

Sweep cần embed ~8.000 đoạn bằng bge-m3; trên CPU mất 30 đến 60 phút, hoặc chạy ô đó trên Kaggle bằng cách thêm một cell gọi lệnh trên sau khi có file nhãn (upload `labels_*.json` như một Dataset nhỏ).

## Lỗi thường gặp

| Triệu chứng | Nguyên nhân / cách sửa |
|---|---|
| `Không thấy dataset ở /kaggle/input/coffee-rag-data` | Chưa Add Input, hoặc tên dataset khác → sửa `DATA_INPUT` cho khớp đường dẫn hiện trong panel Input |
| `Thêm ANTHROPIC_API_KEY trong Add-ons → Secrets` | Secret chưa tạo hoặc chưa bật "attach" cho notebook này |
| Nhiều dòng `error` chứa `429` / `rate_limit` | Giảm `WORKERS` xuống 4 hoặc 2; script resumable, chạy lại là nối tiếp |
| `git clone` fail 403 | Repo private, cần `GITHUB_TOKEN` secret |
| Kết thúc đúng 12 giờ, chưa xong | Bình thường. Resume theo mục trên |
| `CUDA out of memory` | Hiếm; đặt `WORKERS=4`, bge-m3 encode theo lô nhỏ hơn |
