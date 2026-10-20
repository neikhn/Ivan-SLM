# Dataset pilot — Lab 2.1 đến Lab 2.3

Cập nhật 18/10/2026. Chốt theo trao đổi: English và tiếng Việt song song; giữ training hai pha. Thu thập dữ liệu là nội dung Lab 2.1, không phải bước phải hoàn tất trước Lab 2.

## 1. Nguồn và mục tiêu

| Nhóm | Nguồn | Mẫu sạch cho pilot |
|---|---|---:|
| Tầng 1 EN | Wikipedia `20231101.en`, 1–3 đoạn mở đầu giải thích | 1.000 |
| Tầng 1 VI | Wikipedia `20231101.vi`, 1–3 đoạn mở đầu giải thích | 1.000 |
| Tầng 2 EN | SmolLM-Corpus `cosmopedia-v2`, lọc CS/Python/AI | 500 |
| Tầng 2 VI | Wikipedia `20231101.vi`, lọc CS/Python/AI | 500 |
| Tầng 3 EN/VI | Tự biên soạn 30 cặp giải thích về Transformer và dự án | 30 mỗi tiếng |

Nguồn chính thức: [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia), [SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus), [streaming](https://huggingface.co/docs/datasets/stream).

Cấu hình và revision đã quan sát nằm ở `configs/data_sources.json`. Tầng 2 VI hiện chốt Wikipedia, chưa bổ sung website ngoài hoặc dịch hàng loạt. Wikipedia VI dùng cho hai tầng nhưng chỉ cần đọc một lần và phân tuyến nội dung; một đoạn không được đưa vào cả hai tầng.

Tầng 3 là văn bản giải thích, không phải QA SFT. Biên soạn 150–300 từ/bản, giữ quan hệ bản dịch và duyệt trước khi đưa vào corpus. Các số liệu training chưa đo không xuất hiện trong tài liệu học. 30 cặp là mục tiêu pilot, chưa phải 1.500–3.000 mục đầy đủ.

## 2. Đã kiểm tra nguồn đến đâu?

Ngày 18/10 đã đọc metadata và 3 bản ghi đầu của mỗi nguồn: **9 bản ghi thô**, lưu tại `data/source_preview/` (được Git bỏ qua). Chưa lọc chủ đề, chưa kiểm tra chất lượng thủ công, chưa đạt pilot và chưa có tokenizer.

| Nguồn | Schema đã quan sát | Lưu ý |
|---|---|---|
| Wikipedia EN/VI | `id`, `url`, `title`, `text` | Giữ ID bài gốc khi cắt thành đoạn |
| Cosmopedia | `prompt`, `text`, `token_length`, `audience`, `format`, `seed_data` | Chỉ học từ `text`; không dùng `prompt` làm văn bản học |

Cosmopedia không cung cấp trường ID bài trong schema này. Manifest phải định danh theo revision/subset/vị trí gốc **trước khi shuffle**, hoặc vị trí shard ổn định nếu truy cập được; thêm hash nội dung. `seed_data` là nhãn loại nguồn, không phải ID duy nhất. `token_length` dùng tokenizer của nguồn, không phải số token của BPE 8.192 mình sẽ train.

Viewer API không đảm bảo trả đúng revision đã ghim. Vì vậy các preview được đánh dấu `preview_revision_verified=false`; chỉ dùng để xem cấu trúc. Pipeline thật phải gọi `datasets` với `revision` trong config. Metadata giấy phép hiện quan sát: Wikipedia `cc-by-sa-3.0`/`gfdl`; SmolLM-Corpus `odc-by`; giữ nguồn và attribution, không gộp chúng thành giấy phép MIT của code.

Xem mẫu đã lưu, không cần tải lại:

```powershell
python scripts/preview_sources.py --source wiki_vi --limit 1
python scripts/preview_sources.py --source cosmopedia_en --limit 1
```

Lấy lại tối đa vài mẫu từ Viewer khi cần:

```powershell
python scripts/preview_sources.py --source all --fetch --limit 3
```

Môi trường kiểm tra hiện có Python Anaconda 3.13 và chưa có `datasets` trong interpreter đó. Preview dùng thư viện chuẩn, không cài thêm package. Môi trường training mục tiêu vẫn là Python 3.11; chưa xác nhận môi trường training/CUDA qua bước chuẩn bị này.

## 3. Thứ tự thực hành

1. Đọc các preview và nhận diện trường văn bản, tiêu đề, ID và metadata.
2. Tự viết `normalize_text` trong `labs/lab_2_1.py`: NFC, CRLF → LF, dọn khoảng trắng ngang, giữ ranh giới đoạn; để bộ lọc loại U+FFFD.
3. Chạy `python labs/lab_2_1.py --check-normalization`. Mã thoát 2 hiện tại nghĩa là bài còn TODO, không phải đã đạt kiểm tra.
4. Tự viết generator đọc nguồn tại revision đã ghim; không chuyển cả dataset thành list. Cần định danh gốc trước shuffle, seed 42 và buffer có giới hạn.
5. Tự viết trích đoạn/lọc. Tầng 1 bỏ danh sách, tiểu sử và bảng/code-only. Tầng 2 dùng bộ chủ đề cụ thể: Python programming/lập trình Python, neural network/mạng nơ-ron, machine learning/học máy, deep learning/học sâu, tokenization, gradient/đạo hàm, attention/cơ chế chú ý. Từ chung như `model`/`mô hình` không đủ.
6. Lưu shard JSONL với ID, nguồn/revision, source ID/URL, giấy phép, ngôn ngữ, tầng, text, group ID và liên kết bản dịch nếu có. Có bộ đếm đọc/giữ/loại và lý do; chạy lại phải tiếp tục được mà không nhân bản dữ liệu.
7. Lấy 100 mẫu gần đều sáu nhóm để duyệt. Nếu có từ 10 lỗi nghiêm trọng, sửa bộ lọc và kiểm tra bằng mẫu mới. Không tự đánh dấu review đã đạt.
8. Lab 2.2: khử trùng toàn corpus, gom nhóm rồi chia 90/5/5. Lab 2.3: train tokenizer chỉ trên train và thống kê token thật.

Giới hạn pilot: 100.000 bản ghi được quét cho mỗi nguồn, tối đa 2 GB. Dữ liệu/cache tối đa 10 GB, cả dự án 30 GB. Thiếu mẫu thì báo số lượng và tỷ lệ lọc; chưa tự đổi nguồn hay tải toàn bộ kho.

## 4. Chống rò rỉ và tokenizer

- Hash văn bản NFC để khử trùng; MinHash character 5-gram tạo ứng viên, xác nhận Jaccard từ 0,9 rồi gom nhóm gần trùng.
- Cùng bài gốc, bản dịch đã xác định và bản gần trùng phải cùng nhóm trước khi chia tập. MinHash ký tự không tự nhận ra quan hệ dịch Anh–Việt; dùng metadata/liên kết nguồn hoặc rà soát quan hệ đã biết. Báo cáo giới hạn này thay vì khẳng định đã khử trùng ngữ nghĩa hoàn toàn.
- Không đưa QA test vào pretraining hoặc tokenizer. Chỉ cắt block sau khi đã chia tập.
- Tokenizer lấy mẫu tầng 40/40/20, cân bằng hai tiếng; BPE 8.192 gồm special tokens theo config và `AGENT.md`: `<pad>`, `<bos>`, `<eos>`, `<|en|>`, `<|vi|>`, `<|user|>`, `<|assistant|>`.
- Tỷ lệ lấy mẫu tokenizer không bảo đảm mọi thuật ngữ là một token. Kiểm tra round-trip, byte coverage và mức phân mảnh của từ kỹ thuật; không sửa tokenizer sau khi đã train Base.

## 5. Ngân sách training và điều kiện mở rộng

| Pha | Tokens | Tầng 1 | Tầng 2 | Tầng 3 |
|---|---:|---:|---:|---:|
| 1 | 80M | 55% | 35% | 10% |
| 2 | 20M | 20% | 40% | 40% |
| Tổng | 100M | **48M** | **36M** | **16M** |

Mỗi tầng có 50% token Anh và 50% Việt: ngân sách toàn lượt theo sáu nhóm lần lượt là 24M/24M, 18M/18M, 8M/8M. Đây là **token model đọc khi training**, có thể đọc lặp; không mặc định là token độc nhất cần thu thập.

Không trộn cứng corpus theo pha; giữ riêng sáu nhóm. Sau khi có tokenizer mới thống kê số token độc nhất, tỷ lệ giữ lại và mức lặp dự kiến. Nếu thiếu đáng kể dữ liệu Việt tầng 2 hoặc tầng 3 thì báo phần thiếu trước khi đóng băng; không sao chép file để che thiếu dung lượng.

Pilot chỉ đạt khi đủ mẫu sạch ở sáu nhóm, truy được nguồn, review đạt, khử trùng/chia nhóm được kiểm tra và có báo cáo. Dataset training chỉ đóng băng sau khi có tokenizer cùng thống kê token, manifest và mức lặp. Hiện mới hoàn thành bước cấu hình và xem trước nguồn; phần bài tập và corpus pilot còn chờ thực hành.
