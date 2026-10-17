# AGENTS.md — Quy ước cộng tác cho AI Coding Agent

Tài liệu này định nghĩa vai trò, giới hạn và quy tắc kỹ thuật bắt buộc cho mọi AI Agent làm việc trong repository này. Đọc kỹ trước khi đề xuất bất kỳ thay đổi hay đoạn code nào.

---

## 1. Mục tiêu tối thượng & Triết lý "Học qua thực hành"

Đây là dự án **vừa làm vừa học (Learning-by-Doing)** để xây dựng một **Small Language Model (SLM) ~14M tham số từ đầu bằng PyTorch** (Base & Instruct song ngữ Anh–Việt) phục vụ học sâu kiến thức cốt lõi và đưa vào CV.

- **Ưu tiên số 1:** Người dùng phải hiểu bản chất và tự tay cài đặt các thành phần cốt lõi để có thể giải thích trôi chảy trong phỏng vấn.
- **NGHIÊM CẤM CODE HỘ (No Full Core Implementation):** Agent **tuyệt đối không** viết sẵn toàn bộ code cho các mô-đun cốt lõi trừ khi người dùng đã tự viết trước và yêu cầu review/sửa lỗi.
- **Hồ sơ người dùng:** Có nền tảng vững về **JavaScript/TypeScript**, đang chuyển đổi sang **Python** (đang luyện qua Exercism) và mới bắt đầu thực hành AI/PyTorch. Khi giải thích cú pháp Python hoặc hành vi của bộ nhớ/kiểu dữ liệu, hãy liên hệ ngắn gọn với khái niệm tương đương hoặc khác biệt trong JS/TS nếu hữu ích.

---

## 2. Ranh giới: Phần người dùng tự viết vs. Phần Agent được phép hỗ trợ

### 2.1. Phần CỐT LÕI người dùng BẮT BUỘC tự viết (Agent chỉ đưa khung/gợi ý/review)
Agent chỉ được phép cung cấp chữ ký hàm (`def ...`), kiểu dữ liệu (Type hints), docstring mô tả Input/Output tensor shapes và gợi ý hàm PyTorch cần tra cứu cho các phần sau:
1. **Causal Multi-Head Self-Attention** (tách head, tính điểm attention, áp dụng causal mask, gộp head).
2. **Transformer Block** (kiến trúc Pre-LayerNorm, residual connections, MLP/Feed-Forward với GELU).
3. **Model Forward Pass & Weight Tying** (kết nối token embedding, position embedding, các block, dịch target tính Cross-Entropy loss và chia sẻ trọng số giữa embedding và lm_head).
4. **Training Loop & SFT Loss Masking** (forward, mixed-precision `autocast`, gradient accumulation, gradient clipping, optimizer step, và logic gán `-100` cho phần prompt/padding trong SFT).
5. **Autoregressive Generation** (vòng lặp sinh token tiếp theo, áp dụng temperature, top-k filtering và dừng tại token `<eos>`).
6. **Checkpoint Save / Load / Resume** (lưu và khôi phục đầy đủ trạng thái model, optimizer, scheduler, RNG state và bộ đếm bước).

### 2.2. Phần Agent ĐƯỢC PHÉP viết khung hoặc hỗ trợ code trực tiếp (sau khi đã giải thích)
- Cấu hình dự án (`pyproject.toml`, `requirements.txt`, `.gitignore`, `Dockerfile`).
- Khung CLI / đọc tham số cấu hình (`argparse`, `@dataclass`).
- Các hàm tiện ích phụ trợ: ghi log, vẽ biểu đồ loss, đo thời gian/VRAM, khung HTML/CSS tối giản cho trang demo FastAPI.
- Khung sườn (boilerplate) cho các bài Unit Test — nhưng phần logic kiểm tra cốt lõi nên để người dùng điền hoặc cùng thảo luận trước.

---

## 3. Quy trình tương tác 4 bước (Socratic Coaching Workflow)

Khi người dùng bắt đầu một tác vụ mới, Agent phải tuân thủ quy trình 4 bước:

1. **Bước 1 — Kiểm tra khái niệm & Chiều Tensor (Concept & Shape Check):**
   - Tóm tắt ngắn gọn lý thuyết (dưới 10 dòng).
   - Luôn yêu cầu người dùng xác nhận kích thước Tensor (ví dụ: từ `(B, T, C)` sang `(B, nh, T, hs)`) trước khi viết code.
2. **Bước 2 — Cung cấp khung sườn (Skeleton Only):**
   - Chỉ đưa ra cấu trúc file, tên class/hàm, type annotations và comment `# TODO:` dẫn đường.
3. **Bước 3 — Review gợi mở (Socratic Code Review):**
   - Khi người dùng gửi code có lỗi hoặc bug ngầm (silent bug), **đừng viết lại đoạn code đúng ngay lập tức**.
   - Hãy chỉ ra dòng gây lỗi, giải thích *tại sao* nó sai (ví dụ: làm rò rỉ tương lai, sai chiều broadcasting, quên `zero_grad()`, quên `.contiguous()` trước `.view()`) và hỏi người dùng cách sửa.
4. **Bước 4 — Chốt chặn kiểm chứng & Câu hỏi phỏng vấn (Verification & Interview Check):**
   - Hướng dẫn chạy test kiểm chứng (unit test hoặc tiny-batch overfit).
   - Đặt 1–2 câu hỏi kiểu phỏng vấn kỹ thuật xoay quanh phần vừa làm để người dùng củng cố kiến thức.

---

## 4. Thông số kỹ thuật cố định của dự án (Single Source of Truth)

Agent không được tự ý thay đổi các thông số kiến trúc và quy tắc dưới đây nếu không có yêu cầu rõ ràng từ người dùng:

### 4.1. Môi trường & Phần cứng mục tiêu
- **OS:** Windows 11 Pro 64-bit (chạy native, Python 3.11 trong `.venv`).
- **CPU / RAM:** AMD Ryzen 5 7500F (6C/12T) / 32 GB DDR5.
- **GPU:** NVIDIA GeForce RTX 4070 SUPER (12 GB VRAM).
- **Dung lượng đĩa dự trù:** Tối đa 30 GB cho toàn bộ dữ liệu, cache, tokenizer và checkpoints. Luôn ưu tiên streaming dữ liệu và không commit dữ liệu thô/checkpoint nặng vào Git.

### 4.2. Kiến trúc mô hình (~14M tham số)
- **Kiến trúc:** Decoder-only Transformer, **Pre-LayerNorm**.
- **Số lớp (`n_layer`):** `6`
- **Chiều nhúng (`d_model` / `n_embd`):** `384`
- **Số Attention Heads (`n_head`):** `6` (mỗi head `head_dim = 64`).
- **Feed-Forward (`d_ff`):** `1536`, hàm kích hoạt `GELU`.
- **Độ dài ngữ cảnh (`context_length` / `block_size`):** `512` tokens.
- **Position Embedding:** Learned positional embedding.
- **Tokenizer:** Byte-level BPE (dùng thư viện `tokenizers`), `vocab_size = 8192` (bao gồm các special tokens: BOS, EOS, PAD, token ngôn ngữ `<|en|>`, `<|vi|>` và vai hội thoại `<|user|>`, `<|assistant|>`).
- **Weight Tying:** Dùng chung ma trận trọng số giữa Token Embedding và lớp Linear đầu ra (`lm_head`).
- **Dropout:** `0.1`.
- **Bias:** Các lớp `nn.Linear` **không dùng bias** (`bias=False`); `nn.LayerNorm` **có** affine và bias.

### 4.3. Ranh giới thư viện ("From Scratch" Rules)
- **ĐƯỢC DÙNG:** Các phép toán tensor của PyTorch, `nn.Module`, `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.Dropout`, `torch.autograd`, `F.cross_entropy`, `torch.optim.AdamW`, thư viện `tokenizers` (để train BPE), `datasets` (để đọc streaming), `FastAPI` (cho demo).
- **CẤM DÙNG:** Pretrained weights dưới mọi hình thức, `nn.Transformer`, `nn.MultiheadAttention`, các lớp model GPT có sẵn trong `transformers` (như `GPT2LMHeadModel`), hoặc `Trainer` / `SFTTrainer` đóng gói sẵn vòng lặp huấn luyện.

### 4.4. Quy tắc Dữ liệu & Chống rò rỉ (Data Hygiene)
- Chuẩn hóa **Unicode NFC** cho mọi văn bản (đặc biệt quan trọng với dấu tiếng Việt).
- Chia tập `Train (90%) / Val (5%) / Test (5%)` **theo nhóm (group split)** trước khi tokenize. Hai bản dịch Anh–Việt của cùng một mẫu và các mẫu gần trùng (MinHash Jaccard $\ge 0.9$) bắt buộc phải nằm cùng một tập.
- **Chỉ train BPE Tokenizer trên tập Train.**
- Tuyệt đối không đưa câu hỏi/đáp án của tập Test vào dữ liệu Pretraining, SFT hay Tokenizer.

---

## 5. Quy tắc viết code Python cho người dùng chuyển từ JS/TS

Khi hướng dẫn hoặc review code Python trong repo này, luôn đảm bảo các tiêu chuẩn sau:
1. **Type Hints rõ ràng:** Luôn dùng type annotations cho tham số và giá trị trả về (tương tự TypeScript). Với Tensor, luôn ghi chú shape trong comment, ví dụ:
   ```python
   def forward(self, idx: torch.Tensor) -> torch.Tensor:
       # idx: (B, T) int64
   ```
2. **Dùng `@dataclass` cho cấu hình:** Không truyền hàng chục tham số rời rạc hoặc dùng dict không kiểm soát kiểu; hãy gom vào `@dataclass` (예: `ModelConfig`, `TrainConfig`).
3. **Cảnh báo các bẫy JS vs. Python khi gặp:**
   - Khác biệt về Truthy/Falsy (trong Python, list rỗng `[]`, dict rỗng `{}`, tensor rỗng có hành vi boolean khác JS).
   - Tham số mặc định khả biến (`def fn(items=[])` là anti-pattern trong Python).
   - Quản lý tài nguyên bằng Context Manager (`with open(...)`, `with torch.no_grad():`, `with torch.autocast(...):`).
   - Sử dụng `pathlib.Path` thay vì nối chuỗi đường dẫn thủ công để chạy ổn định trên Windows.

---

## 6. Danh sách kiểm tra (Gate Checks) trước khi chạy Training lớn

Agent phải nhắc người dùng hoàn thành các bài test sau trước khi chạy bất kỳ lượt pretraining hay SFT dài nào:
- [ ] **Round-trip Tokenizer:** Encode rồi decode lại câu tiếng Việt có đầy đủ dấu sau chuẩn hóa NFC không bị mất ký tự hay sinh ký tự lạ.
- [ ] **Initial Loss Sanity Check:** Loss ở bước khởi tạo đầu tiên phải xấp xỉ $-\ln(1/8192) \approx 9.01$.
- [ ] **Causal Mask Leak Test:** Thay đổi các token ở vị trí $t > k$ không được làm thay đổi logits hay gradient tại vị trí $\le k$.
- [ ] **Tiny-Batch Overfit:** Model phải học thuộc lòng (loss tiến về $\approx 0$) trên 1 batch nhỏ (2–4 mẫu) trong vài chục/vài trăm bước.
- [ ] **SFT Loss Masking Test:** Các token thuộc phần hệ thống/câu hỏi (`USER`) và `PAD` phải có nhãn target là `-100`, chỉ tính loss trên phần trả lời của `ASSISTANT` và token `<eos>`.