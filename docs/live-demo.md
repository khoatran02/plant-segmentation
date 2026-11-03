# Live demo SAM 3: image + text → mask

Giao diện HTML/CSS/JavaScript tiếng Việt. Người dùng chọn ảnh, nhập một text
prompt, rồi bấm **Tạo mask**. Prompt được mã hóa trực tiếp qua text encoder của
SAM 3; demo không tra bảng nhãn bệnh và không tự chọn prompt từ nội dung ảnh.

## Chạy

Từ thư mục gốc project với môi trường đã cài `requirements.txt`:

```bash
.venv/bin/python demo.py
```

Mở **http://127.0.0.1:8000**, đợi trạng thái **Sẵn sàng**. Chọn một trong năm thẻ
mẫu để điền sẵn ảnh và prompt, hoặc tải ảnh riêng. Nhập/sửa prompt và bấm **Tạo
mask** (phím tắt Ctrl/⌘ + Enter). Dừng server bằng `Ctrl+C`.

```bash
.venv/bin/python demo.py \
  --checkpoint outputs/best.pt \
  --dataset_dir datasets/plantseg \
  --samples_dir demo-samples \
  --cache_dir outputs/hf-cache \
  --device auto \
  --port 8000
```

Server dùng thư viện chuẩn, không cần Node, Flask hay FastAPI. Mặc định đọc base
model từ cache; thêm `--allow_download` nếu cần tải (cần quyền truy cập
`facebook/sam3` và xác thực Hugging Face như README). Thiết bị tự chọn CUDA → MPS
→ CPU. Checkpoint adaptation không chứa toàn bộ base weights.

## Folder ảnh và prompt

`demo-samples/` chứa đúng **5 ảnh JPG** từ PlantSeg test split, mỗi ảnh có file
`.txt` cùng tên chứa prompt tiếng Anh. `prompts.json` lưu ánh xạ ảnh/prompt, chú
thích và tên nguồn. `README.md` trong folder giải thích cách dùng. Bản ZIP sẵn có
ở `demo-samples.zip`; trên giao diện cũng có nút **Tải 5 ảnh + prompt (.zip)**.

| Ảnh | Prompt |
| --- | --- |
| `01_apple_rust.jpg` | `apple rust` |
| `02_bean_mosaic_virus.jpg` | `bean mosaic virus` |
| `03_corn_smut.jpg` | `corn smut` |
| `04_blueberry_rust.jpg` | `blueberry rust` |
| `05_squash_powdery_mildew.jpg` | `squash powdery mildew` |

Server đọc chính folder này, nên ảnh hiển thị và bộ tải về luôn cùng nội dung.
Các ảnh được copy nguyên bản, không resize và không phải ground truth/prediction.
Đây là bộ để thao tác demo, không thay thế benchmark toàn test split.

## Cách hiểu text-to-segment

SAM 3 nhận khái niệm mô tả bằng văn bản và ảnh, rồi dự đoán vùng khớp khái niệm
([tài liệu SAM 3 của Transformers](https://huggingface.co/docs/transformers/model_doc/sam3)).
Trong demo này, text là **đầu vào**, mask là **đầu ra**. Người dùng có thể giữ
nguyên ảnh, thay prompt, rồi chạy lại để quan sát kết quả.

Demo chấp nhận prompt tự do (không ép chọn một trong 115 tên bệnh). Nên dùng cụm
từ ngắn bằng tiếng Anh; các prompt mẫu giữ đúng tên bệnh trong metadata lúc train.
Tối đa 200 ký tự, đồng thời kiểm tra giới hạn token thật của text encoder; prompt
quá dài sẽ được báo lỗi thay vì âm thầm cắt bớt. Demo không dịch hay sửa nghĩa
prompt; chỉ gộp khoảng trắng.

Checkpoint mặc định đã fine-tune cho **known-disease semantic segmentation**.
Demo giữ protocol của checkpoint: nội suy semantic logits về kích thước gốc và
threshold, không dùng presence/confidence để kết luận bệnh có xuất hiện hay không.
Kết quả là **semantic union mask theo prompt**, không phải danh sách instance hay
kết quả chẩn đoán tên bệnh. Prompt khác miền hoặc khái niệm không xuất hiện trong
ảnh vẫn được xử lý, nhưng khả năng từ chối/khái quát hóa chưa được đánh giá. Không
được diễn giải một mask không rỗng thành bằng chứng bệnh chắc chắn hiện diện.

Để hỗ trợ prompt mới, demo giữ CLIP text encoder sau khi nạp checkpoint, thay vì
cache 115 tên bệnh rồi bỏ encoder như pipeline đánh giá. Điều này tăng bộ nhớ so
với evaluation dùng cache. Text tensors của mỗi request được giải phóng sau khi
chạy; demo không tích lũy cache cho số lượng prompt không giới hạn.

## Giao diện và đầu ra

- JPG, PNG, WebP, tối đa 12 MB / 20 megapixel; có kéo thả và xử lý hướng EXIF.
- Chọn mẫu tự điền prompt, vẫn cho phép sửa trước khi chạy.
- Tùy chỉnh ngưỡng mask trong mục mở rộng, mặc định theo checkpoint.
- Ba khung hiển thị đồng thời: **Ảnh gốc**, **Predict** (overlay) và **Mask**.
  Không dùng thanh kéo; màn hình nhỏ xếp ba khung theo chiều dọc.
- Ô **Ảnh gốc** giữ nguyên nguồn ảnh đầu vào cả trước và sau khi chạy model, không phủ mask.
- Ô **Predict** phủ màu xanh dương lên vùng dự đoán; ô **Mask** hiển thị mask dự đoán.
  Giao diện tạo overlay từ ảnh gốc và mask nhị phân để màu mới áp dụng sau khi
  tải lại trang; ảnh Predict tải xuống dùng cùng màu với preview.
  Mỗi ô có nút tải PNG tương ứng.
- Đầu trang chỉ hiển thị tiêu đề bài: **Text-Prompted Plant Disease Lesion
  Segmentation with SAM 3 on PlantSeg**, không có caption hay mô tả phụ.
- Sau khi chạy, phần kết quả chỉ giữ ba ảnh và nút tải tương ứng; không hiện
  bảng số liệu hoặc dòng caption hoàn tất.
- Tải mask PNG (0: nền; 255: vùng khớp prompt) và overlay PNG ở kích thước gốc.
- Thay ảnh/prompt/ngưỡng sẽ xóa kết quả cũ; phải chạy lại để tạo kết quả mới.

Tỷ lệ mask tính trên toàn ảnh, không phải riêng diện tích lá. Thời gian suy luận
bao gồm encode ảnh và text, không gồm nạp model, truyền ảnh hay tạo PNG. Không
hiển thị IoU/Dice với ảnh upload vì không có ground truth. Ảnh/kết quả giữ trong
bộ nhớ khi xử lý, không lưu upload xuống đĩa. Mỗi thời điểm chỉ chạy một inference.
Giao diện có font hệ thống dự phòng khi Google Fonts không tải được.

Checkpoint cũ ghi revision `main` có chính sách tương thích như `inference.py`;
không thể xác nhận immutable commit đã dùng khi train. Dùng đúng base cache của
run gốc để tái lập. Các identity còn lại và tập tham số adaptation vẫn được kiểm tra.

## API và kiểm tra

`GET /api/status`, `GET /api/samples`, `GET /api/sample/{id}`,
`GET /api/samples.zip`, `POST /api/predict`.

Predict nhận JSON chứa `prompt`, `threshold` và một trong `sample_id` hoặc
`image` (base64 thuần). Ví dụ:

```json
{"sample_id": 0, "prompt": "apple rust", "threshold": 0.5}
```

Trả về `prompt` đã dùng, data URL PNG cho `original`, `overlay`, `mask`, kích thước,
số pixel, coverage và thời gian. Không nhận `category_id` thay cho prompt và không
gán mask cho một disease ID. Backend tái sử dụng `Sam3SemanticPredictor` mà không
thay đổi pipeline train/evaluate.

```bash
.venv/bin/python -m pytest tests/test_demo.py -q
node --check demo/app.js
```

Test gồm upload/EXIF, prompt rỗng/quá dài, giới hạn token, nhãn mask nhị phân,
khóa inference, giải phóng text cache, file ZIP và xác nhận hai prompt khác nhau
được encode thành input IDs khác nhau qua text encoder của SAM 3 nhỏ thật.

