# 5 ảnh mẫu cho demo SAM 3 text-to-segment

Mỗi ảnh JPG có một file `.txt` cùng tên chứa prompt tiếng Anh để copy/paste.
`prompts.json` lưu các cặp ảnh/prompt và tên ảnh nguồn. Ảnh được sao chép nguyên
vẹn từ **PlantSeg test split** đang có trong project, không phải ảnh do model sinh.
Các tên prompt giữ đúng tên bệnh trong metadata đã dùng khi fine-tune.

| Ảnh | Text prompt | Vùng cần phân đoạn |
| --- | --- | --- |
| `01_apple_rust.jpg` | `apple rust` | Vùng bệnh gỉ sắt trên lá táo |
| `02_bean_mosaic_virus.jpg` | `bean mosaic virus` | Vùng biểu hiện bệnh khảm trên lá đậu |
| `03_corn_smut.jpg` | `corn smut` | Vùng bệnh than đen trên ngô |
| `04_blueberry_rust.jpg` | `blueberry rust` | Vùng bệnh gỉ sắt trên lá việt quất |
| `05_squash_powdery_mildew.jpg` | `squash powdery mildew` | Vùng bệnh phấn trắng trên lá bí |

## Chạy nhanh

1. Từ thư mục gốc project, chạy `.venv/bin/python demo.py`.
2. Mở http://127.0.0.1:8000 và chọn một thẻ ảnh mẫu: ảnh và prompt sẽ được điền sẵn.
3. Bấm **Tạo mask**. Có thể sửa trực tiếp prompt rồi chạy lại trên cùng ảnh.

Hoặc tải ảnh JPG từ folder này lên demo rồi copy nội dung file `.txt` tương ứng.
Trong demo có nút tải ZIP của cả bộ ảnh và prompt.

## Ý nghĩa prompt

SAM 3 nhận **ảnh + text prompt** và tạo mask theo nội dung văn bản. Text là đầu
vào do người dùng cung cấp; demo không tự suy ra tên bệnh từ ảnh. Có thể nhập
prompt mới ngoài danh sách. Checkpoint trong project đã fine-tune cho phân đoạn
vùng bệnh cây, nên khả năng với mô tả khác miền cần được đánh giá riêng. Ngay cả
với đúng prompt, kết quả vẫn là dự đoán, không phải ground truth.

Năm ảnh này để thao tác demo; không dùng chúng thay cho đánh giá toàn test split.
Nguồn ảnh và quyền sử dụng theo bộ dữ liệu PlantSeg gốc.
