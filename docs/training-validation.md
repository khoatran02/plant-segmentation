# Kiểm tra pipeline Transformers SAM3 — 2026-10-31

## Dọn source và format — 2026-11-01

Đã bỏ checkout/submodule SAM3 upstream không dùng, visualizer COCO độc lập, hướng dẫn Hydra cũ, collector báo cáo không còn tương thích và các file build/cache thừa. Bản thảo, bằng chứng lịch sử, dataset và output được giữ. Ruff format và import sorting áp dụng cho toàn bộ Python source/tests, có cấu hình duy trì trong `pyproject.toml`. Sau khi xóa upstream: **45 tests passed**, lint/format check, train/evaluate `--help`, cú pháp shell và `git diff --check` đều thành công.

## Cập nhật độ phân giải — 2026-11-01

Launcher Kaggle chuyển mặc định sang 672 × 672, validation mỗi 3 epoch, batch prompt đánh giá 2. CLI thêm `--image_size`; thay đổi đồng bộ processor, global RoPE grids và FPN metadata trước khi khởi tạo model, giữ nguyên learned positional weights của pretrained. `--init_from` nạp adaptation weights để bắt đầu run ở độ phân giải mới; `--resume`/evaluate từ chối kích thước khác checkpoint.

Kiểm thử cập nhật: **45 passed**, lint và kiểm tra cú pháp shell thành công. Đã kiểm tra safetensors 32 → 16/48, backward cả scope heads/full với checkpointing, processor save/load, inference mask ở kích thước ảnh gốc và checkpoint cũ chưa ghi độ phân giải. Test đầu vào **672 × 672** dùng Sam3Model thật với patch **14**, pretrain image **336**, window **24**, cả window/global attention; hidden size/query/layer được thu nhỏ để chạy CPU. Loss hữu hạn, gradient qua global attention hữu hạn, validation all-classes chạy trên ảnh gốc. DDP Gloo kiểm tra độ phân giải tùy chỉnh 16 thay cho kích thước gốc 32 của fixture.

Chưa benchmark pretrained SAM3 đầy đủ trên CUDA/T4, chưa đo IoU/Dice sau fine-tuning 672; mức giảm 55,6% là số pixel đầu vào, không phải số đo giảm thời gian hoặc VRAM. Các kết quả kiểm thử lịch sử bên dưới giữ nguyên theo thời điểm ghi nhận.

Môi trường local: macOS ARM64, Python 3.12.14, torch 2.8.0, torchvision 0.23.0, Transformers 5.22.1, NumPy 1.26.4. Không có CUDA.

## Kiểm tra đã hoàn tất

- `python -m pytest -q`: **29 passed**.
- `ruff check src tests train.py evaluate.py`: passed.
- `git diff --check`, Python compile và kiểm tra cú pháp shell scripts: passed.
- Các entry point train/evaluate hỗ trợ `--model_id`, `--model_revision`, `--cache_dir`, `--local_files_only`; không còn tham số pretrained `--checkpoint`.

Các kiểm thử model dùng **Sam3Model thật của Transformers với cấu hình thu nhỏ**: ảnh 32 × 32, hidden size 32, một layer/block, 4 queries, khoảng 219 nghìn parameters. Model được khởi tạo ngẫu nhiên rồi lưu bằng `save_pretrained(..., safe_serialization=True)` và load lại bằng `from_pretrained(..., use_safetensors=True, local_files_only=True)`. Tokenizer/processor cũng được save/load qua thư viện. Không tải pretrained weights lớn hoặc truy cập repo gated trong các kiểm thử này; đây không phải đánh giá chất lượng mô hình.

Đã kiểm tra:

- Load safetensors, processor, forward, loss và backward trong khi chặn import package `sam3` gốc.
- Scope `heads`: ViT/CLIP frozen, bỏ CLIP encoder sau khi cache hidden states; text projection và mask decoder nhận gradient.
- Text cache cho cùng output mask với text encoder trực tiếp; cache không vô tình đóng băng text projection.
- Scope `full`: backward qua CLIP và ViT với activation checkpointing.
- Cả prompt positive/negative, target rỗng, matching bất biến với thứ tự target.
- Inference nhiều class dùng cùng vision features và nội suy mask về kích thước ảnh gốc.
- Save/resume adaptation format 3 tái tạo cùng output; optimizer/RNG và identity guard.
- DataLoader spawn worker với processor/tokenizer thực tế.
- DDP Gloo hai process với Sam3Model thu nhỏ, activation checkpointing và optimizer update; weights sau update khớp giữa các rank.
- DDP gradient accumulation đối chiếu batch tham chiếu, xử lý nhóm cuối epoch và evaluation 3 ảnh/2 process không đếm trùng.
- Confusion/macro metrics, lớp vắng, mask trống, nội suy logits, PNG→RLE, gộp >200 components, cache invalidation và EXIF.

## Dữ liệu thật đã kiểm tra

`train.py --prepare_only` trên `datasets/plantseg`, nguồn PNG và `label_policy=mask`: kiểm tra đủ 7.774 cặp RGB/PNG và tạo RLE cache thành công. Bước chuẩn bị dữ liệu/metric giữ nguyên qua lần chuyển backend này.

| Split | Ảnh | Pseudo-instance liên thông 8 hướng | Bất đồng nhãn PNG/metadata |
|---|---:|---:|---:|
| train | 5.367 | 38.889 | 8 |
| val | 846 | 6.516 | 0 |
| test | 1.561 | 11.204 | 4 |

Không có ảnh cần gộp quá 200 components trong bản dataset này. Không sửa ảnh, PNG hoặc metadata gốc. Mọi ảnh test vẫn được giữ, bao gồm ảnh không có polygon COCO hợp lệ.

Self-check evaluator đã chạy trên toàn bộ 1.561 PNG test bằng cách dùng chính PNG ground truth như prediction: mIoU = mAcc = foreground IoU = image-mean Dice = **1.0**. Đây là kiểm tra evaluator, không phải model score. JSON/CSV/confusion matrix ở `outputs/validation-ground-truth/` (git-ignored).

## Giới hạn xác minh

Chưa chạy pretrained facebook/sam3 kích thước đầy đủ, chưa kiểm tra tải gated model bằng tài khoản người dùng, NCCL, FP16 trên T4, peak VRAM hoặc throughput. Chưa có model test score. Chạy smoke test Kaggle 2 × T4 trong README trước run dài; kiểm thử CPU/Gloo với model nhỏ không xác nhận rằng model đầy đủ vừa 16 GB VRAM.

Backend mới dùng objective `transformers-hungarian-v1` (final-layer instance/semantic supervision), khác recipe auxiliary/one-to-many upstream cũ. Checkpoint format 2 cũ không dùng để resume backend Transformers; format mới là 3. Thay đổi này được ghi trong README và model identity.
