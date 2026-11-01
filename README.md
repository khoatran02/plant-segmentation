# PlantSeg: fine-tune và đánh giá SAM3

Pipeline dùng **Hugging Face Transformers `Sam3Model` và `Sam3Processor`**, tải pretrained weights dạng safetensors bằng `from_pretrained`, hỗ trợ một GPU hoặc DDP qua `torchrun`. Dataset mặc định là `dataset/plantseg`; tự nhận `datasets/plantseg` nếu thư mục thứ nhất không tồn tại. Có thể truyền đường dẫn bất kỳ bằng `--dataset_dir`.

## Cài đặt trên Kaggle 2 × T4

Toàn bộ dependencies cho train, evaluate, test và lint được gộp trong một file `requirements.txt`.

Chọn accelerator **GPU T4 x2**, bật Internet để tải dependencies/model lần đầu. Dùng Python 3.12 hoặc 3.13, NumPy 2.1 trở lên trong nhánh 2.x. `requirements.txt` pin bộ tương thích **torch 2.8.0 / torchvision 0.23.0 / torchaudio 2.8.0** theo [hướng dẫn PyTorch](https://pytorch.org/get-started/previous-versions/) và Transformers **5.22.1**, phiên bản đã kiểm thử API. Trainer chính chạy CUDA; kiểm thử SAM3 nhỏ, chuẩn bị dữ liệu và đánh giá PNG baseline chạy trên CPU/macOS.

```bash
# Chạy tại thư mục gốc source. Cài Transformers và dependencies; không clone SAM3 upstream.
bash scripts/setup_sam3.sh
python -c "import torch; print(torch.__version__, torch.cuda.device_count()); assert torch.cuda.device_count() == 2"
```

Trong notebook, cài bằng `%pip install -r requirements.txt` tại thư mục gốc source. Nếu đã chạy requirements cũ và thấy NumPy bị hạ xuống `1.26.4`, cập nhật source/requirements trước rồi cài lại. NumPy 1.26 chỉ hỗ trợ Python 3.9–3.12 ([release notes](https://numpy.org/doc/2.1/release/1.26.4-notes.html)); không dùng ràng buộc `numpy<2` trên session Python 3.13. Sau khi cài, restart kernel để nạp lại thư viện rồi chạy lại các cell thiết lập/token. Nếu session đã bị thay đổi nhiều package và vẫn lỗi import, tạo session mới để khôi phục môi trường nền trước khi cài requirements đã sửa; restart kernel đơn thuần không hoàn tác các package đã cài.

Nếu log dừng ở `torchaudio/lib/_torchaudio.abi3.so: undefined symbol: torch_library_impl`, bản torchaudio cài sẵn không tương thích với torch. Transformers import audio utilities khi nạp SAM3 dù pipeline chỉ xử lý ảnh. Cập nhật `requirements.txt` rồi chạy `%pip install -r requirements.txt` và restart kernel. Trước khi chạy `torchrun`, kiểm tra bằng cell sau (không tải model weights):

```python
import torch
import torchvision
import torchaudio
from src.sam3_adapter import load_sam3_api

load_sam3_api()
print("torch / torchvision / torchaudio:", torch.__version__, torchvision.__version__, torchaudio.__version__)
print("CUDA:", torch.version.cuda, "GPU count:", torch.cuda.device_count())
assert torch.cuda.is_available() and torch.cuda.device_count() == 2
```

Entry point train kiểm tra import SAM3 trước khi khởi tạo DDP hoặc chuẩn bị annotations để lỗi môi trường xuất hiện ngay. Launcher cũng kiểm tra trước bước `--prepare_only`. Log `[data]` cho biết đang chuẩn bị dữ liệu, `[model]` cho biết đang tải model; vòng train chỉ bắt đầu khi có `[run]` và thanh tiến trình `epoch`.

Model được khởi tạo qua thư viện:

```python
from transformers import Sam3Model, Sam3Processor

model = Sam3Model.from_pretrained("facebook/sam3", use_safetensors=True)
processor = Sam3Processor.from_pretrained("facebook/sam3")
```

Luồng train/evaluate không import package `sam3`, không cần checkout `src/sam3`, không dùng `--checkpoint` hoặc biến `SAM3_CHECKPOINT`. Thư mục upstream cũ chỉ là tài liệu tham chiếu của repo. Transformers tự tải config, tokenizer/processor và safetensors vào cache. Pretrained weights vẫn cần được tải ở lần đầu; không cần tự tìm hoặc upload file `sam3.pt`.

Model [facebook/sam3](https://huggingface.co/facebook/sam3) yêu cầu tài khoản được cấp quyền truy cập. Sau khi chấp nhận điều kiện và có quyền, thêm Kaggle Secret tên **HF_TOKEN**, bật quyền secret cho notebook, rồi chạy cell Python sau trước `torchrun`:

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN").strip()
```

Nếu `get_secret` báo `BackendError`, lỗi xảy ra ở bước lấy secret từ Kaggle, trước khi Hugging Face kiểm tra token. Trong **Add-ons → Secrets**, kiểm tra label chính xác là `HF_TOKEN` (phân biệt hoa/thường, không có khoảng trắng) và secret đã được bật cho notebook hiện tại. Nếu vẫn lỗi dù đã kiểm tra, có thể thay cell trên bằng cách nhập token ẩn cho session hiện tại:

```python
import os
from getpass import getpass

os.environ["HF_TOKEN"] = getpass("Hugging Face token: ").strip()
```

Không in giá trị token. Kiểm tra xác thực và quyền đọc config của model trước khi tải weights hoặc chạy DDP. Truyền token tường minh và buộc tải lại config nhỏ để cache cũ không che khuất lỗi quyền:

```python
from huggingface_hub import HfApi, get_token, hf_hub_download

token = get_token()
if not token:
    raise RuntimeError("Chưa có HF_TOKEN. Chạy lại cell đọc Kaggle Secret hoặc nhập token.")
print("Hugging Face user:", HfApi(token=token).whoami()["name"])
hf_hub_download("facebook/sam3", "config.json", token=token, force_download=True)
print("Đã đọc được config SAM3.")
```

Nếu `whoami()` báo 401, token không được Hugging Face chấp nhận: cập nhật secret bằng token hợp lệ rồi chạy lại cell thiết lập. Nếu `whoami()` thành công nhưng tải `config.json` báo 401/403 hoặc `GatedRepoError`, mở [facebook/sam3](https://huggingface.co/facebook/sam3) bằng đúng tài khoản vừa in ra, chấp nhận điều kiện/gửi yêu cầu truy cập và kiểm tra yêu cầu đã được duyệt. Token cũng cần quyền đọc model đó; với token fine-grained, kiểm tra phạm vi truy cập gated model ([tài liệu Hugging Face](https://huggingface.co/docs/hub/security-tokens)). Token hợp lệ không tự cấp quyền cho model bị giới hạn. Cell dùng `getpass` chỉ giữ token trong môi trường của session, cần chạy lại sau khi restart kernel.

Token được truyền qua môi trường tới worker và được truyền tường minh vào các lệnh `from_pretrained`, kể cả khi tắt implicit token. Token không ghi trong arguments/checkpoint/run-config. Train và launcher kiểm tra quyền đọc config qua HTTP trước khi khởi tạo DDP/chuẩn bị annotations; model local, `--local_files_only` hoặc `HF_HUB_OFFLINE=1` bỏ qua kiểm tra mạng. Có thể dùng `hf auth login` thay cho Kaggle Secrets nếu chạy terminal. Library không thay thế yêu cầu quyền truy cập model.

## Dữ liệu và nhãn

```text
plantseg/
  Metadata.csv
  images/{train,val,test}/*.jpg
  annotations/{train,val,test}/*.png
  annotation_{train,val,test}.json     # chỉ bắt buộc khi --annotation_source coco
```

PNG là nguồn giám sát mặc định, nền `0`, bệnh `1..115`. Metadata có chỉ số bệnh `0..114`; đọc CSV với UTF-8 BOM. Ảnh RGB được sửa hướng EXIF để khớp khung tọa độ của mask. Sai kích thước, thiếu ảnh/mask, mask nhiều lớp hoặc chỉ số ngoài miền sẽ báo lỗi.

Dataset hiện có **5.367 train / 846 val / 1.561 test**, 115 lớp. Audit thực tế phát hiện **12 ảnh** có mã bệnh trong PNG khác metadata: 8 train, 4 test. Tham số `--label_policy` phải giống nhau giữa train, validation và baseline:

- `mask` (mặc định): giữ nguyên nhãn PNG; prompt và category train lấy theo nhãn PNG khi bất đồng. Phù hợp đối chiếu baseline dùng mask gốc.
- `metadata`: giữ nguyên vùng foreground, gán lại mã bệnh theo metadata trong bộ nhớ/cache. Không sửa dữ liệu gốc.
- `strict`: dừng khi hai nguồn nhãn bất đồng.

Mọi bất đồng được ghi vào cache COCO và JSON đánh giá. Có thể chọn `--annotation_source coco` để đối chiếu với polygon gốc; chế độ này lọc annotation của ảnh không còn trong split và polygon suy biến, có thể giảm số ảnh train. Đánh giá cuối cùng vẫn dùng PNG. Các ghi chú audit cũ trong `docs/` và `reports/` mô tả pipeline trước khi refactor, không thay thế hướng dẫn chạy này.

Để dùng loss instance của SAM3 với mask semantic, mỗi thành phần liên thông 8 hướng được chuyển thành một **pseudo-instance** RLE không mất pixel, không khẳng định đây là lesion instance đã được chuyên gia xác nhận. Nếu vượt 200 thành phần/ảnh, giữ 199 thành phần lớn nhất và gộp phần còn lại thành một pseudo-instance. Cache ghi lại số ảnh được gộp. Không dùng pseudo-instance này để tuyên bố instance AP.

```bash
# CPU: kiểm tra tất cả cặp RGB/PNG và tạo cache tại output có quyền ghi.
python train.py --prepare_only \
  --dataset_dir datasets/plantseg --output_dir outputs/sam3-plantseg \
  --annotation_source png --label_policy mask
```

Cache có fingerprint gồm đường dẫn dataset, nội dung metadata, tên/kích thước/mtime file, nguồn annotation và chính sách nhãn. Fingerprint phục vụ phát hiện thay đổi và tránh dùng nhầm cache; không phải kiểm toán hash nội dung từng ảnh. `--rebuild_annotations` tạo lại cache.

## Lệnh training tương tự mẫu torchrun

Trong Kaggle notebook, đặt `!` trước `torchrun` (hoặc dùng cell `%%bash`). Thay `<dataset>` theo mount thực tế; đường dẫn dataset phải chứa trực tiếp `Metadata.csv`.

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py \
  --dataset_dir /kaggle/input/<dataset>/plantseg \
  --model_id facebook/sam3 --model_revision main \
  --cache_dir /kaggle/working/hf-cache \
  --output_dir /kaggle/working/plantseg_sam3_transformers_s1105 \
  --device cuda --scope heads --annotation_source png --label_policy mask \
  --amp fp16 --amp_init_scale 1024 --frozen_dtype fp16 --gradient_checkpointing \
  --batch_size 1 --gradient_accumulation_steps 8 \
  --epochs 20 --warmup_ratio 0.05 --early_stopping_patience 4 \
  --lr 8e-5 --lr_vision 2.5e-5 --lr_language 5e-6 --weight_decay 0.1 \
  --gradient_clip 0.1 --prompts_per_image 1 --pos_prob 0.7 --loss_mask_size 144 \
  --eval_mode all_classes --selection_metric miou \
  --score_threshold 0.5 --mask_threshold 0.5 --val_every 1 \
  --num_workers 2 --log_every_updates 25 --seed 1105
```

Hoặc dùng launcher đã cấu hình sẵn:

```bash
DATASET_DIR=/kaggle/input/<dataset>/plantseg \
MODEL_ID=facebook/sam3 \
bash scripts/train_kaggle_2xt4.sh
```

Cấu hình khởi đầu dùng batch **1 ảnh/GPU**, accumulation 8, effective batch **16 ảnh/update**. T4 dùng FP16. `heads` đóng băng ViT backbone và CLIP text encoder, vẫn train vision neck, text projection, fusion/decoder và segmentation heads. CLIP hidden states của 115 tên bệnh được cache rồi bỏ text encoder khỏi model đang chạy; text projection vẫn nhận gradient. `full` train toàn model, cần đo lại bộ nhớ và điều chỉnh cấu hình. Chưa có số đo xác nhận cấu hình SAM3 này vừa 16 GB T4; phải chạy smoke test dưới đây trước. DDP chia model thành hai bản, mỗi GPU vẫn phải đủ bộ nhớ cho một bản và activation của nó.

```bash
# Smoke test GPU: hoàn thành train, validation và lưu checkpoint trước khi chạy dài.
# Dùng output riêng, không resume checkpoint smoke vào run đầy đủ.
torchrun --standalone --nnodes=1 --nproc_per_node=2 train.py \
  --dataset_dir /kaggle/input/<dataset>/plantseg \
  --model_id facebook/sam3 --model_revision main \
  --cache_dir /kaggle/working/hf-cache \
  --output_dir /kaggle/working/plantseg_smoke \
  --epochs 1 --limit_train 16 --limit_val 4 --eval_mode oracle \
  --batch_size 1 --gradient_accumulation_steps 2 --num_workers 0 --amp fp16
```

Train dùng AdamW, warmup tuyến tính + cosine decay theo optimizer update thành công. AMP overflow không tăng step/scheduler. Nhóm accumulation còn dư cuối epoch vẫn được cập nhật và chia loss theo số microbatch thực tế. Sampler shuffle theo epoch; DDP sampler bỏ tối đa `world_size - 1` ảnh mỗi epoch để mỗi rank có cùng số batch. Validation/test chia chỉ số ảnh theo rank, không padding và không đếm trùng. Nếu loss không hữu hạn, tất cả rank dừng cùng nhau.

`--eval_mode all_classes` chạy tất cả 115 prompt cho mỗi ảnh, chia sẻ image features, ghép nhãn pixel bằng điểm `query confidence × mask probability` lớn nhất trong các instance qua threshold; hòa điểm ưu tiên class ID nhỏ hơn. Không nhận nhãn thật làm đầu vào dự đoán. Chế độ này tốn thời gian hơn nhiều so với `oracle`. `oracle` chỉ query lớp bệnh thật để chẩn đoán chất lượng mask nhanh; kết quả phải ghi rõ **disease-conditioned/oracle**, không đặt ngang số baseline tự dự đoán lớp. Mặc định CLI train là `oracle`; launcher Kaggle và lệnh đầy đủ ở trên chọn `all_classes` rõ ràng.

Có thể đặt `EVAL_MODE=oracle` cho launcher trong giai đoạn phát triển; vẫn phải chọn threshold trên validation và chạy đánh giá `all_classes` sau cùng nếu so sánh baseline. Mọi tham số/tên metric của ví dụ SciVQA như BLEU/ROUGE, Qwen, OT, xLSTM không áp dụng cho segmentation.

### Giảm thời gian train + validation trên 2 × T4

Validation `all_classes` thử 115 prompt/ảnh. Evaluator cache text embeddings trong mỗi lần đánh giá (tạo lại sau mỗi epoch có validation để dùng text projection/CLIP mới nhất), và hỗ trợ gộp prompt bằng `--eval_prompt_batch_size`. Image backbone vẫn chạy một lần/ảnh; mask vẫn được nội suy về kích thước ảnh gốc, giữ đủ các lớp và quy tắc ghép nhãn. Mặc định batch prompt là 1 để giữ nhu cầu VRAM thấp. Có thể thử 2 rồi 4 trên T4; batch lớn tăng bộ nhớ và không bảo đảm tăng tốc tuyến tính. Sai số số thực giữa các batch có thể làm thay đổi pixel sát threshold hoặc sát điểm hòa.

Ưu tiên giảm tần suất validation bằng `--val_every 3`; epoch cuối luôn được đánh giá. Ví dụ tiếp tục **run đã lưu checkpoint** với cùng dataset, output và cấu hình train ban đầu:

```bash
DATASET_DIR=/kaggle/input/<dataset>/plantseg \
OUTPUT_DIR=/kaggle/working/plantseg_sam3_transformers_s1105 \
RESUME=/kaggle/working/plantseg_sam3_transformers_s1105/last.pt \
bash scripts/train_kaggle_2xt4.sh \
  --val_every 3 --resume_allow_val_schedule_change \
  --eval_prompt_batch_size 2
```

`--resume_allow_val_schedule_change` chỉ cho phép đổi `val_every`: giữ optimizer, GradScaler, step/lịch LR và best metric, đặt lại bộ đếm early stopping. Lịch validation thay đổi nên kết quả chọn best checkpoint/early stopping có thể khác run cũ. Không dùng cờ này để đổi `eval_mode`, `limit_val`, batch train hoặc hyperparameter. `eval_prompt_batch_size` có thể đổi khi resume vì chỉ thay cách chia batch inference. Với run mới, bỏ `RESUME` và cờ cho phép đổi lịch. Launcher cũng nhận `VAL_EVERY` và `EVAL_PROMPT_BATCH_SIZE`.

Theo log 42 phút train + 68 phút val, kế hoạch 20 epoch đổi từ val mỗi epoch sang mỗi 3 epoch giảm số lần val từ 20 xuống 7, tiết kiệm khoảng 14 giờ 44 phút, chưa tính hiệu quả batching. Đây là ước tính khi chạy đủ 20 epoch và tốc độ giữ nguyên; không phải benchmark GPU của code mới. Validation thưa hơn giảm số checkpoint được so sánh và kéo dài số epoch trước khi early stopping đủ số lần không cải thiện.

Đo trên checkpoint trước khi chọn batch prompt cho run dài: chạy evaluation cùng một tập ảnh với batch 1, 2, 4 trong các output riêng, giữ nguyên `eval_mode`, threshold và `limit_val`. JSON đánh giá ghi `evaluation_seconds` và `eval_prompt_batch_size`; kiểm tra cả metric và VRAM. Log epoch in riêng `train`, `val`, `save`; `log.jsonl` giữ `seconds` cho train, thêm `val_evaluation_seconds`, `checkpoint_seconds`, `epoch_seconds` (`val_evaluation_seconds` chỉ có ở epoch được validation).

## Loss và preprocessing của backend Transformers

Transformers `Sam3Model` trả về prediction tensors, chưa cung cấp loss training cho API này. `src/losses.py` triển khai objective `transformers-hungarian-v1` với Hungarian matching theo confidence, L1/GIoU box và BCE/Dice mask. Loss cuối gồm:

| Thành phần | Trọng số |
|---|---:|
| Query focal (alpha 0.25, gamma 2), normalized theo số target | 2 |
| Concept presence BCE | 1 |
| Box L1 theo normalized cxcywh | 5 |
| Box GIoU | 2 |
| Instance mask BCE / Dice | 5 / 5 |
| Semantic foreground BCE / Dice | 5 / 5 |

Loss được lấy trung bình theo cặp ảnh/prompt, tính bằng FP32 dưới AMP. Loss này giám sát output cuối; **khác recipe auxiliary/one-to-many của backend SAM3 upstream trước đó**, nên phải ghi lại backend/loss khi so sánh thí nghiệm. Không tuyên bố hai backend có cùng objective hoặc tái lập nguyên xi số trong paper.

Image preprocessing dùng `Sam3Processor` của đúng model revision (model gốc resize vuông 1008 và normalize mean/std 0.5), thêm horizontal flip khi train. `--loss_mask_size 144` chỉ giảm kích thước mask dùng cho matching/loss; giữ instance target và semantic target bằng area resampling. Evaluation vẫn nội suy logits về kích thước PNG gốc trước threshold, không dùng mask resolution của loss để tính metric. Tăng số prompt sẽ tăng số cặp ảnh/prompt thực sự đưa qua model và tăng nhu cầu bộ nhớ.

`Sam3Model` ở phiên bản pin chưa expose `gradient_checkpointing_enable` trên model ngoài. Project dùng `torch.utils.checkpoint` nonreentrant cho các block vision/DETR và mask decoder; đã kiểm thử backward và DDP trên cấu hình nhỏ.

## Resume và đầu ra

Thêm `--resume /kaggle/input/<saved-run>/last.pt` vào **cùng lệnh train ban đầu**, giữ nguyên số epoch tổng. Ví dụ đang dừng ở epoch 6 của kế hoạch 20 thì vẫn dùng `--epochs 20`. Launcher hỗ trợ biến `RESUME=/path/last.pt`.

Checkpoint là adaptation gồm trainable parameters, optimizer, GradScaler, epoch/update, best metric, patience và RNG từng rank/DataLoader. Cần cùng scope, world size, model/revision/Transformers version, dataset/fingerprint và cấu hình tối ưu ban đầu. Khôi phục ở ranh giới epoch; không bảo đảm bitwise deterministic cho kernel CUDA. Checkpoint mới có **format 3**. Checkpoint format 2 của backend SAM3 upstream không tương thích với tham số/loss Transformers; không resume trực tiếp.

Các file chính trong output:

| File | Nội dung |
|---|---|
| `run-config.json` | Arguments, backend/version, resolved HF commit, SHA-256 metadata/local model files, GPU, effective batch, prompt vocabulary, dataset manifest |
| `log.jsonl` | Loss, learning rate, optimizer updates, AMP scale/overflow, peak GPU memory, metric validation |
| `best.pt`, `last.pt` | Adaptation và trạng thái optimizer/RNG để resume; không phải pretrained sam3.pt |
| `processor/` | Processor/tokenizer tương ứng run |
| `best-metrics.json` | Metric tương ứng best checkpoint |
| `val-latest.json`, `val-latest-per-class.csv`, `val-latest-per-image.csv` | Kết quả validation mới nhất |
| `val-latest-confusion.npy` | Ma trận nhầm lẫn pixel: hàng ground truth, cột prediction |

`--model_revision main` được resolve thành commit và ghi trong `model_identity.resolved_revision` của `run-config.json`. Khi tái lập/resume về sau, truyền `--model_revision <resolved_commit>` để dùng đúng base model dù nhánh main đã thay đổi. Kiểm tra identity sẽ từ chối model/revision không khớp. Tham số `--cache_dir` đổi vị trí cache; `--local_files_only` dùng cache có sẵn mà không truy cập mạng.

Để chạy offline với một thư mục model, dùng output `save_pretrained()` của cả `Sam3Model` (safetensors) và `Sam3Processor`, rồi truyền `--model_id /path/to/saved_model --local_files_only`. Thư viện vẫn cần config, tokenizer và weights đầy đủ trong thư mục đó.

Không chạy test tự động mỗi epoch. Chốt cấu hình/threshold trên validation trước rồi dùng best checkpoint đánh giá test.

## Đánh giá và so sánh baseline

```bash
# Fine-tuned model, không cung cấp lớp bệnh thật cho predictor.
torchrun --standalone --nnodes=1 --nproc_per_node=2 evaluate.py \
  --dataset_dir /kaggle/input/<dataset>/plantseg --split test \
  --model_id facebook/sam3 --model_revision main \
  --cache_dir /kaggle/working/hf-cache \
  --resume /kaggle/working/plantseg_sam3_transformers_s1105/best.pt \
  --scope heads --eval_mode all_classes --label_policy mask --amp fp16 \
  --output_dir /kaggle/working/plantseg_sam3_test

# Zero-shot pretrained baseline: dùng cùng lệnh nhưng bỏ --resume.
# Thêm --save_predictions /kaggle/working/predictions để xuất PNG class-index.
```

Đánh giá output của U-Net, DeepLab, SegFormer hoặc baseline khác bằng cùng evaluator, không cần load SAM3/CUDA:

```bash
python evaluate.py --dataset_dir datasets/plantseg --split test \
  --prediction_dir /path/to/baseline_class_index_pngs \
  --label_policy mask --output_dir outputs/baseline-test
```

Prediction phải có cùng tên với mask gốc, cùng kích thước gốc, một channel số nguyên, `0` nền và `1..115` lớp bệnh. Evaluator không tự đổi màu RGB, không tự resize và không dùng metadata để gán bệnh cho prediction nhị phân. Baseline đầu ra train-ID `0..114` cần chuyển sang `1..115` và ánh xạ nền trước khi xuất PNG. Thiếu prediction sẽ báo lỗi, không bỏ ảnh âm thầm.

Paper [PlantSeg, Scientific Data](https://doi.org/10.1038/s41597-025-06513-4) dùng **mIoU** và **mAcc** tại Eq. (1), Table 3:

`IoU_c = TP_c / (TP_c + FP_c + FN_c)`; `Acc_c = TP_c / (TP_c + FN_c)`.

Đếm pixel trên **toàn bộ split trước**, rồi lấy trung bình theo lớp; không lấy trung bình IoU ảnh thay cho mIoU. Các metric xuất ra:

| Key | Cách tính |
|---|---|
| `miou`, `macc` | Macro trên lớp bệnh, bỏ background |
| `miou_with_background`, `macc_with_background` | Macro bao gồm background; báo cả hai vì paper không nêu rõ quy ước background/absent class |
| `mean_dice`, `mean_precision` | Macro theo lớp bệnh từ pixel confusion matrix |
| `pixel_accuracy` | Tổng pixel đúng / tổng pixel |
| `foreground_iou`, `foreground_dice` | Gộp tất cả bệnh thành foreground rồi đếm pixel |
| `pixel_precision`, `pixel_recall` | Precision/recall foreground pooled |
| `image_mean_iou`, `image_mean_dice` | Trung bình mask foreground từng ảnh |
| `disease_macro_image_iou` | Trung bình IoU ảnh trong mỗi bệnh, rồi trung bình các bệnh có ảnh |

Tất cả giá trị ở thang **0–1**, nhân 100 khi lập bảng phần trăm như paper. IoU lớp có union bằng 0 là `null`; mIoU bỏ lớp đó, nhưng vẫn tính lớp vắng GT nếu model dự đoán sai vào lớp đó. mAcc bỏ lớp không có pixel GT. JSON ghi class support và policy. Precision khi không có prediction là `null`; recall khi GT trống là `null`. Cặp foreground cùng trống có image IoU/Dice = 1.

Cùng công thức metric chưa đủ tái lập Table 3: phải thống nhất phiên bản/split, chính sách nhãn, background/absent classes, input preprocessing, quy trình chọn checkpoint và việc model có được nhận nhãn bệnh thật không. Pipeline SAM3 dùng AdamW và loss `transformers-hungarian-v1`; paper mô tả SGD/CE, LR 0.001, momentum 0.9, weight decay 0.0005, batch 16 cho baseline. Đây là protocol fine-tune SAM3 để đối chiếu, không phải tái lập nguyên xi training baseline trong paper.

## Kiểm thử

```bash
python -m pip install -r requirements.txt
python -m pytest -q
ruff check src tests train.py evaluate.py
```

Tests load một SAM3 nhỏ bằng `from_pretrained` từ safetensors local, chạy loss/backward, text cache/projection gradient, inference và DDP checkpointing bằng model thật của Transformers. Đồng thời kiểm tra confusion/macro metrics, nhãn vắng, mask trống, nội suy logits, PNG→RLE không mất pixel, gộp >200 components, cache invalidation, EXIF, RNG/checkpoint, CLI và DDP Gloo hai process. Test DDP đối chiếu gradient accumulation với batch tham chiếu, đồng thời xác nhận evaluation không nhân đôi ảnh khi split không chia hết cho world size. Test này chạy CPU, không thay thế smoke test SAM3/NCCL trên Kaggle.

Chi tiết kiểm tra đã thực hiện: [training-validation.md](docs/training-validation.md). API tham chiếu từ [Transformers SAM3](https://huggingface.co/docs/transformers/model_doc/sam3).
