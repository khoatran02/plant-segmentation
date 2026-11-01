# Giải thích và phản biện báo cáo SAM 3 trên PlantSeg

Ngày rà soát: **30/10/2026**, theo múi giờ Việt Nam. Phạm vi là thư mục `reports/`, đối chiếu với `src/training.py`, tài liệu audit và các bằng chứng đang được lưu trong repo.

**Đánh giá chung:** bản thảo trình bày khá trung thực một pipeline đã được viết và một kế hoạch đánh giá chưa hoàn tất. Mô tả nhiệm vụ, phạm vi fine-tune, sampling prompt và định nghĩa metric nhìn chung phù hợp với mã nguồn. Tuy nhiên, tài liệu hiện phù hợp với **báo cáo phương pháp và giao thức thực nghiệm đang chuẩn bị**. Chưa có kết quả để xác nhận khả năng học, hiệu quả fine-tune hay tính khả thi trên GPU. Các vấn đề quan trọng nhất cần giải quyết là chất lượng supervision, ảnh trùng giữa các split, sự khác nhau giữa metric chọn checkpoint và metric cuối cùng, cùng khả năng tái lập pipeline.

Tài liệu này phân biệt ba loại căn cứ:

- **Kiểm tra trực tiếp:** nội dung file hiện tại, logic mã nguồn, phép đếm trên CSV lưu trong repo, PDF hiện có và thử nghiệm nhỏ trong thư mục tạm.
- **Bằng chứng lịch sử:** số liệu trong `snapshot.json` và audit ngày 07/10/2026. Không có dữ liệu ảnh gốc trong checkout để chạy lại toàn bộ audit.
- **Đề xuất:** thiết kế thí nghiệm, cách sửa mô tả và điều kiện nghiệm thu. Những nội dung này chưa được triển khai chỉ bằng việc viết bản review.

Trong lượt rà soát này không sửa LaTeX, PDF, mã huấn luyện hay dataset. Mã SAM 3 dùng để đối chiếu được đọc từ kho chính thức tại revision `8f0b7f4d4e7eda2ed606ebde6702c93359ad01da`, khớp gitlink và snapshot; việc đọc mã upstream không đồng nghĩa môi trường hiện tại đã chạy được mô hình.

Để đọc theo nhu cầu: mục 1 giải thích các file; mục 2-3 giải thích bài và metric; mục 4 đối chiếu triển khai; mục 5 chứa 15 nhận xét phản biện; mục 6-8 đề xuất sửa và hoàn thiện thí nghiệm; mục 9-10 ghi phạm vi kiểm chứng và nguồn.

## 1. `reports/` đang chứa những gì?

| File/thư mục | Chức năng | Cách sử dụng đúng |
|---|---|---|
| [main.tex](main.tex) | Điểm vào LaTeX; cấu hình bố cục, tiêu đề, abstract và thứ tự các phần | Bắt đầu từ đây để hiểu cấu trúc bài |
| [main.pdf](main.pdf) | PDF đã biên dịch, hiện có 7 trang A4 hai cột | Xem hình thức và nội dung bản đã xuất; sự tồn tại của PDF không xác nhận build hiện tại thành công |
| [sections/01_introduction.tex](sections/01_introduction.tex) | Cả Introduction và Related Work | Giải thích động cơ, câu hỏi nghiên cứu và công trình nền |
| [sections/02_dataset.tex](sections/02_dataset.tex) | Dataset and Annotation Audit | Phân biệt dữ liệu gốc, dữ liệu được builder giữ lại và các lỗi chất lượng |
| [sections/03_method.tex](sections/03_method.tex) | Method | Mô tả prompt, thành phần đóng băng, biến đổi ảnh, loss và tối ưu |
| [sections/04_protocol.tex](sections/04_protocol.tex) | Experimental Protocol | Phân biệt validation hiện có và đánh giá cuối cùng cần bổ sung |
| [sections/05_results.tex](sections/05_results.tex) | Results | Chỉ có kết quả chuẩn bị dữ liệu; các bảng hiệu năng mô hình còn `Pending` |
| [sections/06_discussion.tex](sections/06_discussion.tex) | Cả Discussion and Limitations và Conclusion | Giới hạn phạm vi kết luận và nêu bước thực nghiệm tiếp theo |
| [references.bib](references.bib) | Nguồn thư mục BibTeX | Sửa thông tin trích dẫn tại đây |
| `main.bbl` (sinh khi build) | Danh mục tài liệu tham khảo do BibTeX sinh | Không lưu trong source; tái sinh từ `.bib` |
| [build.sh](build.sh) | Chạy pdfLaTeX, BibTeX rồi pdfLaTeX hai lần | Giải quyết trích dẫn, số bảng/hình và tham chiếu chéo |
| `collect_evidence.py` (đã gỡ ngày 01/11/2026) | Script thu thập snapshot của pipeline cũ | Snapshot lịch sử được giữ; script không còn tương thích source hiện tại |
| [evidence/snapshot.json](evidence/snapshot.json) | Số lượng, hash và mặc định được lưu từ lần thu thập trước | Là một bản ghi có thời điểm, không tự cập nhật theo máy đang mở repo |
| [evidence/dataset-audit.md](evidence/dataset-audit.md) | Bản lưu audit dữ liệu | Đọc cùng CSV lỗi và ghi nhớ đây là audit lịch sử |
| [evidence/flagged-files.csv](evidence/flagged-files.csv) | Danh sách ảnh có vấn đề | Là đầu vào có thể dùng để xây manifest kiểm soát chất lượng |
| [figures/dataset_samples.jpg](figures/dataset_samples.jpg) | Ảnh minh họa với ground-truth mask | Không phải prediction của mô hình |
| [SOURCE_NOTES.md](SOURCE_NOTES.md) | Ánh xạ phát biểu trong bài sang nguồn chứng minh | Hữu ích để truy vết nhận định |
| [VALIDATION.md](VALIDATION.md) | Ghi nhận kiểm tra ở một môi trường trước đó | Cần ghi thêm môi trường và hash đầu ra để người khác kiểm chứng |
| [README.md](README.md) | Hướng dẫn build, refresh bằng chứng và bổ sung kết quả | Có đường dẫn tuyệt đối của máy khác cần chỉnh khi dùng lại |

Sáu file nội dung tạo thành **tám section** trong PDF: Introduction, Related Work, Dataset, Method, Experimental Protocol, Results, Discussion và Conclusion. Đây là cách tổ chức có chủ ý; số file không phải số section.

## 2. Giải thích nội dung từng phần của bài

Một số thuật ngữ dùng xuyên suốt:

| Thuật ngữ | Nghĩa trong tài liệu này |
|---|---|
| Pretrained / fine-tune | Mô hình đã được học từ trước / tiếp tục cập nhật một phần hoặc toàn bộ trọng số trên dữ liệu đích |
| Semantic segmentation | Gán lớp cho pixel; các vùng cùng lớp được xem chung khi tính mask nhị phân |
| Instance segmentation | Ngoài lớp pixel, còn phân biệt từng đối tượng hoặc vùng được định nghĩa là instance |
| Prompt / query | Prompt là điều kiện văn bản; “query” trong code có thể chỉ lượt hỏi theo prompt hoặc candidate của decoder, cần đọc theo ngữ cảnh |
| Presence | Dự đoán khái niệm được hỏi có xuất hiện trong ảnh hay không |
| Logit / sigmoid | Giá trị thực trước khi chuyển sang thang 0-1 / hàm $\sigma(x)=1/(1+e^{-x})$ dùng cho phép chuyển đó |
| Baseline / ablation | Cấu hình đối chiếu / thí nghiệm thay đổi có kiểm soát một lựa chọn để đo tác động của nó |
| Manifest / checkpoint | Danh sách mẫu và quyết định xử lý dùng cho một run / trạng thái mô hình được lưu tại một thời điểm |

### 2.1. Tiêu đề, abstract và phạm vi nhiệm vụ

Tiêu đề hiện tại là *Text-Prompted Plant Disease Lesion Segmentation with SAM 3 on PlantSeg*. Ý nghĩa là dùng một cụm từ chỉ bệnh để điều kiện hóa việc tìm các vùng bệnh trong ảnh.

Đầu vào của nhiệm vụ gồm hai thành phần: ảnh RGB và tên bệnh. Ví dụ, mô hình nhận ảnh lá táo cùng prompt `apple black rot`. Tên bệnh đúng được lấy từ metadata, không được suy ra bởi một bộ phân loại đang được đánh giá trong dự án.

Abstract nêu bốn nhóm thông tin: mục tiêu phân vùng có điều kiện theo bệnh; pipeline thích nghi SAM 3; quy mô và bất thường annotation của PlantSeg v7; các đánh giá còn phải thực hiện. Câu cuối xác nhận chưa có kết quả về độ chính xác hoặc cải thiện so với baseline. Cách giới hạn kết luận này là đúng với trạng thái repo.

Phân vùng có điều kiện theo tên bệnh cho phép trả lời: “Khi biết bệnh cần tìm, mô hình khoanh vùng triệu chứng tốt đến đâu?”. Khả năng nhận biết bệnh từ một ảnh chưa biết nhãn là một câu hỏi khác, cần thiết kế và dữ liệu đánh giá riêng. Vì vậy, từ khóa “disease” trong tiêu đề không nên được diễn giải thành hệ thống chẩn đoán tự động.

### 2.2. Introduction: vì sao cần bài toán này?

Phần mở đầu phân biệt nhãn cấp ảnh và nhãn cấp pixel. Nhãn cấp ảnh cho biết bệnh được gán cho ảnh; segmentation mô tả vị trí của vùng triệu chứng đã được chú thích. Ảnh ngoài tự nhiên làm nhiệm vụ khó hơn vì nền phức tạp, ánh sáng thay đổi, tổn thương nhỏ và hình dạng phân mảnh.

Ba câu hỏi nghiên cứu được đặt ra là:

1. SAM 3 pretrained phân vùng tổn thương tốt đến đâu khi được cung cấp tên bệnh?
2. Fine-tune với encoder đóng băng có cải thiện chất lượng không?
3. Kiểm soát chất lượng dữ liệu làm thay đổi kết luận như thế nào?

Đây là các câu hỏi phù hợp với pipeline. Hiện chúng là câu hỏi chờ thực nghiệm. Đóng góp được tuyên bố là đặc tả workflow và giao thức đánh giá, chưa có đóng góp kiến trúc mới hay cải thiện định lượng đã chứng minh.

### 2.3. Related Work: các công trình nền đóng vai trò gì?

Phần này giới thiệu U-Net và DeepLabv3+ làm đại diện cho phân vùng có giám sát, SAM 3 cho phân vùng theo khái niệm, và PlantSeg cho dữ liệu vùng bệnh. Các mô tả khái quát phù hợp với nguồn gốc: [U-Net](https://arxiv.org/abs/1505.04597), [DeepLabv3+](https://arxiv.org/abs/1802.02611), [SAM 3](https://arxiv.org/abs/2511.16719) và [PlantSeg](https://arxiv.org/abs/2409.04038).

Tuy nhiên, Related Work hiện chủ yếu cung cấp nền tảng. Nó chưa đủ để xác lập khoảng trống nghiên cứu hoặc tính mới của một phương pháp. Nếu phát triển thành bài báo thực nghiệm, cần bổ sung đối chiếu sát nhiệm vụ: đầu vào có tên bệnh hay không, supervision semantic hay instance, và đánh giá trên phiên bản PlantSeg nào. Không nên suy ra tính mới chỉ từ việc kết hợp tên một foundation model với tên dataset.

### 2.4. Dataset and Annotation Audit: vì sao có nhiều bộ số lượng?

Bài mô tả hai bản dữ liệu local. Theo audit được lưu, v7 giữ lại 7.774 ảnh từ v6, không đổi split và giữ nguyên byte ảnh/mask còn lại. Vì vậy, hai bản có quan hệ tập con, không thể coi là hai tập dữ liệu độc lập để chứng minh khả năng chuyển miền.

| Đại lượng | v6 | v7 |
|---|---:|---:|
| Cặp ảnh - mask | 11.458 | 7.774 |
| Train | 7.916 | 5.367 |
| Validation | 1.247 | 846 |
| Test | 2.295 | 1.561 |
| Loài cây | 34 | 34 |
| Bệnh theo nhãn đã hiệu chỉnh | 115 | 115 |
| ID bệnh trong metadata nguyên bản | 114 | 115 |

Chênh lệch 114/115 của v6 là lỗi metadata: ảnh `soybean rust` bị gán vào `bean rust`. Bảng trong bài đã ghi chú sự khác nhau giữa số bệnh thực và số ID hiện diện trong metadata. Snapshot chạy builder trên metadata v6 nguyên bản vẫn tạo 114 category; đó là kết quả của builder trên nhãn chưa sửa, không chứng minh taxonomy chỉ có 114 bệnh.

Một ảnh có ba nguồn thông tin: ảnh màu, PNG semantic mask và annotation polygon trong JSON. PNG dùng 0 cho background, còn foreground có giá trị `Index + 1`. Với mục tiêu phân vùng nhị phân, ground truth là `mask > 0`. Nhiều vùng rời nhau trong mask không tự chứng minh có từng ấy tổn thương độc lập về sinh học.

V7 đã lọc danh sách ảnh trong JSON nhưng giữ cả annotation của ảnh bị loại. Builder loại các bản ghi mồ côi này:

| Đại lượng v7 | Train | Validation | Test | Tổng |
|---|---:|---:|---:|---:|
| Annotation gốc | 52.898 | 8.927 | 15.569 | 77.394 |
| Annotation trỏ đến ảnh đã bị loại | 14.647 | 2.467 | 4.466 | 21.580 |
| Annotation giữ lại | 38.251 | 6.460 | 11.103 | 55.814 |
| Ảnh gốc có PNG mask | 5.367 | 846 | 1.561 | 7.774 |
| Ảnh sau builder COCO | 5.367 | 846 | 1.560 | 7.773 |

Một ảnh test có PNG mask nhưng không có polygon, nên bị loại khỏi tập do builder tạo. Do đó, kết quả trên 1.560 ảnh polygon và trên 1.561 cặp PNG có mẫu số khác nhau. Entry point train hiện chưa thực hiện đánh giá test, dù builder có khả năng chuẩn bị split test.

Hai tỷ lệ coverage cũng cần phân biệt. Trung bình tỷ lệ vùng bệnh theo ảnh của v7 khoảng 21,4%; tổng số pixel bệnh chia tổng số pixel trên toàn bộ dữ liệu khoảng 19,4% theo audit. Ở phép tính thứ nhất, mỗi ảnh có trọng số ngang nhau; ở phép tính thứ hai, ảnh lớn đóng góp nhiều hơn. Hai con số này không mâu thuẫn.

### 2.5. Method: mô hình học gì và phần nào được cập nhật?

Ký hiệu trong bài là:

$$
f_\theta(I,t)=\{(M_j,b_j,s_j)\}_{j=1}^{Q}.
$$

Ở đây, $I$ là ảnh, $t$ là prompt, $\theta$ là tham số mô hình, $Q$ là số candidate query của decoder, $M_j$ là mask, $b_j$ là box và $s_j$ là điểm chọn candidate. Cần phân biệt **prompt văn bản** và **candidate object query**: một prompt có thể tạo ra nhiều candidate mask.

Để đánh giá semantic, các mask vượt ngưỡng được hợp thành một vùng foreground. Phép hợp làm mất thông tin ranh giới giữa các instance, nhưng phù hợp với việc so sánh một mask bệnh nhị phân.

Cấu hình mặc định `scope=heads` đóng băng visual transformer trunk và language backbone. Những thành phần khác, gồm visual neck và các bộ phận fusion/decoder/prediction, vẫn được cập nhật. Vì vậy, “frozen encoders” cần được hiểu theo đúng phạm vi này; không phải toàn bộ đường xử lý ảnh bị đóng băng.

Khi language backbone đóng băng, script mã hóa trước danh sách tên bệnh, lưu các tensor đặc trưng và gỡ language module khỏi mô hình đang dùng. Trọng số đóng băng vẫn tham gia tạo dự đoán thông qua đặc trưng đã tính. Nếu thay cách diễn đạt prompt, phải tạo lại đặc trưng tương ứng hoặc khôi phục đường text encoder. `scope=full` cho phép cập nhật toàn bộ mô hình.

Checkpoint thích nghi chỉ chứa tham số trainable cùng trạng thái optimizer/scaler. Nó cần pretrained base và cách dựng mô hình tương thích để khôi phục. Đây là lý do một file `best.pt` trong thiết kế này chưa phải gói suy luận độc lập.

### 2.6. Prompt dương và âm được chọn như thế nào?

Prompt dương là tên bệnh đúng của ảnh. Prompt âm là tên một bệnh khác, được gán mục tiêu không có instance cần tìm. Với mặc định `prompts_per_image=1` và `pos_prob=0.7`, mỗi lần lấy ảnh dùng một trong hai loại prompt, với kỳ vọng 70% prompt dương và 30% prompt âm trước các tác động runtime.

Sampler ưu tiên bệnh khác trên cùng cây cho các slot âm. Khi chỉ có một slot âm, nếu có bệnh khác trên cùng cây thì slot này lấy từ nhóm đó. Ví dụ, ảnh bệnh táo có thể được hỏi về một bệnh táo khác. Điều này hướng mô hình đến phân biệt triệu chứng trên cùng host thay vì chỉ nhận biết host.

Validation luôn chứa prompt dương và mặc định hai prompt âm. Random generator được seed bằng chỉ số ảnh, nên có thể lặp lại khi giữ nguyên thứ tự dataset. Nếu lọc hoặc sắp lại dữ liệu, chỉ số có thể đổi và bộ prompt âm cũng đổi. Để so sánh hai thí nghiệm theo cặp, nên lưu prompt cụ thể theo tên ảnh hoặc ID ổn định.

Giám sát âm dựa trên giả định bệnh khác không xuất hiện trong ảnh. Nhãn một bệnh ở cấp ảnh không đủ chứng minh giả định đó. Vì vậy, query âm là phép kiểm tra theo chính sách annotation hiện có; chưa tương đương xác nhận ảnh khỏe hoặc bệnh thực sự vắng mặt.

### 2.7. Biến đổi ảnh, matching và loss

Train thực hiện decode mask, lật ngang ngẫu nhiên, resize vuông với scale jitter, pad đến 1.008 x 1.008, chuyển tensor và normalize bằng mean/std bằng 0,5 cho mỗi kênh. Validation resize vuông về 1.008 x 1.008. Biến đổi kích thước ảnh và nhãn cần nhất quán để giữ quan hệ tọa độ.

Đối chiếu [mã resize tại revision đã ghim][sam3-transforms] xác nhận nhánh `square=True` biến đổi hai chiều về cùng kích thước. Nó có thể làm thay đổi tỷ lệ hình dạng. Báo cáo đã nhận diện giới hạn này; lựa chọn resize vuông là quyết định preprocessing cần kiểm chứng, chưa phải một lỗi triển khai tự thân.

Matching ghép candidate của mô hình với target để tính loss. Hungarian matching tạo phép ghép một-một; nhánh one-to-many cho phép thêm candidate học từ cùng target theo quy tắc của criterion. Những trọng số chính được cấu hình là:

| Thành phần | Trọng số | Vai trò |
|---|---|---|
| Box / GIoU | 5 / 2 | Vị trí và độ khớp box |
| Classification / presence | 20 / 20 | Candidate phù hợp và khái niệm có xuất hiện |
| Instance-mask focal / Dice | 200 / 10 | Hình dạng từng mask được ghép target |
| Semantic focal / Dice | 20 / 30 | Hợp vùng foreground ở mức semantic |

Trọng số 200 không đủ để kết luận mask loss chi phối gấp 200 lần một thành phần khác: giá trị loss, chuẩn hóa và gradient còn phụ thuộc công thức, số target và nhánh matching. Các công thức trong bài mô tả trọng số cấu hình, chưa phải biểu thức đầy đủ của tổng loss thực tế.

Có hai chi tiết upstream cần làm rõ thêm trong bài. Với `presence_head=True`, semantic mask loss bỏ query âm khỏi phép lấy trung bình; `presence_loss=False` tắt riêng presence loss của semantic branch. Đồng thời, loss wrapper nhân core loss với căn bậc hai số find query khi bật `scale_by_find_batch_size`. Vì vậy, không nên giải thích prompt âm là ép mọi pixel của semantic mask về 0, hoặc coi tăng số prompt là thay đổi độc lập với thang loss. Căn cứ: [semantic criterion][sam3-loss-fns] và [loss wrapper][sam3-loss-wrapper].

### 2.8. Tối ưu và quản lý bộ nhớ

Bảng cấu hình trong bài chép các mặc định CLI: 20 epoch, batch 1, accumulation 8, AdamW, learning rate `8e-5` cho transformer/heads, `2.5e-5` cho vision và `5e-6` cho language nếu mở train full. Các giá trị này khớp snapshot và mã hiện tại; chúng chưa phải cấu hình của một run đã hoàn thành.

Accumulation cộng gradient từ nhiều microbatch trước một lần optimizer step. Với 5.367 batch và hệ số 8, số nhóm đầy đủ là 670, dư 7 batch. Vòng lặp hiện không xử lý gradient của nhóm dư. Khi loss không hữu hạn, nó xóa gradient đã tích lũy nhưng vẫn xác định ranh giới nhóm theo chỉ số batch ban đầu. Do đó, effective batch thực tế có thể nhỏ hơn con số danh nghĩa.

Các nhánh AMP, giảm precision của frozen trunk, cache văn bản và kernel fallback nhằm giảm nhu cầu tài nguyên hoặc tăng khả năng tương thích. Tài liệu đã thận trọng khi không coi những nhánh này là benchmark. Muốn báo cáo VRAM, tốc độ hoặc trainable parameter count cần đo trên đúng mô hình và cấu hình chạy.

### 2.9. Experimental Protocol: hiện đang đo gì, cuối cùng cần đo gì?

Phần 5 là một trong những phần rõ nhất của bản thảo: nó phân biệt phép đo validation hiện có với phép đo dự kiến dùng cho kết quả cuối cùng.

Điểm chọn candidate được tính như sau:

$$
s_{qj}=\sigma(z_{qj})\sigma(p_q),
$$

trong đó $q$ là prompt, $j$ là candidate, $z$ là query logit, $p$ là presence logit và $\sigma$ là sigmoid. Candidate có điểm lớn hơn 0,5 được giữ. Mask logit lớn hơn 0 được xem là foreground; các mask được giữ sau đó được hợp lại.

Ví dụ, query score 0,9 và presence score 0,8 tạo điểm 0,72 nên được giữ. Hai điểm đều bằng 0,7 tạo tích 0,49 nên bị loại. Ngưỡng của tích điểm cần được đánh giá trên validation; không nên tự diễn giải nó như một xác suất đã được hiệu chuẩn.

Ground truth validation hiện lấy từ polygon, resize xuống kích thước mask dự đoán bằng area interpolation và threshold ở 0,5 trước khi hợp instance. Một tổn thương rất nhỏ có thể mất khi thu nhỏ. Vì vậy, validation hiện tại chưa đo cùng đối tượng với việc so prediction ở kích thước gốc với PNG mask.

| Tên log | Phép đo thực tế | Điều không thể suy ra trực tiếp |
|---|---|---|
| `val_iou` | IoU trung bình trên prompt dương ở kích thước mask đầu ra, dùng ground truth từ polygon | IoU cuối cùng trên PNG gốc hoặc disease-macro IoU |
| `val_recall` | Tỷ lệ prompt dương có candidate vượt ngưỡng score | Pixel recall, lesion recall hoặc định vị đúng |
| `val_false_positive` | Tỷ lệ prompt âm được lấy mẫu có candidate vượt ngưỡng score | Tỷ lệ pixel dương giả hoặc lỗi trên toàn bộ 114 bệnh âm |

Đặc biệt, `val_recall` chỉ kiểm tra `keep.any(...)`, không kiểm tra overlap và cũng không yêu cầu candidate có pixel foreground sau threshold mask. Báo cáo đã nói rõ điều kiện overlap chưa được xét; nên bổ sung chi tiết empty mask và đổi tên metric khi sửa code để tránh hiểu sai.

Giao thức cuối cùng đề xuất PNG mask trong hệ tọa độ ảnh đã căn chỉnh, disease-macro score, xử lý duplicate, nhiều seed và ghi nhận tài nguyên. Đây là hướng hợp lý, nhưng evaluator và manifest để thực hiện giao thức này hiện chưa có.

### 2.10. Results, Discussion và Conclusion

Results hiện có hai loại nội dung: các số liệu chuẩn bị dữ liệu và vị trí chờ cho kết quả mô hình. Bảng 4 dành cho pretrained, frozen-encoder adaptation và tùy chọn full adaptation. Bảng 5 dành cho GPU, cấu hình, số tham số trainable, checkpoint, thời gian, VRAM và latency. Figure 3 là khung chờ prediction.

`Pending` nghĩa là chưa có phép đo được xác minh; không có nghĩa điểm bằng 0. Figure 1 chứa ground truth của dataset, không chứng minh mô hình phân vùng thành công. Những phân biệt này đã được trình bày đúng.

Discussion nhận diện các hạn chế chính: polygon sai, duplicate, lớp hiếm, được cung cấp tên bệnh đúng và thiếu một cohort ảnh khỏe đã được xác minh cho nghiên cứu này. Phát biểu cuối cùng có phạm vi local: bài gốc [PlantSeg](https://arxiv.org/abs/2409.04038) có mô tả thêm ảnh cây khỏe, còn pipeline và bằng chứng trong repo hiện không xác lập một cohort như vậy cho thí nghiệm đang bàn.

Discussion cũng phân biệt coverage trên toàn ảnh và mức độ bệnh trên mô thực vật. Nếu tính diện tích tổn thương chia toàn bộ ảnh thì nền, khoảng cách chụp và bố cục ảnh ảnh hưởng mẫu số. Muốn nghiên cứu disease severity cần một mẫu số như diện tích lá/cơ quan phù hợp và một định nghĩa annotation được kiểm chứng.

Conclusion giữ kết luận ở mức đặc tả workflow và kế hoạch đánh giá. Cách viết phù hợp với bằng chứng hiện có, nhưng cần cập nhật sau khi có run thật.

## 3. Giải thích các metric bằng ví dụ

Với một ảnh, TP là số pixel bệnh được dự đoán đúng; FP là số pixel background bị dự đoán thành bệnh; FN là số pixel bệnh bị bỏ sót. Khi ground truth có foreground:

$$
\operatorname{IoU}=\frac{TP}{TP+FP+FN},\qquad
\operatorname{Dice}=\frac{2TP}{2TP+FP+FN}.
$$

$$
\operatorname{Precision}=\frac{TP}{TP+FP},\qquad
\operatorname{Recall}=\frac{TP}{TP+FN}.
$$

Nếu TP = 60, FP = 20 và FN = 40 thì IoU = 0,50; Dice khoảng 0,667; precision = 0,75; recall = 0,60. Đây là ví dụ toán học, không phải kết quả SAM 3 trong repo.

Các phép tổng hợp có ý nghĩa khác nhau:

| Phép tổng hợp | Cách tính | Câu hỏi được trả lời |
|---|---|---|
| Image-mean IoU | Tính IoU từng ảnh rồi lấy trung bình | Một ảnh được lấy đều từ tập đánh giá có điểm trung bình bao nhiêu? |
| Pooled foreground IoU | Cộng TP, FP, FN của tất cả ảnh rồi tính IoU | Chất lượng trên tổng lượng pixel foreground và lỗi là bao nhiêu? |
| Disease-macro image IoU | Trung bình IoU theo ảnh trong từng bệnh, rồi lấy trung bình các bệnh có mẫu | Khi mỗi bệnh có trọng số ngang nhau, hiệu năng là bao nhiêu? |

Ví dụ, bệnh A có hai ảnh với IoU 0,9 và 0,9; bệnh B có một ảnh với IoU 0,1. Image-mean IoU khoảng 0,633, còn disease-macro image IoU bằng 0,5. Sự khác nhau xuất phát từ trọng số của bệnh nhiều ảnh.

Trong manuscript, ký hiệu `mIoU_disease` được định nghĩa là trung bình **image IoU theo từng bệnh rồi qua các bệnh**. Nó không phải IoU semantic 116 lớp, cũng không phải trung bình IoU pooled của từng bệnh. Tên cột `Macro` nên được mở rộng để tránh nhập nhằng khi đưa bảng ra ngoài ngữ cảnh bài.

Với một ảnh, Dice có thể tính từ IoU theo $2\operatorname{IoU}/(1+\operatorname{IoU})$. Nhưng không được áp dụng công thức này lên **IoU trung bình** để suy ra **Dice trung bình**, vì phép biến đổi phi tuyến không giao hoán với phép lấy trung bình.

Nếu prediction rỗng còn ground truth không rỗng thì IoU, Dice và recall bằng 0; precision không xác định do mẫu số bằng 0. Protocol cần quy định cách lưu và tổng hợp giá trị không xác định. Không nên bỏ toàn bộ ảnh này khỏi các metric còn lại, vì sẽ làm điểm tăng sai.

## 4. Ma trận giữa nội dung báo cáo và trạng thái triển khai

| Nội dung | Trạng thái trong repo | Nhận xét |
|---|---|---|
| Dựng COCO từ metadata và polygon | Có code | Có thử nghiệm nhỏ xác nhận một số hành vi; chưa chạy lại trên dữ liệu thật trong lượt review |
| Sampling prompt dương/âm | Có code | Mặc định một prompt cho train, ba prompt cho validation |
| Freeze trunk/text, cache text | Có code | Mô tả trong Method phù hợp |
| Composite loss và AdamW | Có code | Chưa có run xác minh hội tụ và đóng góp của từng thành phần |
| Validation polygon ở output resolution | Có code | Tên `val_recall` dễ gây nhầm |
| Native-frame PNG evaluator | Đề xuất | Chưa có entry point thực hiện |
| Quality-controlled manifest | Đề xuất | CSV lỗi có sẵn nhưng chưa được áp dụng vào training pipeline |
| Pretrained baseline | Đề xuất | Chưa có prediction và metric được lưu |
| Frozen/full fine-tune có kết quả | Đề xuất | Bảng đang `Pending` |
| U-Net, DeepLabv3+ | Đề xuất tùy chọn | Chưa được triển khai trong repo |
| Nhiều seed và khoảng bất định | Đề xuất | Chưa có kết quả |
| VRAM và latency | Đề xuất | Chưa có benchmark |
| PDF 7 trang | Có artifact, đã xem lại | Không phát hiện clipping/overlap rõ ở mức render đã kiểm tra |
| Build LaTeX trong môi trường hiện tại | Thất bại | Thiếu `balance.sty` |

## 5. Các điểm chưa hợp lý hoặc cần làm chặt hơn

Mức **cao** là vấn đề ảnh hưởng trực tiếp khả năng trả lời câu hỏi nghiên cứu hoặc chạy lại nghiên cứu. Mức **vừa** là thiếu sót có thể làm kết quả khó diễn giải hoặc khó truy vết. Mức **thấp** chủ yếu liên quan trình bày. Không phải mọi dòng dưới đây đều là lỗi logic trong văn bản; một số là khoảng trống thực nghiệm mà chính báo cáo đã thừa nhận.

| Mã | Mức | Phát hiện | Loại vấn đề |
|---|---|---|---|
| R01 | Cao | Chưa có kết quả trả lời ba câu hỏi nghiên cứu | Mức độ hoàn thiện |
| R02 | Cao | Supervision polygon có lỗi đã biết nhưng chưa được xử lý trong pipeline | Thiết kế dữ liệu |
| R03 | Cao | Chưa khóa manifest chống duplicate leakage | Giá trị đánh giá |
| R04 | Cao | Metric chọn checkpoint và metric cuối dùng reference/resolution khác nhau | Chọn mô hình |
| R05 | Cao | Nhãn một bệnh chưa đủ bảo đảm prompt âm thật sự vắng mặt | Giả định supervision |
| R06 | Vừa | So sánh dữ liệu gốc và đã kiểm soát có thể trộn nhiều yếu tố | Thiết kế ablation |
| R07 | Vừa | Baseline chưa được đặc tả đủ về đầu vào và tuning | Tính so sánh |
| R08 | Vừa | Mô tả loss chưa giải thích rõ semantic gating và query-count scaling | Độ đầy đủ của Method |
| R09 | Vừa | Từ “lesion” có thể rộng hơn đơn vị annotation thực | Định nghĩa nhiệm vụ |
| R10 | Cao | Revision và dữ liệu phụ thuộc chưa tái lập được từ checkout | Hạ tầng thực nghiệm |
| R11 | Vừa | Một số số liệu không có đầu ra evidence có cấu trúc | Truy vết bằng chứng |
| R12 | Vừa | Validation record chưa gắn với môi trường và artifact hash | Kiểm chứng bản build |
| R13 | Vừa | Lỗi accumulation và cache ảnh hưởng run dùng cho báo cáo | Độ tin cậy triển khai |
| R14 | Vừa | Giao thức bất định và lớp hiếm còn ở mức ý định | Thống kê |
| R15 | Thấp | Bibliography, figure chữ nhỏ và đường dẫn README cần hoàn thiện | Biên tập |

### R01. Chưa thể đánh giá thành công của phương pháp

**Căn cứ:** Introduction nêu ba câu hỏi thực nghiệm, còn [Results](sections/05_results.tex) để trống toàn bộ chất lượng mô hình và runtime.

Việc mô tả đúng code chứng minh tác giả đã đặc tả một cách tiếp cận; nó chưa cho biết mô hình có học được hoặc có lợi thế so với pretrained. Điều này không phải che giấu kết quả, vì bài đã ghi rõ `Pending`. Nhưng nếu mục đích là một bài báo kết quả hoàn chỉnh, đây là thiếu sót quyết định.

**Cách xử lý:** trước mắt định danh tài liệu là bản thảo phương pháp/giao thức. Để hoàn thiện phần thực nghiệm, cần pretrained baseline, ít nhất một run thích nghi đã kiểm tra, metric trên test cố định, prediction và bản ghi môi trường. Nếu chỉ đủ tài nguyên cho một seed, nêu rõ giới hạn; không đưa ra kết luận về độ ổn định giữa các lần train.

### R02. “Lọc COCO thành công” chưa giải quyết annotation sai

**Căn cứ:** [builder](../src/training.py) ở hàm `build_coco_split` sửa category và loại bản ghi không hợp cấu trúc, nhưng không sử dụng danh sách `coco_polygons_wrong`. [Discussion](sections/06_discussion.tex) đã nhận diện vấn đề này.

Đếm lại CSV đã lưu cho v7 cho thấy 180 ảnh có polygon được xác định thuộc ảnh khác: 163 train, 5 validation và 12 test. Ngoài ra có 6 ảnh train polygon sai nhưng chưa tìm được ảnh nguồn. Vì vậy, số ảnh có cờ `coco_polygons_wrong` là **186**, gồm **169 train**. Hai bộ số 180/163 và 186/169 có phạm vi khác nhau, không phải mâu thuẫn.

Nếu chỉ nói “loại 180 ảnh lỗi” mà không định nghĩa phạm vi, sáu ảnh chưa xác định nguồn có thể bị bỏ sót. Một policy dựa trên cờ CSV nên lấy đủ 186 ảnh hoặc nêu rõ lý do giữ từng nhóm. Nếu tái sinh instance từ PNG, cần ghi thuật toán connectivity, cách xử lý lỗ và vùng nhỏ, nguồn box/area và định nghĩa pseudo-instance.

**Cách xử lý:** tạo manifest từng ảnh với quyết định giữ/loại/sửa, lý do và nguồn ground truth. Bảng số lượng sau kiểm soát phải lấy từ manifest mới. Không gọi toàn bộ PNG là ground truth đã được tái xác minh nếu lượt xử lý chỉ đọc lại audit lịch sử.

### R03. Duplicate leakage cần được giải quyết bằng manifest cụ thể

**Căn cứ:** có thể tái lập các số 107 ảnh validation và 183 ảnh test có bản trùng trong train từ CSV hiện có, bằng cách giữ các thành viên v7 trong từng cluster rồi đếm theo split. Phép đếm cũng cho ra 601 cluster hoạt động chứa 1.294 ảnh, 295 cluster đi qua nhiều split và **37 cluster có cả validation lẫn test**.

Đếm thẳng mọi dòng duplicate có `in_v7=yes` cho ra 1.331 dòng, vì cluster gốc được định nghĩa trên v6; một số cluster chỉ còn một thành viên ở v7. Muốn tái lập 1.294 ảnh phải loại các cluster chỉ còn một thành viên. Đây là chi tiết nên ghi trong bằng chứng đi kèm.

Protocol đã nói cần xử lý cả train-evaluation và validation-test leakage. Tuy nhiên, chưa có quy tắc quyết định cụ thể cho 37 cluster đi qua hai tập đánh giá. Chỉ bỏ ảnh train trùng test chưa bảo đảm validation độc lập với test.

**Cách xử lý:** rà soát cluster ứng viên, gán mỗi cluster vào một partition theo quy tắc cố định hoặc loại có ghi nhận; công bố manifest và hỗ trợ lớp sau xử lý. Không lựa chọn cluster để loại dựa trên điểm mô hình. Kết quả dùng split gốc và kết quả theo policy mới phải có tên protocol và mẫu số riêng.

### R04. Metric chọn checkpoint chưa thống nhất với mục tiêu công bố

**Căn cứ:** `train()` chọn `best.pt` bằng `val_iou` từ polygon ở output resolution; Protocol đề xuất kết quả cuối trên PNG ở hệ tọa độ gốc và có disease-macro score.

Checkpoint tốt nhất theo polygon có thể ưu tiên sai hình học đã tồn tại trong polygon. Checkpoint tốt theo trung bình ảnh cũng chưa chắc tốt theo trung bình bệnh. Bản thảo đã phân biệt các metric, nhưng chưa quyết định dùng metric nào để chọn mô hình cuối cùng.

**Cách xử lý:** ưu tiên triển khai PNG evaluator cho validation trước run chính. Chốt metric chọn checkpoint, ví dụ image-mean PNG IoU hoặc disease-macro image IoU, cùng quy tắc chọn threshold trước khi xem test. Nếu vẫn dùng polygon proxy, phải công bố đó là tiêu chí chọn checkpoint và thừa nhận sự lệch mục tiêu. Không chọn lại checkpoint dựa trên test.

### R05. Cần kiểm tra giả định về prompt âm

**Căn cứ:** Method mặc định mọi disease category khác nhãn ảnh là âm. Audit ghi nhận duplicate có nhãn khác nhau, còn dự án chưa có annotation đa bệnh đầy đủ để xác minh vắng mặt.

Một bệnh không được ghi trong metadata có thể vẫn có biểu hiện trong ảnh hoặc chia sẻ triệu chứng khó phân biệt. Khi đó, suppression mạnh có thể làm mô hình học quy ước nhãn thay vì phân biệt khái niệm bệnh một cách đáng tin cậy.

**Cách xử lý:** giữ giới hạn “annotation-defined negatives”; kiểm tra một tập prompt âm bằng tay nếu có chuyên môn phù hợp; phân tích riêng âm cùng cây và khác cây; bổ sung ablation positive-only. Nếu có nhãn không chắc chắn, cần một policy riêng thay vì tự động coi là âm chắc chắn. Không suy ra hiệu quả phát hiện ảnh khỏe từ sampled-negative false-alarm rate.

### R06. Ablation về chất lượng dữ liệu còn dễ bị nhiễu

**Căn cứ:** Protocol đề xuất original versus quality-controlled supervision; cả ảnh train, annotation và có thể tập đánh giá đều thay đổi theo policy.

Nếu đồng thời bỏ duplicate, sửa polygon, giảm số ảnh train và đổi test set, chênh lệch điểm không cho biết yếu tố nào gây thay đổi. So raw IoU trước và sau trên hai bộ test khác nhau còn trộn cả độ khó của tập đánh giá.

**Cách xử lý:** xác định một tập PNG evaluation đã kiểm tra và giữ nó cố định cho so sánh nguyên nhân. Trên đó, thay lần lượt policy duplicate và policy annotation của train. Nếu cần đánh giá trên hai tập khác nhau, trình bày hai kết quả như hai protocol; có thể thêm phân tích trên phần giao của các ảnh đủ điều kiện, với điều kiện nêu rõ phạm vi kết luận.

Việc so mô hình train trên split bị rò rỉ với mô hình đã bỏ trùng có thể dùng để đo độ nhạy với leakage; không nên gọi chênh lệch đó là cải tiến thuật toán phân vùng.

### R07. Baseline cần có hợp đồng đầu vào và quy tắc tuning rõ

So pretrained SAM 3 với adapted SAM 3 bằng cùng ảnh, prompt, preprocessing và evaluator là so sánh chính hợp lý. Cần chốt thêm checkpoint gốc, score rule, cách đưa prediction về ảnh gốc và phạm vi validation dùng để chọn threshold.

Có thể chọn một ngưỡng chung hoặc cho mỗi mô hình chọn ngưỡng trên cùng validation theo cùng quy tắc; hai cách trả lời câu hỏi khác nhau và cần được tuyên bố trước. Khi so U-Net/DeepLabv3+ với SAM 3, phải ghi rõ mô hình nào được cung cấp tên bệnh. Có thể trình bày đây là hai hệ thống có đầu vào khác nhau, nhưng không quy toàn bộ chênh lệch cho kiến trúc.

Caching text trong adaptation chỉ hỗ trợ vocabulary đã mã hóa. Khả năng tổng quát hóa sang prompt mới hoặc bệnh chưa thấy chưa được đo; không nên gọi thực nghiệm hiện tại là benchmark open-vocabulary tổng quát.

### R08. Cần giải thích các điều kiện kích hoạt loss

**Căn cứ:** Method có ghi tắt semantic presence loss và chuẩn hóa, nhưng chưa nêu rõ tác động của `presence_head=True` và `scale_by_find_batch_size=True`.

Đối chiếu criterion upstream cho thấy semantic mask loss chỉ lấy trung bình trên query dương; wrapper có hệ số căn bậc hai số find query. Bởi vậy, tăng prompt count làm thay đổi cả thành phần âm/dương lẫn thang core loss. Cần giữ hoặc hiệu chỉnh yếu tố này khi nghiên cứu tác dụng của prompt count. Căn cứ: [semantic criterion][sam3-loss-fns] và [wrapper][sam3-loss-wrapper].

Không có căn cứ để kết luận `FilterEmptyTargets` xóa mọi negative query: tại revision đã ghim, bộ lọc này nhắm đến object có area gần 0, không trực tiếp chọn xóa query theo việc nó có target rỗng. Căn cứ: [filter implementation][sam3-filters]. Đây là điểm cần tránh phản biện sai chỉ dựa trên tên class.

### R09. Cần xác định “lesion” tương ứng đơn vị annotation nào

Audit mô tả có bệnh được chú thích từng đốm, có bệnh được chú thích trên cả vùng lá rộng. Vì vậy, “lesion” trong tiêu đề và công thức không bảo đảm luôn là một tổn thương vật lý độc lập.

**Cách xử lý:** bổ sung định nghĩa rằng đối tượng đánh giá chính là vùng bệnh theo annotation của PlantSeg. Nếu báo cáo instance AP về sau, phải xác nhận ý nghĩa và chất lượng instance. PNG semantic mask không tự cung cấp danh tính từng lesion; connected component chỉ tạo đơn vị vận hành theo một thuật toán đã chọn.

### R10. Checkout chưa đủ để tái lập nghiên cứu

**Căn cứ trực tiếp:** `src/sam3` là gitlink nhưng rỗng, không có `.gitmodules`; không có thư mục dữ liệu và manifest môi trường cấp dự án. Pipeline huấn luyện dùng CUDA cố định. Không chạy được thí nghiệm không chứng minh phương pháp sai, nhưng ngăn việc kiểm chứng các mô tả runtime.

Ngoài ra, collector dùng `git -C src/sam3 rev-parse HEAD` để ghi revision. Với thư mục con hiện không phải repository độc lập, Git tìm lên repo cha. Trong lượt rà soát, lệnh trả `683e53e382f4afeb8ade50fc7471be3758069d71` của dự án, trong khi gitlink SAM 3 là `8f0b7f4d4e7eda2ed606ebde6702c93359ad01da`.

**Cách xử lý:** cấu hình nguồn submodule và revision; kiểm tra repository root trước khi ghi commit; tách revision dự kiến từ gitlink và revision thực sự được cài. Bổ sung môi trường, hướng dẫn lấy dữ liệu, pretrained checkpoint/hash và entry point evaluator. Việc đọc upstream trong bản review này chỉ xác nhận một số ngữ nghĩa nguồn, không thay thế kiểm thử runtime.

### R11. Bằng chứng có hash nhưng chưa đủ để truy vết mọi phát biểu

Snapshot lưu hash training source, audit, CSV lỗi, metadata và JSON annotation. Nó chưa chứa hash/manifest cho toàn bộ file ảnh và PNG mask. Do đó, câu “dataset hashes” nên nói rõ các thành phần được băm; chưa thể coi đó là fingerprint đầy đủ của ảnh/mask dùng để chạy.

Results nêu **67 annotation train và 40 annotation test** v7 bị ghi đè category do khác metadata. Trong JSON snapshot không có trường lưu hai số này hoặc danh sách annotation bị ảnh hưởng. Collector chạy builder và nhận các bản in ra console, nhưng chỉ lưu số lượng cache. Với dataset hiện vắng mặt, không thể xác minh lại hai số từ snapshot. Đây là khoảng trống truy vết, chưa phải bằng chứng hai số sai.

Hai số 67/40 là số **annotation**, trong khi 12 dòng `mask_class_differs_from_metadata` v7 là số **ảnh** liên quan mask; chúng còn so sánh những nguồn nhãn khác nhau. Không được dùng trực tiếp để kết luận mâu thuẫn.

**Cách xử lý:** lưu `builder_stats` theo split, danh sách image/annotation ID bị thay nhãn, lý do loại và hash manifest ảnh/mask. Nên có script audit hoặc lệnh đầy đủ tái sinh các phát hiện pixel/duplicate, vì collector hiện chỉ làm structural check.

### R12. PDF đã có và validation record không bảo đảm build được trên máy này

**Căn cứ trực tiếp:** `pdfinfo` xác nhận PDF có 7 trang A4. Tôi đã render và đọc cả 7 trang; không thấy clipping/overlap rõ hoặc tham chiếu `??`. Tuy nhiên, build từ một bản sao tạm dừng với:

```text
! LaTeX Error: File `balance.sty' not found.
./main.tex:10:  ==> Fatal error occurred, no output PDF file produced!
```

Log có sẵn ở checkout cũng ghi lỗi này. [VALIDATION.md](VALIDATION.md) mô tả một lần build thành công trước đó; PDF ghi producer khác bản TeX hiện được dùng. Những thông tin này tương thích với khả năng PDF được tạo trong môi trường khác, chưa chứng minh bản ghi lịch sử sai.

**Cách xử lý:** bổ sung TeX distribution/version, dependency cụ thể, source revision, thời điểm và hash PDF cho mỗi lần xác nhận build. Cài đủ dependency rồi chạy lại quy trình nếu cần tái sinh PDF. `build.sh` hiện chỉ kiểm tra executable, chưa kiểm tra các package trước khi biên dịch. README nên dùng `cd reports` thay đường dẫn tuyệt đối của máy khác.

### R13. Một số lỗi triển khai cần xử lý trước run dùng cho bài

Ba vấn đề trong [training.py](../src/training.py) ảnh hưởng trực tiếp độ tin cậy thí nghiệm:

| Vấn đề | Hành vi đã kiểm tra | Hướng xử lý |
|---|---|---|
| Cache COCO cũ | Dựng cache từ dataset A, đổi sang B với cùng output và không rebuild thì vẫn nhận category của A | Lưu fingerprint nguồn và policy; từ chối cache không tương thích |
| Split không còn annotation | Builder ghi JSON rỗng rồi lỗi tại `max(per_image.values())` | Kiểm tra split rỗng trước khi ghi; trả thông báo rõ hoặc quy ước hỗ trợ tập rỗng |
| Accumulation và nonfinite loss | Code bỏ gradient cuối epoch; xóa gradient giữa nhóm nhưng không reset bộ đếm microbatch | Đếm nhóm hợp lệ, chuẩn hóa đúng nhóm dư và ghi số optimizer update thực |

Hai lỗi đầu đã được tái hiện bằng dữ liệu giả trong thư mục tạm. Lỗi accumulation được xác định từ logic vòng lặp, chưa chạy tensor/GPU trong lượt này.

Resume còn cần lưu/khôi phục RNG nếu muốn tiếp tục sát quá trình ban đầu, kiểm tra các trainable key bị thiếu và kiểm tra tương thích cấu hình. Snapshot mặc định hoặc chuỗi args trong checkpoint chưa đủ để bảo đảm một run được tiếp tục với cùng data policy và pretrained base.

Báo cáo đã ghi nhận nhóm dư bị bỏ, nhưng chỉ mô tả hành vi không làm run đáng tin hơn. Cần sửa hoặc giữ nguyên có lý do và ghi đúng số batch/step bị ảnh hưởng trước khi diễn giải kết quả.

### R14. Kế hoạch thống kê cần được chốt trước khi xem test

“Nhiều seed nếu khả thi” và “bootstrap theo cluster” là định hướng tốt nhưng chưa đủ để người khác tính lại kết quả. Cần chọn rõ metric chính, số seed, các seed, cách tổng hợp độ lệch giữa run, cách lấy mẫu cluster và mức khoảng tin cậy.

Không được đồng nhất biến thiên do random seed với bất định do tập ảnh test hữu hạn. Hai nguồn biến thiên trả lời hai câu hỏi khác nhau. Disease-macro metric cần công bố tập lớp có hỗ trợ, số ảnh mỗi lớp và policy khi lớp biến mất sau làm sạch. Trước làm sạch, v7 có một bệnh không có ảnh test, nên không mặc định chia macro cho 115.

Để so pretrained và adapted, nên dùng cùng ảnh/prompt và phân tích chênh lệch theo cặp. Nếu báo cáo nhóm tổn thương nhỏ hoặc lớp hiếm, chốt định nghĩa nhóm từ nhãn/manifest trước khi xem kết quả thay vì lựa chọn nhóm theo xu hướng điểm.

### R15. Các điểm biên tập và trình bày

Các nguồn nghiên cứu trong bibliography khớp tiêu đề, danh sách tác giả được ghi và năm preprint trên các trang gốc đã kiểm tra. Record [Zenodo v7](https://zenodo.org/records/17719108) cũng xác nhận version và ngày công bố 25/11/2025. Không phát hiện trích dẫn bịa trong phạm vi đối chiếu này. Việc trích arXiv không tự thân sai; khi hoàn thiện có thể chuẩn hóa metadata theo bản công bố được chọn và ghi revision nếu nội dung phụ thuộc phiên bản.

Figure 1 có chữ tên bệnh và coverage rất nhỏ khi thu vào trang. Nên vẽ lại nhãn đủ lớn hoặc chuyển thông tin sang caption/bảng. Table 1 và Table 2 bị float lên trước phần văn bản giải thích tương ứng; đây là hành vi LaTeX hợp lệ nhưng có thể làm người đọc ngắt mạch. Figure 3 chỉ là khung chờ và chỉ nên thay bằng prediction thực sự có provenance.

Author/affiliation còn trống đúng như README. Khi nộp bài cần điền theo yêu cầu môn học/venue. Cụm “clinically or visually salient” trong phần dataset nên đổi sang diễn đạt về ý nghĩa thị giác hoặc nông học, vì tài liệu không cung cấp đánh giá lâm sàng/chuyên gia cho độ quan trọng của từng vùng. Related Work nên được mở rộng nếu mục tiêu là tuyên bố đóng góp nghiên cứu.

## 6. Gợi ý sửa nội dung bằng các đoạn cụ thể

Các đoạn dưới đây là **đề xuất thay hoặc bổ sung**, chưa được ghi vào LaTeX. Chúng không đưa thêm kết quả chưa đo.

**Định danh bản thảo khi chưa có thí nghiệm:** có thể dùng tiêu đề *A Workflow and Evaluation Protocol for Disease-Conditioned PlantSeg Segmentation with SAM 3*. Tiêu đề hiện tại vẫn mô tả đúng nhiệm vụ, nhưng lựa chọn này giúp người đọc nhận biết mức độ hoàn thiện sớm hơn.

**Bổ sung định nghĩa annotation trong Method:**

> We evaluate disease regions as defined by the PlantSeg annotations. These regions may represent individual spots or broader symptomatic tissue. Connected components derived from semantic masks are treated as operational pseudo-instances, without assuming a one-to-one correspondence with biologically distinct lesions.

**Bổ sung phạm vi negative supervision:**

> Negative prompts denote categories absent from the image-level annotation. Their biological absence is not independently verified. With the configured semantic criterion, negative queries are excluded from semantic mask-loss averaging; suppression is learned through the applicable classification and presence objectives.

**Bổ sung quy tắc chọn mô hình sau khi evaluator được triển khai:**

> The primary validation criterion, checkpoint-selection rule, and threshold-selection procedure will be fixed before test evaluation. Native-frame PNG metrics and the current polygon-based validation score will be reported separately. The test set will not be used to select checkpoints or thresholds.

Đoạn này là cam kết giao thức, không khẳng định code hiện đã chọn checkpoint bằng PNG metric. Sau triển khai, thay phần tương lai bằng quy tắc và tên metric thực tế.

**Làm rõ phạm vi số ảnh lỗi:**

> The archived audit identifies 180 retained images with polygons attributed to another image and six further retained images with unexplained polygon-mask mismatch. A filtering policy based on the shared error flag must account for all 186 records and report the resulting split counts.

**Làm rõ evidence:**

> The structural snapshot fingerprints the training source, audit files, metadata, and annotation JSON files. Image and PNG-mask content manifests are not included in the current snapshot. Historical pixel-level findings are retained as audit evidence and were not recomputed in the present review.

**Giữ Results ở đúng trạng thái:** thay vì để người đọc phải suy ra từ bảng, có thể mở đầu phần định lượng bằng một câu trực tiếp: *No completed model run is currently available for quantitative comparison; the tables below specify the measurements required for the planned experiments.*

## 7. Thiết kế thí nghiệm tối thiểu để hoàn thiện báo cáo

Một kế hoạch vừa đủ để trả lời hai câu hỏi đầu là dựng cùng evaluator cho pretrained và adapted, xác minh pipeline trên một tập nhỏ rồi chạy cấu hình adaptation chính. Câu hỏi về chất lượng dữ liệu cần thêm đối chứng riêng.

| Thí nghiệm | Điều giữ cố định | Yếu tố thay đổi | Kết luận có thể hỗ trợ |
|---|---|---|---|
| E0: kiểm tra evaluator | Ảnh/mask nhân tạo có đáp án biết trước | Empty prediction, sai tọa độ, resize và orientation | Phép tính và mapping tọa độ làm đúng trong các trường hợp đã kiểm |
| E1: pretrained SAM 3 | Manifest, prompt, evaluator và tuning policy | Không cập nhật trọng số | Mốc hiệu năng của checkpoint nền |
| E2: frozen-encoder adaptation | Cùng evaluation manifest và quy tắc tuning với E1 | Cập nhật đúng phạm vi đã mô tả | Mức thay đổi do cấu hình adaptation này |
| E3: positive-only đối chiếu positive/negative | Cùng policy dữ liệu và ngân sách cập nhật được ghi rõ | Chiến lược prompt train | Độ nhạy với giám sát âm |
| E4: kiểm soát chất lượng supervision | Cùng tập evaluation đáng tin cậy | Policy nhãn train hoặc policy duplicate, thay từng yếu tố | Độ nhạy với chất lượng dữ liệu trong thiết kế đối chứng |
| E5: full adaptation, nếu đủ tài nguyên | Cùng giao thức và dữ liệu | Phạm vi tham số mở | Vai trò của việc cập nhật encoder trong cấu hình đã thử |

E0 là bước xác minh công cụ, không phải bằng chứng chất lượng mô hình. Overfit một tập nhỏ trước E2 giúp phát hiện lỗi learning pipeline, nhưng cũng không thay thế đánh giá tổng quát hóa. Với E3, thay tỷ lệ prompt dương còn thay số lần nhận mask supervision; cần ghi số update dương/âm và thiết kế ngân sách so sánh phù hợp trước khi diễn giải nguyên nhân.

Mỗi run nên lưu một cấu hình đã resolve, thay vì chỉ dựa vào mặc định trong bài. Bản ghi cần liên kết được checkpoint nền, source revision, manifest, vocabulary/prompt cụ thể, seed, môi trường, số update thành công, metric chọn checkpoint và artifact prediction. Với runtime, nêu latency có bao gồm text encoding, preprocessing và postprocessing không; cache text có được dùng không; cùng input size, batch, precision và cách đồng bộ GPU.

## 8. Tiêu chí nghiệm thu đề xuất

1. **Tái lập nguồn:** SAM 3 đúng revision, môi trường được ghi nhận, checkpoint nền có định danh và hash.
2. **Tái lập dữ liệu:** có manifest giữ/loại/sửa, số lượng mỗi split và lớp; giải thích 180 so với 186 ảnh polygon lỗi, cùng 1.560 so với 1.561 ảnh test.
3. **Căn chỉnh nhãn:** ảnh, PNG, polygon/box và metadata dùng cùng hệ tọa độ sau xử lý EXIF. PNG vốn đã theo frame hiển thị không bị xoay thêm một lần chỉ vì ảnh cần EXIF transpose.
4. **Độc lập split:** cluster được xử lý theo policy cố định, bao gồm các cluster đi qua validation và test.
5. **Huấn luyện đúng:** accumulation và cache được kiểm tra; số update thực và skipped batch được ghi lại; có kiểm tra hữu hạn loss/gradient trên môi trường chạy.
6. **Đánh giá đúng:** có native-frame PNG evaluator, empty-mask policy, metric tổng hợp và tập lớp macro rõ ràng.
7. **Chọn mô hình đúng:** checkpoint và threshold chỉ dùng validation, có quy tắc cố định trước test.
8. **Kết quả có căn cứ:** baseline và adaptation có log/prediction; bảng trong bài lấy từ artifact, không nhập số theo trí nhớ.
9. **Bản thảo nhất quán:** Method, config thực tế, bảng kết quả, caption và giới hạn nói về cùng một protocol.
10. **PDF build lại được:** đủ dependency, không lỗi tham chiếu, có hash của source và PDF được xác nhận.

## 9. Những gì đã kiểm tra trong lượt review này

| Kiểm tra | Kết quả | Giới hạn |
|---|---|---|
| Đọc toàn bộ LaTeX và tài liệu hỗ trợ trong `reports/` | Hoàn tất | Không có tác giả/affiliation hoặc kết quả mô hình để xác minh |
| Đối chiếu mặc định CLI với snapshot | Khớp các default literal được collector trích | Không xác nhận cấu hình của một run đã hoàn tất |
| Hash training source, audit Markdown và CSV lỗi | Khớp snapshot | Không kiểm lại hash dữ liệu ảnh do dữ liệu vắng mặt |
| So bản audit, CSV và montage được copy vào reports | Byte-identical với nguồn tương ứng trong docs | Không phải pixel re-audit |
| Đếm lại CSV duplicate | Khớp 107 validation, 183 test, 601 cluster/1.294 ảnh, 295 cluster nhiều split | Dùng cluster lịch sử, không tính lại perceptual hash |
| Đếm lại CSV polygon lỗi | 180 ảnh xác định nguồn + 6 ảnh chưa xác định nguồn | Không mở ảnh gốc để xác minh hình học |
| Đọc SAM 3 đúng pinned revision | Xác nhận resize vuông, loader không có EXIF transpose tại chỗ mở ảnh, ngữ nghĩa filter/loss được nêu | Chưa import/run mô hình trong môi trường hiện tại |
| Probe cache và split rỗng bằng hàm builder thật | Tái hiện được hai lỗi | Fixture chỉ kiểm file tồn tại và annotation; không dùng ảnh thật |
| Kiểm tra PDF | 7 trang A4; render và đọc đủ 7 trang, không thấy `??` | Chưa tái sinh được PDF từ môi trường hiện tại |
| Build từ bản sao thư mục reports trong vùng tạm | Thất bại do thiếu `balance.sty` | Không thay đổi dependency hoặc file LaTeX |

Các lệnh kiểm tra điển hình có thể chạy lại khi có môi trường phù hợp:

```bash
pdfinfo reports/main.pdf
pdftotext -layout reports/main.pdf /tmp/plantseg-report-text.txt
kpsewhich balance.sty
git ls-tree HEAD src/sam3
git -C src/sam3 rev-parse --show-toplevel HEAD
```

Lệnh Git cuối giúp phát hiện trường hợp revision được trả về từ repo cha. Để build, dùng `bash build.sh` trong một bản sao của `reports/` nếu muốn giữ nguyên PDF/log hiện tại. `collect_evidence.py` cần cả hai dataset local và SAM 3 hợp lệ để metadata revision có ý nghĩa; không nên coi việc chạy nó là đã tái kiểm toàn bộ audit.

Tại đầu lượt review đã có file `.DS_Store` chưa được theo dõi trong working tree. File đó không thuộc kết quả của công việc này. Thay đổi chủ động duy nhất của lượt này trong repo là tài liệu Markdown này.

## 10. Nguồn đối chiếu

Nguồn local chính là các file được liên kết ở mục 1, [mã huấn luyện](../src/training.py) và [audit dữ liệu](../docs/plantseg-v6-v7-statistics.md). Vị trí quan trọng trong phiên bản mã hiện tại: builder từ dòng 107; sampler từ dòng 186; transforms từ dòng 223; scope/cache từ dòng 348; loss từ dòng 462; evaluation từ dòng 511; checkpoint từ dòng 547; train loop từ dòng 631.

Các nguồn ngoài dưới đây là nguồn gốc được kiểm tra ngày 30/10/2026. Chúng hỗ trợ thông tin mô hình, công bố và ngữ nghĩa dependency; các số đếm đặc thù của checkout vẫn dựa vào bằng chứng local.

| Nguồn gốc | Phần đã đối chiếu |
|---|---|
| [PlantSeg, arXiv:2409.04038](https://arxiv.org/abs/2409.04038) | Tên bài, tác giả, năm và phạm vi dataset gốc |
| [PlantSeg v7, Zenodo record 17719108](https://zenodo.org/records/17719108) | Version, DOI và ngày công bố |
| [SAM 3, arXiv:2511.16719](https://arxiv.org/abs/2511.16719) | Tên bài, tác giả, concept prompting và presence mechanism |
| [U-Net, arXiv:1505.04597](https://arxiv.org/abs/1505.04597) | Thông tin thư mục và mô tả kiến trúc nền |
| [DeepLabv3+, arXiv:1802.02611](https://arxiv.org/abs/1802.02611) | Thông tin thư mục và mô tả kiến trúc nền |
| [SAM 3 transforms tại pinned revision][sam3-transforms] | Hành vi `square=True` |
| [SAM 3 dataset loader tại pinned revision][sam3-dataset] | Cách mở ảnh và chuyển RGB; cần kiểm tra EXIF riêng |
| [SAM 3 target filter tại pinned revision][sam3-filters] | Phân biệt lọc object area gần 0 và negative query |
| [SAM 3 loss functions tại pinned revision][sam3-loss-fns] | Semantic presence gating và tính mask loss |
| [SAM 3 loss wrapper tại pinned revision][sam3-loss-wrapper] | Hệ số query-count scaling |

[sam3-transforms]: https://github.com/facebookresearch/sam3/blob/8f0b7f4d4e7eda2ed606ebde6702c93359ad01da/sam3/train/transforms/basic_for_api.py
[sam3-dataset]: https://github.com/facebookresearch/sam3/blob/8f0b7f4d4e7eda2ed606ebde6702c93359ad01da/sam3/train/data/sam3_image_dataset.py
[sam3-filters]: https://github.com/facebookresearch/sam3/blob/8f0b7f4d4e7eda2ed606ebde6702c93359ad01da/sam3/train/transforms/filter_query_transforms.py
[sam3-loss-fns]: https://github.com/facebookresearch/sam3/blob/8f0b7f4d4e7eda2ed606ebde6702c93359ad01da/sam3/train/loss/loss_fns.py
[sam3-loss-wrapper]: https://github.com/facebookresearch/sam3/blob/8f0b7f4d4e7eda2ed606ebde6702c93359ad01da/sam3/train/loss/sam3_loss.py
