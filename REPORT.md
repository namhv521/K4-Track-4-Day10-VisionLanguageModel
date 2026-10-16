# Báo cáo Lab Day 10

Họ tên: **Hoàng Văn Nam**
Mã học viên: **2A202602853**
`dataset_version`: `d10-21e4070301`

## Bảng tổng hợp

| Mã | Tham số train | VRAM GB | Phút train | F1 | CI95 | FP/ảnh | pothole | crack | manhole |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| e0 | 0 | 3.277 | 0.00 | 0.0638 | [0.0423, 0.0870] | 1.0514 | 0.0417 | 0.0130 | 0.1633 |
| e1 | 33030144 | 5.072 | 64.38 | 0.2835 | [0.2478, 0.3198] | 1.0064 | 0.1000 | 0.2147 | 0.5615 |
| e2 | 142135296 | 6.730 | 58.83 | 0.3469 | [0.3098, 0.3861] | 0.9904 | 0.1530 | 0.2913 | 0.5915 |
| e3 | 148426752 | 6.867 | 75.48 | 0.4384 | [0.3960, 0.4811] | 0.8167 | 0.3077 | 0.3536 | 0.6867 |
| e4_data_size_0.1 | 148426752 | 6.751 | 63.77 | 0.2816 | [0.2498, 0.3174] | 1.5981 | 0.1533 | 0.2119 | 0.5646 |
| e4_data_size_1.0 | 148426752 | 6.867 | 75.48 | 0.4384 | [0.3960, 0.4811] | 0.8167 | 0.3077 | 0.3536 | 0.6867 |

E0 không huấn luyện; thời gian train bằng 0. Các số chỉ xuất hiện khi có evidence thật.
Batch train 1, gradient accumulation 8, effective batch 8; NF4 trên LLM, visual và lm_head không lượng tử hóa.
Không dùng tập test để chọn setup: chọn bằng 40 ảnh val cố định; tập con nhỏ giới hạn độ tin cậy.

## 10 câu hỏi

### Câu 1 — Kiến trúc và khối được cập nhật
Số tham số dense đo từ model thật (khôi phục kích thước Params4bit): `{'visual': 415347712, 'projector': 109105152, 'language_and_head': 4022468096}`. Visual bao gồm projector; không cộng projector lần thứ hai.
E1: LoRA attention/MLP của LLM. E2: E1 và toàn bộ merger/DeepStack projector. E3: E2 và LoRA attention/MLP của ViT.

### Câu 2 — E0
parse_valid_rate = 0.6527; JSON hỏng: 108/311 ảnh.
Ví dụ output nguyên bản:
```json
[{"label":"crack","bbox_2d": [287, 477, 300, 484], "confidence": 0.7}, {"label":"crack","bbox_2d": [300, 477, 312, 484], "confidence": 0.7}, {"label":"crack","bbox_2d": [312, 477, 324, 484], "confidence": 0.7}, {"label":"crack","bbox_2d": [324, 477, 336, 484], "confidence": 0.7}, {"label":"crack","bbox_2d": [336, 477, 348, 484], "confidence": 0.7}, {"label":"crack","bbox_2d": [348, 477, 360, 484], "confidence": 0.7}, {"label":"crack","bbox_2d": [360, 477,
```
FP/ảnh, recall và ảnh minh họa cho biết mức bịa/thừa box và bỏ sót. Không khẳng định sai thang tọa độ chỉ từ F1 thấp.

Hai nhóm lỗi đo bằng scorer: bỏ sót FN=553, dự đoán không khớp FP=327; TP=30. FP có thể do sai vị trí hoặc sai lớp, không đồng nhất mọi FP với hallucination.

### Câu 3 — Thang tọa độ
Cùng raw E0, chấm norm1000: F1=0.0638; cố ý chấm pixel: F1=0.0000. Đây là diagnostic, không phải evidence setup mới.
Thang 0–1000 biểu diễn vị trí tương đối; với ảnh 640×360, pixel x = x_norm×640/1000 và pixel y = y_norm×360/1000. Dùng sai thang làm lệch box và IoU.

### Câu 4 — Mask loss
Mask prompt, token ảnh và padding bằng -100. Chỉ tối ưu JSON đáp án và token kết thúc. Kiểm tra prompt là tiền tố của full chat, 220 image tokens và giải mã phần supervised. Nếu học prompt/ảnh, mục tiêu bị pha loãng bởi token không phải đầu ra cần dự đoán.

### Câu 5 — Projector và lượng tử hóa
Projector train đầy đủ phải có tham số float32 cho optimizer/gradient ổn định trong AMP fp16. Forward chạy autocast fp16. Đây là thiết lập của lab, không phải mọi bài toán đều bắt buộc float32. NF4 Params4bit không hỗ trợ cập nhật dense như tham số float; bỏ visual và lm_head khỏi lượng tử hóa. Projector được lưu/nạp riêng ngoài PEFT adapter.

Đã kiểm chứng Transformers 5.19.0 khớp prefix đầy đủ: dùng llm_int8_skip_modules=['model.visual','lm_head']; tên ngắn 'visual' không giữ được ViT/projector dạng float.

### Câu 6 — Split theo phiên quay
Frame liền kề có nội dung gần trùng. Chia ngẫu nhiên có thể đưa cùng phiên vào train/test, làm đánh giá lạc quan. Kiểm tra giao tập session bằng rỗng; không đổi split của đề.
```json
{
  "train": {
    "images": 1422,
    "boxes": {
      "pothole": 1074,
      "crack": 1750,
      "manhole": 691
    }
  },
  "val": {
    "images": 276,
    "boxes": {
      "pothole": 66,
      "crack": 470,
      "manhole": 103
    }
  },
  "test": {
    "images": 311,
    "boxes": {
      "pothole": 121,
      "crack": 299,
      "manhole": 163
    }
  }
}
```

### Câu 7 — So sánh E1/E2/E3
Đối chiếu bảng trên theo F1, CI95, VRAM, thời gian; chỉ kết luận từ setup đã chạy đủ. Khoảng tin cậy từng setup không phải kiểm định paired difference.

E2 so với E1: ΔF1=+0.0634; ΔVRAM=+1.659 GB; Δthời gian train=-5.55 phút. CI95 chồng lấn; đây không phải kiểm định paired difference.

E3 so với E2: ΔF1=+0.0915; ΔVRAM=+0.136 GB; Δthời gian train=+16.65 phút. CI95 không chồng lấn; đây không phải kiểm định paired difference.

Setup chọn trên 40 ảnh val cố định: E3; F1 val: {'E1': 0.2696629213483146, 'E2': 0.35294117647058826, 'E3': 0.4777777777777778}. Lựa chọn này không sử dụng test.

### Câu 8 — Test nhỏ và khoảng tin cậy
Scorer bootstrap theo ảnh, 1.000 lần. Độ rộng CI được thể hiện trong bảng. Frame còn tương quan trong phiên nên CI theo ảnh có thể lạc quan; chưa có bootstrap theo session. Chênh lệch nhỏ không đủ chứng minh setup tốt hơn.

Độ rộng CI95 thật: e0=0.0447, e1=0.0721, e2=0.0762, e3=0.0852.

### Câu 9 — Phân tích lỗi theo lớp
F1 từng lớp trong bảng. Box nhỏ nhạy với sai tọa độ; số lượng lớp khác nhau giữa train/val/test. Nguyên nhân này là giả thuyết, cần đối chiếu ảnh thay vì coi là kết luận đã chứng minh.

Ở e3, lớp F1 thấp nhất là pothole: F1=0.3077, TP=30, FP=44, FN=91. Xem sáu ảnh có TP và/hoặc nhiều lỗi dưới đây; nguyên nhân cần học viên đối chiếu ảnh trước khi kết luận.

### Câu 10 — E4
Chọn data_size: 10% và 100% trên cùng setup tốt nhất theo val. Giữ seed=42, rank=16, optimizer, prompt, độ phân giải, max_steps=300, effective batch=8. 100% dùng lại baseline đúng cấu hình; 10% lấy prefix sau shuffle seed=42. Với bước tối ưu cố định, tập nhỏ được lặp nhiều lần hơn; đây là một giới hạn diễn giải.

10% so với 100%: ΔF1=-0.1568; ΔVRAM=-0.115 GB; Δthời gian train=-11.71 phút. CI95 không chồng lấn; đây không phải kiểm định paired difference. Số ảnh train: 142 và 1422.

## Ảnh minh họa

Box xanh: nhãn thật. Box đỏ: dự đoán. Ảnh được lấy theo chỉ số cố định, không chỉ chọn ảnh đẹp.

![e1_1](report/images/e1_1.jpg)

![e1_2](report/images/e1_2.jpg)

![e1_3](report/images/e1_3.jpg)

![e1_4](report/images/e1_4.jpg)

![e1_5](report/images/e1_5.jpg)

![e1_6](report/images/e1_6.jpg)

![e2_1](report/images/e2_1.jpg)

![e2_2](report/images/e2_2.jpg)

![e2_3](report/images/e2_3.jpg)

![e2_4](report/images/e2_4.jpg)

![e2_5](report/images/e2_5.jpg)

![e2_6](report/images/e2_6.jpg)

![e3_1](report/images/e3_1.jpg)

![e3_2](report/images/e3_2.jpg)

![e3_3](report/images/e3_3.jpg)

![e3_4](report/images/e3_4.jpg)

![e3_5](report/images/e3_5.jpg)

![e3_6](report/images/e3_6.jpg)

## Nhìn lại
Thiếu evidence: không. E5 GRPO không thực hiện (tùy chọn).
Test/inference/training thật mới chứng minh kết quả; các self-check không thay thế thí nghiệm.

## Sáu ảnh đối chiếu có số TP/FP/FN

Chọn tối đa ba ảnh có TP với F1 cao, các ảnh còn lại có nhiều FP+FN; đây là minh họa có chủ đích, không dùng để ước lượng chất lượng tổng thể.

![vlcsnap-00204](report/examples/e3_1.jpg)

TP=1, FP=0, FN=0, F1=1.0000, JSON hợp lệ=True.

![vlcsnap-00206](report/examples/e3_2.jpg)

TP=1, FP=0, FN=0, F1=1.0000, JSON hợp lệ=True.

![vlcsnap-00208](report/examples/e3_3.jpg)

TP=1, FP=0, FN=0, F1=1.0000, JSON hợp lệ=True.

![vlcsnap-2025-02-18-23h30m29s649](report/examples/e3_4.jpg)

TP=0, FP=6, FN=3, F1=0.0000, JSON hợp lệ=True.

![vlcsnap-2025-02-19-17h03m06s413](report/examples/e3_5.jpg)

TP=1, FP=5, FN=4, F1=0.1818, JSON hợp lệ=True.

![vlcsnap-2025-02-18-23h16m47s802](report/examples/e3_6.jpg)

TP=0, FP=5, FN=3, F1=0.0000, JSON hợp lệ=True.
