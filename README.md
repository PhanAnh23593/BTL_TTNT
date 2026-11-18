# BTL_AI — Phân loại lá cây bằng LeafCNN

Dự án Computer Vision dùng CNN viết bằng PyTorch để phân loại ảnh lá vào **5 loại cây**: `banana`, `guava`, `jackfruit`, `mango`, `neem`. FastAPI cung cấp API dự đoán và giao diện web để tải ảnh, xem nhãn dự đoán và điểm softmax của từng lớp.

**Phạm vi thực tế:** nhãn trong dataset và đầu ra model là loại cây. Dù giao diện ghi “AI-powered plant disease detection”, source hiện tại không có lớp bệnh, đầu ra chẩn đoán bệnh, phát hiện vị trí lá hoặc phân đoạn ảnh.

Repository có dataset đã chia tập, checkpoint và kết quả thực nghiệm. Tuy nhiên, có ảnh trùng giữa các tập và hai tệp kết quả không nhất quán; xem [Current Results](#current-results) trước khi sử dụng số liệu.

## Chạy nhanh

Clone repository:

```bash
git clone https://github.com/nddKhanh/BTL_AI.git
cd BTL_AI
```

Tạo và kích hoạt môi trường ảo trên **Windows PowerShell**:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Trên **Linux/macOS**:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Cài dependencies và chạy từ thư mục gốc repository:

```bash
python -m pip install -r requirements.txt
python app.py
```

Mở **http://127.0.0.1:8000**, chọn hoặc kéo thả ảnh, nhấn **Phân tích ảnh**. Có thể dùng ảnh sẵn có `dataset_split_2/test/banana/20210218_155926.jpg`. Không cần train lại để dùng checkpoint đi kèm.

Ứng dụng cần `best_model.pth`, các thư mục lớp trong `dataset_split_2/train/`, `static/` và `templates/`. Không xóa thư mục train khi chỉ chạy inference: ứng dụng lấy danh sách lớp từ thư mục này khi khởi động.

## Requirements và Installation

| Thành phần | Vai trò trong source |
|---|---|
| Python | Backend, chia dữ liệu, train và vẽ kết quả |
| `torch`, `torchvision` | CNN, training, đọc ảnh theo lớp và transforms |
| `pillow` | Đọc và chuyển ảnh sang RGB |
| `fastapi`, `uvicorn[standard]` | API và web server |
| `jinja2`, `python-multipart` | Render HTML và nhận upload |
| `numpy`, `tqdm`, `scikit-learn` | Lưu metrics, tiến trình training, confusion matrix |
| `pandas`, `matplotlib` | Đọc bảng metrics và vẽ biểu đồ |

`requirements.txt` chỉ khai báo 7 gói phục vụ ứng dụng web, không khóa phiên bản. Để train hoặc chạy `draw.py`, cài bổ sung các gói được import trong source:

```bash
python -m pip install numpy tqdm scikit-learn pandas matplotlib
```

Phiên bản Python, bộ phiên bản dependencies đã dùng khi train, phiên bản CUDA và cấu hình phần cứng huấn luyện: **Chưa xác định được từ repository hiện tại.** Không có lockfile, `pyproject.toml`, `environment.yml` hoặc `setup.py`.

Cả train và inference tự chọn CUDA nếu `torch.cuda.is_available()` trả về `True`, nếu không dùng CPU. GPU không bắt buộc theo code; repository không công bố yêu cầu RAM/VRAM hay thời gian chạy. Không cần camera hoặc cảm biến chuyên dụng để sử dụng ảnh có sẵn.

## Problem Statement và Input / Output

Đây là bài toán phân loại một nhãn cho mỗi ảnh. Model luôn chọn một trong 5 lớp, kể cả khi đầu vào không phải lá cây; chưa có lớp “unknown” hoặc cơ chế từ chối dự đoán.

| Chỉ số lớp | Nhãn | Ý nghĩa |
|---:|---|---|
| 0 | `banana` | Chuối |
| 1 | `guava` | Ổi |
| 2 | `jackfruit` | Mít |
| 3 | `mango` | Xoài |
| 4 | `neem` | Neem |

Training lấy nhãn qua `ImageFolder`; inference dùng tên thư mục train được sắp xếp theo thứ tự chữ cái. Thứ tự hiện tại cũng khớp `class_names` trong `confusion_matrix_data.npz`.

| Thành phần | Định dạng và nội dung |
|---|---|
| Dataset hiện có | Ảnh JPEG trong thư mục theo split và lớp |
| API input | Một file ảnh qua multipart field `file`; đuôi `png`, `jpg`, `jpeg`, `webp`, `bmp` |
| Tensor đầu vào | RGB, `N × 3 × 224 × 224`; API dùng `N = 1` |
| Đầu ra CNN | 5 logits cho mỗi ảnh |
| API output | JSON có `class`, `confidence`, `all_probs` |
| Checkpoint | `best_model.pth`, PyTorch `state_dict` |
| Lịch sử train | `metrics.npz`: loss và accuracy train/validation |
| Dữ liệu đánh giá | `confusion_matrix_data.npz`: `y_true`, `y_pred`, `class_names` |

`confidence` và các giá trị trong `all_probs` là softmax nhân 100, làm tròn đến hai chữ số thập phân. Chúng không phải accuracy của model; tổng các giá trị sau làm tròn có thể lệch nhẹ so với 100.

## Pipeline AI

```text
Dataset2/<class>/*                       [nguồn thô không có trong repo]
    → dataset.py: shuffle, copy theo tỷ lệ 70/15/15
    → train/val/test                     [script tạo "val"]

dataset_split_2/train + valid            [dữ liệu thực tế đã commit]
    → ImageFolder, đọc RGB
    → Resize + augmentation cho train + ToTensor + Normalize
    → LeafCNN → CrossEntropyLoss → Adam → StepLR
    → Validation sau mỗi epoch
    → Lưu best_model.pth nếu validation accuracy tăng
    → Lưu metrics; dự đoán validation bằng model cuối để lưu y_true/y_pred

Ảnh upload → kiểm tra đuôi → đọc RGB → Resize → ToTensor → Normalize
    → LeafCNN đã load checkpoint, eval + no_grad
    → Softmax → argmax → JSON → giao diện web
```

Tiền xử lý và học đặc trưng nằm trong luồng train/inference. Không có `preprocess.py`, `feature_extraction.py`, `predict.py` hoặc `evaluate.py` độc lập. Tập `test/` có sẵn nhưng script train hiện không đọc tập này.

## Dataset

Dataset được lưu trực tiếp trong Git tại `dataset_split_2/`, gồm **2.500 file ảnh JPEG RGB**. Khi kiểm tra bằng Pillow, cả 2.500 ảnh đều đọc được; kích thước gốc không đồng nhất.

| Lớp | Train | Validation (`valid`) | Test | Tổng |
|---|---:|---:|---:|---:|
| banana | 350 | 75 | 75 | 500 |
| guava | 350 | 75 | 75 | 500 |
| jackfruit | 350 | 75 | 75 | 500 |
| mango | 350 | 75 | 75 | 500 |
| neem | 350 | 75 | 75 | 500 |
| **Tổng** | **1.750** | **375** | **375** | **2.500** |

Tỷ lệ số file hiện tại là 70% / 15% / 15%. Đây là số file ảnh, không phải số ảnh độc nhất sau loại trùng. Sampling frequency không áp dụng cho ảnh tĩnh; tensor sử dụng 3 kênh RGB.

Nguồn xuất bản dataset, tác giả thu thập, public hay tự thu thập, URL tải dữ liệu gốc và giấy phép dữ liệu: **Chưa xác định được từ repository hiện tại.** Tên file có thể chứa tên bệnh nhưng nhãn huấn luyện chỉ lấy từ thư mục loại cây.

### Chia dữ liệu gốc bằng `dataset.py`

Đây là luồng tùy chọn khi có dữ liệu gốc; không cần chạy lại đối với dataset đã commit. Script đòi hỏi thư mục `Dataset2/` ở thư mục làm việc, với một thư mục con cho mỗi lớp. `Dataset2/` đang được `.gitignore` bỏ qua và không có trong checkout.

```bash
python dataset.py
```

Script dùng `random.seed(42)`, xáo trộn danh sách file trong từng lớp, lấy `int(n * 0.7)` cho train, `int(n * 0.15)` cho val và phần còn lại cho test, rồi sao chép bằng `shutil.copy2`. Biến `test_ratio` được khai báo nhưng không trực tiếp dùng để tính số file test.

**Cần xử lý trước khi chia mới:** script tạo `val/`, còn train đọc `valid/`. Script không xóa output cũ và không sắp xếp danh sách từ `os.listdir`; chạy lại trên output đang có có thể giữ lại ảnh từ lần chia trước. Nên dùng thư mục output mới bằng cách sửa `output_dir`, thống nhất tên `val`/`valid` trong script và đặt `base_dir` của training tương ứng. Không trộn output mới với dataset đã commit.

### Trùng dữ liệu giữa các tập

Kiểm tra SHA-256 toàn bộ nội dung file phát hiện **92 nhóm ảnh giống hệt theo byte xuất hiện ở nhiều split**, bao gồm **316 file** thuộc các nhóm đó.

| Hai tập có nội dung trùng | Số nhóm hash chung | Số file ở tập thứ hai có nội dung xuất hiện ở tập thứ nhất |
|---|---:|---:|
| train → valid | 49 | 62 |
| train → test | 51 | 58 |
| valid → test | 18 | 21 |

Các hàng có thể giao nhau, không cộng chúng thành số nhóm duy nhất. Hash không phát hiện ảnh gần giống hoặc ảnh khác góc của cùng một lá. Sự trùng lặp làm giảm tính độc lập của đánh giá; cần loại trùng và chia lại trước khi kết luận về khả năng tổng quát hóa.

## Data Preprocessing

| Bước | Training | Validation và inference |
|---|---|---|
| Đọc ảnh | `ImageFolder`, RGB | `ImageFolder` cho validation; Pillow `.convert("RGB")` cho upload |
| Resize | `224 × 224` | `224 × 224` |
| Lật ngang | `RandomHorizontalFlip()` | Không |
| Xoay | `RandomRotation(15)` | Không |
| Chuyển tensor | `ToTensor()` | `ToTensor()` |
| Normalize | Mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]` | Giống training |

Với ảnh RGB 8-bit, `ToTensor` đưa pixel về `[0, 1]`; sau đó mỗi kênh được chuẩn hóa theo `(x - mean) / std`. Resize ép ảnh về hình vuông, không có crop hoặc giữ tỷ lệ bằng padding. Không có bước tách nền hoặc định vị lá.

## Feature Extraction

Không có handcrafted feature extraction. Các lớp convolution trong `LeafCNN` học đặc trưng từ tensor ảnh đã tiền xử lý. Đặc trưng cuối trước bộ phân loại có kích thước `256 × 14 × 14`, được flatten thành 50.176 phần tử.

## Model Architecture

Chỉ có một kiến trúc: **LeafCNN**, CNN tự định nghĩa, khởi tạo để train từ đầu; script training không load pretrained backbone. Định nghĩa model được lặp lại trong `app.py` và `train_by_colab.py`.

```text
Input: 3 × 224 × 224
  → Conv2d(3, 32, kernel=3, padding=1)    → ReLU → MaxPool2d(2)
  → Conv2d(32, 64, kernel=3, padding=1)   → ReLU → MaxPool2d(2)
  → Conv2d(64, 128, kernel=3, padding=1)  → ReLU → MaxPool2d(2)
  → Conv2d(128, 256, kernel=3, padding=1) → ReLU → MaxPool2d(2)
  → Features: 256 × 14 × 14
  → Flatten: 50.176
  → Linear(50.176, 512) → ReLU → Dropout(0.5)
  → Linear(512, 5)
  → Logits
```

Training đưa logits vào `CrossEntropyLoss`. Inference áp dụng softmax rồi argmax; dropout được tắt bởi `model.eval()`. Không có baseline hoặc model đối chứng khác.

## Training

Script là `train_by_colab.py`, không nhận CLI arguments. Các tham số phải sửa trong source.

| Tham số | Giá trị hiện tại |
|---|---|
| Epochs | 20 |
| Batch size | 32, cho train và validation |
| Loss | `nn.CrossEntropyLoss()` |
| Optimizer | `optim.Adam(model.parameters(), lr=0.001)` |
| Scheduler | `StepLR(step_size=5, gamma=0.5)`, gọi sau mỗi epoch |
| Shuffle | Train `True`; validation `False` |
| DataLoader | `num_workers=4`, `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=2` |
| Chọn checkpoint | Lưu khi `val_acc > best_acc`, ban đầu `best_acc = 0.0` |
| Seed training | Không được thiết lập trong script |

### Training trên Colab theo đường dẫn hiện tại

1. Đưa project vào Google Drive tại `/content/drive/MyDrive/BTL_AI`, gồm `dataset_split_2/train/` và `dataset_split_2/valid/`.
2. Trong notebook Google Colab, mount Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

3. Chạy cell sau. Cần tạo `metrics/` vì script không tự tạo thư mục này:

```python
%cd /content/drive/MyDrive/BTL_AI
%pip install -r requirements.txt
%pip install numpy tqdm scikit-learn pandas matplotlib

from pathlib import Path
Path("metrics").mkdir(exist_ok=True)

!python train_by_colab.py
```

Script tự train, đánh giá validation và lưu kết quả. **Train lại sẽ ghi đè `best_model.pth` và các tệp cùng tên trong `metrics/`; sao lưu nếu cần giữ lần chạy cũ.**

### Training trên máy local

Không thể dùng nguyên script ở một đường dẫn local bất kỳ. Sau khi cài dependencies và kiểm tra dataset, sửa các đường dẫn trong `train_by_colab.py`:

| Vị trí | Giá trị local khi chạy từ thư mục gốc |
|---|---|
| `base_dir` | `"dataset_split_2"` |
| Đường dẫn trong `torch.save` | `"best_model.pth"` |
| Thư mục trong cả hai `np.savez` | `"metrics"` thay cho `/content/drive/MyDrive/BTL_AI/metrics` |

Trên Windows, training chạy ở cấp module và thiếu main guard cho DataLoader nhiều worker. Điều chỉnh tối thiểu: ở **cả hai** DataLoader, đặt `num_workers=0`, bỏ `persistent_workers=True` và `prefetch_factor=2`. Cách sửa cấu trúc lâu dài là đưa luồng training vào `main()` với `if __name__ == "__main__":`.

Sau khi chỉnh source:

```bash
python -c "from pathlib import Path; Path('metrics').mkdir(exist_ok=True)"
python train_by_colab.py
```

Đây là hướng dẫn thích nghi với local, **chưa được áp dụng vào source trong repository**. Không có bước preprocess hoặc extract feature riêng cần chạy trước training.

## Model Evaluation

Training tính accuracy train/validation và trung bình loss theo batch sau mỗi epoch. Loss là tổng `loss.item()` chia số batch, không phải trung bình có trọng số theo số ảnh của từng batch. Accuracy là số dự đoán đúng chia số ảnh.

Sau epoch cuối, script chạy lại validation và lưu nhãn thật/dự đoán. Nó **không load lại `best_model.pth`** trước bước này: theo implementation, confusion matrix dùng model cuối, còn checkpoint được chọn theo validation accuracy tốt nhất.

Không có đánh giá test độc lập, cross-validation, ROC-AUC hoặc tính precision/recall/F1 trong các script hiện tại. Không có command `evaluate.py`.

### Xem kết quả đã lưu

Từ thư mục gốc, sau khi cài dependencies bổ sung:

```bash
python draw.py
```

`draw.py` đọc hai `.npz` ở thư mục làm việc, hiển thị biểu đồ loss/accuracy, in 5 hàng metrics đầu, in confusion matrix và hiển thị hình ma trận. Script dùng `plt.show()`, không tự lưu PNG; tùy backend, cần đóng biểu đồ đầu để tiếp tục.

Kết quả train mới nằm trong `metrics/`. Muốn xem chúng bằng chính `draw.py`:

```bash
cd metrics
python ../draw.py
cd ..
```

Trong Colab có thể dùng `%cd` tới `metrics` rồi `%run ../draw.py` để chạy trong notebook. `draw.py` chỉ đọc kết quả đã lưu, không chạy lại model.

## Current Results

Các số dưới đây được đọc/tính từ artifacts đã commit, **không phải kết quả train hay inference mới khi viết README**.

### Lịch sử trong `metrics.npz`

Tệp có bốn mảng, mỗi mảng 20 phần tử: `train_losses`, `val_losses`, `train_acc`, `val_acc`. Biến `best_acc` có trong script nhưng không được ghi vào `.npz`.

| Mốc | Train accuracy | Validation accuracy | Train loss | Validation loss |
|---|---:|---:|---:|---:|
| Epoch 1 | 69,8857% | 83,4667% | 0,737574 | 0,343882 |
| Epoch 7, lần đầu đạt validation accuracy cao nhất | 98,4571% | 99,4667% | 0,052047 | 0,033182 |
| Epoch 20 | 99,7714% | 98,6667% | 0,010996 | 0,034953 |

Validation accuracy cao nhất là **99,4667%**, xuất hiện ở epoch 7, 14 và 18. Với điều kiện lưu `>` trong source, nếu tệp này thuộc cùng lần chạy thì checkpoint được giữ từ lần đạt mức tốt nhất đầu tiên. Chưa có metadata để xác nhận quan hệ giữa checkpoint và lần chạy này.

### Dữ liệu trong `confusion_matrix_data.npz`

Tệp lưu 375 cặp nhãn, 75 nhãn thật mỗi lớp. Ma trận tính từ các cặp nhãn, hàng là nhãn thật và cột là nhãn dự đoán:

| Thật / Dự đoán | banana | guava | jackfruit | mango | neem |
|---|---:|---:|---:|---:|---:|
| banana | 74 | 1 | 0 | 0 | 0 |
| guava | 0 | 75 | 0 | 0 | 0 |
| jackfruit | 0 | 0 | 75 | 0 | 0 |
| mango | 0 | 0 | 0 | 75 | 0 |
| neem | 0 | 0 | 0 | 0 | 75 |

Accuracy suy ra là **374/375 = 99,7333%**. Source tạo dữ liệu này từ validation, nhưng `.npz` không lưu đường dẫn ảnh, split identifier, run ID hoặc hash checkpoint để xác minh nguồn gốc độc lập.

**Không nhất quán:** 99,7333% không khớp validation accuracy cuối (98,6667%) và cao hơn mức cao nhất (99,4667%) trong `metrics.npz`. Không đủ bằng chứng để coi hai artifacts và checkpoint là cùng một lần chạy. Kết hợp với ảnh trùng giữa các split, đây chưa phải benchmark độc lập đáng tin cậy. Repository chưa có kết quả đánh giá chính thức trên tập test.

### Checkpoint và phạm vi kiểm chứng

`best_model.pth` có kích thước **104.331.701 byte**. Cấu trúc lưu trữ có các tên trọng số convolution/linear tương ứng LeafCNN. Ứng dụng có code load checkpoint, preprocess, dự đoán và trả kết quả cho web.

Trong lần rà soát tài liệu này đã kiểm tra cú pháp toàn bộ Python, đọc/kiểm tra ảnh, phân tích `.npz`, kiểm tra cấu trúc checkpoint và chạy `draw.py` với backend không mở cửa sổ. Môi trường rà soát chưa cài `torch`, `torchvision`, `fastapi`; **chưa kiểm chứng thực thi load checkpoint, endpoint hoặc giao diện bằng inference mới**, và chưa train lại. Không có output dự đoán mẫu được lưu sẵn trong repository.

## Inference / Prediction và Application Integration

Khởi động `python app.py`, sau đó dùng giao diện hoặc gửi HTTP request. Hai route được định nghĩa trực tiếp trong `app.py`:

| Route | Chức năng |
|---|---|
| `GET /` | Render `templates/index.html` cùng danh sách lớp |
| `POST /predict` | Nhận ảnh ở field `file`, trả dự đoán JSON |

### Ví dụ hoàn chỉnh với ảnh có trong repository

Giữ server đang chạy, mở terminal khác tại thư mục gốc. Trên Windows PowerShell:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" -F "file=@dataset_split_2/test/banana/20210218_155926.jpg"
```

Trên Linux/macOS:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@dataset_split_2/test/banana/20210218_155926.jpg"
```

```text
20210218_155926.jpg
    → RGB → 224 × 224 → ToTensor → Normalize
    → best_model.pth / LeafCNN
    → 5 logits → softmax → chọn lớp điểm cao nhất
    → JSON: class + confidence + all_probs
```

Schema đầu ra dưới đây không phải số liệu dự đoán thực tế:

```text
class: một trong banana, guava, jackfruit, mango, neem
confidence: điểm softmax của lớp được chọn, theo %
all_probs: ánh xạ đủ 5 tên lớp sang điểm softmax theo %
```

Không thể khẳng định nhãn dự đoán từ tên thư mục của ảnh. Output lỗi xác định được từ source là HTTP 400 với `{"error":"Invalid file type"}` nếu đuôi file không được hỗ trợ. Lỗi giải mã ảnh hoặc lỗi trong khối xử lý trả HTTP 500 kèm `error`.

JavaScript gửi `FormData` đến `/predict`, hiển thị lớp cao nhất, điểm số và các thanh xác suất giảm dần. `CLASS_ICONS` ánh xạ nhãn sang biểu tượng cây/quả. Ngưỡng 70% và 40% chỉ đổi màu, không từ chối dự đoán thấp. Không có điều khiển phần cứng, camera stream hoặc xử lý real-time liên tục.

## Saved Models / Checkpoints và Configuration

Ứng dụng dựng `LeafCNN(NUM_CLASSES)`, gọi `torch.load(MODEL_PATH, map_location=device)`, nạp `load_state_dict`, chuyển model sang device và gọi `eval()`. Checkpoint là `state_dict`, không chứa optimizer, scheduler, danh sách lớp hoặc trạng thái resume đầy đủ.

| Cấu hình | Vị trí và giá trị |
|---|---|
| Model inference | `app.py`: `best_model.pth`, tương đối với vị trí file Python |
| Danh sách lớp | `app.py`: thư mục trong `dataset_split_2/train`, sắp xếp chữ cái |
| Static/templates | `app.py`: `static`, `templates`, tương đối với thư mục làm việc |
| Server phát triển | `app.py`: `127.0.0.1:8000`, `reload=True` |
| Dữ liệu train | `/content/drive/MyDrive/BTL_AI/dataset_split_2` |
| Model train xuất ra | `/content/drive/MyDrive/BTL_AI/best_model.pth` |
| Metrics train xuất ra | `/content/drive/MyDrive/BTL_AI/metrics/` |
| Chia tập | `dataset.py`: `Dataset2` → `dataset_split_2` |

Không có YAML, JSON config hoặc `.env.example`; các giá trị nằm trực tiếp trong Python. Đổi tên/thêm/xóa thư mục lớp có thể sai mapping hoặc làm lớp cuối không khớp checkpoint. Giữ nguyên 5 thư mục và thứ tự lớp khi dùng model đi kèm.

