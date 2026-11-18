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

