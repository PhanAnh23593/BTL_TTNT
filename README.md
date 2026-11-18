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

