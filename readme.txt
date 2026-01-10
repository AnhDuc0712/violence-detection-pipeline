pip install -r requirements.txt
// tải để sử dụng
python src/step1_detect_track.py
// chạy mẫu cắt bbox người
lệnh để chạy venv
.\venv\Scripts\activate
lệnh chạy model xgb ( lgb thì đổi lại chữ cuối)
python -m training.step4_prepare_train --model xgb

lệnh chạy step 3
python -m pipeline.step3_feature



# Violence Detection System (DTB)

## 📌 Giới thiệu
Dự án **Violence Detection System (DTB)** là một hệ thống ứng dụng trí tuệ nhân tạo nhằm
phát hiện và phân tích hành vi bạo lực trong video.  
Hệ thống được phát triển trong khuôn khổ **Thực tập tốt nghiệp**, sử dụng Python làm nền tảng chính.

---

## 🎯 Mục tiêu
- Phát hiện người và tư thế (pose) trong video bằng YOLOv8
- Theo dõi đối tượng xuyên suốt các khung hình (tracking)
- Cắt bounding box chi tiết phục vụ phân loại hành vi
- Phân loại hành vi **bạo lực / không bạo lực**
- Lưu trữ sự kiện và thông tin phân tích vào cơ sở dữ liệu

---

## 🏗️ Kiến trúc tổng quát
1. **Input**: Video / Camera
2. **Detect & Pose**: YOLOv8 Pose
3. **Tracking**: ByteTrack / BoT-SORT
4. **Feature Extraction**: Bounding box + keypoints
5. **Behavior Classification**: Violence / Non-violence
6. **Storage**: PostgreSQL + video bằng chứng

---

## 🧠 Công nghệ sử dụng
- **Python 3.11**
- **YOLOv8 (Ultralytics)** – phát hiện người & pose
- **BoxMOT** – tracking đa đối tượng
- **PyTorch** – deep learning framework
- **OpenCV** – xử lý video
- **NumPy / SciPy** – xử lý số liệu
- **PostgreSQL** – lưu trữ dữ liệu (psycopg2)

---

## ⚙️ Yêu cầu hệ thống
- Windows 10/11
- Python 3.11 (khuyến nghị)
- GPU NVIDIA + CUDA 12.1 (không bắt buộc, có thể chạy CPU)

---

## 🚀 Hướng dẫn cài đặt

### 1️⃣ Clone hoặc tải dự án
```bash
cd C:\Thực tập tốt nghiệp\dtb(demo)


