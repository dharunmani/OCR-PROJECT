# 🧠 Industrial Text Recognition System using PaddleOCR

### 🎯 Objective
A real-time, offline OCR system designed to read **engraved or embossed text on metallic surfaces** (such as serial numbers on machine parts).

---

## ⚙️ Features
- Uses **PaddleOCR (PP-OCRv3)** for accurate text recognition  
- Works on **metal surfaces** with glare and low contrast  
- **CLAHE + Sharpening + Thresholding** preprocessing  
- **Voting mechanism** ensures stable, consistent results  
- Saves results to **Excel** and captures ROI images  
- Works completely **offline (CPU only)**  

---

## 🧩 Libraries Used
- `opencv-python`
- `paddleocr`
- `numpy`
- `pandas`

---

## 🧠 Working Flow
1. Captures live video from camera  
2. Defines ROI area for OCR detection  
3. Applies preprocessing (CLAHE, thresholding, sharpening)  
4. Runs PaddleOCR for recognition  
5. Uses a voting mechanism to confirm stable text  
6. Logs all results in Excel (`paddle_results.xlsx`)  

---

## 🔧 Run the Project
```bash
pip install -r requirements.txt
python main.py
