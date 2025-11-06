import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["GLOG_minloglevel"] = "3"
os.environ["FLAGS_minloglevel"] = "2"

import logging
logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("ppocr").propagate = False

import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter
from paddleocr import PaddleOCR

CAM_INDEX = 0
TARGET_W, TARGET_H = 1280, 720
ROI = (0.30, 0.72, 0.70, 0.92)
CONF_THRESHOLD = 1
MIN_CHAR_LEN = 2
VOTE_BUFFER = 5
VOTE_MATCH = 3
EXCEL_FILE = "paddle_results.xlsx"
SAVE_IMG = True
IMG_DIR = "captures"
os.makedirs(IMG_DIR, exist_ok=True)

print("⚙️ Loading PaddleOCR...")
ocr = PaddleOCR(lang="en", use_angle_cls=True, use_gpu=False, ocr_version="PP-OCRv3", show_log=False)
print("✅ OCR Ready\n")

def preprocess_emboss(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    contrast = clahe.apply(blur)
    sharpen = cv2.filter2D(contrast, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
    _, th = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = 255 - th
    th = cv2.resize(th, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

def save_excel_row(row):
    try:
        df_new = pd.DataFrame([row])
        if os.path.exists(EXCEL_FILE):
            df_existing = pd.read_excel(EXCEL_FILE)
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_final = df_new
        df_final.to_excel(EXCEL_FILE, index=False)
        print(f"📁 Saved to {EXCEL_FILE} ({len(df_final)} total entries)")
    except PermissionError:
        print("⚠️ Excel file is open! Please close it and try again.")
    except Exception as e:
        print(f"❌ Error while saving to Excel: {e}")

def draw_roi(frame, roi):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = int(W * roi[0]), int(H * roi[1]), int(W * roi[2]), int(H * roi[3])
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return x1, y1, x2, y2

frame_count = 0

def scan(roi):
    global frame_count
    frame_count += 1
    result = ocr.ocr(roi, cls=True)
    conf_values, texts = [], []

    if result:
        for block in result:
            if not block:
                continue
            for line in block:
                try:
                    txt = line[1][0].upper().strip()
                    conf = float(line[1][1]) * 100
                    if len(txt) >= MIN_CHAR_LEN and conf >= CONF_THRESHOLD:
                        texts.append(txt)
                        conf_values.append(conf)
                except:
                    pass

    if not texts:
        proc = preprocess_emboss(roi)
        cv2.imshow("Processed ROI", proc)
        result = ocr.ocr(proc, cls=True)
        if result:
            for block in result:
                if not block:
                    continue
                for line in block:
                    try:
                        txt = line[1][0].upper().strip()
                        conf = float(line[1][1]) * 100
                        if len(txt) >= MIN_CHAR_LEN and conf >= CONF_THRESHOLD:
                            texts.append(txt)
                            conf_values.append(conf)
                    except:
                        pass
    else:
        cv2.imshow("Processed ROI", roi)

    if texts:
        avg_conf = np.mean(conf_values)
        detected_text = " ".join(texts).strip()
        print(f"\n🎥 Frame #{frame_count}")
        print(f"✅ Detected Text: {detected_text}")
        print(f"🧩 Confidence: {avg_conf:.2f}%")
        print("-" * 40)

        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_excel_row({
            "Text": detected_text,
            "Confidence": f"{avg_conf:.2f}%",
            "Frame": frame_count,
            "Time": t,
            "Type": "Detection"   
        })

        return detected_text, avg_conf
    else:
        print(f"❌ Frame #{frame_count}: No text detected")
        return "", 0

def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(3, TARGET_W)
    cap.set(4, TARGET_H)
    if not cap.isOpened():
        print("❌ Camera not found")
        return

    print("🎥 Ready — press S to scan | Q to quit\n")
    mem, accepted = [], ""

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        x1, y1, x2, y2 = draw_roi(frame, ROI)
        cv2.putText(frame, f"Last: {accepted}" if accepted else "Press S to scan", (10, 40), 0, 1, (255, 255, 255), 2)
        cv2.imshow("PaddleOCR LIVE", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        if key == ord('s'):
            roi = frame[y1:y2, x1:x2].copy()
            text, conf = scan(roi)
            mem.append(text if text else None)
            if len(mem) > VOTE_BUFFER:
                mem.pop(0)
            if len(mem) >= VOTE_MATCH:
                best, c = Counter(mem).most_common(1)[0]
                if best and c >= VOTE_MATCH:
                    accepted = best
                    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{t}] ✅ FINAL: {best}")
                    
                   
                    save_excel_row({
                        "Text": best,
                        "Confidence": f"{conf:.2f}%",
                        "Frame": frame_count,
                        "Time": t,
                        "Votes": c,
                        "Type": "Final"
                    })
                    
                    if SAVE_IMG:
                        cv2.imwrite(f"{IMG_DIR}/{t.replace(':', '-')}.png", roi)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Done. Saved: {EXCEL_FILE}")

if __name__ == "__main__":
    main()
