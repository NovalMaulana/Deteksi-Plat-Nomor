import os
import time
import re
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
from ultralytics import YOLO
import easyocr
import numpy as np
import cv2

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Konfigurasi folder
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Pastikan folder untuk upload dan hasil ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Memuat model YOLO
try:
    model = YOLO('best.pt')
    print("Model 'best.pt' berhasil dimuat.")
except Exception as e:
    print(f"Gagal memuat model YOLO: {e}")
    model = None

# Memuat model EasyOCR
try:
    reader = easyocr.Reader(['en'])
    print("Model EasyOCR berhasil dimuat.")
except Exception as e:
    print(f"Gagal memuat model EasyOCR: {e}")
    reader = None

# Fungsi untuk memeriksa ekstensi file
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_for_ocr(pil_image, base_path):
    """
    Memproses gambar untuk OCR dan menyimpan setiap langkah preprocessing
    """
    img_np = np.array(pil_image)
    
    # Langkah 1: Resize
    h, w = img_np.shape[:2]
    new_h = 100
    new_w = int(w * (new_h / h))
    resized_img = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f"{base_path}_resized.jpg", resized_img)
    
    # Langkah 2: Grayscale
    gray = cv2.cvtColor(resized_img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(f"{base_path}_grayscale.jpg", gray)
    
    # Langkah 3: Blur
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    cv2.imwrite(f"{base_path}_blurred.jpg", blurred)
    
    # Langkah 4: Threshold
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    cv2.imwrite(f"{base_path}_threshold.jpg", thresh)
    
    # Langkah 5: Kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cv2.imwrite(f"{base_path}_kernel.jpg", kernel * 255)  # Skala untuk visualisasi
    
    # Langkah 6: Morphology (Close)
    closed_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(f"{base_path}_closed.jpg", closed_img)
    
    return closed_img

# --- FUNGSI EKSTRAK YANG SANGAT DIPERBAIKI ---
def extract_plate_from_text(text):
    """
    Mencari semua substring yang berpotensi merupakan plat nomor,
    lalu memilih yang terpanjang sebagai kandidat terbaik.
    TANPA koreksi karakter untuk menghindari kesalahan.
    """
    if not text:
        return ""
    
    # 1. Hapus semua karakter yang BUKAN huruf atau angka, lalu ubah ke huruf besar
    text = re.sub(r'[^A-Za-z0-9]', '', text).upper()
    
    # 2. Cari SEMUA substring yang cocok dengan format plat nomor
    all_matches = re.findall(r'[A-Z]{1,2}\d{1,4}[A-Z]{1,3}', text)
    
    if not all_matches:
        print("--- Tidak ada format plat yang ditemukan ---")
        return ""
        
    print(f"Semua kandidat plat yang ditemukan: {all_matches}")

    # 3. Dari semua kandidat, pilih yang terpanjang.
    best_match = max(all_matches, key=len)
    
    print(f"--- Kandidat terbaik (terpanjang) dipilih: '{best_match}' ---")
    return best_match

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('index'))
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    if model is None or reader is None:
        return "Model tidak dapat dimuat.", 500

    results = model(filepath)
    detection_results = []
    
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                img = Image.open(filepath)
                cropped_img = img.crop((x1, y1, x2, y2))
                
                base_filename = f"result_{int(time.time())}_{len(detection_results)}"
                base_path = os.path.join(app.config['RESULT_FOLDER'], base_filename)
                
                original_filename = f"{base_filename}_original.jpg"
                original_path = os.path.join(app.config['RESULT_FOLDER'], original_filename)
                cropped_img.save(original_path)

                # Proses gambar dan simpan setiap langkah
                processed_img = preprocess_for_ocr(cropped_img, base_path)
                
                ocr_results = reader.readtext(processed_img, detail=0)
                
                detected_text = "Tidak dapat membaca"
                combined_raw_text = ""
                
                if ocr_results:
                    combined_raw_text = "".join(ocr_results)
                    print(f"OCR Mentah Digabung: '{combined_raw_text}'")

                    # Panggil fungsi ekstraksi yang baru
                    detected_text = extract_plate_from_text(combined_raw_text)
                    
                    if not detected_text:
                        detected_text = "Tidak dapat membaca"
                
                # Kumpulkan semua nama file preprocessing
                preprocessing_files = {
                    'resized': f"{base_filename}_resized.jpg",
                    'grayscale': f"{base_filename}_grayscale.jpg",
                    'blurred': f"{base_filename}_blurred.jpg",
                    'threshold': f"{base_filename}_threshold.jpg",
                    'closed': f"{base_filename}_closed.jpg"
                }
                
                detection_results.append({
                    'original_filename': original_filename,
                    'preprocessing_files': preprocessing_files,
                    'text': detected_text,
                    'raw_text': combined_raw_text
                })

    return render_template('result.html', 
                           original_image=filename, 
                           detection_results=detection_results)

if __name__ == '__main__':
    app.run(debug=True)