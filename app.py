import os, io, uuid
from pathlib import Path
from typing import List
from flask import Flask, request, send_file, jsonify, abort
from werkzeug.utils import secure_filename

# PDF + imaging
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
import fitz  # PyMuPDF
from PyPDF2 import PdfReader, PdfWriter
import numpy as np
import cv2

try:
    import pytesseract
    TESSERACT_OK = True
except Exception:
    TESSERACT_OK = False

BASE_DIR = Path.cwd()
PUBLIC_DIR = BASE_DIR
UPLOAD_DIR = BASE_DIR / "_uploads"
CACHE_DIR = BASE_DIR / "_cache"
for d in (UPLOAD_DIR, CACHE_DIR):
    d.mkdir(exist_ok=True)

ALLOWED_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

app = Flask(__name__, static_folder=str(PUBLIC_DIR), static_url_path="/static")

# ---------------- Helpers -----------------
def _page_size_from_name(name: str):
    return LETTER if (name or "A4").upper() == "LETTER" else A4

def _clean_filename(name: str) -> str:
    name = secure_filename(name or "file")
    return name or "file"

def _rasterize_pdf_to_images(pdf_bytes: bytes, scale: float = 2.0) -> List[np.ndarray]:
    imgs = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            mat = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
            imgs.append(img[:, :, ::-1])
    return imgs

def _deskew_and_enhance(img_bgr: np.ndarray, do_deskew=True, do_threshold=True) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if do_deskew:
        thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thr > 0))
        if coords.size > 0:
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            angle = -(90 + angle) if angle < -45 else -angle
            (h, w) = img_bgr.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img_bgr = cv2.warpAffine(img_bgr, M, (w, h),
                                     flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if do_threshold:
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 35, 15)
        return cv2.medianBlur(th, 3)
    return gray

def _images_to_pdf_bytes(images: List[np.ndarray], ocr: bool=False) -> bytes:
    if ocr and TESSERACT_OK:
        writer = PdfWriter()
        for img in images:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if len(img.shape) == 2 \
                  else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pdf_bytes = pytesseract.image_to_pdf_or_hocr(rgb, extension='pdf')
            writer.append(PdfReader(io.BytesIO(pdf_bytes)))
        bio = io.BytesIO()
        writer.write(bio)
        bio.seek(0)
        return bio.read()
    else:
        bio = io.BytesIO()
        c = canvas.Canvas(bio, pagesize=A4)
        pw, ph = A4
        for img in images:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if len(img.shape) == 2 \
                  else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            margin = 36
            max_w, max_h = pw - 2*margin, ph - 2*margin
            h, w = rgb.shape[:2]
            scale = min(max_w / w, max_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img_resized = cv2.resize(rgb, (new_w, new_h))
            img_reader = ImageReader(img_resized)
            x = (pw - new_w) / 2
            y = (ph - new_h) / 2
            c.drawImage(img_reader, x, y, width=new_w, height=new_h)
            c.showPage()
        c.save()
        bio.seek(0)
        return bio.read()

# ---------------- Routes -----------------
@app.route("/")
def root():
    return send_file(PUBLIC_DIR / "index.html", mimetype="text/html")

@app.route("/app.js")
def app_js():
    return send_file(PUBLIC_DIR / "app.js", mimetype="application/javascript")

@app.post("/api/note-pdf")
def note_pdf():
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "My Note").strip()
    body = (data.get("body") or "").strip()
    if not body:
        return abort(400, "Empty body")
    bio = io.BytesIO()
    c = canvas.Canvas(bio, pagesize=_page_size_from_name(data.get("page_size")))
    pw, ph = _page_size_from_name(data.get("page_size"))
    y = ph - 60
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, y, title)
    y -= 28
    c.setFont("Helvetica", 12)
    for line in body.splitlines():
        c.drawString(50, y, line)
        y -= 16
    c.showPage()
    c.save()
    bio.seek(0)
    return send_file(bio, mimetype="application/pdf", as_attachment=True,
                     download_name=f"Note_{uuid.uuid4().hex[:6]}.pdf")

@app.post("/api/view/upload")
def view_upload():
    f = request.files.get("file")
    if not f: abort(400, "No file")
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXTS: abort(400, f"Bad type: {ext}")
    fid = uuid.uuid4().hex
    safe = _clean_filename(f.filename)
    out = UPLOAD_DIR / f"{fid}__{safe}"
    f.save(out)
    return jsonify({"file_id": fid, "filename": safe, "url": f"/api/view/{fid}"})

@app.get("/api/view/<fid>")
def view_get(fid):
    for p in UPLOAD_DIR.glob(f"{fid}__*"):
        ext = p.suffix.lower()
        mime = "application/pdf" if ext == ".pdf" else f"image/{ext.strip('.')}"
        return send_file(p, mimetype=mime)
    abort(404, "Not found")

@app.post("/api/scan")
def scan_endpoint():
    do_ocr = request.args.get("ocr", "0") == "1"
    do_deskew = request.args.get("deskew", "1") == "1"
    do_threshold = request.args.get("threshold", "1") == "1"
    files = request.files.getlist("files")
    if not files: abort(400, "No files")
    first_ext = Path(files[0].filename).suffix.lower()
    images = []
    if first_ext == ".pdf":
        if len(files) > 1: abort(400, "One PDF only")
        pdf_bytes = files[0].read()
        for p in _rasterize_pdf_to_images(pdf_bytes):
            images.append(_deskew_and_enhance(p, do_deskew, do_threshold))
    else:
        for f in files:
            data = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            images.append(_deskew_and_enhance(img, do_deskew, do_threshold))
    pdf_out = _images_to_pdf_bytes(images, ocr=do_ocr)
    return send_file(io.BytesIO(pdf_out), mimetype="application/pdf",
                     as_attachment=True, download_name=f"Scan_{uuid.uuid4().hex[:6]}.pdf")

@app.get("/healthz")
def health(): return {"ok": True}

if __name__ == "__main__":
    app.run(port=5000, debug=True)
