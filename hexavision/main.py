import sys, os, math, cmath, time, json, base64, zipfile, datetime
from collections import defaultdict

# ── Data Matrix decoder (distutils_patch must be imported first) ──
import types as _types
sys.modules.setdefault('distutils', _types.ModuleType('distutils'))
sys.modules.setdefault('distutils.version', _types.ModuleType('distutils.version'))
try:
    from setuptools._distutils.version import LooseVersion as _LooseVersion
    sys.modules['distutils.version'].LooseVersion = _LooseVersion
except Exception:
    pass
try:
    from pylibdmtx.pylibdmtx import decode as _dmtx_decode
    _DMTX_AVAILABLE = True
except Exception:
    _dmtx_decode = None
    _DMTX_AVAILABLE = False
import cv2
import numpy as np
from itertools import permutations
from inference import get_model

# reportlab — PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, HRFlowable)

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog,
    QMessageBox, QSizePolicy, QProgressBar, QFrame, QScrollArea,
    QSplitter, QTextEdit, QStackedWidget, QGraphicsDropShadowEffect,
    QSpacerItem, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView
)
from PyQt5.QtGui  import (QPixmap, QImage, QPalette, QColor, QFont,
                           QPainter, QLinearGradient, QBrush, QPen,
                           QFontDatabase, QIcon, QFontMetrics)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve, QSize


# ══════════════════════════════════════════════════════════════
# LIGHT THEME — Enterprise SaaS Dashboard
# ══════════════════════════════════════════════════════════════
T = {
    # Content area
    'bg':         '#f4f4f5',      # Very light grey background
    'surface':    '#ffffff',      # Pure white cards
    'surface2':   '#f4f4f5',      # Nested element bg
    'border':     '#e4e4e7',      # Subtle border
    'accent':     '#09090b',      # Near-black primary accent / buttons
    'accent_bg':  '#f4f4f5',      # Light hover hint
    'accent2':    '#3f3f46',      # Medium dark grey
    'pass_color': '#16a34a',      # Green
    'pass_bg':    '#f0fdf4',
    'fail_color': '#dc2626',      # Red
    'fail_bg':    '#fef2f2',
    'warn':       '#d97706',      # Amber
    'warn_bg':    '#fffbeb',
    'text':       '#09090b',      # Near-black text
    'muted':      '#71717a',      # Medium grey muted
    # Sidebar (dark / near-black)
    'sidebar_bg':          '#09090b',   # Near-black
    'sidebar_hover':       '#27272a',   # Zinc-800
    'sidebar_active':      '#27272a',   # Active item bg
    'sidebar_active_text': '#ffffff',   # White active text
    'sidebar_text':        '#a1a1aa',   # Muted light grey for inactive
    'sidebar_section':     '#52525b',   # Section label colour
    'sidebar_border':      '#27272a',   # Internal sidebar dividers
    # Stage progress bar accent colours (stay colourful — data indicators)
    'stage1':     '#3b82f6',
    'stage2':     '#8b5cf6',
    'stage3':     '#06b6d4',
}


# ══════════════════════════════════════════════════════════════
# IMAGE HELPERS
# ══════════════════════════════════════════════════════════════

def cv2_to_qpixmap(cv_img, max_w=500, max_h=420):
    if cv_img is None:
        return QPixmap()
    if len(cv_img.shape) == 2:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
    rgb  = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg).scaled(
        max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

def resize_to(img, long_side):
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    return cv2.resize(img, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

def pure_rotate(img, angle_deg):
    h, w = img.shape[:2]
    corners = [img[0,0], img[0,-1], img[-1,0], img[-1,-1]]
    bg = np.median(corners, axis=0).astype(np.uint8).tolist()
    M  = cv2.getRotationMatrix2D((w/2.0, h/2.0), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=bg)


# ══════════════════════════════════════════════════════════════
# DATA MATRIX DECODER
# ══════════════════════════════════════════════════════════════

def decode_data_matrix(cv_img):
    """
    Attempt to decode a Data Matrix barcode from a BGR OpenCV image.
    Returns a dict:
        { 'found': bool, 'data': str or None, 'error': str or None }
    """
    if not _DMTX_AVAILABLE:
        return {'found': False, 'data': None, 'error': 'pylibdmtx not installed'}
    try:
        from PIL import Image as _PILImage
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = _PILImage.fromarray(rgb)
        decoded = _dmtx_decode(pil_img)
        if decoded:
            data_str = decoded[0].data.decode('utf-8', errors='replace')
            return {'found': True, 'data': data_str, 'error': None}
        else:
            return {'found': False, 'data': None, 'error': None}
    except Exception as exc:
        return {'found': False, 'data': None, 'error': str(exc)}


# ══════════════════════════════════════════════════════════════
# STAGE 1 — ORIENTATION  (heatmap correlation method)
# ══════════════════════════════════════════════════════════════

def _heatmap_rotate(img, angle):
    """Rotate image with white background fill."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255
    )

def _create_heatmap(img):
    """Gradient-magnitude heatmap of an image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    heat = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return heat

def _heatmap_similarity(a, b):
    """Pearson correlation between two same-size heatmaps."""
    b = cv2.resize(b, (a.shape[1], a.shape[0]))
    return float(np.corrcoef(a.flatten(), b.flatten())[0, 1])

def run_orientation(ref_img, test_img, emit=None):
    """
    Heatmap-correlation orientation correction.

    1. Coarse sweep  0–360 deg in 5-degree steps (downscaled for speed)
    2. Fine   sweep  ±5 deg around best coarse angle in 0.2-degree steps
    3. Rotate the full-resolution test image by the best angle found.
    """
    SCALE = 0.25
    COARSE_STEP = 5          # degrees
    FINE_STEP   = 0.2        # degrees
    FINE_WINDOW = 5          # ± degrees around coarse best

    emit and emit(2, "Downscaling images for coarse search (scale=%.2f)…" % SCALE)
    ref_small  = cv2.resize(ref_img,  None, fx=SCALE, fy=SCALE)
    test_small = cv2.resize(test_img, None, fx=SCALE, fy=SCALE)

    ref_heat = _create_heatmap(ref_small)

    # ── Coarse sweep ───────────────────────────────────────────
    emit and emit(5, "Coarse sweep 0–360 deg (step=%d deg)…" % COARSE_STEP)
    coarse_angles = range(0, 360, COARSE_STEP)
    n_coarse = len(coarse_angles)
    best_score = -1.0
    best_angle = 0.0

    for idx, angle in enumerate(coarse_angles):
        rot  = _heatmap_rotate(test_small, angle)
        heat = _create_heatmap(rot)
        score = _heatmap_similarity(ref_heat, heat)
        if score > best_score:
            best_score = score
            best_angle = float(angle)
        pct = 5 + int(55 * (idx + 1) / n_coarse)
        if emit:
            emit(pct, "Coarse [%d/%d] angle=%d  score=%.4f  best=%.1f (%.4f)"
                 % (idx + 1, n_coarse, angle, score, best_angle, best_score))

    emit and emit(60, "Coarse best: %.1f deg  score=%.4f" % (best_angle, best_score))

    # ── Fine sweep ─────────────────────────────────────────────
    emit and emit(62, "Fine sweep ±%g deg around %.1f (step=%.1f deg)…"
                  % (FINE_WINDOW, best_angle, FINE_STEP))
    fine_angles = np.arange(best_angle - FINE_WINDOW,
                             best_angle + FINE_WINDOW + FINE_STEP,
                             FINE_STEP)
    n_fine = len(fine_angles)

    for idx, angle in enumerate(fine_angles):
        rot  = _heatmap_rotate(test_small, float(angle))
        heat = _create_heatmap(rot)
        score = _heatmap_similarity(ref_heat, heat)
        if score > best_score:
            best_score = score
            best_angle = float(angle)
        pct = 62 + int(28 * (idx + 1) / n_fine)
        if emit:
            emit(pct, "Fine [%d/%d] angle=%.1f  score=%.4f  best=%.2f (%.4f)"
                 % (idx + 1, n_fine, angle, score, best_angle, best_score))

    emit and emit(92, "Fine best: %.2f deg  score=%.4f" % (best_angle, best_score))

    # ── Apply rotation to full-resolution image ────────────────
    emit and emit(94, "Rotating full-resolution image by %.2f deg…" % best_angle)
    aligned = _heatmap_rotate(test_img, best_angle)

    emit and emit(100, "Orientation done — total rotation: %.2f deg" % best_angle)

    info = {
        'total_angle': best_angle,
        'ncc_score':   best_score,
        'sift_inliers': 0,
        'strip_conf':   0.0,
    }
    return aligned, best_angle, info


# ══════════════════════════════════════════════════════════════
# STAGE 2 — STEPHOLE EXTRACTION  (unchanged)
# ══════════════════════════════════════════════════════════════

def detect_blue_dots(ref_img):
    H_ref,W_ref = ref_img.shape[:2]
    hsv  = cv2.cvtColor(ref_img,cv2.COLOR_BGR2HSV)
    mask1= cv2.inRange(hsv,(90,60,40),(115,255,255))
    mask2= cv2.inRange(hsv,(115,60,40),(135,255,255))
    mask = cv2.bitwise_or(mask1,mask2)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN, kernel)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    cnts,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    pts=[]; areas=[]
    for c in cnts:
        area=cv2.contourArea(c)
        if area<5: continue
        M=cv2.moments(c)
        if M["m00"]==0: continue
        pts.append([M["m10"]/M["m00"], M["m01"]/M["m00"]])
        areas.append(area)
    if not pts: return np.array([])
    pts=np.array(pts); areas=np.array(areas)
    order=np.argsort(-areas); pts=pts[order]
    if len(pts)>92: pts=pts[:92]
    return pts

def run_extraction(ref_img, aligned_img, output_dir, emit=None):
    emit and emit(5,"Detecting blue dots on reference...")
    ref_pts=detect_blue_dots(ref_img)
    if len(ref_pts)==0:
        raise RuntimeError("No blue dots detected on reference image")
    emit and emit(15,"Detected %d blue dots"%len(ref_pts))

    emit and emit(20,"ORB feature matching (ref -> aligned)...")
    MAX_DIM=2000
    def downscale(img):
        h,w=img.shape[:2]; scale=MAX_DIM/max(h,w)
        if scale>=1: return img,1.0
        return cv2.resize(img,(int(w*scale),int(h*scale))),scale

    ref_small,scale_ref=downscale(ref_img)
    img_small,scale_img=downscale(aligned_img)
    gray_ref=cv2.cvtColor(ref_small,cv2.COLOR_BGR2GRAY)
    gray_img=cv2.cvtColor(img_small,cv2.COLOR_BGR2GRAY)

    orb=cv2.ORB_create(nfeatures=12000,scaleFactor=1.2,nlevels=8)
    kp1,des1=orb.detectAndCompute(gray_ref,None)
    kp2,des2=orb.detectAndCompute(gray_img,None)
    bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
    matches=sorted(bf.match(des1,des2),key=lambda x:x.distance)[:2000]
    emit and emit(35,"ORB matches: %d"%len(matches))

    pts1=np.float32([kp1[m.queryIdx].pt for m in matches])/scale_ref
    pts2=np.float32([kp2[m.trainIdx].pt for m in matches])/scale_img
    H,mask=cv2.findHomography(pts1,pts2,cv2.RANSAC,5.0)
    emit and emit(45,"Homography inliers: %d"%int(mask.sum()))
    if H is None: raise RuntimeError("Homography failed")

    def ref_to_target(x,y):
        pt=np.array([[[x,y]]],dtype=np.float32)
        return cv2.perspectiveTransform(pt,H)[0][0]

    os.makedirs(output_dir,exist_ok=True)
    # PATCH=180
    # NEW — scale patch size relative to image dimensions
# 180px was designed for a ~2000px long-side image; scale proportionally
    REF_LONG_SIDE = 2000   # the resolution PATCH=180 was tuned for
    actual_long_side = max(aligned_img.shape[:2])
    PATCH = int(180 * actual_long_side / REF_LONG_SIDE)
    PATCH = max(180, min(PATCH, 500))   # clamp: never smaller than 180 or bigger than 1200
    h_img,w_img=aligned_img.shape[:2]
    saved=[]; n=len(ref_pts)
    overlay=aligned_img.copy()
    font_scale=max(0.5,w_img/4000)
    thickness=max(1,int(w_img/2000))

    for i,(x,y) in enumerate(ref_pts):
        tx,ty=ref_to_target(x,y)
        px,py=int(tx),int(ty)
        x1=max(0,px-PATCH//2); y1=max(0,py-PATCH//2)
        x2=min(w_img,px+PATCH//2); y2=min(h_img,py+PATCH//2)
        crop = aligned_img[y1:y2, x1:x2]
        # ── Quality enhancement for better wirebond visibility ──
        # 1. Upscale 2× with Lanczos for sub-pixel detail
        crop = cv2.resize(crop, (crop.shape[1]*2, crop.shape[0]*2),
                        interpolation=cv2.INTER_LANCZOS4)
        # 2. CLAHE on L-channel to boost local contrast (wirebonds are thin & dark)
        lab  = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
        l     = clahe.apply(l)
        crop  = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        # 3. Gentle unsharp mask to crisp up wire edges
        blur  = cv2.GaussianBlur(crop, (0, 0), sigmaX=1.2)
        crop  = cv2.addWeighted(crop, 1.6, blur, -0.6, 0)
        name = "S%03d" % (i + 1)
        path=os.path.join(output_dir,name+".png")
        cv2.imwrite(path,crop)
        saved.append(path)
        cv2.circle(overlay,(px,py),int(20*font_scale),(0,255,0),thickness)
        cv2.putText(overlay,name,(px+10,py-10),cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,(0,255,0),thickness,cv2.LINE_AA)
        pct=45+int(50*(i+1)/n)
        if emit: emit(pct,"Extracted %d/%d: %s"%(i+1,n,name))

    overlay_path=os.path.join(output_dir,"_overlay.png")
    cv2.imwrite(overlay_path,overlay)
    emit and emit(100,"Extraction done -- %d stepholes saved"%len(saved))
    return saved, overlay

# ══════════════════════════════════════════════════════════════
# STAGE 3 — AI INFERENCE  (unchanged)
# ══════════════════════════════════════════════════════════════

def analyse_stephole_batch(paths, emit=None):
    model = get_model(
        model_id="hexaboard-gold-pads/11",
        api_key="TwXID7NRGs9CuZvRS4PM"
    )

    results = []
    n = len(paths)

    for i, path in enumerate(paths):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            response   = model.infer(path)[0]
            detections = response.predictions

            raw_preds = [
                {'class': d.class_name, 'confidence': d.confidence}
                for d in detections
            ]

            if len(detections) == 0:
                status     = "FAIL"
                confidence = 0
            else:
                status     = "PASS"
                confidence = int(max(d.confidence for d in detections) * 100)

            result = {
                "name":        name,
                "path":        path,
                "status":      status,
                "confidence":  confidence,
                "notes":       "%d object (s) detected" % len(detections),
                "predictions": raw_preds,
            }

        except Exception as e:
            result = {
                "name":        name,
                "path":        path,
                "status":      "ERROR",
                "confidence":  0,
                "notes":       str(e),
                "predictions": [],
            }

        results.append(result)
        pct = int(100 * (i + 1) / n)
        if emit:
            emit(pct, "Analysed %d/%d: %s -> %s" % (i+1, n, name, result['status']))

    return results


# ══════════════════════════════════════════════════════════════
# PDF REPORT  (unchanged)
# ══════════════════════════════════════════════════════════════

def build_pdf_report(results, output_dir, module_name=None, dm_result=None):
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, "inspection_report.pdf")

    if module_name is None:
        module_name = os.path.basename(output_dir.rstrip('/\\'))

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.2*cm, bottomMargin=2*cm,
    )
    W = A4[0] - 4*cm

    base = getSampleStyleSheet()

    s_module = ParagraphStyle(
        'ModTitle',
        parent=base['Heading1'],
        fontSize=15, fontName='Helvetica-Bold',
        textColor=colors.black,
        spaceBefore=0, spaceAfter=4,
    )
    s_image = ParagraphStyle(
        'ImgTitle',
        parent=base['Heading2'],
        fontSize=12, fontName='Helvetica-Bold',
        textColor=colors.black,
        spaceBefore=8, spaceAfter=3,
    )
    s_meta = ParagraphStyle(
        'Meta',
        parent=base['Normal'],
        fontSize=10, fontName='Helvetica',
        textColor=colors.HexColor('#555555'),
        spaceAfter=10,
    )

    HDR_BG = colors.HexColor('#dde3ea')
    ALT_BG = colors.HexColor('#f4f6f8')
    BORDER = colors.HexColor('#b0bbc7')

    col_w = [W * 0.55, W * 0.16, W * 0.29]

    def make_det_table(raw_preds):
        by_cls = defaultdict(list)
        for p in raw_preds:
            by_cls[p['class']].append(p['confidence'])

        header = ['Class', 'Count', 'Avg Confidence']
        rows = [header]
        for cls in sorted(by_cls):
            confs = by_cls[cls]
            rows.append([cls, str(len(confs)),
                         '%.2f' % (sum(confs) / len(confs))])

        t = Table(rows, colWidths=col_w)
        style = [
            ('BACKGROUND',    (0, 0), (-1, 0), HDR_BG),
            ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME',      (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE',      (0, 0), (-1, -1), 10),
            ('ALIGN',         (1, 0), (-1, -1), 'CENTER'),
            ('ALIGN',         (0, 0), (0, -1), 'LEFT'),
            ('LEFTPADDING',   (0, 0), (-1, -1), 5),
            ('RIGHTPADDING',  (0, 0), (-1, -1), 5),
            ('TOPPADDING',    (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('GRID',          (0, 0), (-1, -1), 0.5, BORDER),
        ]
        for row_idx in range(2, len(rows), 2):
            style.append(('BACKGROUND', (0, row_idx), (-1, row_idx), ALT_BG))

        t.setStyle(TableStyle(style))
        return t

    story = []
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    story.append(Paragraph('1  Module: %s' % module_name, s_module))
    story.append(Paragraph('Generated: %s' % ts, s_meta))
    story.append(HRFlowable(width=W, thickness=0.5, color=BORDER, spaceAfter=6))

    # ── Data Matrix section ────────────────────────────────────
    dm_label_style = ParagraphStyle(
        'DMLabel',
        parent=base['Heading2'],
        fontSize=12, fontName='Helvetica-Bold',
        textColor=colors.black,
        spaceBefore=4, spaceAfter=3,
    )
    dm_val_style = ParagraphStyle(
        'DMVal',
        parent=base['Normal'],
        fontSize=11, fontName='Helvetica',
        textColor=colors.HexColor('#222222'),
        spaceAfter=8,
    )
    if dm_result is None:
        dm_text = 'Not scanned'
        dm_color = colors.HexColor('#888888')
    elif dm_result.get('found'):
        dm_text = dm_result['data'] or '(empty)'
        dm_color = colors.HexColor('#1a7a35')
    elif dm_result.get('error'):
        dm_text = 'Error: %s' % dm_result['error']
        dm_color = colors.HexColor('#c0392b')
    else:
        dm_text = 'No Data Matrix code detected'
        dm_color = colors.HexColor('#c0392b')

    dm_rows = [
        ['Data Matrix Code', dm_text],
    ]
    dm_table = Table(dm_rows, colWidths=[W * 0.30, W * 0.70])
    dm_table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (0, 0), HDR_BG),
        ('FONTNAME',      (0, 0), (0, 0), 'Helvetica-Bold'),
        ('FONTNAME',      (1, 0), (1, 0), 'Helvetica'),
        ('FONTSIZE',      (0, 0), (-1, -1), 11),
        ('ALIGN',         (0, 0), (-1, -1), 'LEFT'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 8),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 8),
        ('TOPPADDING',    (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('GRID',          (0, 0), (-1, -1), 0.5, BORDER),
        ('TEXTCOLOR',     (1, 0), (1, 0), dm_color),
    ]))
    story.append(Paragraph('Data Matrix', dm_label_style))
    story.append(dm_table)
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width=W, thickness=0.5, color=BORDER, spaceAfter=6))

    total  = len(results)
    passed = sum(1 for r in results if r.get('status') == 'PASS')
    failed = sum(1 for r in results if r.get('status') == 'FAIL')

    for idx, r in enumerate(results, 1):
        name   = r.get('name', 'H%03d' % idx)
        preds  = r.get('predictions', [])
        story.append(Paragraph('%d  Image: %s' % (idx + 1, name), s_image))
        story.append(make_det_table(preds))
        story.append(Spacer(1, 2))

    story.append(Spacer(1, 16))
    story.append(HRFlowable(width=W, thickness=1.2, color=colors.black, spaceAfter=8))

    rate = ('%d%%' % round(100 * passed / total)) if total else 'N/A'
    sum_rows = [
        ['Total Holes', 'PASS', 'FAIL', 'Pass Rate'],
        [str(total),    str(passed), str(failed), rate],
    ]
    st = Table(sum_rows, colWidths=[W/4]*4)
    st.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0), HDR_BG),
        ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME',      (0, 1), (-1, 1), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, -1), 12),
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
        ('GRID',          (0, 0), (-1, -1), 0.5, BORDER),
        ('TOPPADDING',    (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TEXTCOLOR', (1, 1), (1, 1), colors.HexColor('#1a7a35')),
        ('TEXTCOLOR', (2, 1), (2, 1), colors.HexColor('#c0392b')),
    ]))
    story.append(st)

    doc.build(story)
    return pdf_path


# ══════════════════════════════════════════════════════════════
# WORKER THREADS  (unchanged)
# ══════════════════════════════════════════════════════════════

class Stage1Worker(QThread):
    progress= pyqtSignal(int,str)
    finished= pyqtSignal(object,float,dict)
    error   = pyqtSignal(str)
    def __init__(self,ref_img,test_img):
        super().__init__(); self.ref_img=ref_img; self.test_img=test_img
    def run(self):
        try:
            aligned,angle,info=run_orientation(
                self.ref_img,self.test_img,
                emit=lambda p,m: self.progress.emit(p if p else 0,m))
            self.finished.emit(aligned,angle,info)
        except Exception as e:
            import traceback; self.error.emit("%s\n\n%s"%(e,traceback.format_exc()))

class Stage2Worker(QThread):
    progress= pyqtSignal(int,str)
    finished= pyqtSignal(list,object)
    error   = pyqtSignal(str)
    def __init__(self,ref_img,aligned_img,output_dir):
        super().__init__()
        self.ref_img=ref_img; self.aligned_img=aligned_img; self.output_dir=output_dir
    def run(self):
        try:
            saved,overlay=run_extraction(
                self.ref_img,self.aligned_img,self.output_dir,
                emit=lambda p,m: self.progress.emit(p,m))
            self.finished.emit(saved,overlay)
        except Exception as e:
            import traceback; self.error.emit("%s\n\n%s"%(e,traceback.format_exc()))

class Stage3Worker(QThread):
    progress= pyqtSignal(int,str)
    finished= pyqtSignal(list,str)
    error   = pyqtSignal(str)
    def __init__(self,paths,output_dir,module_name=None,dm_result=None):
        super().__init__()
        self.paths=paths
        self.output_dir=output_dir
        self.module_name=module_name
        self.dm_result=dm_result
    def run(self):
        try:
            results=analyse_stephole_batch(
                self.paths,
                emit=lambda p,m: self.progress.emit(p,m))
            pdf_path=build_pdf_report(results, self.output_dir,
                                      module_name=self.module_name,
                                      dm_result=self.dm_result)
            self.finished.emit(results,pdf_path)
        except Exception as e:
            import traceback; self.error.emit("%s\n\n%s"%(e,traceback.format_exc()))


# ══════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════

def _divider():
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet(f"color: {T['border']}; background: {T['border']}; max-height: 1px;")
    return line

def _label(text, color=None, size="13px", weight="400", parent=None):
    lbl = QLabel(text, parent)
    color = color or T['text']
    lbl.setStyleSheet(f"color:{color};font-size:{size};font-weight:{weight};background:transparent;")
    return lbl


# ══════════════════════════════════════════════════════════════
# SIDEBAR FRAME
# ══════════════════════════════════════════════════════════════

class SidebarFrame(QFrame):
    page_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(240)
        self.setObjectName("SidebarFrame")
        self.setStyleSheet(f"""
            #SidebarFrame {{
                background: {T['sidebar_bg']};
                border-right: 1px solid #2d2d30;
            }}
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Brand header (TIFR logo on white pill) ────────────
        brand = QFrame()
        brand.setFixedHeight(76)
        brand.setStyleSheet(f"background:{T['sidebar_bg']};border-bottom:1px solid {T['sidebar_border']};")
        bl = QHBoxLayout(brand)
        bl.setContentsMargins(14, 10, 14, 10)

        logo_container = QFrame()
        logo_container.setStyleSheet(
            "background:#ffffff;border-radius:8px;border:none;"
        )
        logo_cl = QHBoxLayout(logo_container)
        logo_cl.setContentsMargins(8, 5, 8, 5)
        logo_lbl = QLabel()
        logo_lbl.setStyleSheet("background:transparent;")
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tifr-logo-black.png")
        if os.path.exists(logo_path):
            logo_pix = QPixmap(logo_path).scaled(130, 42, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_lbl.setPixmap(logo_pix)
        else:
            logo_lbl.setText("HexaVision")
            logo_lbl.setStyleSheet("color:#09090b;font-size:14px;font-weight:700;background:transparent;")
        logo_cl.addWidget(logo_lbl)
        bl.addWidget(logo_container)
        bl.addStretch()
        root.addWidget(brand)

        # ── Scrollable nav area ────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background:transparent;border:none;")

        nav_widget = QWidget()
        nav_widget.setStyleSheet("background:transparent;")
        nav_layout = QVBoxLayout(nav_widget)
        nav_layout.setContentsMargins(8, 4, 8, 8)
        nav_layout.setSpacing(1)

        nav_layout.addWidget(self._section_label("PAGES"))
        self._page_buttons = []
        page_items = [
            ("⚙️", "Pipeline",         0),
            ("🔬", "Stephole Viewer",  1),
            ("🤖", "AI Results",       2),
            ("📄", "Reports",          3),
        ]
        for icon, name, page_idx in page_items:
            btn = self._nav_item(icon, name, page_idx)
            self._page_buttons.append((btn, page_idx))
            nav_layout.addWidget(btn)

        nav_layout.addStretch()
        scroll.setWidget(nav_widget)
        root.addWidget(scroll, 1)

        # ── "Made with ❤️ for TIFR" footer ────────────────────
        footer = QFrame()
        footer.setFixedHeight(40)
        footer.setStyleSheet("background:#09090b;border-top:1px solid #27272a;")
        fl = QHBoxLayout(footer)
        fl.setContentsMargins(10, 0, 10, 0)
        footer_lbl = QLabel()
        footer_lbl.setTextFormat(Qt.RichText)
        footer_lbl.setText("Made with <span style='color:#ef4444;'>&#10084;</span> for TIFR")
        footer_lbl.setAlignment(Qt.AlignCenter)
        footer_lbl.setStyleSheet("color:#e4e4e7;font-size:12px;font-weight:500;background:transparent;")
        fl.addWidget(footer_lbl)
        root.addWidget(footer)

        # ── Bottom user card ───────────────────────────────────
        user_frame = QFrame()
        user_frame.setFixedHeight(56)
        user_frame.setStyleSheet("background:#09090b;border-top:1px solid #27272a;")
        ul = QHBoxLayout(user_frame)
        ul.setContentsMargins(14, 8, 14, 8)

        avatar = QLabel("U")
        avatar.setFixedSize(32, 32)
        avatar.setAlignment(Qt.AlignCenter)
        avatar.setStyleSheet("""
            background:#3f3f46;color:#ffffff;
            border-radius:16px;font-weight:700;font-size:13px;
        """)
        user_info = QVBoxLayout()
        user_info.setSpacing(0)
        name_lbl = QLabel("Operator")
        name_lbl.setStyleSheet("color:#ffffff;font-size:12px;font-weight:600;background:transparent;")
        role_lbl = QLabel("Quality Control")
        role_lbl.setStyleSheet("color:#a1a1aa;font-size:10px;background:transparent;")
        user_info.addWidget(name_lbl)
        user_info.addWidget(role_lbl)

        ul.addWidget(avatar)
        ul.addSpacing(8)
        ul.addLayout(user_info)
        ul.addStretch()
        root.addWidget(user_frame)

        self._set_active(0)

    def _section_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("""
            color: #a1a1aa;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.8px;
            background: transparent;
            padding: 8px 8px 2px 8px;
        """)
        return lbl

    def _nav_item(self, icon, name, page_idx=None):
        btn = QPushButton(f"  {icon}  {name}")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedHeight(36)
        btn.setCheckable(True)
        btn.setProperty("page_idx", page_idx)
        btn.setStyleSheet(self._nav_style(False))
        if page_idx is not None:
            btn.clicked.connect(lambda checked, idx=page_idx: self._on_nav_click(idx))
        return btn

    def _nav_style(self, active):
        if active:
            return """
                QPushButton {
                    background: #ffffff;
                    color: #09090b;
                    border: none;
                    border-radius: 7px;
                    text-align: left;
                    font-size: 13px;
                    font-weight: 700;
                    padding-left: 8px;
                }
            """
        return """
            QPushButton {
                background: transparent;
                color: #d4d4d8;
                border: none;
                border-radius: 7px;
                text-align: left;
                font-size: 13px;
                font-weight: 500;
                padding-left: 8px;
            }
            QPushButton:hover {
                background: #1c1c1e;
                color: #ffffff;
            }
        """

    def _on_nav_click(self, idx):
        self._set_active(idx)
        self.page_changed.emit(idx)

    def _set_active(self, active_idx):
        for btn, page_idx in self._page_buttons:
            is_active = (page_idx == active_idx)
            btn.setStyleSheet(self._nav_style(is_active))
            btn.setChecked(is_active)


# ══════════════════════════════════════════════════════════════
# STAT CARD
# ══════════════════════════════════════════════════════════════

class StatCard(QFrame):
    def __init__(self, title, value="—", icon="", accent_color=None, parent=None):
        super().__init__(parent)
        self.accent = accent_color or T['accent']
        self.setObjectName("StatCard")
        self.setStyleSheet(f"""
            #StatCard {{
                background: {T['surface']};
                border: 1px solid {T['border']};
                border-radius: 10px;
            }}
            #StatCard:hover {{
                border-color: {self.accent}88;
            }}
        """)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFixedHeight(110)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 14)
        lay.setSpacing(6)

        top_row = QHBoxLayout()
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"color:{T['muted']};font-size:12px;font-weight:500;background:transparent;")
        icon_lbl = QLabel(icon)
        icon_lbl.setStyleSheet(f"color:{self.accent};font-size:20px;background:transparent;")
        top_row.addWidget(title_lbl)
        top_row.addStretch()
        top_row.addWidget(icon_lbl)
        lay.addLayout(top_row)

        self.value_lbl = QLabel(str(value))
        self.value_lbl.setStyleSheet(f"color:{T['text']};font-size:28px;font-weight:700;background:transparent;")
        lay.addWidget(self.value_lbl)

        self.sub_lbl = QLabel("")
        self.sub_lbl.setStyleSheet(f"color:{T['muted']};font-size:11px;background:transparent;")
        lay.addWidget(self.sub_lbl)

    def set_value(self, val, sub=""):
        self.value_lbl.setText(str(val))
        self.sub_lbl.setText(sub)


# ══════════════════════════════════════════════════════════════
# STAGE PROGRESS CARD
# ══════════════════════════════════════════════════════════════

class StageProgressCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("StageProgressCard")
        self.setStyleSheet(f"""
            #StageProgressCard {{
                background: {T['surface']};
                border: 1px solid {T['border']};
                border-radius: 10px;
            }}
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(14)

        hdr = QHBoxLayout()
        title = QLabel("Pipeline Progress")
        title.setStyleSheet(f"color:{T['text']};font-size:14px;font-weight:700;background:transparent;")
        self.status_pill = QLabel("Idle")
        self.status_pill.setStyleSheet(f"""
            color:{T['muted']};
            background:{T['surface2']};
            border:1px solid {T['border']};
            border-radius:10px;
            font-size:11px;
            font-weight:600;
            padding:2px 10px;
        """)
        hdr.addWidget(title)
        hdr.addStretch()
        hdr.addWidget(self.status_pill)
        lay.addLayout(hdr)

        self.stage_bars = []
        stages = [
            ("Stage 1", "Orientation Correction", T['stage1']),
            ("Stage 2", "Stephole Extraction",    T['stage2']),
            ("Stage 3", "AI Inspection Report",   T['stage3']),
        ]
        for tag, label, color in stages:
            row = QVBoxLayout()
            row.setSpacing(4)

            top = QHBoxLayout()
            tag_lbl = QLabel(tag)
            tag_lbl.setStyleSheet(f"color:{T['text']};font-size:12px;font-weight:600;background:transparent;")
            desc_lbl = QLabel(label)
            desc_lbl.setStyleSheet(f"color:{T['muted']};font-size:11px;background:transparent;")
            pct_lbl = QLabel("0%")
            pct_lbl.setStyleSheet(f"color:{color};font-size:11px;font-weight:600;background:transparent;")
            top.addWidget(tag_lbl)
            top.addSpacing(6)
            top.addWidget(desc_lbl)
            top.addStretch()
            top.addWidget(pct_lbl)
            row.addLayout(top)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFixedHeight(6)
            bar.setTextVisible(False)
            bar.setStyleSheet(f"""
                QProgressBar {{
                    border: none;
                    border-radius: 3px;
                    background: {T['surface2']};
                }}
                QProgressBar::chunk {{
                    background: {color};
                    border-radius: 3px;
                }}
            """)
            row.addWidget(bar)
            lay.addLayout(row)
            self.stage_bars.append((bar, pct_lbl))

        lay.addWidget(_divider())
        global_row = QHBoxLayout()
        global_lbl = QLabel("Overall")
        global_lbl.setStyleSheet(f"color:{T['text']};font-size:12px;font-weight:600;background:transparent;")
        self.global_pct_lbl = QLabel("0%")
        self.global_pct_lbl.setStyleSheet(f"color:{T['accent']};font-size:11px;font-weight:600;background:transparent;")
        global_row.addWidget(global_lbl)
        global_row.addStretch()
        global_row.addWidget(self.global_pct_lbl)
        lay.addLayout(global_row)

        self.global_bar = QProgressBar()
        self.global_bar.setRange(0, 100)
        self.global_bar.setValue(0)
        self.global_bar.setFixedHeight(8)
        self.global_bar.setTextVisible(False)
        self.global_bar.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 4px;
                background: {T['surface2']};
            }}
            QProgressBar::chunk {{
                background: {T['accent']};
                border-radius: 4px;
            }}
        """)
        lay.addWidget(self.global_bar)

    def set_stage(self, stage_idx, value):
        bar, pct_lbl = self.stage_bars[stage_idx]
        bar.setValue(value)
        pct_lbl.setText(f"{value}%")

    def set_global(self, value):
        self.global_bar.setValue(value)
        self.global_pct_lbl.setText(f"{value}%")

    def set_status(self, text, color=None):
        color = color or T['muted']
        self.status_pill.setText(text)
        self.status_pill.setStyleSheet(f"""
            color:{color};
            background:{T['accent_bg'] if color == T['accent'] else T['surface2']};
            border:1px solid {color}44;
            border-radius:10px;
            font-size:11px;
            font-weight:600;
            padding:2px 10px;
        """)


# ══════════════════════════════════════════════════════════════
# VIEWER CARD
# ══════════════════════════════════════════════════════════════

class ViewerCard(QFrame):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setObjectName("ViewerCard")
        self.setStyleSheet(f"""
            #ViewerCard {{
                background: {T['surface']};
                border: 1px solid {T['border']};
                border-radius: 10px;
            }}
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 16)
        lay.setSpacing(10)

        hdr = QHBoxLayout()
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"color:{T['text']};font-size:14px;font-weight:700;background:transparent;")
        self.badge = QLabel("")
        self.badge.setStyleSheet("background:transparent;")
        hdr.addWidget(title_lbl)
        hdr.addStretch()
        hdr.addWidget(self.badge)
        lay.addLayout(hdr)

        self.img_lbl = QLabel("No image loaded")
        self.img_lbl.setAlignment(Qt.AlignCenter)
        self.img_lbl.setFixedHeight(200)
        self.img_lbl.setStyleSheet(f"""
            color:{T['muted']};
            background:{T['surface2']};
            border: 1px dashed {T['border']};
            border-radius: 8px;
            font-size: 12px;
        """)
        lay.addWidget(self.img_lbl)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedHeight(4)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 2px;
                background: {T['surface2']};
            }}
            QProgressBar::chunk {{
                background: {T['accent']};
                border-radius: 2px;
            }}
        """)
        lay.addWidget(self.progress)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(90)
        self.log.setStyleSheet(f"""
            QTextEdit {{
                background: {T['surface2']};
                color: {T['muted']};
                border: 1px solid {T['border']};
                border-radius: 6px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 10px;
                padding: 6px;
            }}
        """)
        lay.addWidget(self.log)

    def set_image(self, cv_img, max_w=400, max_h=195):
        if cv_img is None:
            self.img_lbl.setText("No image loaded")
            return
        pix = cv2_to_qpixmap(cv_img, max_w, max_h)
        self.img_lbl.setPixmap(pix)
        self.img_lbl.setText("")

    def set_badge(self, text, status):
        colors_map = {
            'pass':    (T['pass_color'],  T['pass_bg']),
            'fail':    (T['fail_color'],  T['fail_bg']),
            'running': (T['warn'],        T['warn_bg']),
            'idle':    (T['muted'],       T['surface2']),
            'done':    (T['pass_color'],  T['pass_bg']),
            'error':   (T['fail_color'],  T['fail_bg']),
        }
        c, bg = colors_map.get(status.lower(), (T['muted'], T['surface2']))
        self.badge.setText(text)
        self.badge.setStyleSheet(f"""
            color:{c};
            background:{bg};
            border:1px solid {c}44;
            border-radius:9px;
            font-size:11px;
            font-weight:700;
            padding:2px 10px;
        """)

    def log_msg(self, msg):
        self.log.append(msg)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())


# ══════════════════════════════════════════════════════════════
# STATS SIDE CARD
# ══════════════════════════════════════════════════════════════

class StatsCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("StatsCard")
        self.setStyleSheet(f"""
            #StatsCard {{
                background: {T['surface']};
                border: 1px solid {T['border']};
                border-radius: 10px;
            }}
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 16)
        lay.setSpacing(12)

        title_lbl = QLabel("Detection Stats")
        title_lbl.setStyleSheet(f"color:{T['text']};font-size:14px;font-weight:700;background:transparent;")
        lay.addWidget(title_lbl)

        self._rows = {}
        stat_defs = [
            ("confidence",      "Avg Confidence",    T['accent']),
            ("objects",         "Objects Detected",  T['stage2']),
            ("pass_count",      "PASS Count",        T['pass_color']),
            ("fail_count",      "FAIL Count",        T['fail_color']),
            ("pass_rate",       "Pass Rate",         T['warn']),
        ]
        for key, label, color in stat_defs:
            row = QFrame()
            row.setStyleSheet(f"background:{T['surface2']};border-radius:7px;border:1px solid {T['border']};")
            rl = QHBoxLayout(row)
            rl.setContentsMargins(12, 8, 12, 8)
            lbl = QLabel(label)
            lbl.setStyleSheet(f"color:{T['muted']};font-size:11px;background:transparent;")
            val = QLabel("—")
            val.setStyleSheet(f"color:{color};font-size:15px;font-weight:700;background:transparent;")
            rl.addWidget(lbl)
            rl.addStretch()
            rl.addWidget(val)
            lay.addWidget(row)
            self._rows[key] = val

        lay.addStretch()

    def update_stats(self, results):
        total = len(results)
        passed = sum(1 for r in results if r.get('status') == 'PASS')
        failed = sum(1 for r in results if r.get('status') == 'FAIL')
        confs = [r.get('confidence', 0) for r in results if r.get('status') == 'PASS']
        avg_conf = f"{sum(confs)//len(confs)}%" if confs else "—"
        total_obj = sum(
            int(r.get('notes','0 pad').split()[0]) for r in results
            if r.get('status') == 'PASS'
        )
        rate = f"{round(100*passed/total)}%" if total else "—"
        self._rows['confidence'].setText(avg_conf)
        self._rows['objects'].setText(str(total_obj))
        self._rows['pass_count'].setText(str(passed))
        self._rows['fail_count'].setText(str(failed))
        self._rows['pass_rate'].setText(rate)


# ══════════════════════════════════════════════════════════════
# TABLE CARD
# ══════════════════════════════════════════════════════════════

class TableCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TableCard")
        self.setStyleSheet(f"""
            #TableCard {{
                background: {T['surface']};
                border: 1px solid {T['border']};
                border-radius: 10px;
            }}
        """)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 16)
        lay.setSpacing(10)

        hdr = QHBoxLayout()
        title_lbl = QLabel("Inspection Results")
        title_lbl.setStyleSheet(f"color:{T['text']};font-size:14px;font-weight:700;background:transparent;")
        self.count_lbl = QLabel("")
        self.count_lbl.setStyleSheet(f"color:{T['muted']};font-size:12px;background:transparent;")
        hdr.addWidget(title_lbl)
        hdr.addStretch()
        hdr.addWidget(self.count_lbl)
        lay.addLayout(hdr)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Hole ID", "Status", "Confidence", "Notes"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background: {T['surface']};
                border: none;
                font-size: 12px;
                color: {T['text']};
                outline: none;
            }}
            QTableWidget::item {{
                padding: 6px 10px;
                border-bottom: 1px solid {T['surface2']};
            }}
            QTableWidget::item:selected {{
                background: {T['accent_bg']};
                color: {T['text']};
            }}
            QHeaderView::section {{
                background: {T['surface2']};
                color: {T['muted']};
                font-size: 11px;
                font-weight: 600;
                border: none;
                border-bottom: 1px solid {T['border']};
                padding: 6px 10px;
            }}
            QTableWidget QTableCornerButton::section {{
                background: {T['surface2']};
                border: none;
            }}
        """)
        lay.addWidget(self.table)

    def populate(self, results):
        self.table.setRowCount(0)
        self.count_lbl.setText(f"{len(results)} holes")
        for r in results:
            row = self.table.rowCount()
            self.table.insertRow(row)

            name_item = QTableWidgetItem(r.get('name', ''))
            name_item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
            self.table.setItem(row, 0, name_item)

            status = r.get('status', '?')
            s_item = QTableWidgetItem(status)
            s_item.setTextAlignment(Qt.AlignVCenter | Qt.AlignCenter)
            if status == 'PASS':
                s_item.setForeground(QColor(T['pass_color']))
            elif status == 'FAIL':
                s_item.setForeground(QColor(T['fail_color']))
            else:
                s_item.setForeground(QColor(T['warn']))
            self.table.setItem(row, 1, s_item)

            conf_item = QTableWidgetItem(f"{r.get('confidence',0)}%")
            conf_item.setTextAlignment(Qt.AlignVCenter | Qt.AlignCenter)
            self.table.setItem(row, 2, conf_item)

            notes_item = QTableWidgetItem(r.get('notes', '')[:50])
            notes_item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
            self.table.setItem(row, 3, notes_item)


# ══════════════════════════════════════════════════════════════
# IMAGE INPUT CARD
# ══════════════════════════════════════════════════════════════

class ImageInputCard(QFrame):
    def __init__(self, title, btn_text, color, parent=None):
        super().__init__(parent)
        self.setObjectName("ImageInputCard")
        self.setStyleSheet(f"""
            #ImageInputCard {{
                background: {T['surface']};
                border: 1px solid {T['border']};
                border-radius: 10px;
            }}
        """)
        self.setFixedHeight(130)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(6)

        top = QHBoxLayout()
        lbl = QLabel(title)
        lbl.setStyleSheet(f"color:{T['muted']};font-size:11px;font-weight:500;background:transparent;")
        top.addWidget(lbl)
        top.addStretch()
        lay.addLayout(top)

        self.img_lbl = QLabel("Click to load →")
        self.img_lbl.setAlignment(Qt.AlignCenter)
        self.img_lbl.setStyleSheet(f"color:{T['muted']};font-size:11px;background:transparent;")
        lay.addWidget(self.img_lbl, 1)

        self.btn = QPushButton(btn_text)
        self.btn.setCursor(Qt.PointingHandCursor)
        self.btn.setFixedHeight(30)
        self.btn.setStyleSheet(f"""
            QPushButton {{
                background: {color}18;
                color: {color};
                border: 1px solid {color}44;
                border-radius: 6px;
                font-size: 11px;
                font-weight: 600;
                padding: 0 10px;
            }}
            QPushButton:hover {{
                background: {color}28;
            }}
        """)
        lay.addWidget(self.btn)

    def set_preview(self, cv_img):
        pix = cv2_to_qpixmap(cv_img, 200, 60)
        self.img_lbl.setPixmap(pix)
        self.img_lbl.setText("")


# ══════════════════════════════════════════════════════════════
# PAGE: PIPELINE
# ══════════════════════════════════════════════════════════════

class PipelinePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{T['bg']};")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet(f"background:{T['bg']};border:none;")

        scroll_content = QWidget()
        scroll_content.setStyleSheet(f"background:{T['bg']};")
        root = QVBoxLayout(scroll_content)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(16)

        page_hdr = QHBoxLayout()
        page_title = QLabel("Pipeline")
        page_title.setStyleSheet(f"color:{T['text']};font-size:22px;font-weight:700;background:transparent;")
        page_sub = QLabel("Hexaboard Inspection Pipeline")
        page_sub.setStyleSheet(f"color:{T['muted']};font-size:13px;background:transparent;")
        hdr_left = QVBoxLayout()
        hdr_left.setSpacing(2)
        hdr_left.addWidget(page_title)
        hdr_left.addWidget(page_sub)
        page_hdr.addLayout(hdr_left)
        page_hdr.addStretch()

        self.run_all_btn = QPushButton("▶  Run Pipeline")
        self.run_all_btn.setCursor(Qt.PointingHandCursor)
        self.run_all_btn.setFixedHeight(40)
        self.run_all_btn.setEnabled(False)
        self.run_all_btn.setStyleSheet(f"""
            QPushButton {{
                background: {T['accent']};
                color: #fff;
                border: none;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 700;
                padding: 0 20px;
            }}
            QPushButton:hover {{ background: #27272a; }}
            QPushButton:disabled {{
                background: {T['surface2']};
                color: {T['muted']};
                border: 1px solid {T['border']};
            }}
        """)
        page_hdr.addWidget(self.run_all_btn)
        root.addLayout(page_hdr)

        stat_row = QHBoxLayout()
        stat_row.setSpacing(12)
        self.stat_total = StatCard("Total Holes",  "—", "🔢", T['accent'])
        self.stat_pass  = StatCard("Pass",         "—", "✅", T['pass_color'])
        self.stat_fail  = StatCard("Fail",         "—", "❌", T['fail_color'])
        self.stat_rate  = StatCard("Pass Rate",    "—", "📊", T['warn'])
        for card in (self.stat_total, self.stat_pass, self.stat_fail, self.stat_rate):
            stat_row.addWidget(card)
        root.addLayout(stat_row)

        input_row = QHBoxLayout()
        input_row.setSpacing(12)

        self.ref_card  = ImageInputCard("Reference Image", "Load Reference",  T['stage1'])
        self.test_card = ImageInputCard("Test Image",      "Load Test Image", T['stage2'])
        input_row.addWidget(self.ref_card)
        input_row.addWidget(self.test_card)

        dir_card = QFrame()
        dir_card.setObjectName("DirCard")
        dir_card.setStyleSheet(f"""
            #DirCard {{
                background: {T['surface']};
                border: 1px solid {T['border']};
                border-radius: 10px;
            }}
        """)
        dir_card.setFixedHeight(130)
        dcl = QVBoxLayout(dir_card)
        dcl.setContentsMargins(14, 10, 14, 10)
        dcl.setSpacing(6)
        dcl.addWidget(_label("Output Folder", T['muted'], "11px", "500"))
        self.dir_lbl = QLabel("(not set)")
        self.dir_lbl.setStyleSheet(f"color:{T['accent']};font-size:11px;font-family:'Consolas','Courier New';background:transparent;")
        self.dir_lbl.setWordWrap(True)
        dcl.addWidget(self.dir_lbl, 1)
        self.pick_dir_btn = QPushButton("📁  Choose Folder")
        self.pick_dir_btn.setCursor(Qt.PointingHandCursor)
        self.pick_dir_btn.setFixedHeight(30)
        self.pick_dir_btn.setStyleSheet(f"""
            QPushButton {{
                background: {T['surface2']};
                color: {T['text']};
                border: 1px solid {T['border']};
                border-radius: 6px;
                font-size: 11px;
                font-weight: 500;
                padding: 0 10px;
            }}
            QPushButton:hover {{ background: #e4e4e7; }}
        """)
        dcl.addWidget(self.pick_dir_btn)
        input_row.addWidget(dir_card)
        root.addLayout(input_row)

        self.progress_card = StageProgressCard()
        root.addWidget(self.progress_card)

        viewer_row = QHBoxLayout()
        viewer_row.setSpacing(12)
        self.card1 = ViewerCard("Stage 1 — Orientation Correction")
        self.card2 = ViewerCard("Stage 2 — Stephole Extraction")
        self.card3 = ViewerCard("Stage 3 — AI Inspection")
        viewer_row.addWidget(self.card1)
        viewer_row.addWidget(self.card2)
        viewer_row.addWidget(self.card3)
        root.addLayout(viewer_row)
        root.addStretch(1)

        scroll_area.setWidget(scroll_content)
        outer.addWidget(scroll_area, 1)

        status_bar = QFrame()
        status_bar.setFixedHeight(38)
        status_bar.setStyleSheet(f"""
            background:{T['surface']};
            border-top:1px solid {T['border']};
        """)
        sbl = QHBoxLayout(status_bar)
        sbl.setContentsMargins(24, 0, 24, 0)
        self.status_lbl = QLabel("Ready — load reference and test images to begin")
        self.status_lbl.setStyleSheet(f"color:{T['accent']};font-size:12px;font-weight:600;background:transparent;")
        sbl.addWidget(self.status_lbl)
        sbl.addStretch()
        outer.addWidget(status_bar)


# ══════════════════════════════════════════════════════════════
# PAGE: AI RESULTS
# ══════════════════════════════════════════════════════════════

class AIResultsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{T['bg']};")
        self._results = []

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(16)

        hdr = QHBoxLayout()
        title = QLabel("AI Results")
        title.setStyleSheet(f"color:{T['text']};font-size:22px;font-weight:700;background:transparent;")
        hdr.addWidget(title)
        hdr.addStretch()

        self.open_pdf_btn = QPushButton("📄  Open PDF Report")
        self.open_pdf_btn.setCursor(Qt.PointingHandCursor)
        self.open_pdf_btn.setEnabled(False)
        self.open_pdf_btn.setFixedHeight(38)
        self.open_pdf_btn.setStyleSheet(f"""
            QPushButton {{
                background:{T['accent']};
                color:#fff;
                border:none;
                border-radius:8px;
                font-size:12px;
                font-weight:600;
                padding:0 16px;
            }}
            QPushButton:hover {{ background:#27272a; }}
            QPushButton:disabled {{
                background:{T['surface2']};
                color:{T['muted']};
                border:1px solid {T['border']};
            }}
        """)
        hdr.addWidget(self.open_pdf_btn)
        root.addLayout(hdr)

        mid_row = QHBoxLayout()
        mid_row.setSpacing(12)

        self.viewer = ViewerCard("Detection Viewer — click a row below to inspect")
        self.viewer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.viewer.setFixedHeight(340)
        mid_row.addWidget(self.viewer, 2)

        self.stats_card = StatsCard()
        self.stats_card.setFixedWidth(240)
        mid_row.addWidget(self.stats_card)
        root.addLayout(mid_row)

        self.table_card = TableCard()
        self.table_card.table.cellClicked.connect(self._on_row_clicked)
        root.addWidget(self.table_card, 1)

    def _on_row_clicked(self, row, col):
        if row < 0 or row >= len(self._results):
            return
        r = self._results[row]
        try:
            img = cv2.imread(r['path'])
            self.viewer.set_image(img, 400, 210)
        except Exception:
            self.viewer.img_lbl.setText("Could not load image")
        status = r.get('status', 'IDLE')
        self.viewer.set_badge(f"{r['name']}  —  {status}", status.lower())
        self.viewer.log.clear()
        self.viewer.log.append(f"Hole: {r['name']}")
        self.viewer.log.append(f"Status: {status}")
        self.viewer.log.append(f"Confidence: {r.get('confidence', 0)}%")
        self.viewer.log.append(f"Notes: {r.get('notes', '')}")
        preds = r.get('predictions', [])
        if preds:
            self.viewer.log.append(f"Detections: {len(preds)} pad(s)")
            for p in preds[:5]:
                self.viewer.log.append(f"  {p['class']}: {p['confidence']:.2f}")

    def populate(self, results, pdf_path):
        self._results = results
        self.table_card.populate(results)
        self.stats_card.update_stats(results)

        if results:
            first_pass = next((r for r in results if r.get('status') == 'PASS'), results[0])
            try:
                img = cv2.imread(first_pass['path'])
                self.viewer.set_image(img, 400, 210)
            except Exception:
                pass
            status = first_pass.get('status', 'IDLE')
            self.viewer.set_badge(f"{first_pass['name']}  —  {status}", status.lower())

        self.open_pdf_btn.setEnabled(True)
        self._pdf_path = pdf_path
        try:
            self.open_pdf_btn.clicked.disconnect()
        except TypeError:
            pass
        self.open_pdf_btn.clicked.connect(self._open_pdf)

    def _open_pdf(self):
        if not hasattr(self, '_pdf_path'): return
        if sys.platform == 'win32':
            os.startfile(self._pdf_path)
        elif sys.platform == 'darwin':
            os.system(f'open "{self._pdf_path}"')
        else:
            os.system(f'xdg-open "{self._pdf_path}"')


# ══════════════════════════════════════════════════════════════
# PAGE: STEPHOLE VIEWER
# ══════════════════════════════════════════════════════════════

class StepholeViewerPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{T['bg']};")
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(16)

        title = QLabel("Stephole Viewer")
        title.setStyleSheet(f"color:{T['text']};font-size:22px;font-weight:700;background:transparent;")
        root.addWidget(title)

        card = QFrame()
        card.setObjectName("SVCard")
        card.setStyleSheet(f"""
            #SVCard {{
                background:{T['surface']};
                border:1px solid {T['border']};
                border-radius:10px;
            }}
        """)
        cl = QVBoxLayout(card)
        cl.setContentsMargins(20, 16, 20, 16)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setStyleSheet("background:transparent;")
        self.grid_widget = QWidget()
        self.grid_widget.setStyleSheet("background:transparent;")
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(8)
        self.scroll.setWidget(self.grid_widget)
        cl.addWidget(self.scroll)
        root.addWidget(card, 1)

        self.empty_lbl = QLabel("Run the pipeline to view extracted stepholes here.")
        self.empty_lbl.setAlignment(Qt.AlignCenter)
        self.empty_lbl.setStyleSheet(f"color:{T['muted']};font-size:13px;background:transparent;")
        self.grid_layout.addWidget(self.empty_lbl, 0, 0)

    def populate(self, paths, results=None):
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        results_map = {}
        if results:
            for r in results:
                results_map[r['name']] = r

        COLS = 6
        for i, path in enumerate(paths):
            name = os.path.splitext(os.path.basename(path))[0]
            cell = QFrame()
            cell.setObjectName("SVCell")
            cell.setStyleSheet(f"""
                #SVCell {{
                    background:{T['surface']};
                    border:1px solid {T['border']};
                    border-radius:8px;
                }}
                #SVCell:hover {{
                    border-color:{T['accent']};
                }}
            """)
            cell.setFixedSize(110, 120)
            cl = QVBoxLayout(cell)
            cl.setContentsMargins(4, 4, 4, 4)
            cl.setSpacing(3)

            img_lbl = QLabel()
            img_lbl.setAlignment(Qt.AlignCenter)
            img_lbl.setFixedHeight(80)
            img_lbl.setStyleSheet(f"background:{T['surface2']};border-radius:5px;")
            try:
                img = cv2.imread(path)
                if img is not None:
                    pix = cv2_to_qpixmap(img, 100, 76)
                    img_lbl.setPixmap(pix)
            except Exception:
                pass
            cl.addWidget(img_lbl)

            name_lbl = QLabel(name)
            name_lbl.setAlignment(Qt.AlignCenter)
            name_lbl.setStyleSheet(f"color:{T['muted']};font-size:9px;background:transparent;")
            cl.addWidget(name_lbl)

            r_info = results_map.get(name)
            if r_info:
                status = r_info.get('status', '')
                color = T['pass_color'] if status == 'PASS' else T['fail_color']
                badge = QLabel(status)
                badge.setAlignment(Qt.AlignCenter)
                badge.setStyleSheet(f"color:{color};font-size:9px;font-weight:700;background:transparent;")
                cl.addWidget(badge)

            row, col = divmod(i, COLS)
            self.grid_layout.addWidget(cell, row, col)


# ══════════════════════════════════════════════════════════════
# PAGE: REPORTS
# ══════════════════════════════════════════════════════════════

class ReportsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{T['bg']};")
        self._pdf_path = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.setSpacing(16)

        hdr = QHBoxLayout()
        title = QLabel("Reports")
        title.setStyleSheet(f"color:{T['text']};font-size:22px;font-weight:700;background:transparent;")
        hdr.addWidget(title)
        hdr.addStretch()

        self._open_pdf_btn = QPushButton("📄  Open PDF Report")
        self._open_pdf_btn.setCursor(Qt.PointingHandCursor)
        self._open_pdf_btn.setEnabled(False)
        self._open_pdf_btn.setFixedHeight(38)
        self._open_pdf_btn.setStyleSheet(f"""
            QPushButton {{
                background:{T['accent']};
                color:#fff;
                border:none;
                border-radius:8px;
                font-size:12px;
                font-weight:600;
                padding:0 16px;
            }}
            QPushButton:hover {{ background:#27272a; }}
            QPushButton:disabled {{
                background:{T['surface2']};
                color:{T['muted']};
                border:1px solid {T['border']};
            }}
        """)
        self._open_pdf_btn.clicked.connect(self._open_pdf)
        hdr.addWidget(self._open_pdf_btn)
        outer.addLayout(hdr)

        self._stats_widget = QWidget()
        self._stats_widget.setStyleSheet("background:transparent;")
        stat_row = QHBoxLayout(self._stats_widget)
        stat_row.setSpacing(12)
        stat_row.setContentsMargins(0, 0, 0, 0)
        self._r_total = StatCard("Total Holes", "—", "🔢", T['accent'])
        self._r_pass  = StatCard("Pass",        "—", "✅", T['pass_color'])
        self._r_fail  = StatCard("Fail",        "—", "❌", T['fail_color'])
        self._r_rate  = StatCard("Pass Rate",   "—", "📊", T['warn'])
        for c in (self._r_total, self._r_pass, self._r_fail, self._r_rate):
            stat_row.addWidget(c)
        self._stats_widget.hide()
        outer.addWidget(self._stats_widget)

        self._table_card = TableCard()
        self._table_card.hide()
        outer.addWidget(self._table_card, 1)

        self._placeholder = QFrame()
        self._placeholder.setObjectName("RPlaceholder")
        self._placeholder.setStyleSheet(f"""
            #RPlaceholder {{
                background:{T['surface']};
                border:1px solid {T['border']};
                border-radius:12px;
            }}
        """)
        pl = QVBoxLayout(self._placeholder)
        pl.setAlignment(Qt.AlignCenter)
        pl.setSpacing(8)
        pl_icon = QLabel("📋")
        pl_icon.setAlignment(Qt.AlignCenter)
        pl_icon.setStyleSheet("font-size:52px;background:transparent;")
        pl_lbl = QLabel("No report yet")
        pl_lbl.setAlignment(Qt.AlignCenter)
        pl_lbl.setStyleSheet(f"color:{T['text']};font-size:18px;font-weight:700;background:transparent;")
        pl_sub = QLabel("Complete the pipeline to generate an inspection report.")
        pl_sub.setAlignment(Qt.AlignCenter)
        pl_sub.setStyleSheet(f"color:{T['muted']};font-size:13px;background:transparent;")
        pl.addStretch()
        pl.addWidget(pl_icon)
        pl.addWidget(pl_lbl)
        pl.addWidget(pl_sub)
        pl.addStretch()
        outer.addWidget(self._placeholder, 1)

    def populate(self, results, pdf_path):
        self._pdf_path = pdf_path
        total  = len(results)
        passed = sum(1 for r in results if r.get('status') == 'PASS')
        failed = total - passed
        rate   = f"{round(100 * passed / total)}%" if total else "—"

        self._r_total.set_value(total,  "holes inspected")
        self._r_pass.set_value(passed,  "holes passed")
        self._r_fail.set_value(failed,  "holes failed")
        self._r_rate.set_value(rate,    f"{passed}/{total} pass")

        self._table_card.populate(results)

        self._placeholder.hide()
        self._stats_widget.show()
        self._table_card.show()
        self._open_pdf_btn.setEnabled(True)

    def _open_pdf(self):
        if not self._pdf_path: return
        if sys.platform == 'win32':
            os.startfile(self._pdf_path)
        elif sys.platform == 'darwin':
            os.system(f'open "{self._pdf_path}"')
        else:
            os.system(f'xdg-open "{self._pdf_path}"')


# ══════════════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════════════

class HexaPipeline(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HexaVision — Inspection Pipeline")
        self.resize(1420, 900)
        self._apply_palette()

        self.ref_img     = None
        self.test_img    = None
        self.aligned_img = None
        self.saved_paths = []
        self.results     = []
        self.dm_result   = None   # Data Matrix decode result for test image
        self.output_dir  = os.path.join(os.path.expanduser("~"), "hexaboard_output")
        self._worker     = None

        self._build_ui()

    def _apply_palette(self):
        pal = QPalette()
        pal.setColor(QPalette.Window,       QColor(T['bg']))
        pal.setColor(QPalette.WindowText,   QColor(T['text']))
        pal.setColor(QPalette.Base,         QColor(T['surface']))
        pal.setColor(QPalette.AlternateBase,QColor(T['surface2']))
        pal.setColor(QPalette.Text,         QColor(T['text']))
        pal.setColor(QPalette.Button,       QColor(T['surface2']))
        pal.setColor(QPalette.ButtonText,   QColor(T['text']))
        pal.setColor(QPalette.Highlight,    QColor(T['accent']))
        pal.setColor(QPalette.HighlightedText, QColor('#ffffff'))
        self.setPalette(pal)
        self.setAutoFillBackground(True)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main = QHBoxLayout(central)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)

        self.sidebar = SidebarFrame()
        self.sidebar.page_changed.connect(self._on_page_change)
        main.addWidget(self.sidebar)

        self.stack = QStackedWidget()
        self.stack.setStyleSheet(f"background:{T['bg']};")

        self.pipeline_page   = PipelinePage()
        self.stephole_page   = StepholeViewerPage()
        self.ai_results_page = AIResultsPage()
        self.reports_page    = ReportsPage()

        self.stack.addWidget(self.pipeline_page)
        self.stack.addWidget(self.stephole_page)
        self.stack.addWidget(self.ai_results_page)
        self.stack.addWidget(self.reports_page)

        main.addWidget(self.stack, 1)

        pp = self.pipeline_page
        pp.ref_card.btn.clicked.connect(self.load_ref)
        pp.test_card.btn.clicked.connect(self.load_test)
        pp.pick_dir_btn.clicked.connect(self.pick_dir)
        pp.run_all_btn.clicked.connect(self.run_pipeline)

        pp.dir_lbl.setText(self.output_dir)

    def _on_page_change(self, idx):
        self.stack.setCurrentIndex(idx)

    def load_ref(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Reference Image", os.path.expanduser("~"),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not path: return
        img = cv2.imread(path)
        if img is None:
            QMessageBox.critical(self, "Error", "Cannot read image.")
            return
        self.ref_img = img
        self.pipeline_page.ref_card.set_preview(img)
        self._status(f"Reference loaded: {os.path.basename(path)}")
        self._refresh()

    def load_test(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Test Image", os.path.expanduser("~"),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not path: return
        img = cv2.imread(path)
        if img is None:
            QMessageBox.critical(self, "Error", "Cannot read image.")
            return
        self.test_img = img
        self.pipeline_page.test_card.set_preview(img)
        # ── Decode Data Matrix from test image ──────────────────
        self.dm_result = decode_data_matrix(img)
        if self.dm_result['found']:
            dm_msg = f"Data Matrix: {self.dm_result['data']}"
        elif self.dm_result['error']:
            dm_msg = f"Data Matrix: error ({self.dm_result['error']})"
        else:
            dm_msg = "Data Matrix: not detected"
        self._status(f"Test image loaded: {os.path.basename(path)}  |  {dm_msg}")
        self._refresh()

    def pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Choose Output Folder", os.path.expanduser("~"))
        if d:
            self.output_dir = d
            self.pipeline_page.dir_lbl.setText(d)

    def _refresh(self):
        ok = (self.ref_img is not None and self.test_img is not None)
        self.pipeline_page.run_all_btn.setEnabled(ok)

    def _status(self, msg, color=None):
        c = color or T['accent']
        self.pipeline_page.status_lbl.setText(msg)
        self.pipeline_page.status_lbl.setStyleSheet(
            f"color:{c};font-size:12px;font-weight:600;background:transparent;")

    def run_pipeline(self):
        pp = self.pipeline_page
        for card in (pp.card1, pp.card2, pp.card3):
            card.log.clear()
            card.progress.setValue(0)
            card.set_badge("Idle", "idle")
        pp.progress_card.set_global(0)
        for i in range(3):
            pp.progress_card.set_stage(i, 0)
        pp.progress_card.set_status("Running…", T['warn'])
        pp.run_all_btn.setEnabled(False)
        self._run_stage1()

    def _run_stage1(self):
        pp = self.pipeline_page
        self._status("Stage 1 — Correcting orientation…", T['stage1'])
        pp.card1.set_badge("Running", "running")
        self._worker = Stage1Worker(self.ref_img, self.test_img)
        self._worker.progress.connect(lambda p, m: (
            pp.card1.progress.setValue(p),
            pp.card1.log_msg(m),
            pp.progress_card.set_stage(0, p),
            pp.progress_card.set_global(int(p * 0.33)),
        ))
        self._worker.finished.connect(self._on_stage1_done)
        self._worker.error.connect(lambda e: (
            pp.card1.set_badge("Error", "error"),
            pp.card1.log_msg("ERROR: " + e.splitlines()[0]),
            self._status("Stage 1 failed", T['fail_color']),
            pp.progress_card.set_status("Error", T['fail_color']),
        ))
        self._worker.start()

    def _on_stage1_done(self, aligned, angle, info):
        pp = self.pipeline_page
        self.aligned_img = aligned
        pp.card1.set_badge("Done", "done")
        pp.card1.set_image(aligned)
        pp.card1.progress.setValue(100)
        pp.progress_card.set_stage(0, 100)
        pp.card1.log_msg("Total rotation: %.2f deg" % angle)
        pp.card1.log_msg("  Heatmap score: %.4f" % info['ncc_score'])
        self._status("Stage 1 complete — starting extraction…", T['accent'])
        self._run_stage2()

    def _run_stage2(self):
        pp = self.pipeline_page
        self._status("Stage 2 — Extracting stepholes…", T['stage2'])
        pp.card2.set_badge("Running", "running")
        out = os.path.join(self.output_dir, "stepholes")
        self._worker = Stage2Worker(self.ref_img, self.aligned_img, out)
        self._worker.progress.connect(lambda p, m: (
            pp.card2.progress.setValue(p),
            pp.card2.log_msg(m),
            pp.progress_card.set_stage(1, p),
            pp.progress_card.set_global(33 + int(p * 0.33)),
        ))
        self._worker.finished.connect(self._on_stage2_done)
        self._worker.error.connect(lambda e: (
            pp.card2.set_badge("Error", "error"),
            pp.card2.log_msg("ERROR: " + e.splitlines()[0]),
            self._status("Stage 2 failed", T['fail_color']),
            pp.progress_card.set_status("Error", T['fail_color']),
        ))
        self._worker.start()

    def _on_stage2_done(self, saved, overlay):
        pp = self.pipeline_page
        self.saved_paths = saved
        pp.card2.set_badge("Done", "done")
        pp.card2.set_image(overlay)
        pp.card2.progress.setValue(100)
        pp.progress_card.set_stage(1, 100)
        pp.card2.log_msg("%d stepholes saved" % len(saved))
        self.stephole_page.populate(saved)
        self._status("Stage 2 complete — running AI analysis…", T['accent'])
        self._run_stage3()

    def _run_stage3(self):
        pp = self.pipeline_page
        self._status("Stage 3 — AI inspection in progress…", T['stage3'])
        pp.card3.set_badge("Running", "running")
        module_name = os.path.basename(self.output_dir.rstrip('/\\'))
        self._worker = Stage3Worker(self.saved_paths, self.output_dir, module_name,
                                    dm_result=self.dm_result)
        self._worker.progress.connect(lambda p, m: (
            pp.card3.progress.setValue(p),
            pp.card3.log_msg(m),
            pp.progress_card.set_stage(2, p),
            pp.progress_card.set_global(66 + int(p * 0.34)),
        ))
        self._worker.finished.connect(self._on_stage3_done)
        self._worker.error.connect(lambda e: (
            pp.card3.set_badge("Error", "error"),
            pp.card3.log_msg("ERROR: " + e.splitlines()[0]),
            self._status("Stage 3 failed — check API key?", T['fail_color']),
            pp.progress_card.set_status("Error", T['fail_color']),
        ))
        self._worker.start()

    def _on_stage3_done(self, results, pdf_path):
        pp = self.pipeline_page
        self.results = results
        pp.card3.set_badge("Done", "done")
        pp.card3.progress.setValue(100)
        pp.progress_card.set_stage(2, 100)
        pp.progress_card.set_global(100)

        passed = sum(1 for r in results if r.get('status') == 'PASS')
        total  = len(results)
        rate   = round(100 * passed / total) if total else 0

        pp.card3.log_msg("Analysis complete: %d/%d PASS" % (passed, total))
        pp.card3.log_msg("  PDF: %s" % pdf_path)

        pp.stat_total.set_value(total,  "holes inspected")
        pp.stat_pass.set_value(passed,  "holes passed")
        pp.stat_fail.set_value(total - passed, "holes failed")
        pp.stat_rate.set_value(f"{rate}%", f"{passed}/{total} pass")

        pp.progress_card.set_status("Complete ✓", T['pass_color'])
        self._status(
            f"Pipeline complete — {passed}/{total} PASS  |  PDF saved",
            T['pass_color'])

        self.ai_results_page.populate(results, pdf_path)
        self.stephole_page.populate(self.saved_paths, results)
        self.reports_page.populate(results, pdf_path)

        pp.run_all_btn.setEnabled(True)


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    pal = QPalette()
    pal.setColor(QPalette.Window,          QColor(T['bg']))
    pal.setColor(QPalette.WindowText,      QColor(T['text']))
    pal.setColor(QPalette.Base,            QColor(T['surface']))
    pal.setColor(QPalette.AlternateBase,   QColor(T['surface2']))
    pal.setColor(QPalette.Text,            QColor(T['text']))
    pal.setColor(QPalette.Button,          QColor(T['surface2']))
    pal.setColor(QPalette.ButtonText,      QColor(T['text']))
    pal.setColor(QPalette.Highlight,       QColor(T['accent']))
    pal.setColor(QPalette.HighlightedText, QColor('#ffffff'))
    pal.setColor(QPalette.ToolTipBase,     QColor(T['surface']))
    pal.setColor(QPalette.ToolTipText,     QColor(T['text']))
    pal.setColor(QPalette.Mid,             QColor(T['border']))
    pal.setColor(QPalette.Dark,            QColor('#d4d4d8'))
    app.setPalette(pal)

    win = HexaPipeline()
    win.show()
    sys.exit(app.exec_())