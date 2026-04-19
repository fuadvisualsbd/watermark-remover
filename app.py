"""
WatermarkRemover Pro v2.0 — Streamlit Web App
==============================================
Professional watermark removal tool for images and short videos.
Runs entirely in the browser via Streamlit Cloud.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import tempfile
import os
import zipfile
import subprocess
import shutil

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="WatermarkRemover Pro",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global */
.stApp { font-family: 'Inter', sans-serif; }

/* Header */
.app-header {
    text-align: center;
    padding: 1rem 0 0.5rem;
}
.app-header h1 {
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
}
.app-header .brand { color: #E94560; }
.app-header .suffix { color: #8892A0; font-weight: 300; }
.app-header .version {
    font-size: 0.75rem;
    color: #8892A0;
    opacity: 0.6;
}

/* Cards */
.info-card {
    background: rgba(26, 26, 46, 0.7);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
}

/* Warning box */
.video-warn {
    background: rgba(255, 184, 48, 0.1);
    border: 1px solid rgba(255, 184, 48, 0.25);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    color: #FFB830;
    font-size: 0.85rem;
    line-height: 1.6;
}

/* Success box */
.success-box {
    background: rgba(0, 212, 116, 0.1);
    border: 1px solid rgba(0, 212, 116, 0.25);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    color: #00D474;
    font-size: 0.85rem;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: #1A1A2E;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #EAEAEA;
    font-size: 0.9rem;
    font-weight: 600;
    margin-top: 1rem;
}

/* Hide default Streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* File uploader style */
.stFileUploader > div {
    border: 2px dashed #2A2A4A !important;
    border-radius: 12px !important;
}
.stFileUploader > div:hover {
    border-color: #E94560 !important;
}

/* Download button */
.stDownloadButton > button {
    background: #00D474 !important;
    color: #0D0D0D !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
}
.stDownloadButton > button:hover {
    box-shadow: 0 0 20px rgba(0, 212, 116, 0.35) !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────

SUPPORTED_IMAGES = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif'}
SUPPORTED_VIDEOS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
MAX_VIDEO_DURATION = 8  # seconds

POSITION_PRESETS = {
    "Bottom-Right":  (0.70, 0.85, 1.0, 1.0),
    "Bottom-Left":   (0.0,  0.85, 0.30, 1.0),
    "Top-Right":     (0.70, 0.0,  1.0, 0.15),
    "Top-Left":      (0.0,  0.0,  0.30, 0.15),
    "Bottom-Center": (0.25, 0.85, 0.75, 1.0),
    "Top-Center":    (0.25, 0.0,  0.75, 0.15),
}

METHODS = {
    "TELEA (Fast)": cv2.INPAINT_TELEA,
    "Navier-Stokes (Quality)": cv2.INPAINT_NS,
}


# ─── Watermark Engine ────────────────────────────────────────────────────────

def get_roi_from_position(img_h, img_w, position_name):
    ratios = POSITION_PRESETS.get(position_name)
    if not ratios:
        return None
    return (
        int(img_w * ratios[0]), int(img_h * ratios[1]),
        int(img_w * ratios[2]), int(img_h * ratios[3]),
    )


def create_precise_mask(image, roi, threshold_value, dilate_size, dilate_iterations):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = max(0, roi[0]), max(0, roi[1]), min(w, roi[2]), min(h, roi[3])
    mask = np.zeros((h, w), dtype=np.uint8)

    roi_img = image[y1:y2, x1:x2]
    if roi_img.size == 0:
        return mask

    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY) if len(roi_img.shape) == 3 else roi_img.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 1)

    _, bright_mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

    combined = cv2.bitwise_or(bright_mask, edges)
    kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (dilate_size, dilate_size))
    combined = cv2.dilate(combined, kernel, iterations=dilate_iterations)

    mask[y1:y2, x1:x2] = combined
    return mask


def create_region_mask(image, roi):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = max(0, roi[0]), max(0, roi[1]), min(w, roi[2]), min(h, roi[3])
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def remove_watermark(image, mask, inpaint_radius, method):
    if mask.max() == 0:
        return image.copy()

    result = cv2.inpaint(image, mask, inpaint_radius, method)

    # Edge blending
    blend = mask.astype(np.float32) / 255.0
    blend = cv2.GaussianBlur(blend, (15, 15), 5)
    blend = np.clip(blend, 0, 1)
    blend_3ch = blend[:, :, np.newaxis]

    final = (result * blend_3ch + image * (1 - blend_3ch)).astype(np.uint8)
    return final


def process_single_image(image, position, mode, threshold_value, dilate_size,
                         dilate_iterations, inpaint_radius, method, custom_roi=None):
    h, w = image.shape[:2]

    if custom_roi:
        roi = custom_roi
    else:
        roi = get_roi_from_position(h, w, position)

    if mode == "Precise":
        mask = create_precise_mask(image, roi, threshold_value, dilate_size, dilate_iterations)
    else:
        mask = create_region_mask(image, roi)

    result = remove_watermark(image, mask, inpaint_radius, method)
    return result, mask


def process_video(video_path, output_path, position, mode, threshold_value,
                  dilate_size, dilate_iterations, inpaint_radius, method,
                  progress_bar=None, custom_roi=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    mask = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if mask is None:
            h, w = frame.shape[:2]
            roi = custom_roi or get_roi_from_position(h, w, position)
            if mode == "Precise":
                mask = create_precise_mask(frame, roi, threshold_value, dilate_size, dilate_iterations)
            else:
                mask = create_region_mask(frame, roi)

        result = remove_watermark(frame, mask, inpaint_radius, method)
        out.write(result)
        frame_count += 1

        if progress_bar and total_frames > 0:
            progress_bar.progress(frame_count / total_frames)

    cap.release()
    out.release()
    return frame_count


# ─── Helper Functions ─────────────────────────────────────────────────────────

def get_file_ext(filename):
    return os.path.splitext(filename)[1].lower()


def image_to_bytes(image, ext='.png'):
    if ext in ('.jpg', '.jpeg'):
        _, buf = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 98])
    elif ext == '.webp':
        _, buf = cv2.imencode('.webp', image, [cv2.IMWRITE_WEBP_QUALITY, 98])
    else:
        _, buf = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return buf.tobytes()


def draw_roi_overlay(image, position):
    """Draw ROI rectangle on image for preview."""
    overlay = image.copy()
    h, w = overlay.shape[:2]
    roi = get_roi_from_position(h, w, position)
    if roi:
        x1, y1, x2, y2 = roi
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (233, 69, 96), 2)
        # Semi-transparent fill
        sub = overlay[y1:y2, x1:x2]
        red_overlay = np.full_like(sub, (60, 20, 200))
        cv2.addWeighted(red_overlay, 0.15, sub, 0.85, 0, sub)
    return overlay


def create_mask_preview(image, mask):
    """Create mask overlay visualization."""
    overlay = image.copy()
    mask_colored = np.zeros_like(overlay)
    mask_colored[:, :, 2] = mask  # Red channel
    return cv2.addWeighted(overlay, 0.7, mask_colored, 0.5, 0)


# ─── Main App ────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div class="app-header">
        <h1><span class="brand">✦ WatermarkRemover</span> <span class="suffix">Pro</span></h1>
        <span class="version">v2.0 Web · Powered by OpenCV</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar Settings ──
    with st.sidebar:
        st.markdown("## ⚙️ Settings")

        st.markdown("### 📍 Watermark Position")
        position = st.selectbox(
            "Position", list(POSITION_PRESETS.keys()),
            index=0, label_visibility="collapsed"
        )

        st.markdown("### 🔬 Detection Mode")
        mode = st.radio(
            "Mode",
            ["Precise", "Region"],
            captions=["Text Detection — detects watermark pixels", "Full Area Fill — fills entire region"],
            label_visibility="collapsed"
        )

        st.markdown("### ⚡ Sensitivity")
        threshold_value = st.slider("Threshold", 100, 255, 200, help="Higher = only detect brighter watermark text")

        st.markdown("### 🎯 Inpaint Radius")
        inpaint_radius = st.slider("Radius", 1, 20, 7, help="Size of neighborhood for inpainting")

        st.markdown("### 🔲 Mask Expansion")
        dilate_iterations = st.slider("Dilation", 1, 10, 3, help="Expand the detection mask")

        st.markdown("### 🧠 Algorithm")
        method_name = st.selectbox("Algorithm", list(METHODS.keys()), index=0, label_visibility="collapsed")
        method = METHODS[method_name]

        st.divider()

        st.markdown("""
        <div class="info-card">
            <strong>💾 Output Info</strong><br>
            <small style="color:#8892A0">
            • Images download in original format<br>
            • Multiple files → ZIP download<br>
            • Original filenames preserved
            </small>
        </div>
        """, unsafe_allow_html=True)

    # ── Main Content ──
    uploaded_files = st.file_uploader(
        "📁 Drop images or videos here",
        type=["png", "jpg", "jpeg", "webp", "bmp", "tiff", "tif", "mp4", "avi", "mov", "webm"],
        accept_multiple_files=True,
        help="Supported: PNG, JPG, WebP, BMP, TIFF, MP4, WebM, AVI, MOV"
    )

    if not uploaded_files:
        st.markdown("""
        <div class="info-card" style="text-align:center; padding:3rem;">
            <div style="font-size:3rem; margin-bottom:0.5rem; opacity:0.3;">📷</div>
            <div style="color:#8892A0;">Upload images or videos to get started</div>
            <div style="color:#8892A0; font-size:0.75rem; margin-top:0.3rem;">
                Drag & drop or click above · All processing happens in your browser
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Check for videos
    has_video = any(get_file_ext(f.name) in SUPPORTED_VIDEOS for f in uploaded_files)
    if has_video:
        st.markdown("""
        <div class="video-warn">
            <strong>⚠️ Video Processing Info</strong><br>
            • Maximum 8 seconds video supported<br>
            • Output will have <strong>NO AUDIO</strong><br>
            • Processing may take a moment
        </div>
        """, unsafe_allow_html=True)

    # File info
    img_count = sum(1 for f in uploaded_files if get_file_ext(f.name) in SUPPORTED_IMAGES)
    vid_count = len(uploaded_files) - img_count
    st.caption(f"📋 {img_count} image{'s' if img_count != 1 else ''} · {vid_count} video{'s' if vid_count != 1 else ''}")

    # ── Preview first image ──
    first_image_file = next((f for f in uploaded_files if get_file_ext(f.name) in SUPPORTED_IMAGES), None)
    first_video_file = next((f for f in uploaded_files if get_file_ext(f.name) in SUPPORTED_VIDEOS), None)

    preview_file = first_image_file or first_video_file

    if preview_file:
        ext = get_file_ext(preview_file.name)

        if ext in SUPPORTED_IMAGES:
            file_bytes = np.frombuffer(preview_file.read(), np.uint8)
            preview_file.seek(0)
            preview_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        else:
            # Video: extract first frame
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(preview_file.read())
                preview_file.seek(0)
                tmp_path = tmp.name
            cap = cv2.VideoCapture(tmp_path)
            ret, preview_image = cap.read()
            cap.release()
            os.unlink(tmp_path)
            if not ret:
                preview_image = None

        if preview_image is not None:
            st.markdown("### 🔍 Preview")

            col1, col2, col3 = st.columns(3)

            # Original with ROI overlay
            with col1:
                st.caption("📷 Original + ROI")
                overlay = draw_roi_overlay(preview_image, position)
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                st.image(overlay_rgb, use_container_width=True)

            # Preview removal on first image
            if st.button("🔍 Preview Removal", type="secondary", use_container_width=True):
                with st.spinner("Generating preview..."):
                    result, mask = process_single_image(
                        preview_image, position, mode, threshold_value, 5,
                        dilate_iterations, inpaint_radius, method
                    )

                    with col2:
                        st.caption("🎭 Mask Detected")
                        mask_vis = create_mask_preview(preview_image, mask)
                        mask_rgb = cv2.cvtColor(mask_vis, cv2.COLOR_BGR2RGB)
                        st.image(mask_rgb, use_container_width=True)

                    with col3:
                        st.caption("✅ Result")
                        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                        st.image(result_rgb, use_container_width=True)

    st.divider()

    # ── Process All Button ──
    if st.button("▶  REMOVE WATERMARKS", type="primary", use_container_width=True):
        progress_bar = st.progress(0, text="Starting...")
        status = st.empty()
        results = []
        errors = []
        total = len(uploaded_files)

        for i, uploaded_file in enumerate(uploaded_files):
            name = uploaded_file.name
            ext = get_file_ext(name)
            status.text(f"Processing ({i+1}/{total}): {name}")

            try:
                if ext in SUPPORTED_IMAGES:
                    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                    uploaded_file.seek(0)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

                    if image is None:
                        raise ValueError("Cannot read image")

                    has_alpha = len(image.shape) == 3 and image.shape[2] == 4
                    if has_alpha:
                        alpha = image[:, :, 3]
                        image = image[:, :, :3]

                    result, _ = process_single_image(
                        image, position, mode, threshold_value, 5,
                        dilate_iterations, inpaint_radius, method
                    )

                    if has_alpha:
                        result = np.dstack([result, alpha])

                    out_bytes = image_to_bytes(result, ext)
                    results.append({"name": name, "data": out_bytes, "type": "image"})

                elif ext in SUPPORTED_VIDEOS:
                    # Save uploaded video to temp file
                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_in:
                        tmp_in.write(uploaded_file.read())
                        uploaded_file.seek(0)
                        input_path = tmp_in.name

                    # Check duration
                    cap = cv2.VideoCapture(input_path)
                    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1)
                    cap.release()

                    if duration > MAX_VIDEO_DURATION:
                        os.unlink(input_path)
                        raise ValueError(f"Video too long ({duration:.1f}s). Max {MAX_VIDEO_DURATION}s.")

                    output_path = input_path + "_out.mp4"
                    frame_count = process_video(
                        input_path, output_path, position, mode,
                        threshold_value, 5, dilate_iterations,
                        inpaint_radius, method, progress_bar
                    )

                    with open(output_path, 'rb') as f:
                        video_data = f.read()

                    results.append({
                        "name": os.path.splitext(name)[0] + "_processed.mp4",
                        "data": video_data,
                        "type": "video"
                    })

                    os.unlink(input_path)
                    os.unlink(output_path)

            except Exception as e:
                errors.append(f"{name}: {str(e)}")

            progress_bar.progress((i + 1) / total, text=f"{i+1}/{total} files processed")

        # ── Results ──
        if errors:
            status.warning(f"⚠️ {len(results)} succeeded, {len(errors)} errors")
            for err in errors:
                st.error(f"❌ {err}")
        else:
            status.success(f"✅ All {len(results)} files processed successfully!")

        if not results:
            return

        st.markdown("### 📥 Download Results")

        if len(results) == 1:
            r = results[0]
            mime = "video/mp4" if r["type"] == "video" else f"image/{get_file_ext(r['name'])[1:]}"
            st.download_button(
                f"⬇️ Download {r['name']}",
                data=r["data"],
                file_name=r["name"],
                mime=mime,
                use_container_width=True,
            )

            if r["type"] == "image":
                result_img = cv2.imdecode(np.frombuffer(r["data"], np.uint8), cv2.IMREAD_COLOR)
                if result_img is not None:
                    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                             caption="✅ Processed Result", use_container_width=True)
        else:
            # Create ZIP
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                for r in results:
                    zf.writestr(f"watermark_removed/{r['name']}", r["data"])
            zip_buffer.seek(0)

            st.download_button(
                f"⬇️ Download All ({len(results)} files) as ZIP",
                data=zip_buffer.getvalue(),
                file_name="watermark_removed.zip",
                mime="application/zip",
                use_container_width=True,
            )

            # Show thumbnails
            cols = st.columns(min(len(results), 4))
            for idx, r in enumerate(results[:4]):
                if r["type"] == "image":
                    result_img = cv2.imdecode(np.frombuffer(r["data"], np.uint8), cv2.IMREAD_COLOR)
                    if result_img is not None:
                        with cols[idx % 4]:
                            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB),
                                     caption=r["name"], use_container_width=True)

        if has_video:
            st.markdown("""
            <div class="video-warn" style="margin-top:1rem;">
                <strong>📢 Note:</strong> Video output has <strong>no audio</strong>.
                This is a browser limitation.
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
