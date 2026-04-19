"""
WatermarkRemover Pro v1.0
========================
Professional watermark removal tool for images and videos.
Supports batch processing, custom region selection, and multiple detection modes.
Built with CustomTkinter for a modern dark UI.

Usage:
    python watermark_remover.py

Build EXE:
    pyinstaller --onefile --windowed --name "WatermarkRemover Pro" watermark_remover.py
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import sys
import threading
from pathlib import Path
import time
import subprocess
import shutil

# ─── Constants ────────────────────────────────────────────────────────────────

APP_NAME = "WatermarkRemover Pro"
APP_VERSION = "1.0"

SUPPORTED_IMAGES = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif'}
SUPPORTED_VIDEOS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
SUPPORTED_ALL = SUPPORTED_IMAGES | SUPPORTED_VIDEOS

# Color palette
COLORS = {
    "bg_dark": "#0D0D0D",
    "bg_panel": "#1A1A2E",
    "bg_card": "#16213E",
    "accent": "#E94560",
    "accent_hover": "#FF6B6B",
    "accent_green": "#00D474",
    "accent_blue": "#0F3460",
    "text": "#EAEAEA",
    "text_dim": "#8892A0",
    "border": "#2A2A4A",
    "success": "#00D474",
    "warning": "#FFB830",
    "error": "#E94560",
    "preview_bg": "#0A0A1A",
}

# Position presets (x1_ratio, y1_ratio, x2_ratio, y2_ratio)
POSITION_PRESETS = {
    "Bottom-Right": (0.70, 0.85, 1.0, 1.0),
    "Bottom-Left": (0.0, 0.85, 0.30, 1.0),
    "Top-Right": (0.70, 0.0, 1.0, 0.15),
    "Top-Left": (0.0, 0.0, 0.30, 0.15),
    "Bottom-Center": (0.25, 0.85, 0.75, 1.0),
    "Top-Center": (0.25, 0.0, 0.75, 0.15),
}


# ─── Watermark Removal Engine ────────────────────────────────────────────────

class WatermarkEngine:
    """Core engine for watermark detection and removal using OpenCV inpainting."""

    METHODS = {
        "TELEA": cv2.INPAINT_TELEA,
        "Navier-Stokes": cv2.INPAINT_NS,
    }

    def __init__(self):
        self.inpaint_radius = 7
        self.threshold_value = 200
        self.dilate_size = 5
        self.dilate_iterations = 3
        self.method_name = "TELEA"
        self.mode = "Precise"  # "Precise" or "Region"

    @property
    def method(self):
        return self.METHODS.get(self.method_name, cv2.INPAINT_TELEA)

    def get_roi_from_position(self, img_h, img_w, position_name):
        """Convert position preset name to pixel coordinates."""
        ratios = POSITION_PRESETS.get(position_name)
        if not ratios:
            return None
        x1 = int(img_w * ratios[0])
        y1 = int(img_h * ratios[1])
        x2 = int(img_w * ratios[2])
        y2 = int(img_h * ratios[3])
        return (x1, y1, x2, y2)

    def create_precise_mask(self, image, roi):
        """Create a mask that detects text-like watermark pixels within the ROI."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = roi

        # Clamp ROI
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        roi_img = image[y1:y2, x1:x2]
        if roi_img.size == 0:
            return np.zeros((h, w), dtype=np.uint8)

        # Convert to grayscale
        if len(roi_img.shape) == 3:
            gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_img.copy()

        # Slight blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 1)

        # Binary threshold to detect bright watermark text
        _, bright_mask = cv2.threshold(
            gray, self.threshold_value, 255, cv2.THRESH_BINARY
        )

        # Also try edge detection for semi-transparent watermarks
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)

        # Combine both detections
        combined = cv2.bitwise_or(bright_mask, edges)

        # Morphological dilation to expand the mask
        kernel = cv2.getStructuringElement(
            cv2.MORPH_DILATE, (self.dilate_size, self.dilate_size)
        )
        combined = cv2.dilate(combined, kernel, iterations=self.dilate_iterations)

        # Create full-size mask
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = combined

        return mask

    def create_region_mask(self, image, roi):
        """Create a solid mask covering the entire ROI (for smooth backgrounds)."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = roi

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255

        return mask

    def create_mask(self, image, roi):
        """Create watermark mask based on current mode."""
        if self.mode == "Precise":
            return self.create_precise_mask(image, roi)
        else:
            return self.create_region_mask(image, roi)

    def remove_watermark(self, image, mask):
        """Remove watermark using OpenCV inpainting."""
        if mask.max() == 0:
            return image.copy()

        result = cv2.inpaint(image, mask, self.inpaint_radius, self.method)

        # Edge blending for smoother result
        # Create a feathered blend mask
        blend = mask.astype(np.float32) / 255.0
        blend = cv2.GaussianBlur(blend, (15, 15), 5)
        blend = np.clip(blend, 0, 1)
        blend_3ch = blend[:, :, np.newaxis]

        # Blend inpainted result with original at the edges
        final = (result * blend_3ch + image * (1 - blend_3ch)).astype(np.uint8)

        return final

    def process_image(self, image_path, output_path, position="Bottom-Right", custom_roi=None):
        """Process a single image file."""
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")

        # Handle alpha channel
        has_alpha = len(image.shape) == 3 and image.shape[2] == 4
        if has_alpha:
            alpha = image[:, :, 3]
            image = image[:, :, :3]

        h, w = image.shape[:2]

        if custom_roi:
            roi = custom_roi
        else:
            roi = self.get_roi_from_position(h, w, position)
            if not roi:
                raise ValueError(f"Unknown position: {position}")

        mask = self.create_mask(image, roi)
        result = self.remove_watermark(image, mask)

        if has_alpha:
            result = np.dstack([result, alpha])

        # Determine output quality
        ext = Path(output_path).suffix.lower()
        if ext in ('.jpg', '.jpeg'):
            cv2.imwrite(str(output_path), result, [cv2.IMWRITE_JPEG_QUALITY, 98])
        elif ext == '.png':
            cv2.imwrite(str(output_path), result, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        elif ext == '.webp':
            cv2.imwrite(str(output_path), result, [cv2.IMWRITE_WEBP_QUALITY, 98])
        else:
            cv2.imwrite(str(output_path), result)

        return result

    def process_video(self, video_path, output_path, position="Bottom-Right",
                      custom_roi=None, progress_callback=None, remove_audio=False):
        """Process a video file frame by frame."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Use mp4v codec (widely compatible)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_output = str(output_path) + ".temp.mp4"
        out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

        if not out.isOpened():
            cap.release()
            raise ValueError("Cannot create output video writer")

        frame_count = 0
        mask = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Create mask once from first frame (watermark is fixed position)
            if mask is None:
                if custom_roi:
                    roi = custom_roi
                else:
                    roi = self.get_roi_from_position(height, width, position)
                mask = self.create_mask(frame, roi)

            result = self.remove_watermark(frame, mask)
            out.write(result)

            frame_count += 1
            if progress_callback and total_frames > 0:
                progress_callback(frame_count / total_frames)

        cap.release()
        out.release()

        final_output = str(output_path)
        ffmpeg_path = shutil.which("ffmpeg")

        if remove_audio:
            # No audio needed — just rename temp file
            if ffmpeg_path:
                try:
                    cmd = [
                        ffmpeg_path, "-y",
                        "-i", temp_output,
                        "-c:v", "copy",
                        "-an",
                        final_output
                    ]
                    subprocess.run(cmd, capture_output=True, timeout=300)
                    os.remove(temp_output)
                except Exception:
                    if os.path.exists(final_output):
                        os.remove(final_output)
                    os.rename(temp_output, final_output)
            else:
                # No ffmpeg, OpenCV output has no audio anyway
                if os.path.exists(final_output):
                    os.remove(final_output)
                os.rename(temp_output, final_output)
        else:
            # Keep original audio — merge using ffmpeg
            if ffmpeg_path:
                try:
                    cmd = [
                        ffmpeg_path, "-y",
                        "-i", temp_output,
                        "-i", str(video_path),
                        "-c:v", "copy",
                        "-c:a", "aac",
                        "-map", "0:v:0",
                        "-map", "1:a:0?",
                        "-shortest",
                        final_output
                    ]
                    subprocess.run(cmd, capture_output=True, timeout=300)
                    os.remove(temp_output)
                except Exception:
                    if os.path.exists(final_output):
                        os.remove(final_output)
                    os.rename(temp_output, final_output)
            else:
                # No ffmpeg available, output without audio
                if os.path.exists(final_output):
                    os.remove(final_output)
                os.rename(temp_output, final_output)

        return frame_count


# ─── Preview Canvas ──────────────────────────────────────────────────────────

class PreviewCanvas(tk.Canvas):
    """Custom canvas for image preview with interactive ROI selection."""

    def __init__(self, parent, on_roi_change=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.cv_image = None
        self.display_image = None
        self.photo = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        # ROI selection state
        self.roi_start = None
        self.roi_end = None
        self.roi_rect_id = None
        self.preset_rect_id = None
        self.on_roi_change = on_roi_change
        self.selection_enabled = False

        # Bind mouse events
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Configure>", self._on_resize)

        # Placeholder text
        self._show_placeholder()

    def _show_placeholder(self):
        self.delete("all")
        self.create_text(
            self.winfo_reqwidth() // 2, self.winfo_reqheight() // 2,
            text="📷 Select a file to preview",
            fill=COLORS["text_dim"], font=("Segoe UI", 14)
        )

    def set_image(self, cv_image):
        """Display an OpenCV image on the canvas."""
        self.cv_image = cv_image.copy()
        self._render()

    def _render(self):
        """Render the current image to the canvas."""
        if self.cv_image is None:
            return

        canvas_w = self.winfo_width()
        canvas_h = self.winfo_height()
        if canvas_w <= 1 or canvas_h <= 1:
            return

        # Convert BGR to RGB
        if len(self.cv_image.shape) == 3:
            rgb = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(self.cv_image, cv2.COLOR_GRAY2RGB)

        pil_img = Image.fromarray(rgb)
        img_w, img_h = pil_img.size

        # Calculate scale to fit canvas
        self.scale = min(canvas_w / img_w, canvas_h / img_h, 1.0)
        new_w = max(1, int(img_w * self.scale))
        new_h = max(1, int(img_h * self.scale))

        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        self.display_image = pil_img
        self.photo = ImageTk.PhotoImage(pil_img)

        self.offset_x = (canvas_w - new_w) // 2
        self.offset_y = (canvas_h - new_h) // 2

        self.delete("all")
        self.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.photo, tags="image")

    def show_preset_roi(self, position_name):
        """Show a rectangle overlay for the preset position."""
        if self.cv_image is None:
            return

        if self.preset_rect_id:
            self.delete(self.preset_rect_id)
            self.preset_rect_id = None

        ratios = POSITION_PRESETS.get(position_name)
        if not ratios:
            return

        h, w = self.cv_image.shape[:2]
        x1 = int(w * ratios[0] * self.scale) + self.offset_x
        y1 = int(h * ratios[1] * self.scale) + self.offset_y
        x2 = int(w * ratios[2] * self.scale) + self.offset_x
        y2 = int(h * ratios[3] * self.scale) + self.offset_y

        self.preset_rect_id = self.create_rectangle(
            x1, y1, x2, y2,
            outline=COLORS["accent"], width=2, dash=(6, 4), tags="roi_preset"
        )

    def _on_press(self, event):
        if not self.selection_enabled or self.cv_image is None:
            return
        self.roi_start = (event.x, event.y)
        if self.roi_rect_id:
            self.delete(self.roi_rect_id)
            self.roi_rect_id = None

    def _on_drag(self, event):
        if not self.selection_enabled or self.roi_start is None:
            return
        if self.roi_rect_id:
            self.delete(self.roi_rect_id)
        self.roi_rect_id = self.create_rectangle(
            self.roi_start[0], self.roi_start[1], event.x, event.y,
            outline=COLORS["warning"], width=2, dash=(5, 3), tags="roi_custom"
        )

    def _on_release(self, event):
        if not self.selection_enabled or self.roi_start is None:
            return
        self.roi_end = (event.x, event.y)
        roi = self.get_image_roi()
        if roi and self.on_roi_change:
            self.on_roi_change(roi)

    def _on_resize(self, event):
        if self.cv_image is not None:
            self.after(50, self._render)

    def get_image_roi(self):
        """Convert canvas ROI coordinates to original image coordinates."""
        if not self.roi_start or not self.roi_end or self.cv_image is None:
            return None

        h, w = self.cv_image.shape[:2]

        x1 = int((min(self.roi_start[0], self.roi_end[0]) - self.offset_x) / self.scale)
        y1 = int((min(self.roi_start[1], self.roi_end[1]) - self.offset_y) / self.scale)
        x2 = int((max(self.roi_start[0], self.roi_end[0]) - self.offset_x) / self.scale)
        y2 = int((max(self.roi_start[1], self.roi_end[1]) - self.offset_y) / self.scale)

        # Clamp to image bounds
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 - x1 < 5 or y2 - y1 < 5:
            return None

        return (x1, y1, x2, y2)

    def enable_selection(self, enabled=True):
        self.selection_enabled = enabled
        if enabled:
            self.config(cursor="crosshair")
        else:
            self.config(cursor="")

    def clear_custom_roi(self):
        if self.roi_rect_id:
            self.delete(self.roi_rect_id)
            self.roi_rect_id = None
        self.roi_start = None
        self.roi_end = None

    def clear(self):
        self.cv_image = None
        self.photo = None
        self.clear_custom_roi()
        if self.preset_rect_id:
            self.delete(self.preset_rect_id)
            self.preset_rect_id = None
        self._show_placeholder()


# ─── Main Application ────────────────────────────────────────────────────────

class App(ctk.CTk):
    """WatermarkRemover Pro main application."""

    def __init__(self):
        super().__init__()

        self.title(f"✦ {APP_NAME}")
        self.geometry("1280x820")
        self.minsize(1000, 650)

        # Dark theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.configure(fg_color=COLORS["bg_dark"])

        # State
        self.engine = WatermarkEngine()
        self.file_list = []
        self.current_index = -1
        self.current_image = None
        self.current_result = None
        self.current_mask = None
        self.custom_roi = None
        self.processing = False
        self.view_mode = "original"  # "original", "mask", "result"

        # Build UI
        self._create_ui()

    # ── UI Creation ──────────────────────────────────────────────────────────

    def _create_ui(self):
        """Build the complete user interface."""
        # Header
        self._create_header()

        # Main content area
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.content.grid_columnconfigure(0, weight=3)
        self.content.grid_columnconfigure(1, weight=1)
        self.content.grid_rowconfigure(0, weight=1)

        self._create_left_panel()
        self._create_right_panel()

        # Status bar
        self._create_status_bar()

    def _create_header(self):
        """Create the app header with title and file controls."""
        header = ctk.CTkFrame(self, fg_color=COLORS["bg_panel"], corner_radius=0, height=56)
        header.pack(fill="x", padx=0, pady=0)
        header.pack_propagate(False)

        # App title
        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.pack(side="left", padx=16)

        ctk.CTkLabel(
            title_frame, text="✦ WatermarkRemover",
            font=("Segoe UI", 20, "bold"),
            text_color=COLORS["accent"]
        ).pack(side="left")

        ctk.CTkLabel(
            title_frame, text="  Pro",
            font=("Segoe UI", 20),
            text_color=COLORS["text_dim"]
        ).pack(side="left")

        ctk.CTkLabel(
            title_frame, text=f"  v{APP_VERSION}",
            font=("Segoe UI", 11),
            text_color=COLORS["text_dim"]
        ).pack(side="left", padx=(4, 0))

        # File control buttons
        btn_frame = ctk.CTkFrame(header, fg_color="transparent")
        btn_frame.pack(side="right", padx=16)

        ctk.CTkButton(
            btn_frame, text="📁 Add Files", width=110, height=34,
            font=("Segoe UI", 12), corner_radius=8,
            fg_color=COLORS["accent_blue"], hover_color=COLORS["accent"],
            command=self._add_files
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            btn_frame, text="📂 Add Folder", width=120, height=34,
            font=("Segoe UI", 12), corner_radius=8,
            fg_color=COLORS["accent_blue"], hover_color=COLORS["accent"],
            command=self._add_folder
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            btn_frame, text="🗑 Clear All", width=100, height=34,
            font=("Segoe UI", 12), corner_radius=8,
            fg_color="#333345", hover_color=COLORS["error"],
            command=self._clear_files
        ).pack(side="left", padx=4)

    def _create_left_panel(self):
        """Create the left panel with file list and preview."""
        left = ctk.CTkFrame(self.content, fg_color=COLORS["bg_panel"], corner_radius=12)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=0)
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        # ── File list (top part) ──
        list_frame = ctk.CTkFrame(left, fg_color="transparent", height=160)
        list_frame.grid(row=0, column=0, sticky="new", padx=10, pady=(10, 4))
        list_frame.grid_columnconfigure(0, weight=1)
        list_frame.pack_propagate(False)

        ctk.CTkLabel(
            list_frame, text="📋 Files",
            font=("Segoe UI", 13, "bold"),
            text_color=COLORS["text"]
        ).pack(anchor="w", padx=4, pady=(2, 4))

        # Scrollable file list
        self.file_listbox = tk.Listbox(
            list_frame,
            bg=COLORS["bg_card"], fg=COLORS["text"],
            selectbackground=COLORS["accent"],
            selectforeground="white",
            font=("Cascadia Code", 10),
            borderwidth=0, highlightthickness=1,
            highlightcolor=COLORS["border"],
            highlightbackground=COLORS["border"],
            activestyle="none",
            relief="flat"
        )
        self.file_listbox.pack(fill="both", expand=True, padx=4)
        self.file_listbox.bind("<<ListboxSelect>>", self._on_file_select)

        # ── Preview area (bottom part, takes most space) ──
        preview_frame = ctk.CTkFrame(left, fg_color="transparent")
        preview_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(4, 10))
        preview_frame.grid_rowconfigure(1, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)

        # Preview header with view toggle buttons
        prev_header = ctk.CTkFrame(preview_frame, fg_color="transparent")
        prev_header.grid(row=0, column=0, sticky="ew", pady=(0, 4))

        ctk.CTkLabel(
            prev_header, text="🔍 Preview",
            font=("Segoe UI", 13, "bold"),
            text_color=COLORS["text"]
        ).pack(side="left", padx=4)

        # View mode buttons
        self.btn_view_result = ctk.CTkButton(
            prev_header, text="✅ Result", width=80, height=28,
            font=("Segoe UI", 11), corner_radius=6,
            fg_color="#333345", hover_color=COLORS["success"],
            command=lambda: self._set_view("result")
        )
        self.btn_view_result.pack(side="right", padx=2)

        self.btn_view_mask = ctk.CTkButton(
            prev_header, text="🎭 Mask", width=80, height=28,
            font=("Segoe UI", 11), corner_radius=6,
            fg_color="#333345", hover_color=COLORS["warning"],
            command=lambda: self._set_view("mask")
        )
        self.btn_view_mask.pack(side="right", padx=2)

        self.btn_view_original = ctk.CTkButton(
            prev_header, text="📷 Original", width=90, height=28,
            font=("Segoe UI", 11), corner_radius=6,
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            command=lambda: self._set_view("original")
        )
        self.btn_view_original.pack(side="right", padx=2)

        # Preview canvas
        self.preview = PreviewCanvas(
            preview_frame,
            on_roi_change=self._on_custom_roi,
            bg=COLORS["preview_bg"],
            highlightthickness=1,
            highlightbackground=COLORS["border"]
        )
        self.preview.grid(row=1, column=0, sticky="nsew")

    def _create_right_panel(self):
        """Create the right settings panel."""
        right = ctk.CTkFrame(self.content, fg_color=COLORS["bg_panel"], corner_radius=12)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=0)

        # Scrollable settings
        settings_scroll = ctk.CTkScrollableFrame(
            right, fg_color="transparent",
            scrollbar_button_color=COLORS["border"],
            scrollbar_button_hover_color=COLORS["accent"]
        )
        settings_scroll.pack(fill="both", expand=True, padx=6, pady=10)

        # ── Position ──
        self._section_label(settings_scroll, "📍 Watermark Position")

        self.position_var = ctk.StringVar(value="Bottom-Right")
        self.position_menu = ctk.CTkOptionMenu(
            settings_scroll,
            values=list(POSITION_PRESETS.keys()) + ["✏️ Custom (Draw on Preview)"],
            variable=self.position_var,
            font=("Segoe UI", 12),
            dropdown_font=("Segoe UI", 11),
            fg_color=COLORS["bg_card"],
            button_color=COLORS["accent_blue"],
            button_hover_color=COLORS["accent"],
            dropdown_fg_color=COLORS["bg_card"],
            dropdown_hover_color=COLORS["accent"],
            corner_radius=8,
            command=self._on_position_change
        )
        self.position_menu.pack(fill="x", padx=8, pady=(2, 10))

        # ── Detection Mode ──
        self._section_label(settings_scroll, "🔬 Detection Mode")

        self.mode_var = ctk.StringVar(value="Precise")
        mode_frame = ctk.CTkFrame(settings_scroll, fg_color="transparent")
        mode_frame.pack(fill="x", padx=8, pady=(2, 6))

        ctk.CTkRadioButton(
            mode_frame, text="Precise (Text Detection)",
            variable=self.mode_var, value="Precise",
            font=("Segoe UI", 11),
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            border_color=COLORS["border"],
            command=self._on_settings_change
        ).pack(anchor="w", pady=2)

        ctk.CTkRadioButton(
            mode_frame, text="Region (Full Area Fill)",
            variable=self.mode_var, value="Region",
            font=("Segoe UI", 11),
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            border_color=COLORS["border"],
            command=self._on_settings_change
        ).pack(anchor="w", pady=2)

        # ── Sensitivity ──
        self._section_label(settings_scroll, "⚡ Sensitivity")

        self.threshold_var = ctk.IntVar(value=200)
        self.threshold_label = ctk.CTkLabel(
            settings_scroll, text="Threshold: 200",
            font=("Segoe UI", 11), text_color=COLORS["text_dim"]
        )
        self.threshold_label.pack(anchor="w", padx=12)

        self.threshold_slider = ctk.CTkSlider(
            settings_scroll, from_=100, to=255,
            variable=self.threshold_var,
            fg_color=COLORS["bg_card"],
            progress_color=COLORS["accent"],
            button_color=COLORS["accent"],
            button_hover_color=COLORS["accent_hover"],
            command=self._on_threshold_change
        )
        self.threshold_slider.pack(fill="x", padx=8, pady=(2, 10))

        # ── Inpaint Radius ──
        self._section_label(settings_scroll, "🎯 Inpaint Radius")

        self.radius_var = ctk.IntVar(value=7)
        self.radius_label = ctk.CTkLabel(
            settings_scroll, text="Radius: 7",
            font=("Segoe UI", 11), text_color=COLORS["text_dim"]
        )
        self.radius_label.pack(anchor="w", padx=12)

        self.radius_slider = ctk.CTkSlider(
            settings_scroll, from_=1, to=20,
            variable=self.radius_var,
            fg_color=COLORS["bg_card"],
            progress_color=COLORS["accent_green"],
            button_color=COLORS["accent_green"],
            button_hover_color="#33FF99",
            command=self._on_radius_change
        )
        self.radius_slider.pack(fill="x", padx=8, pady=(2, 10))

        # ── Dilation ──
        self._section_label(settings_scroll, "🔲 Mask Expansion")

        self.dilate_var = ctk.IntVar(value=3)
        self.dilate_label = ctk.CTkLabel(
            settings_scroll, text="Dilation: 3",
            font=("Segoe UI", 11), text_color=COLORS["text_dim"]
        )
        self.dilate_label.pack(anchor="w", padx=12)

        self.dilate_slider = ctk.CTkSlider(
            settings_scroll, from_=1, to=10,
            variable=self.dilate_var,
            fg_color=COLORS["bg_card"],
            progress_color=COLORS["warning"],
            button_color=COLORS["warning"],
            button_hover_color="#FFD700",
            command=self._on_dilate_change
        )
        self.dilate_slider.pack(fill="x", padx=8, pady=(2, 10))

        # ── Inpaint Method ──
        self._section_label(settings_scroll, "🧠 Algorithm")

        self.method_var = ctk.StringVar(value="TELEA")
        self.method_menu = ctk.CTkOptionMenu(
            settings_scroll,
            values=["TELEA", "Navier-Stokes"],
            variable=self.method_var,
            font=("Segoe UI", 12),
            fg_color=COLORS["bg_card"],
            button_color=COLORS["accent_blue"],
            button_hover_color=COLORS["accent"],
            dropdown_fg_color=COLORS["bg_card"],
            dropdown_hover_color=COLORS["accent"],
            corner_radius=8,
            command=self._on_settings_change
        )
        self.method_menu.pack(fill="x", padx=8, pady=(2, 10))

        # ── Video Audio ──
        self._section_label(settings_scroll, "🔊 Video Audio")

        self.remove_audio_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            settings_scroll, text="Remove Audio from Video",
            variable=self.remove_audio_var,
            font=("Segoe UI", 11),
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            border_color=COLORS["border"],
            checkmark_color="white"
        ).pack(anchor="w", padx=8, pady=(2, 2))

        ctk.CTkLabel(
            settings_scroll,
            text="Checked = no audio in output\nUnchecked = keep original audio",
            font=("Segoe UI", 10), text_color=COLORS["text_dim"],
            justify="left"
        ).pack(anchor="w", padx=28, pady=(0, 6))

        # ── Output Settings ──
        self._section_label(settings_scroll, "💾 Output")

        ctk.CTkLabel(
            settings_scroll,
            text='📂 Saves to "watermark removed" folder\n     inside input file\'s directory.\n     Original filenames are preserved.',
            font=("Segoe UI", 10), text_color=COLORS["text_dim"],
            justify="left"
        ).pack(anchor="w", padx=12, pady=(2, 6))

        # ── Spacer ──
        ctk.CTkFrame(settings_scroll, fg_color="transparent", height=10).pack()

        # ── Action Buttons ──
        self.preview_btn = ctk.CTkButton(
            settings_scroll, text="🔍 Preview Removal", height=40,
            font=("Segoe UI", 13, "bold"), corner_radius=10,
            fg_color=COLORS["accent_blue"],
            hover_color="#1A5276",
            command=self._preview_removal
        )
        self.preview_btn.pack(fill="x", padx=8, pady=(6, 4))

        self.process_btn = ctk.CTkButton(
            settings_scroll, text="▶  REMOVE WATERMARKS", height=48,
            font=("Segoe UI", 14, "bold"), corner_radius=10,
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            command=self._start_processing
        )
        self.process_btn.pack(fill="x", padx=8, pady=(4, 6))

        # Progress bar
        self.progress_var = ctk.DoubleVar(value=0)
        self.progress_bar = ctk.CTkProgressBar(
            settings_scroll,
            variable=self.progress_var,
            fg_color=COLORS["bg_card"],
            progress_color=COLORS["accent_green"],
            corner_radius=6,
            height=8
        )
        self.progress_bar.pack(fill="x", padx=8, pady=(2, 4))
        self.progress_bar.set(0)

        self.progress_label = ctk.CTkLabel(
            settings_scroll, text="",
            font=("Segoe UI", 10), text_color=COLORS["text_dim"]
        )
        self.progress_label.pack(anchor="w", padx=12)

    def _create_status_bar(self):
        """Create bottom status bar."""
        self.status_bar = ctk.CTkFrame(
            self, fg_color=COLORS["bg_panel"],
            corner_radius=0, height=30
        )
        self.status_bar.pack(fill="x", side="bottom")
        self.status_bar.pack_propagate(False)

        self.status_label = ctk.CTkLabel(
            self.status_bar, text="  ✦ Ready — Add files to get started",
            font=("Segoe UI", 11), text_color=COLORS["text_dim"],
            anchor="w"
        )
        self.status_label.pack(side="left", padx=8, fill="x", expand=True)

        self.file_count_label = ctk.CTkLabel(
            self.status_bar, text="0 files",
            font=("Segoe UI", 11), text_color=COLORS["text_dim"]
        )
        self.file_count_label.pack(side="right", padx=12)

    def _section_label(self, parent, text):
        """Create a styled section label."""
        ctk.CTkLabel(
            parent, text=text,
            font=("Segoe UI", 12, "bold"),
            text_color=COLORS["text"]
        ).pack(anchor="w", padx=8, pady=(12, 2))

    # ── File Management ──────────────────────────────────────────────────────

    def _add_files(self):
        """Open file dialog to add files."""
        filetypes = [
            ("Supported Files", " ".join(f"*{ext}" for ext in SUPPORTED_ALL)),
            ("Images", " ".join(f"*{ext}" for ext in SUPPORTED_IMAGES)),
            ("Videos", " ".join(f"*{ext}" for ext in SUPPORTED_VIDEOS)),
            ("All Files", "*.*")
        ]
        files = filedialog.askopenfilenames(
            title="Select Images/Videos",
            filetypes=filetypes
        )
        if files:
            for f in files:
                if f not in self.file_list:
                    self.file_list.append(f)
            self._update_file_listbox()

    def _add_folder(self):
        """Add all supported files from a folder."""
        folder = filedialog.askdirectory(title="Select Folder")
        if folder:
            count = 0
            for f in Path(folder).iterdir():
                if f.is_file() and f.suffix.lower() in SUPPORTED_ALL:
                    fpath = str(f)
                    if fpath not in self.file_list:
                        self.file_list.append(fpath)
                        count += 1
            self._update_file_listbox()
            self._set_status(f"Added {count} files from folder")

    def _clear_files(self):
        """Clear all files from the list."""
        self.file_list.clear()
        self.current_index = -1
        self.current_image = None
        self.current_result = None
        self.current_mask = None
        self.custom_roi = None
        self.preview.clear()
        self._update_file_listbox()
        self._set_status("File list cleared")

    def _update_file_listbox(self):
        """Refresh the file listbox display."""
        self.file_listbox.delete(0, tk.END)
        for f in self.file_list:
            ext = Path(f).suffix.lower()
            icon = "🎬" if ext in SUPPORTED_VIDEOS else "📷"
            name = Path(f).name
            self.file_listbox.insert(tk.END, f"  {icon}  {name}")

        count = len(self.file_list)
        imgs = sum(1 for f in self.file_list if Path(f).suffix.lower() in SUPPORTED_IMAGES)
        vids = count - imgs
        self.file_count_label.configure(
            text=f"{imgs} img{'s' if imgs != 1 else ''} · {vids} vid{'s' if vids != 1 else ''}"
        )

    def _on_file_select(self, event):
        """Handle file selection in the listbox."""
        selection = self.file_listbox.curselection()
        if not selection:
            return

        idx = selection[0]
        if idx == self.current_index:
            return

        self.current_index = idx
        self.current_result = None
        self.current_mask = None
        self.view_mode = "original"
        self._update_view_buttons()

        filepath = self.file_list[idx]
        ext = Path(filepath).suffix.lower()

        try:
            if ext in SUPPORTED_IMAGES:
                self.current_image = cv2.imread(filepath, cv2.IMREAD_COLOR)
                if self.current_image is None:
                    raise ValueError("Cannot read image")
                self.preview.set_image(self.current_image)
                self._show_position_overlay()
                self._set_status(f"Loaded: {Path(filepath).name}")
            elif ext in SUPPORTED_VIDEOS:
                cap = cv2.VideoCapture(filepath)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    self.current_image = frame
                    self.preview.set_image(frame)
                    self._show_position_overlay()
                    self._set_status(f"Loaded video preview: {Path(filepath).name}")
                else:
                    raise ValueError("Cannot read video")
        except Exception as e:
            self._set_status(f"Error: {e}", error=True)

    # ── Settings Handlers ────────────────────────────────────────────────────

    def _on_position_change(self, value):
        """Handle position preset change."""
        if "Custom" in value:
            self.preview.enable_selection(True)
            self.preview.clear_custom_roi()
            self.custom_roi = None
            self._set_status("🎯 Draw a rectangle on the preview to select watermark region")
        else:
            self.preview.enable_selection(False)
            self.preview.clear_custom_roi()
            self.custom_roi = None
            self._show_position_overlay()
            self._set_status(f"Position set to: {value}")

    def _on_custom_roi(self, roi):
        """Handle custom ROI selection from preview canvas."""
        self.custom_roi = roi
        self._set_status(f"Custom region selected: ({roi[0]}, {roi[1]}) → ({roi[2]}, {roi[3]})")

    def _show_position_overlay(self):
        """Show position rectangle on preview."""
        pos = self.position_var.get()
        if "Custom" not in pos:
            self.preview.show_preset_roi(pos)

    def _on_threshold_change(self, value):
        val = int(value)
        self.threshold_label.configure(text=f"Threshold: {val}")

    def _on_radius_change(self, value):
        val = int(value)
        self.radius_label.configure(text=f"Radius: {val}")

    def _on_dilate_change(self, value):
        val = int(value)
        self.dilate_label.configure(text=f"Dilation: {val}")

    def _on_settings_change(self, *args):
        pass



    # ── Preview ──────────────────────────────────────────────────────────────

    def _set_view(self, mode):
        """Switch preview view mode."""
        self.view_mode = mode
        self._update_view_buttons()

        if mode == "original" and self.current_image is not None:
            self.preview.set_image(self.current_image)
            self._show_position_overlay()
        elif mode == "mask" and self.current_mask is not None:
            # Show mask as colored overlay on original
            overlay = self.current_image.copy()
            mask_colored = np.zeros_like(overlay)
            mask_colored[:, :, 2] = self.current_mask  # Red channel
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.5, 0)
            self.preview.set_image(overlay)
        elif mode == "result" and self.current_result is not None:
            self.preview.set_image(self.current_result)

    def _update_view_buttons(self):
        """Update view button appearances."""
        active_fg = COLORS["accent"]
        inactive_fg = "#333345"

        self.btn_view_original.configure(
            fg_color=active_fg if self.view_mode == "original" else inactive_fg
        )
        self.btn_view_mask.configure(
            fg_color=COLORS["warning"] if self.view_mode == "mask" else inactive_fg
        )
        self.btn_view_result.configure(
            fg_color=COLORS["success"] if self.view_mode == "result" else inactive_fg
        )

    def _apply_engine_settings(self):
        """Apply current UI settings to the engine."""
        self.engine.threshold_value = self.threshold_var.get()
        self.engine.inpaint_radius = self.radius_var.get()
        self.engine.dilate_iterations = self.dilate_var.get()
        self.engine.method_name = self.method_var.get()
        self.engine.mode = self.mode_var.get()

    def _preview_removal(self):
        """Preview watermark removal on current image."""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please select an image from the file list first.")
            return

        self._apply_engine_settings()

        try:
            h, w = self.current_image.shape[:2]
            pos = self.position_var.get()

            if "Custom" in pos:
                if self.custom_roi is None:
                    messagebox.showinfo(
                        "Select Region",
                        "Please draw a rectangle on the preview to select the watermark region."
                    )
                    return
                roi = self.custom_roi
            else:
                roi = self.engine.get_roi_from_position(h, w, pos)

            mask = self.engine.create_mask(self.current_image, roi)
            result = self.engine.remove_watermark(self.current_image, mask)

            self.current_mask = mask
            self.current_result = result

            # Show result
            self._set_view("result")
            self._set_status("✅ Preview generated — switch views to compare")

        except Exception as e:
            self._set_status(f"Preview error: {e}", error=True)

    # ── Processing ───────────────────────────────────────────────────────────

    def _get_output_path(self, input_path):
        """Generate output path for a file.
        Creates 'watermark removed' subfolder inside the input file's parent directory.
        Keeps the original filename unchanged.
        """
        p = Path(input_path)

        # Create "watermark removed" folder inside input file's directory
        out_dir = p.parent / "watermark removed"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Keep original filename
        output = out_dir / p.name

        return str(output)

    def _start_processing(self):
        """Start batch processing in a separate thread."""
        if not self.file_list:
            messagebox.showwarning("No Files", "Please add files to process.")
            return

        if self.processing:
            messagebox.showinfo("Processing", "Already processing. Please wait.")
            return

        self.processing = True
        self.process_btn.configure(text="⏳ Processing...", state="disabled")
        self.progress_bar.set(0)

        thread = threading.Thread(target=self._process_all, daemon=True)
        thread.start()

    def _process_all(self):
        """Process all files (runs in separate thread)."""
        self._apply_engine_settings()
        total = len(self.file_list)
        success = 0
        errors = []
        pos = self.position_var.get()

        for i, filepath in enumerate(self.file_list):
            try:
                ext = Path(filepath).suffix.lower()
                output_path = self._get_output_path(filepath)
                name = Path(filepath).name

                self.after(0, self._set_status, f"Processing ({i+1}/{total}): {name}")

                if "Custom" in pos:
                    roi = self.custom_roi
                else:
                    roi = None

                if ext in SUPPORTED_IMAGES:
                    self.engine.process_image(
                        filepath, output_path,
                        position=pos if "Custom" not in pos else "Bottom-Right",
                        custom_roi=roi
                    )
                    success += 1

                elif ext in SUPPORTED_VIDEOS:
                    def vid_progress(p, idx=i):
                        overall = (idx + p) / total
                        self.after(0, self.progress_bar.set, overall)
                        self.after(0, self.progress_label.configure,
                                   {"text": f"Frame progress: {int(p * 100)}%"})

                    self.engine.process_video(
                        filepath, output_path,
                        position=pos if "Custom" not in pos else "Bottom-Right",
                        custom_roi=roi,
                        progress_callback=vid_progress,
                        remove_audio=self.remove_audio_var.get()
                    )
                    success += 1

            except Exception as e:
                errors.append(f"{Path(filepath).name}: {e}")

            # Update overall progress
            progress = (i + 1) / total
            self.after(0, self.progress_bar.set, progress)
            self.after(0, self.progress_label.configure,
                       {"text": f"{i+1}/{total} files processed"})

        # Done
        self.processing = False
        self.after(0, self._processing_complete, success, errors, total)

    def _processing_complete(self, success, errors, total):
        """Handle processing completion (called on main thread)."""
        self.process_btn.configure(text="▶  REMOVE WATERMARKS", state="normal")

        if errors:
            error_text = "\n".join(errors[:10])
            if len(errors) > 10:
                error_text += f"\n... and {len(errors) - 10} more"
            messagebox.showwarning(
                "Completed with Errors",
                f"✅ Success: {success}/{total}\n❌ Errors: {len(errors)}\n\n{error_text}"
            )
        else:
            messagebox.showinfo(
                "✅ Complete!",
                f"Successfully processed {success}/{total} files!\n\n"
                f"Output: 'watermark removed' folder"
            )

        self._set_status(f"✅ Done — {success}/{total} files processed successfully")

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _set_status(self, text, error=False):
        """Update the status bar text."""
        color = COLORS["error"] if error else COLORS["text_dim"]
        self.status_label.configure(text=f"  ✦ {text}", text_color=color)

    # ── Keyboard Shortcuts ───────────────────────────────────────────────────

    def _bind_shortcuts(self):
        self.bind("<Control-o>", lambda e: self._add_files())
        self.bind("<Control-d>", lambda e: self._add_folder())
        self.bind("<Delete>", lambda e: self._remove_selected())

    def _remove_selected(self):
        """Remove selected file from list."""
        selection = self.file_listbox.curselection()
        if selection:
            idx = selection[0]
            del self.file_list[idx]
            self._update_file_listbox()
            if not self.file_list:
                self.preview.clear()
                self.current_image = None
                self.current_index = -1


# ─── Main Entry ──────────────────────────────────────────────────────────────

def main():
    app = App()
    app._bind_shortcuts()
    app.mainloop()


if __name__ == "__main__":
    main()
