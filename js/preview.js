/**
 * WatermarkRemover Pro — Preview Canvas Controller
 * Interactive ROI selection and image display
 */

class PreviewCanvas {
  constructor(canvasId, wrapId, placeholderId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.wrap = document.getElementById(wrapId);
    this.placeholder = document.getElementById(placeholderId);

    this.currentImage = null; // HTMLImageElement or ImageData
    this.currentMat = null;   // cv.Mat reference (not owned)
    this.displayScale = 1;
    this.offsetX = 0;
    this.offsetY = 0;
    this.imgW = 0;
    this.imgH = 0;

    // ROI selection
    this.selectionEnabled = false;
    this.isDrawing = false;
    this.roiStart = null;
    this.roiEnd = null;
    this.customRoi = null;
    this.onRoiChange = null;

    // Preset overlay
    this.presetPosition = null;

    this._bindEvents();
    this._resizeObserver();
  }

  _bindEvents() {
    this.canvas.addEventListener('mousedown', e => this._onMouseDown(e));
    this.canvas.addEventListener('mousemove', e => this._onMouseMove(e));
    this.canvas.addEventListener('mouseup', e => this._onMouseUp(e));
    this.canvas.addEventListener('mouseleave', e => this._onMouseUp(e));

    // Touch events for mobile
    this.canvas.addEventListener('touchstart', e => {
      e.preventDefault();
      const t = e.touches[0];
      this._onMouseDown({ offsetX: t.clientX - this.canvas.getBoundingClientRect().left, offsetY: t.clientY - this.canvas.getBoundingClientRect().top });
    }, { passive: false });
    this.canvas.addEventListener('touchmove', e => {
      e.preventDefault();
      const t = e.touches[0];
      this._onMouseMove({ offsetX: t.clientX - this.canvas.getBoundingClientRect().left, offsetY: t.clientY - this.canvas.getBoundingClientRect().top });
    }, { passive: false });
    this.canvas.addEventListener('touchend', e => this._onMouseUp(e));
  }

  _resizeObserver() {
    const ro = new ResizeObserver(() => {
      if (this.currentImage) {
        requestAnimationFrame(() => this._render());
      }
    });
    ro.observe(this.wrap);
  }

  showImage(imgElement) {
    this.currentImage = imgElement;
    this.imgW = imgElement.naturalWidth || imgElement.width;
    this.imgH = imgElement.naturalHeight || imgElement.height;
    this.placeholder.style.display = 'none';
    this.canvas.style.display = 'block';
    this._render();
  }

  showMat(mat) {
    // Convert cv.Mat to image and display
    const tempCanvas = document.createElement('canvas');
    cv.imshow(tempCanvas, mat);
    const img = new Image();
    img.onload = () => {
      this.showImage(img);
    };
    img.src = tempCanvas.toDataURL();
  }

  _render() {
    if (!this.currentImage) return;

    const wrapRect = this.wrap.getBoundingClientRect();
    const cw = wrapRect.width;
    const ch = wrapRect.height;

    if (cw < 2 || ch < 2) return;

    this.canvas.width = cw;
    this.canvas.height = ch;

    // Calculate fit scale
    this.displayScale = Math.min(cw / this.imgW, ch / this.imgH, 1.0);
    const drawW = Math.round(this.imgW * this.displayScale);
    const drawH = Math.round(this.imgH * this.displayScale);
    this.offsetX = Math.round((cw - drawW) / 2);
    this.offsetY = Math.round((ch - drawH) / 2);

    // Clear and draw
    this.ctx.clearRect(0, 0, cw, ch);
    this.ctx.drawImage(this.currentImage, this.offsetX, this.offsetY, drawW, drawH);

    // Draw preset ROI overlay
    if (this.presetPosition && POSITION_PRESETS[this.presetPosition]) {
      this._drawPresetOverlay();
    }

    // Draw custom ROI
    if (this.roiStart && this.roiEnd) {
      this._drawRoiRect(this.roiStart, this.roiEnd, '#FFB830');
    }
  }

  _drawPresetOverlay() {
    const ratios = POSITION_PRESETS[this.presetPosition];
    if (!ratios) return;

    const x1 = Math.round(this.imgW * ratios[0] * this.displayScale) + this.offsetX;
    const y1 = Math.round(this.imgH * ratios[1] * this.displayScale) + this.offsetY;
    const x2 = Math.round(this.imgW * ratios[2] * this.displayScale) + this.offsetX;
    const y2 = Math.round(this.imgH * ratios[3] * this.displayScale) + this.offsetY;

    this.ctx.save();
    this.ctx.strokeStyle = '#E94560';
    this.ctx.lineWidth = 2;
    this.ctx.setLineDash([6, 4]);
    this.ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    // Semi-transparent fill
    this.ctx.fillStyle = 'rgba(233, 69, 96, 0.08)';
    this.ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
    this.ctx.restore();
  }

  _drawRoiRect(start, end, color) {
    this.ctx.save();
    this.ctx.strokeStyle = color;
    this.ctx.lineWidth = 2;
    this.ctx.setLineDash([5, 3]);
    this.ctx.strokeRect(
      start.x, start.y,
      end.x - start.x, end.y - start.y
    );
    this.ctx.fillStyle = color.replace(')', ', 0.08)').replace('rgb', 'rgba');
    this.ctx.fillRect(start.x, start.y, end.x - start.x, end.y - start.y);
    this.ctx.restore();
  }

  showPresetRoi(positionName) {
    this.presetPosition = positionName;
    if (this.currentImage) this._render();
  }

  clearPresetRoi() {
    this.presetPosition = null;
    if (this.currentImage) this._render();
  }

  enableSelection(enabled) {
    this.selectionEnabled = enabled;
    this.canvas.style.cursor = enabled ? 'crosshair' : 'default';
    if (!enabled) {
      this.roiStart = null;
      this.roiEnd = null;
      this.customRoi = null;
    }
  }

  getCustomRoi() {
    return this.customRoi;
  }

  _onMouseDown(e) {
    if (!this.selectionEnabled || !this.currentImage) return;
    this.isDrawing = true;
    this.roiStart = { x: e.offsetX, y: e.offsetY };
    this.roiEnd = null;
    this.customRoi = null;
  }

  _onMouseMove(e) {
    if (!this.isDrawing) return;
    this.roiEnd = { x: e.offsetX, y: e.offsetY };
    this._render();
  }

  _onMouseUp(e) {
    if (!this.isDrawing) return;
    this.isDrawing = false;
    if (!this.roiEnd) return;

    // Convert to image coordinates
    const roi = this._canvasToImageCoords(this.roiStart, this.roiEnd);
    if (roi && (roi[2] - roi[0] > 5) && (roi[3] - roi[1] > 5)) {
      this.customRoi = roi;
      if (this.onRoiChange) this.onRoiChange(roi);
    }
  }

  _canvasToImageCoords(start, end) {
    if (!start || !end) return null;

    const x1 = Math.round((Math.min(start.x, end.x) - this.offsetX) / this.displayScale);
    const y1 = Math.round((Math.min(start.y, end.y) - this.offsetY) / this.displayScale);
    const x2 = Math.round((Math.max(start.x, end.x) - this.offsetX) / this.displayScale);
    const y2 = Math.round((Math.max(start.y, end.y) - this.offsetY) / this.displayScale);

    return [
      Math.max(0, Math.min(x1, this.imgW)),
      Math.max(0, Math.min(y1, this.imgH)),
      Math.max(0, Math.min(x2, this.imgW)),
      Math.max(0, Math.min(y2, this.imgH)),
    ];
  }

  clear() {
    this.currentImage = null;
    this.currentMat = null;
    this.roiStart = null;
    this.roiEnd = null;
    this.customRoi = null;
    this.presetPosition = null;
    this.canvas.style.display = 'none';
    this.placeholder.style.display = 'flex';
  }
}
