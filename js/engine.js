/**
 * WatermarkRemover Pro — Watermark Removal Engine
 * Uses OpenCV.js for inpainting (TELEA / Navier-Stokes)
 */

const POSITION_PRESETS = {
  'Bottom-Right':  [0.70, 0.85, 1.0, 1.0],
  'Bottom-Left':   [0.0,  0.85, 0.30, 1.0],
  'Top-Right':     [0.70, 0.0,  1.0, 0.15],
  'Top-Left':      [0.0,  0.0,  0.30, 0.15],
  'Bottom-Center': [0.25, 0.85, 0.75, 1.0],
  'Top-Center':    [0.25, 0.0,  0.75, 0.15],
};

const SUPPORTED_IMAGES = new Set(['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif']);
const SUPPORTED_VIDEOS = new Set(['.mp4', '.webm']);
const MAX_VIDEO_DURATION = 8; // seconds

class WatermarkEngine {
  constructor() {
    this.inpaintRadius = 7;
    this.thresholdValue = 200;
    this.dilateSize = 5;
    this.dilateIterations = 3;
    this.methodName = 'TELEA'; // 'TELEA' or 'NS'
    this.mode = 'Precise'; // 'Precise' or 'Region'
  }

  get method() {
    return this.methodName === 'TELEA' ? cv.INPAINT_TELEA : cv.INPAINT_NS;
  }

  getRoiFromPosition(imgH, imgW, positionName) {
    const ratios = POSITION_PRESETS[positionName];
    if (!ratios) return null;
    return [
      Math.round(imgW * ratios[0]),
      Math.round(imgH * ratios[1]),
      Math.round(imgW * ratios[2]),
      Math.round(imgH * ratios[3]),
    ];
  }

  createPreciseMask(src, roi) {
    const [x1, y1, x2, y2] = this._clampRoi(roi, src.cols, src.rows);
    const h = src.rows, w = src.cols;
    const mask = new cv.Mat.zeros(h, w, cv.CV_8UC1);

    // Extract ROI
    const roiRect = new cv.Rect(x1, y1, x2 - x1, y2 - y1);
    if (roiRect.width < 2 || roiRect.height < 2) return mask;

    const roiMat = src.roi(roiRect);
    const gray = new cv.Mat();
    cv.cvtColor(roiMat, gray, cv.COLOR_RGBA2GRAY);

    // Gaussian blur
    const blurred = new cv.Mat();
    cv.GaussianBlur(gray, blurred, new cv.Size(3, 3), 1);

    // Binary threshold for bright watermark text
    const brightMask = new cv.Mat();
    cv.threshold(blurred, brightMask, this.thresholdValue, 255, cv.THRESH_BINARY);

    // Canny edge detection for semi-transparent watermarks
    const edges = new cv.Mat();
    cv.Canny(blurred, edges, 50, 150);
    const dilKernel2 = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(2, 2));
    cv.dilate(edges, edges, dilKernel2, new cv.Point(-1, -1), 1);

    // Combine
    const combined = new cv.Mat();
    cv.bitwise_or(brightMask, edges, combined);

    // Morphological dilation
    const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(this.dilateSize, this.dilateSize));
    cv.dilate(combined, combined, kernel, new cv.Point(-1, -1), this.dilateIterations);

    // Place into full mask
    const maskRoi = mask.roi(roiRect);
    combined.copyTo(maskRoi);

    // Cleanup
    roiMat.delete(); gray.delete(); blurred.delete();
    brightMask.delete(); edges.delete(); combined.delete();
    kernel.delete(); dilKernel2.delete(); maskRoi.delete();

    return mask;
  }

  createRegionMask(src, roi) {
    const [x1, y1, x2, y2] = this._clampRoi(roi, src.cols, src.rows);
    const mask = new cv.Mat.zeros(src.rows, src.cols, cv.CV_8UC1);
    const white = new cv.Scalar(255);
    cv.rectangle(mask, new cv.Point(x1, y1), new cv.Point(x2, y2), white, -1);
    return mask;
  }

  createMask(src, roi) {
    return this.mode === 'Precise'
      ? this.createPreciseMask(src, roi)
      : this.createRegionMask(src, roi);
  }

  removeWatermark(src, mask) {
    if (cv.countNonZero(mask) === 0) {
      return src.clone();
    }

    // Convert RGBA to BGR for inpainting
    const bgr = new cv.Mat();
    cv.cvtColor(src, bgr, cv.COLOR_RGBA2RGB);

    const result = new cv.Mat();
    cv.inpaint(bgr, mask, result, this.inpaintRadius, this.method);

    // Edge blending
    const blend = new cv.Mat();
    mask.convertTo(blend, cv.CV_32F, 1.0 / 255.0);
    const blendBlurred = new cv.Mat();
    cv.GaussianBlur(blend, blendBlurred, new cv.Size(15, 15), 5);

    // Create 3-channel blend
    const channels = new cv.MatVector();
    channels.push_back(blendBlurred);
    channels.push_back(blendBlurred);
    channels.push_back(blendBlurred);
    const blend3 = new cv.Mat();
    cv.merge(channels, blend3);

    // Convert images to float
    const resultF = new cv.Mat();
    const bgrF = new cv.Mat();
    result.convertTo(resultF, cv.CV_32FC3);
    bgr.convertTo(bgrF, cv.CV_32FC3);

    // blended = result * blend + original * (1 - blend)
    const invBlend = new cv.Mat();
    const ones = new cv.Mat(blend3.rows, blend3.cols, cv.CV_32FC3, new cv.Scalar(1, 1, 1));
    cv.subtract(ones, blend3, invBlend);

    const part1 = new cv.Mat();
    const part2 = new cv.Mat();
    cv.multiply(resultF, blend3, part1);
    cv.multiply(bgrF, invBlend, part2);

    const final32 = new cv.Mat();
    cv.add(part1, part2, final32);

    const finalU8 = new cv.Mat();
    final32.convertTo(finalU8, cv.CV_8UC3);

    // Convert back to RGBA
    const finalRGBA = new cv.Mat();
    cv.cvtColor(finalU8, finalRGBA, cv.COLOR_RGB2RGBA);

    // Cleanup
    bgr.delete(); result.delete(); blend.delete(); blendBlurred.delete();
    channels.delete(); blend3.delete(); resultF.delete(); bgrF.delete();
    invBlend.delete(); ones.delete(); part1.delete(); part2.delete();
    final32.delete(); finalU8.delete();

    return finalRGBA;
  }

  /**
   * Process an image from an HTMLImageElement or canvas ImageData
   * Returns { result: cv.Mat, mask: cv.Mat }
   */
  processImageMat(src, positionName, customRoi) {
    const h = src.rows, w = src.cols;
    let roi;
    if (customRoi) {
      roi = customRoi;
    } else {
      roi = this.getRoiFromPosition(h, w, positionName);
      if (!roi) throw new Error('Unknown position: ' + positionName);
    }

    const mask = this.createMask(src, roi);
    const result = this.removeWatermark(src, mask);
    return { result, mask };
  }

  _clampRoi(roi, w, h) {
    return [
      Math.max(0, Math.min(roi[0], w)),
      Math.max(0, Math.min(roi[1], h)),
      Math.max(0, Math.min(roi[2], w)),
      Math.max(0, Math.min(roi[3], h)),
    ];
  }
}

/**
 * Load image file as cv.Mat (RGBA)
 */
function loadImageAsMat(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      const mat = cv.imread(canvas);
      URL.revokeObjectURL(img.src);
      resolve(mat);
    };
    img.onerror = () => {
      URL.revokeObjectURL(img.src);
      reject(new Error('Cannot load image: ' + file.name));
    };
    img.src = URL.createObjectURL(file);
  });
}

/**
 * Convert cv.Mat to canvas data URL (PNG)
 */
function matToDataURL(mat, mimeType = 'image/png') {
  const canvas = document.createElement('canvas');
  cv.imshow(canvas, mat);
  return canvas.toDataURL(mimeType, 0.98);
}

/**
 * Convert cv.Mat to Blob
 */
function matToBlob(mat, mimeType = 'image/png') {
  return new Promise(resolve => {
    const canvas = document.createElement('canvas');
    cv.imshow(canvas, mat);
    canvas.toBlob(blob => resolve(blob), mimeType, 0.98);
  });
}

/**
 * Get file extension
 */
function getFileExt(filename) {
  const dot = filename.lastIndexOf('.');
  return dot >= 0 ? filename.substring(dot).toLowerCase() : '';
}

/**
 * Check if file is image
 */
function isImageFile(filename) {
  return SUPPORTED_IMAGES.has(getFileExt(filename));
}

/**
 * Check if file is video
 */
function isVideoFile(filename) {
  return SUPPORTED_VIDEOS.has(getFileExt(filename));
}
