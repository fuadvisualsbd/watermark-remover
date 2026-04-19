/**
 * WatermarkRemover Pro — Main Application Controller
 * Handles file management, UI interactions, processing pipeline
 */

(function () {
  'use strict';

  // ── State ──────────────────────────────────────────────────
  let engine = null;
  let preview = null;
  let cvReady = false;

  let files = [];           // Array of File objects
  let currentIndex = -1;
  let currentSrcMat = null; // cv.Mat of current loaded image (RGBA)
  let currentResultMat = null;
  let currentMaskMat = null;
  let viewMode = 'original';
  let processing = false;

  // ── DOM References ─────────────────────────────────────────
  const $ = id => document.getElementById(id);

  const dom = {};
  function cacheDom() {
    dom.loadingOverlay = $('loadingOverlay');
    dom.dropOverlay = $('dropOverlay');
    dom.fileInput = $('fileInput');
    dom.btnAddFiles = $('btnAddFiles');
    dom.btnClearAll = $('btnClearAll');
    dom.fileList = $('fileList');
    dom.btnViewOriginal = $('btnViewOriginal');
    dom.btnViewMask = $('btnViewMask');
    dom.btnViewResult = $('btnViewResult');
    dom.positionSelect = $('positionSelect');
    dom.thresholdSlider = $('thresholdSlider');
    dom.thresholdValue = $('thresholdValue');
    dom.radiusSlider = $('radiusSlider');
    dom.radiusValue = $('radiusValue');
    dom.dilateSlider = $('dilateSlider');
    dom.dilateValue = $('dilateValue');
    dom.algorithmSelect = $('algorithmSelect');
    dom.videoWarning = $('videoWarning');
    dom.btnPreview = $('btnPreview');
    dom.btnProcess = $('btnProcess');
    dom.progressWrap = $('progressWrap');
    dom.progressFill = $('progressFill');
    dom.progressText = $('progressText');
    dom.statusText = $('statusText');
    dom.fileCount = $('fileCount');
    dom.statusBar = $('statusBar');
  }

  // ── OpenCV.js Ready ────────────────────────────────────────
  function waitForOpenCV() {
    return new Promise(resolve => {
      if (typeof cv !== 'undefined' && cv.Mat) {
        resolve();
        return;
      }
      // OpenCV.js sets cv as Module, then calls onRuntimeInitialized
      const checkInterval = setInterval(() => {
        if (typeof cv !== 'undefined' && cv.Mat) {
          clearInterval(checkInterval);
          resolve();
        }
      }, 100);

      // Also listen for the global callback
      if (typeof cv === 'undefined') {
        window.cv = {};
      }
      const origOnReady = cv.onRuntimeInitialized;
      cv.onRuntimeInitialized = () => {
        if (origOnReady) origOnReady();
        clearInterval(checkInterval);
        resolve();
      };
    });
  }

  // ── Init ───────────────────────────────────────────────────
  async function init() {
    cacheDom();

    // Wait for OpenCV.js
    await waitForOpenCV();
    cvReady = true;
    engine = new WatermarkEngine();
    preview = new PreviewCanvas('previewCanvas', 'previewWrap', 'previewPlaceholder');
    preview.onRoiChange = onCustomRoi;

    // Hide loading
    dom.loadingOverlay.classList.add('hidden');
    setTimeout(() => dom.loadingOverlay.style.display = 'none', 500);

    bindEvents();
    setStatus('Ready — Add files to get started');
  }

  // ── Event Binding ──────────────────────────────────────────
  function bindEvents() {
    // File input
    dom.btnAddFiles.addEventListener('click', () => dom.fileInput.click());
    dom.fileInput.addEventListener('change', e => addFiles(Array.from(e.target.files)));
    dom.btnClearAll.addEventListener('click', clearAll);

    // Drag & drop
    document.addEventListener('dragover', e => { e.preventDefault(); dom.dropOverlay.classList.add('visible'); });
    document.addEventListener('dragleave', e => {
      if (e.relatedTarget === null) dom.dropOverlay.classList.remove('visible');
    });
    document.addEventListener('drop', e => {
      e.preventDefault();
      dom.dropOverlay.classList.remove('visible');
      if (e.dataTransfer.files.length) addFiles(Array.from(e.dataTransfer.files));
    });

    // View toggles
    dom.btnViewOriginal.addEventListener('click', () => setView('original'));
    dom.btnViewMask.addEventListener('click', () => setView('mask'));
    dom.btnViewResult.addEventListener('click', () => setView('result'));

    // Position select
    dom.positionSelect.addEventListener('change', onPositionChange);

    // Sliders
    dom.thresholdSlider.addEventListener('input', e => {
      dom.thresholdValue.textContent = e.target.value;
    });
    dom.radiusSlider.addEventListener('input', e => {
      dom.radiusValue.textContent = e.target.value;
    });
    dom.dilateSlider.addEventListener('input', e => {
      dom.dilateValue.textContent = e.target.value;
    });

    // Action buttons
    dom.btnPreview.addEventListener('click', previewRemoval);
    dom.btnProcess.addEventListener('click', startProcessing);
  }

  // ── File Management ────────────────────────────────────────
  function addFiles(newFiles) {
    const validExts = new Set([...SUPPORTED_IMAGES, ...SUPPORTED_VIDEOS]);
    let added = 0;

    for (const f of newFiles) {
      const ext = getFileExt(f.name);
      if (!validExts.has(ext)) continue;
      // Avoid duplicates by name
      if (files.some(existing => existing.name === f.name && existing.size === f.size)) continue;
      files.push(f);
      added++;
    }

    renderFileList();
    updateFileCount();
    if (added > 0) setStatus(`Added ${added} file${added > 1 ? 's' : ''}`);
    dom.fileInput.value = '';

    // Show video warning if any videos
    const hasVideo = files.some(f => isVideoFile(f.name));
    dom.videoWarning.style.display = hasVideo ? 'block' : 'none';
  }

  function clearAll() {
    files = [];
    currentIndex = -1;
    freeMats();
    preview.clear();
    renderFileList();
    updateFileCount();
    setStatus('File list cleared');
    dom.videoWarning.style.display = 'none';
  }

  function removeFile(index) {
    files.splice(index, 1);
    if (currentIndex === index) {
      currentIndex = -1;
      freeMats();
      preview.clear();
    } else if (currentIndex > index) {
      currentIndex--;
    }
    renderFileList();
    updateFileCount();
  }

  function renderFileList() {
    dom.fileList.innerHTML = '';
    files.forEach((f, i) => {
      const li = document.createElement('li');
      li.className = 'file-item' + (i === currentIndex ? ' active' : '');
      const isVid = isVideoFile(f.name);
      li.innerHTML = `
        <span class="icon">${isVid ? '🎬' : '📷'}</span>
        <span class="name" title="${f.name}">${f.name}</span>
        <button class="remove-file" title="Remove">✕</button>
      `;
      li.querySelector('.name').addEventListener('click', () => selectFile(i));
      li.querySelector('.remove-file').addEventListener('click', e => {
        e.stopPropagation();
        removeFile(i);
      });
      dom.fileList.appendChild(li);
    });
  }

  function updateFileCount() {
    const imgs = files.filter(f => isImageFile(f.name)).length;
    const vids = files.length - imgs;
    dom.fileCount.textContent = `${imgs} img${imgs !== 1 ? 's' : ''} · ${vids} vid${vids !== 1 ? 's' : ''}`;
  }

  async function selectFile(index) {
    if (index === currentIndex) return;
    currentIndex = index;
    freeMats();
    viewMode = 'original';
    updateViewButtons();
    renderFileList();

    const file = files[index];
    try {
      if (isImageFile(file.name)) {
        currentSrcMat = await loadImageAsMat(file);
        showMatOnPreview(currentSrcMat);
        showPositionOverlay();
        setStatus('Loaded: ' + file.name);
      } else if (isVideoFile(file.name)) {
        // Load first frame
        const firstFrame = await loadVideoFirstFrame(file);
        currentSrcMat = firstFrame;
        showMatOnPreview(currentSrcMat);
        showPositionOverlay();
        setStatus('Loaded video preview: ' + file.name);
      }
    } catch (e) {
      setStatus('Error: ' + e.message, true);
    }
  }

  // ── Preview Display ────────────────────────────────────────
  function showMatOnPreview(mat) {
    const tempCanvas = document.createElement('canvas');
    cv.imshow(tempCanvas, mat);
    const img = new Image();
    img.onload = () => preview.showImage(img);
    img.src = tempCanvas.toDataURL();
  }

  function setView(mode) {
    viewMode = mode;
    updateViewButtons();

    if (mode === 'original' && currentSrcMat) {
      showMatOnPreview(currentSrcMat);
      showPositionOverlay();
    } else if (mode === 'mask' && currentMaskMat && currentSrcMat) {
      // Show mask as red overlay
      const overlay = currentSrcMat.clone();
      const maskColored = new cv.Mat(overlay.rows, overlay.cols, cv.CV_8UC4, new cv.Scalar(0, 0, 255, 0));
      // Where mask is non-zero, blend red
      for (let r = 0; r < overlay.rows; r++) {
        for (let c = 0; c < overlay.cols; c++) {
          if (currentMaskMat.ucharAt(r, c) > 0) {
            const idx = (r * overlay.cols + c) * 4;
            overlay.data[idx] = Math.min(255, overlay.data[idx] * 0.7);
            overlay.data[idx + 1] = Math.min(255, overlay.data[idx + 1] * 0.5);
            overlay.data[idx + 2] = Math.min(255, Math.round(overlay.data[idx + 2] * 0.5 + 128));
          }
        }
      }
      showMatOnPreview(overlay);
      maskColored.delete();
      overlay.delete();
    } else if (mode === 'result' && currentResultMat) {
      showMatOnPreview(currentResultMat);
    }
  }

  function updateViewButtons() {
    dom.btnViewOriginal.className = 'view-btn' + (viewMode === 'original' ? ' active-original' : '');
    dom.btnViewMask.className = 'view-btn' + (viewMode === 'mask' ? ' active-mask' : '');
    dom.btnViewResult.className = 'view-btn' + (viewMode === 'result' ? ' active-result' : '');
  }

  // ── Position / ROI ─────────────────────────────────────────
  function onPositionChange() {
    const val = dom.positionSelect.value;
    if (val === 'Custom') {
      preview.enableSelection(true);
      preview.clearPresetRoi();
      setStatus('🎯 Draw a rectangle on the preview to select watermark region');
    } else {
      preview.enableSelection(false);
      showPositionOverlay();
      setStatus('Position set to: ' + val);
    }
  }

  function showPositionOverlay() {
    const pos = dom.positionSelect.value;
    if (pos !== 'Custom') {
      preview.showPresetRoi(pos);
    }
  }

  function onCustomRoi(roi) {
    setStatus(`Custom region: (${roi[0]}, ${roi[1]}) → (${roi[2]}, ${roi[3]})`);
  }

  // ── Engine Settings ────────────────────────────────────────
  function applySettings() {
    engine.thresholdValue = parseInt(dom.thresholdSlider.value);
    engine.inpaintRadius = parseInt(dom.radiusSlider.value);
    engine.dilateIterations = parseInt(dom.dilateSlider.value);
    engine.methodName = dom.algorithmSelect.value;
    engine.mode = document.querySelector('input[name="detectionMode"]:checked').value;
  }

  function getPositionAndRoi() {
    const pos = dom.positionSelect.value;
    if (pos === 'Custom') {
      const roi = preview.getCustomRoi();
      return { position: null, customRoi: roi };
    }
    return { position: pos, customRoi: null };
  }

  // ── Preview Removal ────────────────────────────────────────
  function previewRemoval() {
    if (!currentSrcMat) {
      alert('Please select an image from the file list first.');
      return;
    }

    applySettings();
    const { position, customRoi } = getPositionAndRoi();

    if (!position && !customRoi) {
      alert('Please draw a rectangle on the preview to select the watermark region.');
      return;
    }

    try {
      setStatus('Generating preview…');
      const { result, mask } = engine.processImageMat(
        currentSrcMat,
        position || 'Bottom-Right',
        customRoi
      );

      // Free old results
      if (currentResultMat) currentResultMat.delete();
      if (currentMaskMat) currentMaskMat.delete();

      currentResultMat = result;
      currentMaskMat = mask;

      setView('result');
      setStatus('✅ Preview generated — switch views to compare');
    } catch (e) {
      setStatus('Preview error: ' + e.message, true);
    }
  }

  // ── Batch Processing ───────────────────────────────────────
  async function startProcessing() {
    if (!files.length) {
      alert('Please add files to process.');
      return;
    }
    if (processing) {
      alert('Already processing. Please wait.');
      return;
    }

    processing = true;
    dom.btnProcess.disabled = true;
    dom.btnProcess.textContent = '⏳ Processing…';
    dom.progressWrap.style.display = 'block';
    setProgress(0);

    applySettings();
    const { position, customRoi } = getPositionAndRoi();
    const total = files.length;
    let success = 0;
    const errors = [];
    const results = []; // { name, blob }

    for (let i = 0; i < total; i++) {
      const file = files[i];
      const name = file.name;

      try {
        setStatus(`Processing (${i + 1}/${total}): ${name}`);

        if (isImageFile(name)) {
          const src = await loadImageAsMat(file);
          const { result } = engine.processImageMat(
            src,
            position || 'Bottom-Right',
            customRoi
          );

          const ext = getFileExt(name);
          const mime = ext === '.webp' ? 'image/webp'
            : (ext === '.jpg' || ext === '.jpeg') ? 'image/jpeg'
            : 'image/png';
          const blob = await matToBlob(result, mime);
          results.push({ name, blob });

          src.delete();
          result.delete();
          success++;
        } else if (isVideoFile(name)) {
          // Video processing
          setStatus(`Processing video (${i + 1}/${total}): ${name}`);
          const videoBlob = await processVideoFile(file, position, customRoi, p => {
            const overall = (i + p) / total;
            setProgress(overall);
            dom.progressText.textContent = `Frame: ${Math.round(p * 100)}%`;
          });
          if (videoBlob) {
            results.push({ name: name.replace(/\.[^.]+$/, '.webm'), blob: videoBlob });
            success++;
          }
        }
      } catch (e) {
        errors.push(`${name}: ${e.message}`);
      }

      setProgress((i + 1) / total);
      dom.progressText.textContent = `${i + 1}/${total} files`;
    }

    // Download results
    if (results.length === 1) {
      downloadBlob(results[0].blob, results[0].name);
    } else if (results.length > 1) {
      await downloadAsZip(results);
    }

    // Done
    processing = false;
    dom.btnProcess.disabled = false;
    dom.btnProcess.textContent = '▶ REMOVE WATERMARKS';

    if (errors.length) {
      setStatus(`⚠️ Done — ${success}/${total} succeeded, ${errors.length} errors`, true);
      alert(`Completed with errors:\n\n✅ Success: ${success}/${total}\n❌ Errors:\n${errors.slice(0, 5).join('\n')}`);
    } else {
      setStatus(`✅ Done — ${success}/${total} files processed successfully`);
    }

    setTimeout(() => {
      dom.progressWrap.style.display = 'none';
    }, 2000);
  }

  // ── Video Processing ───────────────────────────────────────
  function loadVideoFirstFrame(file) {
    return new Promise((resolve, reject) => {
      const video = document.getElementById('hiddenVideo');
      const url = URL.createObjectURL(file);
      video.src = url;
      video.currentTime = 0.01;
      video.onloadeddata = () => {
        const canvas = document.getElementById('hiddenCanvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);
        const mat = cv.imread(canvas);
        URL.revokeObjectURL(url);
        resolve(mat);
      };
      video.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error('Cannot load video: ' + file.name));
      };
    });
  }

  function processVideoFile(file, position, customRoi, progressCb) {
    return new Promise((resolve, reject) => {
      const video = document.getElementById('hiddenVideo');
      const url = URL.createObjectURL(file);
      video.src = url;
      video.currentTime = 0;

      video.onloadedmetadata = () => {
        const duration = video.duration;
        if (duration > MAX_VIDEO_DURATION) {
          URL.revokeObjectURL(url);
          reject(new Error(`Video too long (${Math.round(duration)}s). Max ${MAX_VIDEO_DURATION}s allowed.`));
          return;
        }

        video.onloadeddata = () => {
          processVideoFrames(video, position, customRoi, progressCb)
            .then(blob => {
              URL.revokeObjectURL(url);
              resolve(blob);
            })
            .catch(err => {
              URL.revokeObjectURL(url);
              reject(err);
            });
        };
      };

      video.onerror = () => {
        URL.revokeObjectURL(url);
        reject(new Error('Cannot load video'));
      };
    });
  }

  async function processVideoFrames(video, position, customRoi, progressCb) {
    const w = video.videoWidth;
    const h = video.videoHeight;
    const canvas = document.getElementById('hiddenCanvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');

    // Setup output canvas for recording
    const outCanvas = document.createElement('canvas');
    outCanvas.width = w;
    outCanvas.height = h;
    const outCtx = outCanvas.getContext('2d');

    const fps = 30;
    const duration = video.duration;
    const totalFrames = Math.ceil(duration * fps);

    // Create MediaRecorder
    const stream = outCanvas.captureStream(fps);
    const recorder = new MediaRecorder(stream, {
      mimeType: 'video/webm;codecs=vp9',
      videoBitsPerSecond: 5000000,
    });

    const chunks = [];
    recorder.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };

    return new Promise((resolve, reject) => {
      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'video/webm' });
        resolve(blob);
      };

      recorder.start();

      let mask = null;
      let frameIdx = 0;

      function processNextFrame() {
        if (video.ended || frameIdx >= totalFrames) {
          recorder.stop();
          return;
        }

        const time = frameIdx / fps;
        video.currentTime = Math.min(time, duration);
      }

      video.onseeked = () => {
        ctx.drawImage(video, 0, 0);
        const src = cv.imread(canvas);

        // Create mask once
        if (!mask) {
          const roi = customRoi || engine.getRoiFromPosition(h, w, position || 'Bottom-Right');
          mask = engine.createMask(src, roi);
        }

        const result = engine.removeWatermark(src, mask);
        cv.imshow(outCanvas, result);

        src.delete();
        result.delete();

        frameIdx++;
        if (progressCb) progressCb(frameIdx / totalFrames);

        if (frameIdx < totalFrames) {
          requestAnimationFrame(processNextFrame);
        } else {
          if (mask) mask.delete();
          recorder.stop();
        }
      };

      processNextFrame();
    });
  }

  // ── Download Helpers ───────────────────────────────────────
  function downloadBlob(blob, filename) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      URL.revokeObjectURL(a.href);
      document.body.removeChild(a);
    }, 100);
  }

  async function downloadAsZip(results) {
    setStatus('Creating ZIP archive…');
    // Dynamic import JSZip from CDN
    if (typeof JSZip === 'undefined') {
      await loadScript('https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js');
    }
    const zip = new JSZip();
    const folder = zip.folder('watermark_removed');
    for (const { name, blob } of results) {
      folder.file(name, blob);
    }
    const zipBlob = await zip.generateAsync({ type: 'blob' });
    downloadBlob(zipBlob, 'watermark_removed.zip');
  }

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = src;
      s.onload = resolve;
      s.onerror = reject;
      document.head.appendChild(s);
    });
  }

  // ── Helpers ────────────────────────────────────────────────
  function freeMats() {
    if (currentSrcMat) { currentSrcMat.delete(); currentSrcMat = null; }
    if (currentResultMat) { currentResultMat.delete(); currentResultMat = null; }
    if (currentMaskMat) { currentMaskMat.delete(); currentMaskMat = null; }
  }

  function setStatus(text, error = false) {
    dom.statusText.textContent = '✦ ' + text;
    dom.statusBar.classList.toggle('error', error);
  }

  function setProgress(ratio) {
    const pct = Math.round(ratio * 100);
    dom.progressFill.style.width = pct + '%';
    dom.progressText.textContent = pct + '%';
  }

  // ── Boot ───────────────────────────────────────────────────
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
