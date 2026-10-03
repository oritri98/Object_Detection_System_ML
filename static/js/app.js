/**
 * Project HYDRA Cam - Client Controller
 * Obsidian & Electrum Gold Command Center
 */

document.addEventListener('DOMContentLoaded', () => {
  // --- View Containers ---
  const splashScreen = document.getElementById('splashScreen');
  const commandCenter = document.getElementById('commandCenter');

  // --- Splash Controls ---
  const btnConnectIp = document.getElementById('btnConnectIp');
  const inputIpUrl = document.getElementById('inputIpUrl');
  const btnConnectPhone = document.getElementById('btnConnectPhone');
  const inputPhoneUrl = document.getElementById('inputPhoneUrl');
  const btnQuickWebcam = document.getElementById('btnQuickWebcam');
  const btnQuickDemo = document.getElementById('btnQuickDemo');
  const presetChips = document.querySelectorAll('.preset-chip');

  // --- Command Center Elements ---
  const hydraVideoFeed = document.getElementById('hydraVideoFeed');
  const videoFallbackOverlay = document.getElementById('videoFallbackOverlay');
  const sourceBadgeText = document.getElementById('sourceBadgeText');
  const hudFpsText = document.getElementById('hudFpsText');
  const inVideoFps = document.getElementById('inVideoFps');
  const inVideoStatus = document.getElementById('inVideoStatus');
  const inVideoStatusText = document.getElementById('inVideoStatusText');

  // --- ML Switch & Controls ---
  const switchMlToggle = document.getElementById('switchMlToggle');
  const feedTypeBadge = document.getElementById('feedTypeBadge');
  const toggleModeDesc = document.getElementById('toggleModeDesc');
  const sliderConfidence = document.getElementById('sliderConfidence');
  const confidenceValText = document.getElementById('confidenceValText');
  const detectedCountPill = document.getElementById('detectedCountPill');
  const detectedListScroller = document.getElementById('detectedListScroller');
  const detectedEmptyState = document.getElementById('detectedEmptyState');

  // --- Viewport Actions ---
  const btnActionPause = document.getElementById('btnActionPause');
  const iconPause = document.getElementById('iconPause');
  const labelPause = document.getElementById('labelPause');
  const btnActionFullscreen = document.getElementById('btnActionFullscreen');
  const btnActionSnapshot = document.getElementById('btnActionSnapshot');
  const btnActionRecord = document.getElementById('btnActionRecord');
  const iconRecord = document.getElementById('iconRecord');
  const labelRecord = document.getElementById('labelRecord');
  const btnTopDisconnect = document.getElementById('btnTopDisconnect');
  const btnSidebarDisconnect = document.getElementById('btnSidebarDisconnect');

  // --- Gallery Modal ---
  const btnGalleryModal = document.getElementById('btnGalleryModal');
  const capturesBadgeCount = document.getElementById('capturesBadgeCount');
  const galleryModal = document.getElementById('galleryModal');
  const btnCloseGalleryModal = document.getElementById('btnCloseGalleryModal');
  const modalGalleryGrid = document.getElementById('modalGalleryGrid');
  const toastContainer = document.getElementById('toastContainer');

  // --- State Variables ---
  let isMlAnalysisActive = true;
  let isPaused = false;
  let isRecording = false;
  let telemetryInterval = null;

  // -------------------------------------------------------------------------
  // Preset Chips Listener
  // -------------------------------------------------------------------------
  presetChips.forEach(chip => {
    chip.addEventListener('click', () => {
      presetChips.forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      inputIpUrl.value = chip.getAttribute('data-url');
    });
  });

  // -------------------------------------------------------------------------
  // Connection Handlers
  // -------------------------------------------------------------------------
  btnConnectIp.addEventListener('click', () => {
    const url = inputIpUrl.value.trim();
    if (!url) {
      showToast('⚠️ Please enter an IP camera stream URL');
      return;
    }
    establishConnection('ip_cam', url, 'IP Camera');
  });

  btnConnectPhone.addEventListener('click', () => {
    const url = inputPhoneUrl.value.trim();
    if (!url) {
      showToast('⚠️ Please enter your Phone Camera URL');
      return;
    }
    establishConnection('ip_cam', url, 'Phone Camera');
  });

  btnQuickWebcam.addEventListener('click', () => {
    establishConnection('webcam', 0, 'Laptop Webcam (USB 0)');
  });

  btnQuickDemo.addEventListener('click', () => {
    establishConnection('demo', 'demo', 'Simulation Demo');
  });

  async function establishConnection(sourceType, sourceValue, displayName) {
    showToast(`Connecting to ${displayName}...`);
    try {
      const resp = await fetch('/api/camera/connect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source_type: sourceType, source_value: sourceValue })
      });

      const data = await resp.json();
      if (data.success) {
        sourceBadgeText.textContent = `${displayName} (Live)`;
        switchToCommandCenter();
        showToast(`✅ Connected to ${displayName}!`);
      } else {
        showToast(`❌ Connection failed: ${data.message || 'Error'}`);
      }
    } catch (err) {
      console.error(err);
      showToast('❌ Connection error. Check server logs.');
    }
  }

  function switchToCommandCenter() {
    splashScreen.style.display = 'none';
    commandCenter.style.display = 'flex';
    updateStreamSource();
    startTelemetryPolling();
    refreshCapturesBadge();
  }

  function switchToSplashScreen() {
    stopTelemetryPolling();
    commandCenter.style.display = 'none';
    splashScreen.style.display = 'flex';
    hydraVideoFeed.src = '';
  }

  // -------------------------------------------------------------------------
  // Disconnect Handlers
  // -------------------------------------------------------------------------
  btnTopDisconnect.addEventListener('click', disconnectCamera);
  btnSidebarDisconnect.addEventListener('click', disconnectCamera);

  async function disconnectCamera() {
    try {
      await fetch('/api/camera/disconnect', { method: 'POST' });
      showToast('⏹ Camera disconnected.');
    } catch (err) {
      console.error(err);
    }
    switchToSplashScreen();
  }

  // -------------------------------------------------------------------------
  // Video Stream & Toggle ML Analysis
  // -------------------------------------------------------------------------
  function updateStreamSource() {
    if (isPaused) return;

    if (isMlAnalysisActive) {
      hydraVideoFeed.src = '/video_feed?' + Date.now();
      feedTypeBadge.textContent = 'AI PROCESSED';
      toggleModeDesc.textContent = 'Annotated Bounding Boxes Active';
      inVideoStatusText.textContent = 'AI detection active';
      inVideoStatus.style.color = 'var(--accent-teal)';
    } else {
      hydraVideoFeed.src = '/raw_feed?' + Date.now();
      feedTypeBadge.textContent = 'RAW FEED';
      toggleModeDesc.textContent = 'Direct Camera Stream (Bypassed)';
      inVideoStatusText.textContent = 'Raw camera feed';
      inVideoStatus.style.color = 'var(--text-pewter-blue)';
    }
  }

  switchMlToggle.addEventListener('change', (e) => {
    isMlAnalysisActive = e.target.checked;
    updateStreamSource();
    showToast(isMlAnalysisActive ? '⚡ AI Object Detection Enabled' : '📷 Switched to Raw Camera Stream');
  });

  // Confidence Gate Slider
  sliderConfidence.addEventListener('input', async (e) => {
    const val = parseInt(e.target.value);
    confidenceValText.textContent = `${val}%`;
    try {
      await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ threshold: val / 100.0 })
      });
    } catch (err) {
      console.error(err);
    }
  });

  // -------------------------------------------------------------------------
  // Viewport Action Buttons
  // -------------------------------------------------------------------------
  btnActionPause.addEventListener('click', () => {
    isPaused = !isPaused;
    if (isPaused) {
      hydraVideoFeed.src = '';
      iconPause.textContent = '▶';
      labelPause.textContent = 'Resume';
      showToast('⏸ Video Feed Paused');
    } else {
      iconPause.textContent = '⏸';
      labelPause.textContent = 'Pause';
      updateStreamSource();
      showToast('▶ Video Feed Resumed');
    }
  });

  btnActionFullscreen.addEventListener('click', () => {
    const wrapper = document.getElementById('videoWrapper');
    if (!document.fullscreenElement) {
      wrapper.requestFullscreen().catch(err => alert(`Fullscreen error: ${err.message}`));
    } else {
      document.exitFullscreen();
    }
  });

  btnActionSnapshot.addEventListener('click', async () => {
    try {
      const resp = await fetch('/api/snapshot', { method: 'POST' });
      const data = await resp.json();
      if (data.success) {
        showToast(`📸 Snapshot saved to E:\\HYDRA REC (${data.filename})`);
        refreshCapturesBadge();
      } else {
        showToast(`❌ Snapshot failed: ${data.error}`);
      }
    } catch (err) {
      showToast('❌ Error capturing snapshot');
    }
  });

  btnActionRecord.addEventListener('click', async () => {
    try {
      const resp = await fetch('/api/record/toggle', { method: 'POST' });
      const data = await resp.json();
      isRecording = data.recording;
      if (isRecording) {
        iconRecord.textContent = '⏹';
        labelRecord.textContent = 'Stop Rec';
        btnActionRecord.style.background = 'var(--accent-ruby)';
        btnActionRecord.style.color = '#FFFFFF';
        showToast('🔴 Recording started! Saving to E:\\HYDRA REC');
      } else {
        iconRecord.textContent = '🔴';
        labelRecord.textContent = 'Record Video';
        btnActionRecord.style.background = '';
        btnActionRecord.style.color = '';
        showToast(`💾 Recording saved to E:\\HYDRA REC (${data.filename})`);
        refreshCapturesBadge();
      }
    } catch (err) {
      showToast('❌ Recording toggle error');
    }
  });

  // -------------------------------------------------------------------------
  // Telemetry Polling & Live Detected Objects List (Matching Inspo)
  // -------------------------------------------------------------------------
  function startTelemetryPolling() {
    stopTelemetryPolling();
    telemetryInterval = setInterval(fetchTelemetry, 250);
  }

  function stopTelemetryPolling() {
    if (telemetryInterval) {
      clearInterval(telemetryInterval);
      telemetryInterval = null;
    }
  }

  async function fetchTelemetry() {
    try {
      const resp = await fetch('/api/status');
      const data = await resp.json();

      // Update FPS Telemetry
      const fps = (data.fps || 0).toFixed(1);
      hudFpsText.textContent = `${fps} FPS`;
      inVideoFps.textContent = `FPS: ${fps}`;

      // Update Detected Objects List
      const detections = data.detections || [];
      const count = detections.length;
      detectedCountPill.textContent = `${count} object${count === 1 ? '' : 's'}`;

      if (count === 0) {
        detectedEmptyState.style.display = 'flex';
        detectedListScroller.innerHTML = '';
        detectedListScroller.appendChild(detectedEmptyState);
      } else {
        detectedEmptyState.style.display = 'none';
        detectedListScroller.innerHTML = detections.map(d => `
          <div class="detected-item-row">
            <div class="item-left">
              <span class="item-dot"></span>
              <div class="item-titles">
                <span class="item-main-title">Object ${d.id}</span>
                <span class="item-sub-title">${d.raw_label}</span>
              </div>
            </div>
            <span class="item-score">${Math.round(d.score)}%</span>
          </div>
        `).join('');
      }

    } catch (err) {
      console.warn('Telemetry poll failed:', err);
    }
  }

  // -------------------------------------------------------------------------
  // Captures Gallery Modal & Badge
  // -------------------------------------------------------------------------
  async function refreshCapturesBadge() {
    try {
      const resp = await fetch('/api/captures');
      const data = await resp.json();
      const captures = data.captures || [];
      capturesBadgeCount.textContent = captures.length;
    } catch (err) { }
  }

  btnGalleryModal.addEventListener('click', async () => {
    galleryModal.style.display = 'flex';
    try {
      const resp = await fetch('/api/captures');
      const data = await resp.json();
      const captures = data.captures || [];

      if (captures.length === 0) {
        modalGalleryGrid.innerHTML = '<p style="color:var(--text-pewter-blue); grid-column:1/-1; text-align:center; padding:2rem;">No captures in E:\\HYDRA REC yet. Use Snapshot (Alt+S) or Record.</p>';
      } else {
        modalGalleryGrid.innerHTML = captures.map(c => `
          <div class="gallery-thumb-card">
            ${c.is_video ? '<div style="height:120px; background:#111; display:flex; align-items:center; justify-content:center; font-size:2rem;">🎬</div>' : `<img src="${c.url}" class="gallery-img-preview" alt="${c.filename}" />`}
            <div class="gallery-info">
              <span class="gallery-filename">${c.filename}</span>
              <div class="gallery-meta">
                <span>${c.size_kb} KB</span>
                <span>${c.created_time.split(' ')[1] || ''}</span>
              </div>
            </div>
          </div>
        `).join('');
      }
    } catch (err) {
      modalGalleryGrid.innerHTML = '<p style="color:red;">Failed to load captures.</p>';
    }
  });

  btnCloseGalleryModal.addEventListener('click', () => {
    galleryModal.style.display = 'none';
  });

  // -------------------------------------------------------------------------
  // Toast Helper
  // -------------------------------------------------------------------------
  function showToast(msg) {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = msg;
    toastContainer.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      setTimeout(() => toast.remove(), 300);
    }, 3200);
  }

});
