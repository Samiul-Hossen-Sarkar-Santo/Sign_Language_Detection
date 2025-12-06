// ===============================================
//  SIMPLE MEDIA-PIPE STYLE HAND CONNECTIONS
//  (We use this everywhere, including preview.)
// ===============================================
const MP_HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],         // Thumb
  [0,5],[5,6],[6,7],[7,8],         // Index
  [5,9],[9,10],[10,11],[11,12],    // Middle
  [9,13],[13,14],[14,15],[15,16],  // Ring
  [13,17],[17,18],[18,19],[19,20], // Pinky
  [0,17]                            // Palm closure
];

// ===============================================
//  BASIC KNN CLASSIFIER
// ===============================================
const dataset = []; // { label, vector(Float32Array), meta }

function addSample(label, vector, meta = null) {
  dataset.push({ label, vector: new Float32Array(vector), meta });
  updateSamplesInfo();
  saveSampleToIndexedDB(label, vector, meta);
}

function euclidean(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s);
}

function predict(vec, k = 7) {
  if (dataset.length === 0) return null;

  const dist = dataset.map((s, idx) => ({
    idx,
    label: s.label,
    dist: euclidean(vec, s.vector)
  }));

  dist.sort((a, b) => a.dist - b.dist);
  const top = dist.slice(0, Math.min(k, dist.length));

  const count = {};
  let best = null, max = 0;
  for (const d of top) {
    count[d.label] = (count[d.label] || 0) + 1;
    if (count[d.label] > max) {
      max = count[d.label];
      best = d.label;
    }
  }

  return {
    label: best,
    confidence: max / top.length,
    neighbors: top
  };
}

function getDatasetStats() {
  const s = {};
  for (const x of dataset) {
    s[x.label] = (s[x.label] || 0) + 1;
  }
  return s;
}

// ===============================================
//  DOM ELEMENTS
// ===============================================
const videoElement = document.getElementById("input-video");
const canvasElement = document.getElementById("output-canvas");
const canvasCtx = canvasElement.getContext("2d");

const samplesInfo = document.getElementById("samples-info");
const labelSelect = document.getElementById("label-select");
const btnAddSample = document.getElementById("btn-add-sample");
const btnAddBurst = document.getElementById("btn-add-burst");

const predictionMain = document.getElementById("prediction-main");
const predictionConf = document.getElementById("prediction-conf");
const predictionSamples = document.getElementById("prediction-samples");
const toggleAutoType = document.getElementById("toggle-auto-type");
const statusCamera = document.getElementById("status-camera");
const statusHands = document.getElementById("status-hands");

const btnCommitCurrent = document.getElementById("btn-commit-current");
const btnClearText = document.getElementById("btn-clear-text");
const outputText = document.getElementById("output-text");

const btnSpeak = document.getElementById("btn-speak");
const btnStopSpeak = document.getElementById("btn-stop-speak");
const btnDownloadPdf = document.getElementById("btn-download-pdf");

const btnExport = document.getElementById("btn-export");
const btnImport = document.getElementById("btn-import");
const fileImport = document.getElementById("file-import");
const btnReset = document.getElementById("btn-reset");

const btnManageSamples = document.getElementById("btn-manage-samples");
const sampleManagerOverlay = document.getElementById("sample-manager-overlay");
const btnCloseManager = document.getElementById("btn-close-manager");
const btnHeaderDelete = document.getElementById("btn-header-delete");

const managerLabelList = document.getElementById("manager-label-list");
const managerSamplesContainer = document.getElementById("manager-samples-container");
const managerPreviewCanvas = document.getElementById("manager-preview-canvas");
const managerPreviewMeta = document.getElementById("manager-preview-meta");
const previewCtx = managerPreviewCanvas.getContext("2d");

// ===============================================
//  LANDMARK → VECTOR NORMALIZATION
// ===============================================
function landmarksToVector(lm) {
  const xs = lm.map(p => p.x);
  const ys = lm.map(p => p.y);
  const zs = lm.map(p => p.z || 0);

  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const minZ = Math.min(...zs), maxZ = Math.max(...zs);

  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const cz = (minZ + maxZ) / 2;

  const scale = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1e-5);

  const vec = new Float32Array(lm.length * 3);
  for (let i = 0; i < lm.length; i++) {
    vec[i*3]   = (lm[i].x - cx) / scale;
    vec[i*3+1] = (lm[i].y - cy) / scale;
    vec[i*3+2] = (lm[i].z - cz) / scale;
  }
  return vec;
}

function updateSamplesInfo() {
  const stats = getDatasetStats();
  const arr = Object.entries(stats).sort();
  samplesInfo.textContent =
    arr.length === 0
      ? "No samples yet."
      : "Samples: " + arr.map(([l,c]) => `${l}: ${c}`).join(" | ");

  predictionSamples.textContent = dataset.length;
}

// ===============================================
//  MEDIAPIPE HANDS SETUP
// ===============================================
let lastLandmarks = null;
let handsModelReady = false;

const hands = new Hands({
  locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`
});

hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.7
});

hands.onResults(onResults);

function onResults(res) {
  if (!handsModelReady && statusHands) {
    statusHands.textContent = "Hands model: ready";
    handsModelReady = true;
  }
  canvasElement.width = videoElement.videoWidth;
  canvasElement.height = videoElement.videoHeight;

  canvasCtx.save();
  canvasCtx.clearRect(0,0,canvasElement.width,canvasElement.height);
  canvasCtx.drawImage(res.image,0,0,canvasElement.width,canvasElement.height);

  if (res.multiHandLandmarks && res.multiHandLandmarks.length > 0) {
    const lm = res.multiHandLandmarks[0];
    lastLandmarks = lm;

    drawConnectors(canvasCtx, lm, HAND_CONNECTIONS, {lineWidth:3});
    drawLandmarks(canvasCtx, lm, {radius:3});

    if (dataset.length > 0) {
      const vec = landmarksToVector(lm);
      const pred = predict(vec);
      handlePrediction(pred);
    }
  } else {
    lastLandmarks = null;
    handlePrediction(null);
  }
  canvasCtx.restore();
}

// ===============================================
//  CAMERA INIT
// ===============================================
async function initCamera() {
  const cam = new Camera(videoElement, {
    width: 640, height: 480,
    onFrame: async () => hands.send({ image: videoElement })
  });
  try {
    await cam.start();
    btnAddSample.disabled = false;
    btnAddBurst.disabled = false;
    if (statusCamera) statusCamera.textContent = "Camera: active";
  } catch (err) {
    console.error("Camera failed to start", err);
    if (statusCamera) statusCamera.textContent = "Camera: unavailable";
  }
}
initCamera();

// ===============================================
//  PREDICTION STABILIZATION
// ===============================================
const windowPred = [];
const WINDOW_SIZE = 8;
let lastCommit = null;
let lastTime = 0;

function handlePrediction(pred) {
  if (!pred) {
    predictionMain.textContent = "–";
    predictionConf.textContent = "–";
    windowPred.length = 0;
    return;
  }

  predictionMain.textContent = pred.label;
  predictionConf.textContent = (pred.confidence*100).toFixed(0)+"%";

  // Stabilization
  windowPred.push(pred.label);
  if (windowPred.length > WINDOW_SIZE) windowPred.shift();

  const counts = {};
  windowPred.forEach(l => counts[l] = (counts[l]||0)+1);

  let best = null, max = 0;
  for (const [l,c] of Object.entries(counts)) {
    if (c > max) { max = c; best = l; }
  }

  const stability = max / windowPred.length;

  if (toggleAutoType.checked && best && stability >= 0.7) {
    const now = performance.now();
    const gap = best === "Delete" ? 450 : 900;
    if ((best !== lastCommit || now - lastTime > gap*2) &&
        now - lastTime > gap) {
      commitLabel(best);
      lastCommit = best;
      lastTime = now;
    }
  }
}

function commitLabel(label) {
  if (label === "Space") outputText.value += " ";
  else if (label === "Delete") outputText.value = outputText.value.slice(0,-1);
  else outputText.value += label;
}

// ===============================================
//  BUTTONS: ADD SAMPLE / BURST
// ===============================================
btnAddSample.addEventListener("click", () => {
  if (!lastLandmarks) return alert("No hand detected.");
  saveLiveSample(labelSelect.value);
});

btnAddBurst.addEventListener("click", () => {
  if (!lastLandmarks) return alert("No hand detected.");
  let n = 0;
  const label = labelSelect.value;
  const t = setInterval(() => {
    if (!lastLandmarks) return;
    saveLiveSample(label);
    if (++n >= 10) clearInterval(t);
  }, 100);
});

function saveLiveSample(label) {
  const vec = landmarksToVector(lastLandmarks);
  const meta = {
    timestamp: Date.now(),
    landmarks: lastLandmarks.map(p => ({ x:p.x, y:p.y, z:p.z||0 }))
  };
  addSample(label, vec, meta);
}
// ===============================================
//  TEXT CONTROLS + SPEECH
// ===============================================
btnCommitCurrent.addEventListener("click", () => {
  const l = predictionMain.textContent;
  if (l && l !== "–") commitLabel(l);
});

btnClearText.addEventListener("click", () => outputText.value = "");

let currentUtter = null;

btnSpeak.addEventListener("click", () => {
  const t = outputText.value.trim();
  if (!t) return;

  if (currentUtter) {
    speechSynthesis.cancel();
    currentUtter = null;
  }

  const u = new SpeechSynthesisUtterance(t);
  u.lang = "en-US";
  u.rate = 1;
  currentUtter = u;
  speechSynthesis.speak(u);
});

btnStopSpeak.addEventListener("click", () => {
  if (currentUtter) {
    speechSynthesis.cancel();
    currentUtter = null;
  }
});

btnDownloadPdf.addEventListener("click", () => {
  const text = outputText.value.trim();
  if (!text) {
    alert("No recognized text to export yet.");
    return;
  }
  const jspdfGlobal = window.jspdf;
  if (!jspdfGlobal?.jsPDF) {
    alert("PDF generator not loaded. Please retry.");
    return;
  }
  const doc = new jspdfGlobal.jsPDF({ unit: "pt", format: "a4" });
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  const marginX = 64;
  const usableWidth = pageWidth - marginX * 2;
  const headerHeight = 120;
  const summaryGap = 26;
  const subsequentBodyBase = headerHeight + 60;
  const bottomMargin = 60;
  const lineHeight = 22;
  const stampDate = new Date();
  const timestampLabel = stampDate.toLocaleString();
  const safeFileName = `asl_text_${stampDate
    .toISOString()
    .replace(/[:.]/g, "-")}.pdf`;
  const wordCount = text.split(/\s+/).filter(Boolean).length;
  const colors = {
    headerBg: [6, 10, 25],
    headerText: [244, 247, 255],
    headerSub: [167, 189, 214],
    divider: [82, 109, 255],
    summaryTitle: [17, 24, 39],
    summaryMeta: [75, 85, 99],
    summaryValue: [17, 94, 89],
    bodyText: [26, 32, 44]
  };

  const renderHeader = pageNum => {
    doc.setFillColor(...colors.headerBg);
    doc.rect(0, 0, pageWidth, headerHeight, "F");
    doc.setFont("helvetica", "bold");
    doc.setFontSize(24);
    doc.setTextColor(...colors.headerText);
    doc.text("ASL Live Gesture Tracker", marginX, 55);
    doc.setFontSize(12);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(...colors.headerSub);
    doc.text("Recognized text export", marginX, 78);
    doc.setFont("helvetica", "medium");
    doc.setFontSize(12);
    doc.setTextColor(...colors.headerText);
    doc.text(`Page ${pageNum}`, pageWidth - marginX, 55, { align: "right" });
    doc.setDrawColor(...colors.divider);
    doc.setLineWidth(1.5);
    doc.line(marginX, headerHeight - 12, pageWidth - marginX, headerHeight - 12);
  };

  const renderFooter = () => {
    doc.setFont("helvetica", "normal");
    doc.setFontSize(10);
    doc.setTextColor(...colors.headerSub);
    doc.text("Generated via ASL Live Gesture Tracker", marginX, pageHeight - 30);
    doc.text(timestampLabel, pageWidth - marginX, pageHeight - 30, { align: "right" });
  };

  const drawSummary = () => {
    const top = headerHeight + summaryGap;
    doc.setFont("helvetica", "bold");
    doc.setFontSize(15);
    doc.setTextColor(...colors.summaryTitle);
    doc.text("Recognition summary", marginX, top);
    doc.setFont("helvetica", "normal");
    doc.setFontSize(12.5);
    doc.setTextColor(...colors.summaryMeta);
    const summaryPairs = [
      ["Generated", timestampLabel],
      ["Characters", String(text.length)],
      ["Words", String(wordCount)],
      ["Auto-type", toggleAutoType.checked ? "Enabled" : "Disabled"]
    ];
    summaryPairs.forEach((pair, idx) => {
      const y = top + 24 + idx * 18;
      doc.text(`${pair[0]}:`, marginX, y);
      doc.setTextColor(...colors.summaryValue);
      doc.text(pair[1], marginX + 85, y);
      doc.setTextColor(...colors.summaryMeta);
    });
    return top + 24 + summaryPairs.length * 18;
  };

  const drawBodyText = (chunk, top) => {
    doc.setFont("helvetica", "normal");
    doc.setFontSize(13.5);
    doc.setTextColor(...colors.bodyText);
    chunk.forEach((line, idx) => {
      doc.text(line, marginX, top + idx * lineHeight);
    });
  };

  renderHeader(1);
  const summaryBottom = drawSummary();

  const bodyTextWidth = usableWidth - 40;
  const lines = doc.splitTextToSize(text, bodyTextWidth);

  const writeBodyHeading = top => {
    doc.setFont("helvetica", "bold");
    doc.setFontSize(14);
    doc.setTextColor(...colors.summaryTitle);
    doc.text("Recognized text", marginX, top);
    return top + 24;
  };

  const availableLines = top =>
    Math.max(1, Math.floor((pageHeight - bottomMargin - top) / lineHeight));

  let start = 0;
  let page = 1;
  let bodyTop = writeBodyHeading(summaryBottom + 36);
  let capacity = availableLines(bodyTop);

  while (start < lines.length) {
    const chunk = lines.slice(start, start + capacity);
    drawBodyText(chunk, bodyTop);
    start += chunk.length;
    renderFooter();
    if (start < lines.length) {
      doc.addPage();
      page += 1;
      renderHeader(page);
      bodyTop = writeBodyHeading(subsequentBodyBase);
      capacity = availableLines(bodyTop);
    }
  }

  doc.save(safeFileName);
});

// ===============================================
//  INDEXEDDB STORAGE
// ===============================================
const DB_NAME = "asl_gesture_db";
const DB_VERSION = 1;
const STORE = "samples";

let dbPromise = null;

function openDB() {
  if (dbPromise) return dbPromise;

  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);

    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        const s = db.createObjectStore(STORE, { keyPath: "id", autoIncrement: true });
        s.createIndex("label", "label", { unique: false });
      }
    };

    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });

  return dbPromise;
}

async function saveSampleToIndexedDB(label, vector, meta) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    tx.objectStore(STORE).add({
      label,
      vector: Array.from(vector),
      meta
    });
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function getAllSamplesFromIndexedDB() {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readonly");
    const req = tx.objectStore(STORE).getAll();
    req.onsuccess = () => resolve(req.result || []);
    req.onerror = () => reject(req.error);
  });
}

async function clearAllSamplesFromIndexedDB() {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    tx.objectStore(STORE).clear();
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
}

async function deleteSamplesByIds(ids) {
  if (!ids.length) return;
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    const store = tx.objectStore(STORE);
    ids.forEach(id => store.delete(id));
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
}

async function deleteSamplesByLabel(label) {
  const all = await getAllSamplesFromIndexedDB();
  const ids = all.filter(s => s.label === label).map(s => s.id);
  await deleteSamplesByIds(ids);
}

async function reloadDatasetFromDB() {
  const all = await getAllSamplesFromIndexedDB();
  dataset.length = 0;
  all.forEach(s => {
    if (s.label && Array.isArray(s.vector))
      dataset.push({ label: s.label, vector: new Float32Array(s.vector), meta: s.meta });
  });
  updateSamplesInfo();
}

(async function loadOnStartup() {
  await reloadDatasetFromDB();
})();

// ===============================================
//  EXPORT / IMPORT / RESET
// ===============================================
btnExport.addEventListener("click", async () => {
  const samples = await getAllSamplesFromIndexedDB();
  const blob = new Blob(
    [JSON.stringify({ schema: 1, exportedAt: new Date(), samples }, null, 2)],
    { type: "application/json" }
  );
  const url = URL.createObjectURL(blob);

  const a = document.createElement("a");
  a.href = url;
  a.download = "asl_dataset.json";
  a.click();

  URL.revokeObjectURL(url);
});

btnImport.addEventListener("click", () => fileImport.click());

fileImport.addEventListener("change", async e => {
  const file = e.target.files?.[0];
  if (!file) return;

  const text = await file.text();
  const json = JSON.parse(text);
  const samples = Array.isArray(json.samples) ? json.samples : json;

  await clearAllSamplesFromIndexedDB();
  dataset.length = 0;

  const db = await openDB();
  await new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    const store = tx.objectStore(STORE);
    samples.forEach(s => {
      store.add({
        label: s.label,
        vector: s.vector,
        meta: s.meta || null
      });
    });
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });

  await reloadDatasetFromDB();
});

btnReset.addEventListener("click", async () => {
  if (!confirm("Delete ALL saved samples?")) return;
  await clearAllSamplesFromIndexedDB();
  dataset.length = 0;
  updateSamplesInfo();
});

// ===============================================
//  ASL GUIDE (FLOATING WINDOW)
// ===============================================
const guideWindow = document.getElementById("asl-guide-window");
const guideHeader = document.getElementById("asl-guide-header");
const btnAslGuide = document.getElementById("btn-asl-guide");
const btnCloseGuide = document.getElementById("btn-close-guide");
const btnMinimizeGuide = document.getElementById("btn-minimize-guide");
const guideResizeHandles = document.querySelectorAll("[data-guide-resize]");
const DEFAULT_GUIDE_WIDTH = 420;
const DEFAULT_GUIDE_HEIGHT = 320;
let guideStoredSize = null;

if (guideWindow) {
  const initialRect = guideWindow.getBoundingClientRect();
  const width = initialRect.width || DEFAULT_GUIDE_WIDTH;
  const height = initialRect.height || DEFAULT_GUIDE_HEIGHT;
  guideStoredSize = {
    width: `${width}px`,
    height: `${height}px`
  };
  guideWindow.style.width = guideStoredSize.width;
  guideWindow.style.height = guideStoredSize.height;
}

btnAslGuide.addEventListener("click", () => {
  guideWindow.classList.remove("hidden");
  guideWindow.classList.remove("minimized");
  if (guideStoredSize) {
    guideWindow.style.width = guideStoredSize.width;
    guideWindow.style.height = guideStoredSize.height;
  }
});

btnCloseGuide.addEventListener("click", () => guideWindow.classList.add("hidden"));
btnMinimizeGuide.addEventListener("click", () => {
  const isMinimized = guideWindow.classList.contains("minimized");
  if (!isMinimized) {
    const rect = guideWindow.getBoundingClientRect();
    guideStoredSize = {
      width: guideWindow.style.width || `${rect.width}px`,
      height: guideWindow.style.height || `${rect.height}px`
    };
    guideWindow.style.width = guideStoredSize.width;
    guideWindow.style.height = guideStoredSize.height;
    guideWindow.classList.add("minimized");
    guideWindow.style.height = "";
  } else {
    guideWindow.classList.remove("minimized");
    if (guideStoredSize) {
      guideWindow.style.width = guideStoredSize.width;
      guideWindow.style.height = guideStoredSize.height;
    }
  }
});

// Dragging
let drag = false, dx = 0, dy = 0;

guideHeader.addEventListener("mousedown", e => {
  if (e.target.closest(".guide-header-actions")) return;
  drag = true;
  const r = guideWindow.getBoundingClientRect();
  dx = e.clientX - r.left;
  dy = e.clientY - r.top;
  guideWindow.style.right = "auto";
  guideWindow.style.bottom = "auto";
  e.preventDefault();
});

document.addEventListener("mousemove", e => {
  if (!drag) return;
  guideWindow.style.left = (e.clientX - dx) + "px";
  guideWindow.style.top = (e.clientY - dy) + "px";
});

document.addEventListener("mouseup", () => drag = false);

// Resize (all edges)
const MIN_GUIDE_W = 300;
const MIN_GUIDE_H = 220;
let resizing = false;
let resizeDir = null;
let resizeStartRect = null;
let resizeStartX = 0;
let resizeStartY = 0;

guideResizeHandles.forEach(handle => {
  handle.addEventListener("mousedown", e => {
    if (guideWindow.classList.contains("minimized")) return;
    resizing = true;
    resizeDir = handle.dataset.guideResize;
    resizeStartRect = guideWindow.getBoundingClientRect();
    resizeStartX = e.clientX;
    resizeStartY = e.clientY;
    guideWindow.style.right = "auto";
    guideWindow.style.bottom = "auto";
    document.body.style.userSelect = "none";
    e.preventDefault();
    e.stopPropagation();
  });
});

document.addEventListener("mousemove", e => {
  if (!resizing || !resizeStartRect) return;
  const dx = e.clientX - resizeStartX;
  const dy = e.clientY - resizeStartY;
  let width = resizeStartRect.width;
  let height = resizeStartRect.height;
  let left = resizeStartRect.left;
  let top = resizeStartRect.top;

  if (resizeDir === "corner" || resizeDir.includes("right")) {
    width = Math.max(MIN_GUIDE_W, resizeStartRect.width + dx);
  }
  if (resizeDir === "corner" || resizeDir.includes("bottom")) {
    height = Math.max(MIN_GUIDE_H, resizeStartRect.height + dy);
  }
  if (resizeDir.includes("left")) {
    const newWidth = Math.max(MIN_GUIDE_W, resizeStartRect.width - dx);
    left = resizeStartRect.right - newWidth;
    width = newWidth;
  }
  if (resizeDir.includes("top")) {
    const newHeight = Math.max(MIN_GUIDE_H, resizeStartRect.height - dy);
    top = resizeStartRect.bottom - newHeight;
    height = newHeight;
  }

  guideWindow.style.width = width + "px";
  guideWindow.style.height = height + "px";
  guideWindow.style.left = left + "px";
  guideWindow.style.top = top + "px";
});

document.addEventListener("mouseup", () => {
  if (resizing) {
    resizing = false;
    resizeDir = null;
    resizeStartRect = null;
    document.body.style.userSelect = "";
    guideStoredSize = {
      width: guideWindow.style.width,
      height: guideWindow.style.height
    };
  }
});

// ===============================================
//  SAMPLE MANAGER
// ===============================================
let managerAll = [];
let managerLabel = null;

function getSelectedSampleIds() {
  return [...managerSamplesContainer.querySelectorAll(".manager-sample-row")]
    .map(row => {
      const checkbox = row.querySelector("input[type=checkbox]");
      if (!checkbox || !checkbox.checked) return null;
      const id = Number(row.dataset.sampleId);
      return Number.isNaN(id) ? null : id;
    })
    .filter(id => id !== null);
}

function updateManagerDeleteState() {
  const hasSelection = getSelectedSampleIds().length > 0;
  if (btnHeaderDelete) {
    btnHeaderDelete.disabled = !hasSelection;
  }
}

btnManageSamples.addEventListener("click", async () => {
  sampleManagerOverlay.classList.remove("hidden");
  await refreshManager();
  updateManagerDeleteState();
});

btnCloseManager.addEventListener("click", () => {
  sampleManagerOverlay.classList.add("hidden");
  updateManagerDeleteState();
});

sampleManagerOverlay.addEventListener("click", e => {
  if (e.target === sampleManagerOverlay)
    sampleManagerOverlay.classList.add("hidden");
});

async function refreshManager() {
  managerAll = await getAllSamplesFromIndexedDB();
  renderManagerLabels();

  if (managerLabel) renderManagerSamples(managerLabel);
  else {
    managerSamplesContainer.innerHTML =
      `<div style="opacity:0.7; font-size:0.8rem;">Select a label to view samples.</div>`;
    clearPreview();
    updateManagerDeleteState();
  }
}

function renderManagerLabels() {
  const stats = getDatasetStats();
  const arr = Object.entries(stats).sort();
  if (!arr.length) {
    managerLabelList.innerHTML =
      `<div style="opacity:0.7; font-size:0.8rem;">No samples stored yet.</div>`;
    return;
  }

  managerLabelList.innerHTML = "";
  arr.forEach(([label, count]) => {
    const row = document.createElement("div");
    row.className = "manager-label-row";

    const span = document.createElement("span");
    span.textContent = `${label} — ${count}`;

    const btnView = document.createElement("button");
    btnView.className = "secondary btn-small";
    btnView.textContent = "View";
    btnView.onclick = () => {
      managerLabel = label;
      renderManagerSamples(label);
    };

    const btnDel = document.createElement("button");
    btnDel.className = "secondary btn-small";
    btnDel.textContent = "Delete all";
    btnDel.onclick = async () => {
      if (confirm(`Delete ALL samples for "${label}"?`)) {
        await deleteSamplesByLabel(label);
        await reloadDatasetFromDB();
        await refreshManager();
      }
    };

    row.append(span, btnView, btnDel);
    managerLabelList.appendChild(row);
  });
}

function renderManagerSamples(label) {
  const list = managerAll.filter(s => s.label === label);
  if (!list.length) {
    managerSamplesContainer.innerHTML = `<div>No samples for "${label}".</div>`;
    clearPreview();
    updateManagerDeleteState();
    return;
  }

  const wrap = document.createElement("div");
  wrap.style.display = "flex";
  wrap.style.flexDirection = "column";
  wrap.style.gap = "4px";

  list.forEach(s => {
    const row = document.createElement("div");
    row.className = "manager-sample-row";
    row.dataset.sampleId = s.id;

    const check = document.createElement("input");
    check.type = "checkbox";

    const meta = document.createElement("div");
    meta.className = "manager-sample-meta";
    meta.innerHTML = `
      <div>#${s.id} — ${s.meta?.timestamp ? new Date(s.meta.timestamp).toLocaleString() : "No time"}</div>
      <div style="opacity:0.7;">${s.meta?.landmarks?.length || "?"} points</div>
    `;

    row.onclick = e => {
      if (e.target.tagName !== "INPUT") check.checked = !check.checked;
      renderPreview(s);
      updateManagerDeleteState();
    };

    check.addEventListener("change", updateManagerDeleteState);

    row.append(check, meta);
    wrap.appendChild(row);
  });

  managerSamplesContainer.innerHTML = "";
  managerSamplesContainer.appendChild(wrap);

  clearPreview();
  updateManagerDeleteState();
}

btnHeaderDelete?.addEventListener("click", async () => {
  if (btnHeaderDelete.disabled) return;
  const ids = getSelectedSampleIds();
  if (!ids.length) return;
  if (!confirm(`Delete ${ids.length} selected samples?`)) return;

  await deleteSamplesByIds(ids);
  await reloadDatasetFromDB();
  await refreshManager();
  updateManagerDeleteState();
});

// ===============================================
//  PREVIEW CANVAS
// ===============================================
function clearPreview() {
  managerPreviewMeta.textContent = "No sample selected.";
  previewCtx.clearRect(0, 0, managerPreviewCanvas.width, managerPreviewCanvas.height);
}

function renderPreview(sample) {
  if (!sample || !sample.meta?.landmarks) {
    clearPreview();
    return;
  }

  const lm = sample.meta.landmarks;
  managerPreviewMeta.textContent = `Label: ${sample.label} (#${sample.id})`;

  const w = managerPreviewCanvas.width;
  const h = managerPreviewCanvas.height;

  previewCtx.clearRect(0, 0, w, h);

  // Extract bounds
  const xs = lm.map(p => p.x);
  const ys = lm.map(p => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);

  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;

  const scale = 0.85 * Math.min(
    w / (maxX - minX || 1e-5),
    h / (maxY - minY || 1e-5)
  );

  const tx = p => ({
    x: (p.x - cx) * scale + w / 2,
    y: (p.y - cy) * scale + h / 2
  });

  // Draw connections
  previewCtx.strokeStyle = "#d1d5db";
  previewCtx.lineWidth = 2;
  previewCtx.beginPath();
  MP_HAND_CONNECTIONS.forEach(([i, j]) => {
    const a = tx(lm[i]);
    const b = tx(lm[j]);
    previewCtx.moveTo(a.x, a.y);
    previewCtx.lineTo(b.x, b.y);
  });
  previewCtx.stroke();

  // Draw landmarks
  previewCtx.fillStyle = "#22c55e";
  lm.forEach(p => {
    const t = tx(p);
    previewCtx.beginPath();
    previewCtx.arc(t.x, t.y, 3, 0, Math.PI * 2);
    previewCtx.fill();
  });
}
