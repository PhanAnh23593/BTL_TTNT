// UI behavior scaffold. Owner: Phuc. Replace with the planned source script.
const dropZone       = document.getElementById('dropZone');
const fileInput      = document.getElementById('fileInput');
const browseBtn      = document.getElementById('browseBtn');
const predictBtn     = document.getElementById('predictBtn');
const btnText        = document.getElementById('btnText');
const spinner        = document.getElementById('spinner');

const dropIcon       = document.getElementById('dropIcon');
const dropText       = document.getElementById('dropText');
const dropPreview    = document.getElementById('dropPreview');
const previewImg     = document.getElementById('previewImg');

const resultPlaceholder = document.getElementById('resultPlaceholder');
const resultContent     = document.getElementById('resultContent');
const resultClass       = document.getElementById('resultClass');
const resultIcon        = document.getElementById('resultIcon');
const confidenceBadge   = document.getElementById('confidenceBadge');
const probList          = document.getElementById('probList');
const CLASS_ICONS = {
  banana: '🍌', guava: '🍈', jackfruit: '🌳', mango: '🥭', neem: '🌿'
};

let selectedFile = null;
// ── File Selection ──────────────────────────────────────────────────────────
browseBtn.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('click', (e) => {
  if (e.target === dropZone || e.target.closest('.drop-icon') || e.target.closest('.drop-text')) {
    fileInput.click();
  }
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

// ── Drag & Drop ─────────────────────────────────────────────────────────────