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
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) handleFile(file);
});

// ── Handle File ─────────────────────────────────────────────────────────────
function handleFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    dropIcon.style.display = 'none';
    dropText.style.display = 'none';
    dropPreview.style.display = 'flex';
  };
  reader.readAsDataURL(file);
  predictBtn.disabled = false;

  // Reset results
  resultPlaceholder.style.display = 'block';
  resultContent.style.display = 'none';
}

// ── Predict ─────────────────────────────────────────────────────────────────