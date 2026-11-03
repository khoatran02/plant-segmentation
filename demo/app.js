'use strict';
const $ = id => document.getElementById(id);
let model = null, selected = null, result = null, running = false;
let selectionVersion = 0, readVersion = 0;
let uploadingCheckpoint = false, statusTimer;
const fmt = new Intl.NumberFormat('vi-VN');
const PREDICTION_COLOR = [37, 99, 235];
const PREDICTION_OPACITY = 0.65;

async function predictionOverlay(originalSource, maskSource) {
  const original = new Image();
  const mask = new Image();
  original.src = originalSource;
  mask.src = maskSource;
  await Promise.all([original.decode(), mask.decode()]);
  if (original.naturalWidth !== mask.naturalWidth || original.naturalHeight !== mask.naturalHeight) {
    throw new Error('Kích thước mask không khớp với ảnh gốc.');
  }
  const canvas = document.createElement('canvas');
  canvas.width = original.naturalWidth;
  canvas.height = original.naturalHeight;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  context.drawImage(mask, 0, 0);
  const labels = context.getImageData(0, 0, canvas.width, canvas.height).data;
  context.drawImage(original, 0, 0);
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height);
  for (let offset = 0; offset < labels.length; offset += 4) {
    if (labels[offset] <= 127) continue;
    for (let channel = 0; channel < 3; channel++) {
      pixels.data[offset + channel] = Math.floor(
        pixels.data[offset + channel] * (1 - PREDICTION_OPACITY)
        + PREDICTION_COLOR[channel] * PREDICTION_OPACITY
      );
    }
  }
  context.putImageData(pixels, 0, 0);
  return canvas.toDataURL('image/png');
}

function message(text, error = false) {
  $('message').textContent = text;
  $('message').hidden = !text;
  $('message').classList.toggle('error', error);
}
function updateRun() {
  $('run').disabled = running || uploadingCheckpoint || !selected || model?.status !== 'ready' || !$('prompt').value.trim();
  $('prompt-count').textContent = `${$('prompt').value.length} / 200`;
}
function clearResult() {
  result = null;
  for (const id of ['download-mask', 'download-overlay']) {
    $(id).removeAttribute('href'); $(id).classList.add('disabled'); $(id).setAttribute('aria-disabled', 'true');
  }
  renderView();
}
function renderView() {
  const hasImage = Boolean(selected);
  $('empty').hidden = hasImage;
  $('image-grid').hidden = !hasImage;
  $('overlay').hidden = !result;
  $('mask').hidden = !result;
  $('prediction-empty').hidden = Boolean(result);
  $('mask-empty').hidden = Boolean(result);
  const link = $('download-overlay');
  if (result) link.href = result.overlay; else link.removeAttribute('href');
  link.classList.toggle('disabled', !result);
  link.setAttribute('aria-disabled', String(!result));
}
function invalidate() {
  selectionVersion++;
  clearResult();
  if (selected) message('Bấm “Tạo mask” để phân đoạn ảnh theo prompt hiện tại.');
  updateRun();
}
function setImage(source, selection, name) {
  selected = selection;
  $('original').onload = () => {
    $('image-size').textContent = `${fmt.format($('original').naturalWidth)} × ${fmt.format($('original').naturalHeight)} PX`;
  };
  $('original').src = source;
  $('file-label').textContent = name;
  document.querySelectorAll('.sample').forEach(b => b.classList.toggle('selected', Number(b.dataset.id) === selection.sample_id));
  invalidate();
  message($('prompt').value.trim() ? 'Ảnh và prompt đã sẵn sàng. Bấm “Tạo mask” để chạy.' : 'Ảnh đã sẵn sàng. Nhập text prompt để mô tả vùng muốn tìm.');
}
async function useFile(file) {
  const version = ++readVersion;
  if (!file) return;
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) return message('Hãy chọn ảnh JPG, PNG hoặc WebP.', true);
  if (file.size > 12 * 1024 * 1024) return message('Ảnh vượt quá 12 MB. Vui lòng chọn ảnh nhỏ hơn.', true);
  try {
    const source = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result); reader.onerror = reject; reader.readAsDataURL(file);
    });
    const image = new Image(); image.src = source; await image.decode();
    if (version !== readVersion) return;
    if (image.naturalWidth * image.naturalHeight > 20000000) return message('Ảnh vượt quá 20 megapixel.', true);
    setImage(source, { image: source.split(',')[1] }, file.name);
  } catch { message('Không đọc được ảnh. Vui lòng chọn một ảnh khác.', true); }
}
function renderModelStatus(next) {
  const previousStatus = model?.status;
  model = next;
  const labels = {
    ready: `Sẵn sàng · ${(model.device || '').toUpperCase()}`,
    error: 'Lỗi nạp model', missing_checkpoint: 'Chưa có checkpoint',
    uploading: 'Đang tải checkpoint', loading: 'Đang nạp model',
  };
  $('status').className = `status ${model.status}`;
  $('status').lastElementChild.textContent = labels[model.status] || 'Chưa sẵn sàng';
  const needsCheckpoint = model.checkpoint_upload_allowed;
  $('checkpoint-panel').hidden = !(needsCheckpoint || uploadingCheckpoint || ['uploading', 'loading'].includes(model.status));
  $('checkpoint-title').textContent = model.status === 'missing_checkpoint' ? 'Chưa tìm thấy checkpoint' :
    needsCheckpoint ? 'Không nạp được model' : 'Đang chuẩn bị model';
  $('checkpoint-reason').textContent = model.message;
  $('checkpoint-file').disabled = uploadingCheckpoint || !needsCheckpoint;
  $('checkpoint-submit').disabled = uploadingCheckpoint || !needsCheckpoint || !$('checkpoint-file').files.length;
  if (model.status === 'ready') {
    $('model-detail').textContent = `SAM 3 · ${model.input_size.join(' × ')} · ${model.checkpoint}`;
    if (previousStatus !== 'ready' && !$('threshold').dataset.edited) {
      $('threshold').value = model.mask_threshold;
      $('threshold-value').value = Number(model.mask_threshold).toFixed(2);
    }
    if (previousStatus !== 'ready') {
      message(selected ? 'Model sẵn sàng. Bấm “Tạo mask” để chạy.' : 'Model sẵn sàng. Chọn ảnh mẫu hoặc tải ảnh của bạn lên.');
      $('checkpoint-feedback').hidden = true;
      $('checkpoint-progress').hidden = true;
    }
  } else if (previousStatus !== model.status) {
    message(model.message, needsCheckpoint);
  }
  updateRun();
}
async function pollStatus() {
  clearTimeout(statusTimer);
  try {
    const response = await fetch('/api/status');
    if (!response.ok) throw new Error();
    const next = await response.json();
    if (!uploadingCheckpoint) renderModelStatus(next);
  } catch {
    $('status').className = 'status error';
    $('status').lastElementChild.textContent = 'Mất kết nối';
    model = null; updateRun(); message('Chưa kết nối được server. Kiểm tra terminal đang chạy demo.py.', true);
    $('checkpoint-file').disabled = true;
    $('checkpoint-submit').disabled = true;
  } finally {
    statusTimer = setTimeout(pollStatus, ['loading', 'uploading'].includes(model?.status) ? 2000 : 5000);
  }
}
function checkpointFeedback(text, error = false) {
  $('checkpoint-feedback').textContent = text;
  $('checkpoint-feedback').hidden = !text;
  $('checkpoint-feedback').classList.toggle('error', error);
}
async function uploadCheckpoint(event) {
  event.preventDefault();
  const file = $('checkpoint-file').files[0];
  if (!file || uploadingCheckpoint || !model?.checkpoint_upload_allowed) return;
  if (!/\.(pt|pth|ckpt)$/i.test(file.name)) return checkpointFeedback('Hãy chọn checkpoint .pt, .pth hoặc .ckpt.', true);
  if (!file.size || file.size > model.max_checkpoint_bytes) return checkpointFeedback('Checkpoint phải có dữ liệu và không vượt quá 4 GB.', true);
  uploadingCheckpoint = true;
  invalidate();
  renderModelStatus(model);
  $('checkpoint-progress').value = 0;
  $('checkpoint-progress').hidden = false;
  checkpointFeedback('Đang tải checkpoint… 0%');
  try {
    const state = await new Promise((resolve, reject) => {
      const request = new XMLHttpRequest();
      request.open('POST', '/api/checkpoint');
      request.setRequestHeader('Content-Type', 'application/octet-stream');
      request.setRequestHeader('X-Checkpoint-Name', encodeURIComponent(file.name));
      request.upload.onprogress = progress => {
        if (!progress.lengthComputable) return;
        const percent = Math.round(progress.loaded / progress.total * 100);
        $('checkpoint-progress').value = percent;
        checkpointFeedback(percent < 100 ? `Đang tải checkpoint… ${percent}%` : 'Đã gửi checkpoint. Đang đợi server…');
      };
      request.onload = () => {
        let data;
        try { data = JSON.parse(request.responseText); }
        catch { reject(new Error('Server trả về dữ liệu không hợp lệ.')); return; }
        if (request.status !== 202) reject(new Error(data.error || 'Không tải được checkpoint.'));
        else resolve(data);
      };
      request.onerror = () => reject(new Error('Mất kết nối khi tải checkpoint. Vui lòng thử lại.'));
      request.onabort = () => reject(new Error('Đã hủy tải checkpoint.'));
      request.send(file);
    });
    $('checkpoint-file').value = '';
    checkpointFeedback('Đã tải checkpoint. Đang nạp model…');
    renderModelStatus(state);
  } catch (error) {
    $('checkpoint-progress').hidden = true;
    checkpointFeedback(error.message, true);
  } finally {
    uploadingCheckpoint = false;
    if (model) renderModelStatus(model);
    await pollStatus();
  }
}
async function loadSamples() {
  try {
    const response = await fetch('/api/samples');
    if (!response.ok) throw new Error();
    for (const sample of await response.json()) {
      const button = document.createElement('button');
      button.className = 'sample'; button.dataset.id = sample.id;
      button.title = sample.title; button.setAttribute('aria-label', `Dùng ảnh và prompt ${sample.prompt}`);
      const image = document.createElement('img'); image.src = sample.url; image.alt = sample.title; image.loading = 'lazy';
      const title = document.createElement('strong'); title.textContent = sample.prompt;
      const caption = document.createElement('span'); caption.textContent = sample.title;
      button.append(image, title, caption);
      button.onclick = () => {
        readVersion++;
        $('prompt').value = sample.prompt;
        setImage(sample.url, { sample_id: sample.id }, sample.name);
        $('prompt').focus({ preventScroll: true });
        document.querySelector('.workspace').scrollIntoView({
          behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth',
          block: 'start',
        });
      };
      $('samples').append(button);
    }
  } catch { $('samples').textContent = 'Không tải được ảnh mẫu.'; }
}
async function run() {
  if ($('run').disabled) return;
  running = true; updateRun(); $('busy').hidden = false;
  if (window.matchMedia('(max-width: 760px)').matches) {
    $('result-title').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
  const version = selectionVersion;
  const started = performance.now();
  const timer = setInterval(() => $('elapsed').textContent = `${Math.floor((performance.now() - started) / 1000)} giây · Model đang xử lý…`, 1000);
  const request = { ...selected, prompt: $('prompt').value.trim(), threshold: Number($('threshold').value) };
  message('Đang chạy suy luận trên model đã fine-tune…');
  try {
    const response = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(request) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Không thực hiện được phân đoạn.');
    if (version !== selectionVersion) { message('Đầu vào đã thay đổi trong lúc xử lý. Bấm phân đoạn để chạy lại.'); return; }
    // Render from the raw image and binary mask, independent of an older server's overlay color.
    data.overlay = await predictionOverlay(data.original, data.mask);
    if (version !== selectionVersion) return;
    // Leave the original input untouched; only prediction and mask change after inference.
    const images = [new Image(), new Image()];
    const sources = [data.overlay, data.mask];
    await Promise.all(images.map((image, i) => { image.src = sources[i]; return image.decode(); }));
    if (version !== selectionVersion) return;
    result = data;
    $('overlay').src = data.overlay; $('mask').src = data.mask;
    $('image-size').textContent = `${fmt.format(data.width)} × ${fmt.format(data.height)} PX`;
    for (const [id, source] of [['download-mask', data.mask]]) {
      $(id).href = source; $(id).classList.remove('disabled'); $(id).setAttribute('aria-disabled', 'false');
    }
    message('');
    renderView();
  } catch (error) { if (version === selectionVersion) message(error.message || 'Mất kết nối với server.', true); }
  finally { clearInterval(timer); $('busy').hidden = true; $('elapsed').textContent = 'Đang gửi ảnh…'; running = false; updateRun(); }
}

$('file').onchange = event => { useFile(event.target.files[0]); event.target.value = ''; };
$('dropzone').onkeydown = event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); $('file').click(); } };
for (const name of ['dragenter', 'dragover']) $('dropzone').addEventListener(name, event => { event.preventDefault(); $('dropzone').classList.add('dragging'); });
for (const name of ['dragleave', 'drop']) $('dropzone').addEventListener(name, event => { event.preventDefault(); $('dropzone').classList.remove('dragging'); });
$('dropzone').addEventListener('drop', event => useFile(event.dataTransfer.files[0]));
$('prompt').oninput = invalidate;
$('prompt').onkeydown = event => { if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) { event.preventDefault(); run(); } };
$('threshold').oninput = () => { $('threshold').dataset.edited = 'true'; $('threshold-value').value = Number($('threshold').value).toFixed(2); invalidate(); };
$('run').onclick = run;
$('checkpoint-form').onsubmit = uploadCheckpoint;
$('checkpoint-file').onchange = () => {
  checkpointFeedback('');
  if (model) renderModelStatus(model);
};
pollStatus(); loadSamples(); renderView();
