'use strict';
const $ = id => document.getElementById(id);
let model = null, selected = null, result = null, running = false;
let selectionVersion = 0, readVersion = 0;
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
  $('run').disabled = running || !selected || model?.status !== 'ready' || !$('prompt').value.trim();
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
async function pollStatus() {
  try {
    const response = await fetch('/api/status');
    if (!response.ok) throw new Error();
    const previousStatus = model?.status;
    model = await response.json();
    $('status').className = `status ${model.status}`;
    $('status').lastElementChild.textContent = model.status === 'ready' ? `Sẵn sàng · ${model.device.toUpperCase()}` : model.status === 'error' ? 'Lỗi nạp model' : 'Đang nạp model';
    if (model.status === 'ready') {
      $('model-detail').textContent = `SAM 3 · ${model.input_size.join(' × ')} · ${model.checkpoint}`;
      if (previousStatus !== 'ready' && !$('threshold').dataset.edited) {
        $('threshold').value = model.mask_threshold;
        $('threshold-value').value = Number(model.mask_threshold).toFixed(2);
      }
      if (!selected) message('Model sẵn sàng. Chọn ảnh mẫu hoặc tải ảnh của bạn lên.');
    } else if (model.status === 'error') message(model.message, true);
    updateRun();
    if (model.status === 'loading') setTimeout(pollStatus, 2000);
  } catch {
    $('status').className = 'status error';
    $('status').lastElementChild.textContent = 'Mất kết nối';
    model = null; updateRun(); message('Chưa kết nối được server. Kiểm tra terminal đang chạy demo.py.', true);
    setTimeout(pollStatus, 5000);
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
pollStatus(); loadSamples(); renderView();
