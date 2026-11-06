// Exercise checkpoint UI transitions without downloading or running a full model.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function element() {
  const classes = new Set();
  return {
    hidden: false, disabled: false, value: '', files: [], dataset: {}, textContent: '',
    children: [], append(...children) { this.children.push(...children); },
    scrollIntoView() {},
    lastElementChild: {},
    classList: {
      toggle(name, enabled) { if (enabled) classes.add(name); else classes.delete(name); },
      add(name) { classes.add(name); }, remove(name) { classes.delete(name); },
      contains(name) { return classes.has(name); },
    },
    setAttribute() {}, removeAttribute() {}, addEventListener() {},
  };
}

async function main() {
  const elements = new Map();
  const get = id => {
    if (!elements.has(id)) elements.set(id, element());
    return elements.get(id);
  };
  const missing = {
    status: 'missing_checkpoint', message: 'Không tìm thấy checkpoint best.pt.',
    checkpoint_upload_allowed: true, max_checkpoint_bytes: 4 * 1024 ** 3,
  };
  let serverState = missing;
  let uploadError = false;
  let predictionRequest;
  let samples = [];
  const prediction = {
    original: 'data:image/png;base64,original', mask: 'data:image/png;base64,mask',
    overlay: 'data:image/png;base64,overlay', width: 12, height: 8,
    classification: { predicted_class: { name: 'apple rust' }, method: 'reference_image_lookup', source: 'reference_manifest', known_disease_checkpoint: true },
  };
  const requests = [];
  const readers = [];
  class Upload {
    constructor() { this.upload = {}; this.headers = {}; requests.push(this); }
    open(method, url) { this.method = method; this.url = url; }
    setRequestHeader(name, value) { this.headers[name] = value; }
    send(file) {
      this.body = file;
      this.upload.onprogress({ lengthComputable: true, loaded: file.size, total: file.size });
      this.status = uploadError ? 400 : 202;
      this.responseText = JSON.stringify(uploadError ? { error: 'File tải lên không hợp lệ.' } : {
        status: 'loading', message: 'Đang nạp model…', checkpoint_upload_allowed: false,
      });
      if (!uploadError) serverState = {
        status: 'ready', device: 'cpu', input_size: [32, 32], checkpoint: file.name,
        mask_threshold: 0.6, checkpoint_upload_allowed: false, classification_available: true, prediction_pipeline: 'reference_lookup_segmentation_v1',
      };
      queueMicrotask(() => this.onload());
    }
  }
  const context = vm.createContext({
    document: { getElementById: get, querySelectorAll: () => [],
      createElement: element, querySelector: () => element() },
    Intl, XMLHttpRequest: Upload,
    FileReader: class {
      readAsDataURL(file) { this.file = file; readers.push(this); }
    },
    Image: class { async decode() {} },
    window: { matchMedia: () => ({ matches: false }) },
    performance: { now: () => 0 }, setInterval: () => 1, clearInterval() {},
    setTimeout: () => 1, clearTimeout() {},
    fetch: async (url, options) => {
      if (url === '/api/predict') {
        predictionRequest = JSON.parse(options.body);
        return { ok: true, json: async () => structuredClone(prediction) };
      }
      return { ok: true, json: async () => url === '/api/samples' ? samples : serverState };
    },
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../demo/app.js'), 'utf8'), context);
  await new Promise(setImmediate);
  assert.equal(get('checkpoint-panel').hidden, false);
  assert.equal(get('checkpoint-file').disabled, false);
  assert.equal(get('checkpoint-submit').disabled, true);
  assert.equal(get('run').disabled, true);

  const file = { name: 'my best.pt', size: 1024 };
  get('checkpoint-file').files = [file];
  get('checkpoint-file').onchange();
  assert.equal(get('checkpoint-submit').disabled, false);
  const uploading = get('checkpoint-form').onsubmit({ preventDefault() {} });
  assert.equal(get('checkpoint-file').disabled, true);
  assert.equal(get('checkpoint-submit').disabled, true);
  await uploading;
  assert.equal(requests[0].url, '/api/checkpoint');
  assert.equal(requests[0].headers['Content-Type'], 'application/octet-stream');
  assert.equal(requests[0].headers['X-Checkpoint-Name'], 'my%20best.pt');
  assert.equal(requests[0].body, file);
  assert.equal(get('checkpoint-panel').hidden, true);
  assert.equal(Number(get('threshold').value), 0.6);

  uploadError = true;
  serverState = { ...missing, status: 'error' };
  await vm.runInContext('pollStatus()', context);
  samples = Array.from({ length: 15 }, (_, id) => ({
    id, title: 'Ground truth disease', prompt: 'Ground truth disease',
    name: 'ground-truth-disease.jpg', url: `/api/sample/${id}`,
  }));
  await vm.runInContext('loadSamples()', context);
  assert.equal(get('samples').children.length, 5, 'Only five sample cards should be displayed');
  const sampleButton = get('samples').children[0];
  assert.equal(sampleButton.title, 'Ảnh mẫu 01', 'Even legacy API labels must stay hidden');
  assert.equal(sampleButton.children[0].alt, 'Ảnh mẫu 01');
  sampleButton.onclick();
  assert.equal(get('prompt').value, '', 'Selecting a sample must not prefill its disease');
  assert.equal(get('file-label').textContent, 'Ảnh mẫu 01');
  await get('checkpoint-form').onsubmit({ preventDefault() {} });
  assert.equal(get('checkpoint-feedback').textContent, 'File tải lên không hợp lệ.');
  assert.equal(get('checkpoint-feedback').classList.contains('error'), true);
  assert.equal(get('checkpoint-file').disabled, false);
  assert.equal(get('checkpoint-submit').disabled, false);

  for (const bad of [{ name: 'bad.txt', size: 10 }, { name: 'big.pt', size: 4 * 1024 ** 3 + 1 }]) {
    get('checkpoint-file').files = [bad];
    await get('checkpoint-form').onsubmit({ preventDefault() {} });
  }
  assert.equal(requests.length, 2, 'Invalid files must be rejected before creating a request');

  serverState = {
    status: 'ready', device: 'cpu', input_size: [32, 32], checkpoint: 'best.pt',
    mask_threshold: 0.5, classification_available: true, prediction_pipeline: 'reference_lookup_segmentation_v1',
  };
  await vm.runInContext('pollStatus()', context);
  vm.runInContext("setImage('/api/sample/0', { sample_id: 0 }, 'sample-01.jpg')", context);
  assert.equal(get('run').disabled, true, 'A prompt is required for the single prediction flow');
  assert.equal(get('classification-result').hidden, true, 'No disease may be shown before inference');
  get('prompt').value = 'predict risk on leaf';
  get('prompt').oninput();
  assert.equal(get('run').disabled, false, 'A general prompt must enable class + mask prediction');
  // Isolate prediction state transitions from browser canvas rasterization.
  vm.runInContext('predictionOverlay = async () => "data:image/png;base64,overlay"', context);
  await vm.runInContext('run()', context);
  assert.equal('mode' in predictionRequest, false);
  assert.equal(predictionRequest.prompt, 'predict risk on leaf');
  assert.equal(get('classification-result').hidden, false);
  assert.equal(get('predicted-class').textContent, 'apple rust');
  assert.equal(get('classification-note').textContent, '');
  assert.equal(get('classification-note').hidden, true);

  get('prompt').value = 'diseased regions';
  get('prompt').oninput();
  assert.equal(get('classification-result').hidden, true, 'Changing prompts must clear the old class');
  prediction.classification.predicted_class = null;
  prediction.classification.reason = 'no_prompt_region';
  prediction.classification.source = 'model';
  await vm.runInContext('run()', context);
  assert.equal(get('predicted-class').textContent, 'Chưa xác định được lớp bệnh');
  assert.match(get('classification-note').textContent, /prompt/);
  vm.runInContext("setImage('/api/sample/1', { sample_id: 1 }, 'sample-02.jpg')", context);
  assert.equal(get('classification-result').hidden, true, 'Changing images must clear the old class');

  const reading = vm.runInContext("useFile({ type: 'image/png', size: 12, name: 'new.png' })", context);
  assert.equal(get('run').disabled, true, 'Pending file reads must not run inference on the old image');
  sampleButton.onclick();
  const sampleMessage = get('message').textContent;
  readers[0].onerror(new Error('stale file read'));
  await reading;
  assert.equal(get('message').textContent, sampleMessage, 'Stale read errors must not overwrite the new selection');
  assert.equal(get('run').disabled, false);

  const nextReading = vm.runInContext("useFile({ type: 'image/png', size: 12, name: 'next.png' })", context);
  assert.equal(get('run').disabled, true);
  await vm.runInContext("useFile({ type: 'text/plain', size: 12, name: 'bad.txt' })", context);
  readers[1].onerror(new Error('superseded file read'));
  await nextReading;
  assert.equal(get('run').disabled, false, 'Invalid replacement files must not leave the run button locked');
  console.log('Demo UI passed: checkpoint upload/retry, required general prompt, class + mask response and stale result clearing.');
}

main().catch(error => { console.error(error); process.exitCode = 1; });
