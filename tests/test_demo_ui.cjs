// Exercise checkpoint UI transitions without downloading or running a full model.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function element() {
  const classes = new Set();
  return {
    hidden: false, disabled: false, value: '', files: [], dataset: {}, textContent: '',
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
  const requests = [];
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
        mask_threshold: 0.6, checkpoint_upload_allowed: false,
      };
      queueMicrotask(() => this.onload());
    }
  }
  const context = vm.createContext({
    document: { getElementById: get, querySelectorAll: () => [] },
    Intl, XMLHttpRequest: Upload,
    setTimeout: () => 1, clearTimeout() {},
    fetch: async url => ({ ok: true, json: async () => url === '/api/samples' ? [] : serverState }),
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
  console.log('Checkpoint UI passed: missing file, progress, successful load, failed upload/retry, type and size limits.');
}

main().catch(error => { console.error(error); process.exitCode = 1; });
