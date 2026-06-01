const modelSelect = document.getElementById('modelSelect');
const imageInput = document.getElementById('imageInput');
const runBtn = document.getElementById('runBtn');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const resultDiv = document.getElementById('result');

let model = null;
let imgElement = null;

imageInput.addEventListener('change', (evt) => {
  const file = evt.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => {
      imgElement = img;
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);
      runBtn.disabled = !(model && imgElement);
      resultDiv.textContent = '';
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
});

async function loadModelFromFolder(folderPath) {
  if (!folderPath.endsWith('/')) folderPath += '/';
  resultDiv.textContent = 'モデル読み込み中...';
  runBtn.disabled = true;
  try {
    model = await tf.loadGraphModel(folderPath + "model.json");
    resultDiv.textContent = 'モデル読み込み完了';
  } catch (error) {
    resultDiv.textContent = 'モデルの読み込みに失敗しました';
    model = null;
  }
  runBtn.disabled = !(model && imgElement);
}

async function loadModelList() {
  try {
    const modelList = await (await fetch('models_list.json')).json();
    modelSelect.innerHTML = '';
    modelList.forEach(m => {
      const option = document.createElement('option');
      option.value = m.path;
      option.textContent = m.name;
      modelSelect.appendChild(option);
    });
    if (modelList.length > 0) await loadModelFromFolder(modelSelect.value);
  } catch (error) {
    resultDiv.textContent = 'モデル一覧の読み込みに失敗しました';
  }
}

modelSelect.addEventListener('change', () => loadModelFromFolder(modelSelect.value));

async function runInference() {
  if (!model || !imgElement) {
    alert('モデルまたは画像がありません。');
    return;
  }

  let inputTensor = tf.browser.fromPixels(imgElement).toFloat();
  let resized = tf.image.resizeBilinear(inputTensor, [640, 640]);
  let expanded = resized.expandDims(0);
  let normalized = expanded.div(255);

  try {
    const outputs = await model.executeAsync({ 'x': normalized });
    let outArray;
    if (Array.isArray(outputs)) {
      outArray = await outputs[0].array();
      outputs.forEach(o => o.dispose?.());
    } else {
      outArray = await outputs.array();
      outputs.dispose?.();
    }

    const detections = outArray[0];
    const confs = detections.map(d => d[4]);
    const maxConf = Math.max(...confs);
    const threshold = maxConf * 0.7;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imgElement, 0, 0);

    let count = 0;
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'red';
    ctx.font = '16px Arial';
    ctx.fillStyle = 'red';

    detections.forEach((det) => {
      const conf = det[4];
      if (conf >= threshold) {
        // ===== スケーリングなし =====
        const cx = det[0];
        const cy = det[1];
        const w = det[2];
        const h = det[3];

        const xmin = cx - w / 2;
        const ymin = cy - h / 2;

        if (w > 0 && h > 0) {
          ctx.strokeRect(xmin, ymin, w, h);
          ctx.fillText(`${(conf * 100).toFixed(1)}%`, xmin + 5, ymin + 20);
          count++;
        }
      }
    });

    tf.dispose([inputTensor, resized, expanded, normalized]);
    resultDiv.textContent = `検出数: ${count}`;

  } catch (error) {
    console.error(error);
    resultDiv.textContent = `エラー: ${error.message}`;
  }
}

runBtn.addEventListener('click', runInference);
loadModelList();

console.log('✅ NEW APP.JS LOADED - v8');
