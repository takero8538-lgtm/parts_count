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
  const modelJsonPath = folderPath + "model.json";

  resultDiv.textContent = 'モデル読み込み中...';
  runBtn.disabled = true;

  try {
    model = await tf.loadGraphModel(modelJsonPath);
    resultDiv.textContent = 'モデル読み込み完了: ' + folderPath;
    console.log('✅ モデル読み込み成功');
  } catch (error) {
    console.error('❌ 読み込みエラー:', error);
    resultDiv.textContent = 'モデルの読み込みに失敗しました';
    model = null;
  }

  runBtn.disabled = !(model && imgElement);
}

async function loadModelList() {
  try {
    const response = await fetch('models_list.json');
    if (!response.ok) throw new Error('モデル一覧の取得に失敗');
    const modelList = await response.json();

    modelSelect.innerHTML = '';
    modelList.forEach(m => {
      const option = document.createElement('option');
      option.value = m.path;
      option.textContent = m.name;
      modelSelect.appendChild(option);
    });

    if (modelList.length > 0) {
      await loadModelFromFolder(modelSelect.value);
    }
  } catch (error) {
    console.error(error);
    resultDiv.textContent = 'モデル一覧の読み込みに失敗しました';
    runBtn.disabled = true;
  }
}

modelSelect.addEventListener('change', () => {
  loadModelFromFolder(modelSelect.value);
});

async function runInference() {
  if (!model || !imgElement) {
    alert('モデルまたは画像がありません。');
    return;
  }

  const MODEL_INPUT_SIZE = 640;

  let inputTensor = tf.browser.fromPixels(imgElement).toFloat();
  let resized = tf.image.resizeBilinear(inputTensor, [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
  let expanded = resized.expandDims(0);
  let normalized = expanded.div(255);

  try {
    const outputs = await model.executeAsync({ 'x': normalized });
    
    let outArray;
    if (Array.isArray(outputs)) {
      const firstOutput = outputs[0];
      outArray = await firstOutput.array();
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
    ctx.lineWidth = 3;
    ctx.strokeStyle = 'red';
    ctx.font = '20px Arial';
    ctx.fillStyle = 'yellow';

    console.log('=== 最初の 3 つの検出を詳しく ===');

    detections.forEach((det, idx) => {
      const conf = det[4];
      
      if (conf >= threshold && idx < 3) {
        count++;

        const x = det[0];
        const y = det[1];
        const w = det[2];
        const h = det[3];

        console.log(`[${count}] 生データ: x=${x.toFixed(1)}, y=${y.toFixed(1)}, w=${w.toFixed(1)}, h=${h.toFixed(1)}`);

        // パターン1: 座標が中心座標 + サイズ
        const xmin1 = x - w / 2;
        const ymin1 = y - h / 2;
        console.log(`  パターン1(中心): xmin=${xmin1.toFixed(1)}, ymin=${ymin1.toFixed(1)}, w=${w.toFixed(1)}, h=${h.toFixed(1)}`);

        // パターン2: 座標がそのまま左上座標
        console.log(`  パターン2(左上): x=${x.toFixed(1)}, y=${y.toFixed(1)}, w=${w.toFixed(1)}, h=${h.toFixed(1)}`);

        // 描画（パターン1を試す）
        ctx.strokeRect(xmin1, ymin1, w, h);
        ctx.fillText(`${count}`, xmin1, ymin1 - 5);
      }
    });

    tf.dispose([inputTensor, resized, expanded, normalized]);
    resultDiv.textContent = `検出数: ${count}`;

  } catch (error) {
    console.error('推論エラー:', error);
    resultDiv.textContent = `推論エラー: ${error.message}`;
  }
}

runBtn.addEventListener('click', runInference);
loadModelList();
