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
    resultDiv.textContent = '推論中...';
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
    
    console.log('=== デバッグ情報 ===');
    console.log('canvas サイズ:', canvas.width, '×', canvas.height);
    console.log('検出数:', detections.length);
    
    console.log('=== 最初の 10 個の検出 ===');
    for (let i = 0; i < Math.min(10, detections.length); i++) {
      const det = detections[i];
      console.log(`[${i}] x=${det[0].toFixed(3)}, y=${det[1].toFixed(3)}, w=${det[2].toFixed(3)}, h=${det[3].toFixed(3)}, conf=${det[4].toFixed(2)}`);
    }

    const confs = detections.map(d => d[4]);
    const maxConf = Math.max(...confs);
    const minConf = Math.min(...confs);
    
    console.log('confidence 範囲:', minConf.toFixed(2), '～', maxConf.toFixed(2));
    
    const threshold = maxConf * 0.5;
    console.log('使用閾値:', threshold.toFixed(2));

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imgElement, 0, 0);

    let count = 0;
    let maxConfidence = 0;

    ctx.lineWidth = 2;
    ctx.strokeStyle = 'red';
    ctx.font = '16px Arial';
    ctx.fillStyle = 'red';

    detections.forEach((det) => {
      const x = det[0];
      const y = det[1];
      const w = det[2];
      const h = det[3];
      const conf = det[4];
      
      if (conf >= threshold) {
        maxConfidence = Math.max(maxConfidence, conf);
        count++;

        // 座標が 0-1 の範囲か、0-640 の範囲か自動判定
        let xmin, ymin, width, height;
        
        if (x >= 0 && x <= 1 && y >= 0 && y <= 1 && w >= 0 && w <= 1 && h >= 0 && h <= 1) {
          // 0-1 の正規化座標
          xmin = (x - w / 2) * canvas.width;
          ymin = (y - h / 2) * canvas.height;
          width = w * canvas.width;
          height = h * canvas.height;
        } else {
          // 0-640 の座標
          const normX = x / 640;
          const normY = y / 640;
          const normW = w / 640;
          const normH = h / 640;

          xmin = (normX - normW / 2) * canvas.width;
          ymin = (normY - normH / 2) * canvas.height;
          width = normW * canvas.width;
          height = normH * canvas.height;
        }

        if (xmin >= -100 && ymin >= -100 && width > 0 && height > 0) {
          ctx.strokeRect(xmin, ymin, width, height);
          ctx.fillText(
            `${(conf / 255 * 100).toFixed(1)}%`,
            Math.max(0, xmin),
            Math.max(15, ymin)
          );
        }
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
