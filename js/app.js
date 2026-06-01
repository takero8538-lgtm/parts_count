const modelSelect = document.getElementById('modelSelect');
const imageInput = document.getElementById('imageInput');
const runBtn = document.getElementById('runBtn');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const resultDiv = document.getElementById('result');

let model = null;
let imgElement = null;

// 画像ファイル選択時の処理
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

// モデル読み込み関数
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
    resultDiv.textContent = 'モデルの読み込みに失敗しました: ' + error.message;
    model = null;
  }

  runBtn.disabled = !(model && imgElement);
}

// モデル一覧をJSONから取得
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

// モデル選択時に再読み込み
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
      console.log('出力形状:', firstOutput.shape);
      outArray = await firstOutput.array();
      outputs.forEach(o => o.dispose?.());
    } else {
      outArray = await outputs.array();
      outputs.dispose?.();
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imgElement, 0, 0);

    let count = 0;
    let maxConfidence = 0;

    ctx.lineWidth = 2;
    ctx.strokeStyle = 'red';
    ctx.font = '16px Arial';
    ctx.fillStyle = 'red';

    // 新しいコード: [1, 300, 6] 形式に対応
    console.log('出力形式チェック:', outArray[0]?.length);
    
    const detections = outArray[0]; // [300, 6]
    
    detections.forEach((det) => {
      const x = det[0];
      const y = det[1];
      const w = det[2];
      const h = det[3];
      const conf = det[4];
      
      // confidence を正規化（0-255 → 0-1）
      const normalizedConf = conf > 1 ? conf / 255 : conf;
      
      // 閾値チェック
      if (normalizedConf > 0.3) {
        maxConfidence = Math.max(maxConfidence, normalizedConf);
        count++;

        // 座標変換（640×640 → キャンバス）
        const normX = x / 640;
        const normY = y / 640;
        const normW = w / 640;
        const normH = h / 640;

        const xmin = (normX - normW / 2) * canvas.width;
        const ymin = (normY - normH / 2) * canvas.height;
        const width = normW * canvas.width;
        const height = normH * canvas.height;

        if (xmin >= -100 && ymin >= -100 && width > 0 && height > 0) {
          ctx.strokeRect(xmin, ymin, width, height);
          ctx.fillText(
            `${(normalizedConf * 100).toFixed(1)}%`,
            Math.max(0, xmin),
            Math.max(15, ymin)
          );
        }
      }
    });

    tf.dispose([inputTensor, resized, expanded, normalized]);
    resultDiv.textContent = `検出数: ${count} (最高信頼度: ${(maxConfidence * 100).toFixed(1)}%)`;

  } catch (error) {
    console.error('推論エラー:', error);
    resultDiv.textContent = `推論エラー: ${error.message}`;
  }
}

runBtn.addEventListener('click', runInference);
loadModelList();
