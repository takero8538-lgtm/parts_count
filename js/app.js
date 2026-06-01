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

// 推論実行関数（修正版）
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
    
    console.log('outputs が配列か？', Array.isArray(outputs));
    console.log('出力数:', Array.isArray(outputs) ? outputs.length : 1);

    let outArray;
    if (Array.isArray(outputs)) {
      const firstOutput = outputs[0];
      console.log('firstOutput shape:', firstOutput.shape);
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

    // 出力形式: [1, 300, 6] → [x, y, w, h, conf, class_id]
    if (outArray[0] && Array.isArray(outArray[0]) && outArray[0][0]?.length >= 6) {
      console.log('形式: [batch, num_detections, 6]');
      
      const detections = outArray[0];
      
      // confidence の範囲を調べる
      const confs = detections.map(d => d[4]);
      const maxConf = Math.max(...confs);
      console.log('maxConf:', maxConf, 'minConf:', Math.min(...confs));

      // confidence が 0-255 の場合は 255 で正規化
      const normalizedConfs = confs.map(c => c > 1 ? c / 255 : c);
      const maxNormConf = Math.max(...normalizedConfs);
      
      // 動的閾値（最大値の 50%）
      const threshold = maxNormConf * 0.5;
      console.log('使用閾値:', threshold.toFixed(4));

      detections.forEach((det, idx) => {
        // confidence を正規化
        const conf = det[4] > 1 ? det[4] / 255 : det[4];
        
        if (conf >= threshold) {
          maxConfidence = Math.max(maxConfidence, conf);
          count++;

          // [x, y, w, h, conf, class_id]
          const x = det[0];
          const y = det[1];
          const w = det[2];
          const h = det[3];

          // 座標が 0-640 の範囲と仮定
          const normX = x / 640;
          const normY = y / 640;
          const normW = w / 640;
          const normH = h / 640;

          const xmin = (normX - normW / 2) * canvas.width;
          const ymin = (normY - normH / 2) * canvas.height;
          const width = normW * canvas.width;
          const height = normH * canvas.height;

          if (count <= 5) {
            console.log(`[${count}] conf=${conf.toFixed(4)}, x=${xmin.toFixed(0)}, y=${ymin.toFixed(0)}, w=${width.toFixed(0)}, h=${height.toFixed(0)}`);
          }

          if (xmin >= -100 && ymin >= -100 && width > 0 && height > 0) {
            ctx.strokeRect(xmin, ymin, width, height);
            ctx.fillText(
              `${(conf * 100).toFixed(1)}%`,
              Math.max(0, xmin),
              Math.max(15, ymin)
            );
          }
        }
      });
    }

    tf.dispose([inputTensor, resized, expanded, normalized]);
    resultDiv.textContent = `検出数: ${count} (最高信頼度: ${(maxConfidence * 100).toFixed(1)}%)`;

  } catch (error) {
    console.error('推論エラー:', error);
    resultDiv.textContent = `推論エラー: ${error.message}`;
  }
}

runBtn.addEventListener('click', runInference);

loadModelList();
