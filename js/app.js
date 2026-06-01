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
  } catch (error) {
    console.error(error);
    resultDiv.textContent = 'モデルの読み込みに失敗しました';
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

// 推論実行関数
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
      outArray = await outputs[0].array();
      outputs.forEach(o => o.dispose?.());
    } else {
      outArray = await outputs.array();
      outputs.dispose?.();
    }

    const detections = outArray[0];
    const confs = detections.map(d => d[4]);
    const maxConf = Math.max(...confs);
    const threshold = maxConf * 0.5;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imgElement, 0, 0);

    let count = 0;
    let maxConfidence = 0;

    ctx.lineWidth = 2;
    ctx.strokeStyle = 'red';
    ctx.font = '16px Arial';
    ctx.fillStyle = 'red';

    console.log('=== 推論結果 ===');
    console.log('canvas.width:', canvas.width);
    console.log('canvas.height:', canvas.height);
    console.log('maxConf:', maxConf);
    console.log('threshold:', threshold);

    detections.forEach((det, idx) => {
      const conf = det[4];
      
      if (conf >= threshold) {
        maxConfidence = Math.max(maxConfidence, conf);
        count++;

        // [cx, cy, w, h, conf, class] 形式
        // 座標は 0-640 の範囲
        const cx = det[0];
        const cy = det[1];
        const w = det[2];
        const h = det[3];

        // 0-640 → 0-1 に正規化
        const normCx = cx / 640;
        const normCy = cy / 640;
        const normW = w / 640;
        const normH = h / 640;

        // キャンバス座標に変換
        const canvasCx = normCx * canvas.width;
        const canvasCy = normCy * canvas.height;
        const canvasW = normW * canvas.width;
        const canvasH = normH * canvas.height;

        // 中心座標 → 左上座標
        const xmin = canvasCx - canvasW / 2;
        const ymin = canvasCy - canvasH / 2;

        // 最初の3つを出力
        if (count <= 3) {
          console.log(`[${count}] 元座標: x=${cx.toFixed(2)}, y=${cy.toFixed(2)}, w=${w.toFixed(2)}, h=${h.toFixed(2)}`);
          console.log(`     正規化: cx=${normCx.toFixed(4)}, cy=${normCy.toFixed(4)}, w=${normW.toFixed(4)}, h=${normH.toFixed(4)}`);
          console.log(`     キャンバス: xmin=${xmin.toFixed(2)}, ymin=${ymin.toFixed(2)}, w=${canvasW.toFixed(2)}, h=${canvasH.toFixed(2)}`);
        }

        // 座標が有効な範囲か確認
        if (xmin >= -100 && ymin >= -100 && canvasW > 0 && canvasH > 0) {
          ctx.strokeRect(xmin, ymin, canvasW, canvasH);
          ctx.fillText(
            `${(conf * 100).toFixed(1)}%`,
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

console.log('✅ NEW APP.JS LOADED - FINAL');
console.log(model.outputs);
