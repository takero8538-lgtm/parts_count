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

// モデル読み込み関数（フォルダのmodel.jsonを読み込む想定）
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

// モデル一覧をJSONから取得し<select>にセットする関数
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
    // 新しいモデルの出力に対応
    const outputs = model.execute({ 'x': normalized });
    
    console.log('=== 新しいモデル出力形式 ===');
    console.log('outputs が配列か？', Array.isArray(outputs));

    let outArray;
    
    // 出力が配列の場合（複数出力）
    if (Array.isArray(outputs)) {
      console.log('出力数:', outputs.length);
      
      // 最初の出力を使用
      const firstOutput = outputs[0];
      console.log('firstOutput shape:', firstOutput.shape);
      
      outArray = firstOutput.arraySync();
      
      // 配列内の Tensor を破棄
      outputs.forEach(o => {
        if (o && typeof o.dispose === 'function') {
          o.dispose();
        }
      });
    } else {
      // 単一出力の場合
      console.log('単一出力');
      console.log('output shape:', outputs.shape);
      
      outArray = outputs.arraySync();
      
      if (outputs && typeof outputs.dispose === 'function') {
        outputs.dispose();
      }
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imgElement, 0, 0);

    let count = 0;
    let maxConfidence = 0;

    ctx.lineWidth = 2;
    ctx.strokeStyle = 'red';
    ctx.font = '16px Arial';
    ctx.fillStyle = 'red';

    // 出力形式を判定して処理
    // 形式1: [1, 5, N] の場合（元の形式）
    if (outArray[0] && outArray[0][0] && Array.isArray(outArray[0][0])) {
      const xs = outArray[0][0];
      const ys = outArray[0][1];
      const ws = outArray[0][2];
      const hs = outArray[0][3];
      const confs = outArray[0][4];

      const maxConf = Math.max(...confs);
      const dynamicThreshold = maxConf * 0.5;

      console.log('=== 推論結果（形式1）===');
      console.log('canvas.width:', canvas.width);
      console.log('canvas.height:', canvas.height);
      console.log('maxConf:', maxConf);
      console.log('dynamicThreshold:', dynamicThreshold);

      for (let i = 0; i < confs.length; i++) {
        if (confs[i] >= dynamicThreshold) {
          maxConfidence = Math.max(maxConfidence, confs[i]);
          count++;

          const x = xs[i];
          const y = ys[i];
          const w = ws[i];
          const h = hs[i];

          const normX = x / 640;
          const normY = y / 640;
          const normW = w / 640;
          const normH = h / 640;

          const xmin = (normX - normW / 2) * canvas.width;
          const ymin = (normY - normH / 2) * canvas.height;
          const width = normW * canvas.width;
          const height = normH * canvas.height;

          if (count <= 3) {
            console.log(`[${count}] x=${x.toFixed(2)}, y=${y.toFixed(2)}, w=${w.toFixed(2)}, h=${h.toFixed(2)}`);
            console.log(`     canvas: x=${xmin.toFixed(2)}, y=${ymin.toFixed(2)}, w=${width.toFixed(2)}, h=${height.toFixed(2)}`);
          }

          if (xmin >= -100 && ymin >= -100 && width > 0 && height > 0) {
            ctx.strokeRect(xmin, ymin, width, height);
            ctx.fillText(
              `${(confs[i] * 100).toFixed(1)}%`,
              Math.max(0, xmin),
              Math.max(15, ymin)
            );
          }
        }
      }
    }
    // 形式2: [1, num_detections, 6] の場合（NMS処理済み）
    else if (outArray[0] && Array.isArray(outArray[0][0]) && outArray[0][0].length >= 6) {
      console.log('=== 推論結果（形式2：NMS処理済み）===');
      
      const detections = outArray[0];
      const maxConf = Math.max(...detections.map(d => d[4]));
      const dynamicThreshold = maxConf * 0.5;

      console.log('検出数:', detections.length);
      console.log('maxConf:', maxConf);

      detections.forEach((det, idx) => {
        // [x, y, w, h, conf, class_id] の形式と仮定
        const conf = det[4];
        
        if (conf >= dynamicThreshold) {
          maxConfidence = Math.max(maxConfidence, conf);
          count++;

          const x = det[0];
          const y = det[1];
          const w = det[2];
          const h = det[3];

          const normX = x / 640;
          const normY = y / 640;
          const normW = w / 640;
          const normH = h / 640;

          const xmin = (normX - normW / 2) * canvas.width;
          const ymin = (normY - normH / 2) * canvas.height;
          const width = normW * canvas.width;
          const height = normH * canvas.height;

          if (count <= 3) {
            console.log(`[${count}] conf=${conf.toFixed(4)}, x=${xmin.toFixed(2)}, y=${ymin.toFixed(2)}`);
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
    // 形式3: その他
    else {
      console.log('outArray:', outArray);
      console.log('⚠️ 予期しない出力形式です');
      resultDiv.textContent = '予期しない出力形式です。コンソールを確認してください。';
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
