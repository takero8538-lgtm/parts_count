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
    const output = model.execute({ 'x': normalized });
    const outArray = output.arraySync();

    // ===== デバッグ情報出力 =====
    console.log('shape:', output.shape);
    console.log('最初の検出結果:', outArray[0]?.[0]?.slice(0, 5));
    console.log('すべてのスコア最大値:', Math.max(...outArray.flat()));
    // ===== ここまで =====

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imgElement, 0, 0);

    const threshold = 0.5;
    let count = 0;
    let maxConfidence = 0;

    ctx.lineWidth = 2;
    ctx.strokeStyle = 'red';
    ctx.font = '16px Arial';
    ctx.fillStyle = 'red';

    // 出力形式に応じた処理
    // 形式1: [1, 5, 8400] の場合
    if (output.shape[0] === 1 && output.shape[1] === 5 && output.shape[2] > 1000) {
      console.log('形式: [1, 5, 8400]');
      const xs = outArray[0][0];
      const ys = outArray[0][1];
      const ws = outArray[0][2];
      const hs = outArray[0][3];
      const confs = outArray[0][4];

      for (let i = 0; i < confs.length; i++) {
        maxConfidence = Math.max(maxConfidence, confs[i]);
        if (confs[i] >= threshold) {
          count++;
          const xmin = (xs[i] - ws[i] / 2) * canvas.width;
          const ymin = (ys[i] - hs[i] / 2) * canvas.height;
          const width = ws[i] * canvas.width;
          const height = hs[i] * canvas.height;

          ctx.strokeRect(xmin, ymin, width, height);
          ctx.fillText(
            `${(confs[i] * 100).toFixed(1)}%`,
            xmin,
            ymin > 10 ? ymin - 5 : 10
          );
        }
      }
    }
    // 形式2: [1, num_detections, 5+] の場合
    else if (output.shape[0] === 1 && output.shape[2] >= 5) {
      console.log('形式: [1, num_detections, 5+]');
      const detections = outArray[0];
      detections.forEach((det) => {
        const conf = det[4];
        maxConfidence = Math.max(maxConfidence, conf);
        if (conf >= threshold) {
          count++;
          const x = det[0] * canvas.width;
          const y = det[1] * canvas.height;
          const w = det[2] * canvas.width;
          const h = det[3] * canvas.height;

          ctx.strokeRect(x - w / 2, y - h / 2, w, h);
          ctx.fillText(
            `${(conf * 100).toFixed(1)}%`,
            x,
            y > 10 ? y - 5 : 10
          );
        }
      });
    }
    // 形式3: [num_detections, 5+] の場合
    else if (Array.isArray(outArray[0]) && outArray[0].length >= 5) {
      console.log('形式: [num_detections, 5+]');
      outArray.forEach((det) => {
        const conf = det[4];
        maxConfidence = Math.max(maxConfidence, conf);
        if (conf >= threshold) {
          count++;
          const x = det[0] * canvas.width;
          const y = det[1] * canvas.height;
          const w = det[2] * canvas.width;
          const h = det[3] * canvas.height;

          ctx.strokeRect(x - w / 2, y - h / 2, w, h);
          ctx.fillText(
            `${(conf * 100).toFixed(1)}%`,
            x,
            y > 10 ? y - 5 : 10
          );
        }
      });
    }

    tf.dispose([inputTensor, resized, expanded, normalized, output]);
    resultDiv.textContent = `検出数: ${count} (最高信頼度: ${(maxConfidence * 100).toFixed(1)}%)`;

  } catch (error) {
    console.error('推論エラー:', error);
    resultDiv.textContent = `推論エラー: ${error.message}`;
  }
}

runBtn.addEventListener('click', runInference);

loadModelList();
