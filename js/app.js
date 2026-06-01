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

    const xs = outArray[0][0];
    const ys = outArray[0][1];
    const ws = outArray[0][2];
    const hs = outArray[0][3];
    const confs = outArray[0][4];

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imgElement, 0, 0);

    let count = 0;
    let maxConfidence = 0;

    ctx.lineWidth = 2;
    ctx.strokeStyle = 'red';
    ctx.font = '16px Arial';
    ctx.fillStyle = 'red';

    const maxConf = Math.max(...confs);
    const dynamicThreshold = maxConf * 0.5;

    console.log('=== 推論結果 ===');
    console.log('canvas.width:', canvas.width);
    console.log('canvas.height:', canvas.height);
    console.log('maxConf:', maxConf);
    console.log('dynamicThreshold:', dynamicThreshold);

    for (let i = 0; i < confs.length; i++) {
      if (confs[i] >= dynamicThreshold) {
        maxConfidence = Math.max(maxConfidence, confs[i]);
        count++;

        // 座標値（0～640 の範囲）
        const x = xs[i];
        const y = ys[i];
        const w = ws[i];
        const h = hs[i];

        // 正規化座標に変換（0～1）
        const normX = x / 640;
        const normY = y / 640;
        const normW = w / 640;
        const normH = h / 640;

        // キャンバス座標に変換
        const xmin = (normX - normW / 2) * canvas.width;
        const ymin = (normY - normH / 2) * canvas.height;
        const width = normW * canvas.width;
        const height = normH * canvas.height;

        // 最初の3つを出力
        if (count <= 3) {
          console.log(`[${count}] 元の座標: x=${x.toFixed(2)}, y=${y.toFixed(2)}, w=${w.toFixed(2)}, h=${h.toFixed(2)}`);
          console.log(`     正規化: x=${normX.toFixed(4)}, y=${normY.toFixed(4)}, w=${normW.toFixed(4)}, h=${normH.toFixed(4)}`);
          console.log(`     キャンバス: xmin=${xmin.toFixed(2)}, ymin=${ymin.toFixed(2)}, width=${width.toFixed(2)}, height=${height.toFixed(2)}`);
        }

        // 座標が有効な範囲か確認
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

    tf.dispose([inputTensor, resized, expanded, normalized, output]);
    resultDiv.textContent = `検出数: ${count} (最高信頼度: ${(maxConfidence * 100).toFixed(1)}%)`;

  } catch (error) {
    console.error('推論エラー:', error);
    resultDiv.textContent = `推論エラー: ${error.message}`;
  }
}

runBtn.addEventListener('click', runInference);

loadModelList();
