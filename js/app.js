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

    console.log('=== デバッグ ===');
    console.log('shape:', output.shape);
    console.log('outArray[0][0].length:', outArray[0][0].length);
    console.log('outArray[0][4].length:', outArray[0][4].length);
    console.log('最初の5つ (x):', outArray[0][0].slice(0, 5));
    console.log('最初の5つ (y):', outArray[0][1].slice(0, 5));
    console.log('最初の5つ (w):', outArray[0][2].slice(0, 5));
    console.log('最初の5つ (h):', outArray[0][3].slice(0, 5));
    console.log('最初の5つ (confidence):', outArray[0][4].slice(0, 5));
    console.log('confidence 最大値:', Math.max(...outArray[0][4]));
    console.log('confidence 最小値:', Math.min(...outArray[0][4]));

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imgElement, 0, 0);

    let count = 0;
    let maxConfidence = 0;

    ctx.lineWidth = 2;
    ctx.strokeStyle = 'red';
    ctx.font = '16px Arial';
    ctx.fillStyle = 'red';

    const xs = outArray[0][0];
    const ys = outArray[0][1];
    const ws = outArray[0][2];
    const hs = outArray[0][3];
    const confs = outArray[0][4];

    // confidence の最大値と最小値を取得
    const maxConf = Math.max(...confs);
    const minConf = Math.min(...confs);
    console.log(`confidence 範囲: ${minConf.toFixed(4)} ～ ${maxConf.toFixed(4)}`);

    // 閾値を動的に調整（最大値の 50%）
    const dynamicThreshold = maxConf * 0.5;
    console.log(`使用する閾値: ${dynamicThreshold.toFixed(4)}`);

    for (let i = 0; i < confs.length; i++) {
      if (confs[i] >= dynamicThreshold) {
        maxConfidence = Math.max(maxConfidence, confs[i]);
        count++;

        // 座標がすでにピクセル値と仮定（0～640 の範囲）
        const x = xs[i];
        const y = ys[i];
        const w = ws[i];
        const h = hs[i];

        // キャンバスサイズに合わせてスケーリング
        const scaleX = canvas.width / 640;
        const scaleY = canvas.height / 640;

        const xmin = (x - w / 2) * scaleX;
        const ymin = (y - h / 2) * scaleY;
        const width = w * scaleX;
        const height = h * scaleY;

        ctx.strokeRect(xmin, ymin, width, height);
        ctx.fillText(
          `${(confs[i] * 100).toFixed(1)}%`,
          xmin,
          ymin > 10 ? ymin - 5 : 10
        );
      }
    }

    tf.dispose([inputTensor, resized, expanded, normalized, output]);
    resultDiv.textContent = `検出数: ${count} (最高信頼度: ${maxConfidence.toFixed(4)})`;

  } catch (error) {
    console.error('推論エラー:', error);
    resultDiv.textContent = `推論エラー: ${error.message}`;
  }
}

runBtn.addEventListener('click', runInference);

loadModelList();
