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
    const response = await fetch('models_list.json'); // パスを修正（models/models_list.json → models_list.json）
    if (!response.ok) throw new Error('モデル一覧の取得に失敗');
    const modelList = await response.json();

    modelSelect.innerHTML = ''; // クリア

    modelList.forEach(m => {
      const option = document.createElement('option');
      option.value = m.path;
      option.textContent = m.name;
      modelSelect.appendChild(option);
    });

    // 初期モデルロード
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

  // 入力画像をモデルの期待サイズにリサイズ（モデルに合わせて調整）
  const MODEL_INPUT_SIZE = 320;

  const inputTensor = tf.browser.fromPixels(imgElement).toFloat();
  const resized = tf.image.resizeBilinear(inputTensor, [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
  const expanded = resized.expandDims(0);
  const normalized = expanded.div(255);

  try {
    const output = await model.executeAsync(normalized);

    // モデルの出力構造に合わせて修正が必要です
    // ここでは例：boxes, scores, classes が配列として返る想定
    const boxes = output[0].arraySync();
    const scores = output[1].arraySync();
    const classes = output[2].arraySync();

    // 元の画像をキャンバスに再描画
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imgElement, 0, 0);

    const threshold = 0.5;
    let count = 0;

    ctx.lineWidth = 2;
    ctx.strokeStyle = 'red';
    ctx.font = '16px Arial';
    ctx.fillStyle = 'red';

    for (let i = 0; i < scores[0].length; i++) {
      if (scores[0][i] < threshold) continue;
      count++;
      const [ymin, xmin, ymax, xmax] = boxes[0][i];
      const x = xmin * canvas.width;
      const y = ymin * canvas.height;
      const width = (xmax - xmin) * canvas.width;
      const height = (ymax - ymin) * canvas.height;

      ctx.strokeRect(x, y, width, height);
      ctx.fillText(`#${classes[0][i]} ${(scores[0][i] * 100).toFixed(1)}%`, x, y > 10 ? y - 5 : 10);
    }

    resultDiv.textContent = `検出数: ${count}`;

    // Tensorの解放
    tf.dispose([inputTensor, resized, expanded, normalized]);
    if (Array.isArray(output)) {
      output.forEach(t => t.dispose());
    } else {
      output.dispose();
    }
  } catch (error) {
    console.error('推論エラー:', error);
    alert('推論に失敗しました。');
  }
}

// 推論ボタン押下時
runBtn.addEventListener('click', runInference);

// ページロード時にモデルリスト読み込み開始
loadModelList();
