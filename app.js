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
      ctx.drawImage(img, 0, 0);
      runBtn.disabled = false;
      resultDiv.textContent = '';
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
});

// モデルの読み込み関数
async function loadModel(url) {
  resultDiv.textContent = 'モデル読み込み中...';
  runBtn.disabled = true;
  try {
    model = await tf.loadGraphModel(url);
    resultDiv.textContent = 'モデル読み込み完了';
  } catch (error) {
    console.error(error);
    resultDiv.textContent = 'モデルの読み込みに失敗しました';
    model = null;
  }
  runBtn.disabled = !(model && imgElement);
}

// モデル選択時にロード
modelSelect.addEventListener('change', () => {
  loadModel(modelSelect.value);
});

// 初期モデル読み込み
loadModel(modelSelect.value);

// 推論実行関数
async function runInference() {
  if (!model || !imgElement) {
    alert('モデルまたは画像がありません。');
    return;
  }

  // 画像をTensorに変換、モデルに合わせてサイズ変更（ここは例: 320x320）
  const inputTensor = tf.browser.fromPixels(imgElement).toFloat();
  const resized = tf.image.resizeBilinear(inputTensor, [320, 320]);
  const expanded = resized.expandDims(0);
  const normalized = expanded.div(255);

  // 推論実行
  const output = await model.executeAsync(normalized);

  // 以下、モデルの出力にあわせて処理を調整してください。
  // 例としてバウンディングボックス、スコア、クラス情報を取り出すコードの雛形
  const boxes = output[0].arraySync();
  const scores = output[1].arraySync();
  const classes = output[2].arraySync();

  // 画像表示をクリアし再描画
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

  // Tensorをメモリ解放
  tf.dispose([inputTensor, resized, expanded, normalized]);
  if (Array.isArray(output)) {
    output.forEach(t => t.dispose());
  } else {
    output.dispose();
  }
}

// 推論ボタン押下時
runBtn.addEventListener('click', runInference);
