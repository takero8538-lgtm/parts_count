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
    const response = await fetch('models_list.json'); // index.html と同じ階層にあると想定
    if (!response.ok) throw new Error('モデル一覧の取得に失敗');
    const modelList = await response.json();

    modelSelect.innerHTML = ''; // クリア

    modelList.forEach(m => {
      const option = document.createElement('option');
      option.value = m.path;    // 例: "models/OLWM4/"
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
    // モデルに期待される入力名に合わせてオブジェクト形式で渡す
    // もし単一Tensorでも良いなら model.execute(normalized) に変更可
    const output = model.execute({'x': normalized});
    console.log('output:', output);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imgElement, 0, 0);

    const threshold = 0.5;
    let count = 0;

    ctx.lineWidth = 2;
    ctx.strokeStyle = 'red';
    ctx.font = '16px Arial';
    ctx.fillStyle = 'red';

    if (Array.isArray(output)) {
      // 出力が配列の場合の例
      const boxes = output[0].arraySync();
      const scores = output[1].arraySync();
      const classes = output[2].arraySync();

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

      output.forEach(t => t.dispose());
    } else {
      // 出力が単一Tensorの場合の例
      const outArray = output.arraySync();
      console.log('outArray:', outArray);
      // モデル仕様に応じて描画処理をここで実装してください

      output.dispose();
    }

    tf.dispose([inputTensor, resized, expanded, normalized]);
    resultDiv.textContent = `検出数: ${count}`;

  } catch (error) {
    console.error('推論エラー:', error);
    alert('推論に失敗しました。');
  }
}

runBtn.addEventListener('click', runInference);

loadModelList();
