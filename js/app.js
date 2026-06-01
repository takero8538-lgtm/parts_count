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
    console.log('=== 入力情報 ===');
    console.log('inputTensor shape:', inputTensor.shape);
    console.log('normalized shape:', normalized.shape);

    const output = model.execute({ 'x': normalized });
    
    console.log('=== 出力情報 ===');
    console.log('output:', output);
    console.log('output.shape:', output.shape);
    console.log('output.size:', output.size);
    
    const outArray = output.arraySync();
    
    console.log('=== 生データ出力 ===');
    console.log('typeof outArray:', typeof outArray);
    console.log('Array.isArray(outArray):', Array.isArray(outArray));
    console.log('outArray:', JSON.stringify(outArray));
    console.log('outArray length:', outArray.length);
    console.log('outArray[0]:', outArray[0]);
    console.log('typeof outArray[0]:', typeof outArray[0]);
    console.log('outArray[0] length:', outArray[0]?.length);

    // ネストの深さを確認
    if (outArray[0] && outArray[0][0]) {
      console.log('outArray[0][0]:', outArray[0][0]);
      console.log('outArray[0][0] length:', outArray[0][0]?.length);
      if (outArray[0][0][0] !== undefined) {
        console.log('outArray[0][0][0]:', outArray[0][0][0]);
      }
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imgElement, 0, 0);

    resultDiv.textContent = 'コンソールを確認してください';

    tf.dispose([inputTensor, resized, expanded, normalized, output]);

  } catch (error) {
    console.error('推論エラー:', error);
    resultDiv.textContent = `推論エラー: ${error.message}`;
  }
}
runBtn.addEventListener('click', runInference);

loadModelList();
