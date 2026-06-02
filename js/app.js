const modelSelect = document.getElementById('modelSelect');
const imageInput = document.getElementById('imageInput');
const runBtn = document.getElementById('runBtn');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const resultDiv = document.getElementById('result');

let model = null;
let imgElement = null;

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

async function loadModelFromFolder(folderPath) {
  if (!folderPath.endsWith('/')) folderPath += '/';
  resultDiv.textContent = 'モデル読み込み中...';
  runBtn.disabled = true;
  try {
    model = await tf.loadGraphModel(folderPath + "model.json");
    resultDiv.textContent = 'モデル読み込み完了';
  } catch (error) {
    resultDiv.textContent = 'モデルの読み込みに失敗しました';
    model = null;
  }
  runBtn.disabled = !(model && imgElement);
}

async function loadModelList() {
  try {
    const response = await fetch('models_list.json');
    const modelList = await response.json();
    modelSelect.innerHTML = '';
    modelList.forEach(m => {
      const option = document.createElement('option');
      option.value = m.path;
      option.textContent = m.name;
      modelSelect.appendChild(option);
    });
    if (modelList.length > 0) await loadModelFromFolder(modelSelect.value);
  } catch (error) {
    resultDiv.textContent = 'モデル一覧の読み込みに失敗しました';
  }
}

modelSelect.addEventListener('change', () => loadModelFromFolder(modelSelect.value));

async function runInference() {
  if (!model || !imgElement) {
    alert('モデルまたは画像がありません。');
    return;
  }

  let inputTensor = tf.browser.fromPixels(imgElement).toFloat();
  let resized = tf.image.resizeBilinear(inputTensor, [640, 640]);
  let expanded = resized.expandDims(0);
  let normalized = expanded.div(255);

  try {
    const outputTensor = await model.executeAsync({ 'x': normalized }); // [1, 300, 6]

    const data = await outputTensor.data();

    const origWidth = imgElement.width;
    const origHeight = imgElement.height;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imgElement, 0, 0);

    ctx.lineWidth = 2;
    ctx.strokeStyle = 'red';
    ctx.font = '16px Arial';
    ctx.fillStyle = 'red';

    const threshold = 0.5;

    let count = 0;
    let maxConfidence = 0;

    for (let i = 0; i < 300; i++) {
      const offset = i * 6;
      const x1 = data[offset];
      const y1 = data[offset + 1];
      const x2 = data[offset + 2];
      const y2 = data[offset + 3];
      const score = data[offset + 4];
      const classId = data[offset + 5];

      if (score >= threshold) {
        count++;
        maxConfidence = Math.max(maxConfidence, score);

        const xmin = x1 * origWidth;
        const ymin = y1 * origHeight;
        const boxWidth = (x2 - x1) * origWidth;
        const boxHeight = (y2 - y1) * origHeight;

        if (boxWidth > 0 && boxHeight > 0) {
          ctx.strokeRect(xmin, ymin, boxWidth, boxHeight);
          ctx.fillText(`${classId} ${(score * 100).toFixed(1)}%`, xmin + 5, ymin + 20);
        }
      }
    }

    tf.dispose([inputTensor, resized, expanded, normalized, outputTensor]);
    resultDiv.textContent = `検出数: ${count} (最高信頼度: ${(maxConfidence * 100).toFixed(1)}%)`;

  } catch (error) {
    console.error(error);
    resultDiv.textContent = `エラー: ${error.message}`;
  }
}

runBtn.addEventListener('click', runInference);
loadModelList();
