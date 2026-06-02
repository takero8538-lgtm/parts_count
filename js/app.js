const modelSelect = document.getElementById('modelSelect');
const imageInput = document.getElementById('imageInput');
const runBtn = document.getElementById('runBtn');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const resultDiv = document.getElementById('result');

// カメラ機能用の要素を取得
const startCameraBtn = document.getElementById('startCameraBtn');
const captureBtn = document.getElementById('captureBtn');
const video = document.getElementById('video');

let model = null;
let imgElement = null;
let stream = null; // カメラのストリーム保持用

// フォルダから画像が選択された時の処理
imageInput.addEventListener('change', (evt) => {
  const file = evt.target.files[0];
  if (!file) return;
  
  // カメラが起動中なら停止する
  stopCamera();

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

// カメラ起動ボタンの処理
startCameraBtn.addEventListener('click', async () => {
  // すでに起動している場合は停止して閉じる
  if (stream) {
    stopCamera();
    return;
  }

  try {
    // スマートフォンの背面カメラ（environment）を最優先で要求
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" } },
      audio: false
    });
    video.srcObject = stream;
    video.style.display = 'block'; // 映像を表示する
    startCameraBtn.textContent = '❌ カメラを閉じる';
    captureBtn.disabled = false;
    resultDiv.textContent = 'カメラが起動しました。対象を映して「写真を撮る」を押してください。';
  } catch (error) {
    console.error('カメラ起動エラー:', error);
    resultDiv.textContent = 'カメラの起動に失敗しました。アクセス権限を確認してください。';
  }
});

// 写真を撮るボタンの処理
captureBtn.addEventListener('click', () => {
  if (!stream) return;

  // カメラ映像の実際の解像度をCanvasに適用
  const videoWidth = video.videoWidth;
  const videoHeight = video.videoHeight;
  
  canvas.width = videoWidth;
  canvas.height = videoHeight;

  // Canvasに現在のビデオフレームを描画
  ctx.drawImage(video, 0, 0, videoWidth, videoHeight);

  // 撮影したデータを仮想Imageオブジェクトに変換して既存ロジックに渡す
  const img = new Image();
  img.src = canvas.toDataURL('image/jpeg');
  img.onload = () => {
    imgElement = img;
    runBtn.disabled = !(model && imgElement);
    resultDiv.textContent = '写真を撮影しました。「推論開始」を押してください。';
    
    // 写真を撮ったらカメラのストリームは自動停止させて画面をスッキリさせる
    stopCamera();
  };
});

// カメラを安全に停止させる関数
function stopCamera() {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  video.srcObject = null;
  video.style.display = 'none';
  startCameraBtn.textContent = '📸 カメラを起動';
  captureBtn.disabled = true;
}

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

  const modelWidth = 640;
  const modelHeight = 640;
  const origWidth = imgElement.width;
  const origHeight = imgElement.height;

  const scale = Math.min(modelWidth / origWidth, modelHeight / origHeight);
  const nw = Math.floor(origWidth * scale);
  const nh = Math.floor(origHeight * scale);

  let inputTensor = tf.browser.fromPixels(imgElement).toFloat();
  let resized = tf.image.resizeBilinear(inputTensor, [nh, nw]);
  
  const padTop = Math.floor((modelHeight - nh) / 2);
  const padLeft = Math.floor((modelWidth - nw) / 2);
  let padded = resized.pad([[padTop, modelHeight - nh - padTop], [padLeft, modelWidth - nw - padLeft], [0, 0]]);
  
  let expanded = padded.expandDims(0);
  let normalized = expanded.div(255.0);

  try {
    const outputTensor = await model.executeAsync(normalized);

    let rawOutput;
    if (Array.isArray(outputTensor)) {
      rawOutput = outputTensor[0];
      outputTensor.forEach(t => { if(t !== rawOutput) t.dispose(); });
    } else {
      rawOutput = outputTensor;
    }

    const squeezed = rawOutput.squeeze();
    const transposed = squeezed.transpose([1, 0]);
    const data = await transposed.data();
    const shape = transposed.shape;

    const numBoxes = shape[0];
    const numAttributes = shape[1];
    const numClasses = numAttributes - 4;

    const boxes = [];
    const scores = [];
    const classIds = [];

    const confThreshold = 0.1; // 👈 満足のいく精度を出した0.1をキープ

    for (let i = 0; i < numBoxes; i++) {
      const offset = i * numAttributes;
      
      const cx = data[offset];
      const cy = data[offset + 1];
      const w = data[offset + 2];
      const h = data[offset + 3];

      let maxScore = 0;
      let classId = -1;
      for (let c = 0; c < numClasses; c++) {
        const score = data[offset + 4 + c];
        if (score > maxScore) {
          maxScore = score;
          classId = c;
        }
      }

      if (maxScore >= confThreshold) {
        const ymin = cy - h / 2;
        const xmin = cx - w / 2;
        const ymax = cy + h / 2;
        const xmax = cx + w / 2;

        boxes.push([ymin, xmin, ymax, xmax]);
        scores.push(maxScore);
        classIds.push(classId);
      }
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imgElement, 0, 0);

    let count = 0;
    let maxConfidence = 0;

    if (boxes.length > 0) {
      const boxesTensor = tf.tensor2d(boxes);
      const scoresTensor = tf.tensor1d(scores);
      
      const nmsIndices = await tf.image.nonMaxSuppressionAsync(
        boxesTensor,
        scoresTensor,
        100,
        0.45,
        confThreshold
      );

      const indices = await nmsIndices.data();

      ctx.lineWidth = 2;
      ctx.strokeStyle = 'red';
      ctx.font = '16px Arial';
      ctx.fillStyle = 'red';

      for (let i = 0; i < indices.length; i++) {
        const idx = indices[i];
        const [ymin, xmin, ymax, xmax] = boxes[idx];
        const score = scores[idx];
        const classId = classIds[idx];

        count++;
        maxConfidence = Math.max(maxConfidence, score);

        const realXmin = (xmin - padLeft) / scale;
        const realYmin = (ymin - padTop) / scale;
        const realXmax = (xmax - padLeft) / scale;
        const realYmax = (ymax - padTop) / scale;

        const boxWidth = realXmax - realXmin;
        const boxHeight = realYmax - realYmin;

        if (boxWidth > 0 && boxHeight > 0) {
          ctx.strokeRect(realXmin, realYmin, boxWidth, boxHeight);
          ctx.fillText(`ID:${classId} ${(score * 100).toFixed(1)}%`, realXmin + 5, realYmin + 18);
        }
      }

      tf.dispose([boxesTensor, scoresTensor, nmsIndices]);
    }

    squeezed.dispose();
    transposed.dispose();
    rawOutput.dispose();
    tf.dispose([inputTensor, resized, padded, expanded, normalized]);

    resultDiv.textContent = `検出数: ${count} (最高信頼度: ${(maxConfidence * 100).toFixed(1)}%)`;

  } catch (error) {
    console.error(error);
    resultDiv.textContent = `エラー: ${error.message}`;
    tf.dispose([inputTensor, resized, padded, expanded, normalized]);
  }
}

runBtn.addEventListener('click', runInference);
loadModelList();
