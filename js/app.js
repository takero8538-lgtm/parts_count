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

  const modelWidth = 640;
  const modelHeight = 640;
  const origWidth = imgElement.width;
  const origHeight = imgElement.height;

  // 1. 【精度向上】アスペクト比を維持したレターボックス処理（余白埋め）
  const scale = Math.min(modelWidth / origWidth, modelHeight / origHeight);
  const nw = Math.floor(origWidth * scale);
  const nh = Math.floor(origHeight * scale);

  let inputTensor = tf.browser.fromPixels(imgElement).toFloat();
  // 縦横比を保ってリサイズ
  let resized = tf.image.resizeBilinear(inputTensor, [nh, nw]);
  
  // 640x640の黒画像を作り、中央（または左上）にリサイズ画像を埋め込む
  const padTop = Math.floor((modelHeight - nh) / 2);
  const padLeft = Math.floor((modelWidth - nw) / 2);
  let padded = resized.pad([[padTop, modelHeight - nh - padTop], [padLeft, modelWidth - nw - padLeft], [0, 0]]);
  
  let expanded = padded.expandDims(0);
  let normalized = expanded.div(255.0); // 0.0〜1.0に正規化

  try {
    // 2. 推論の実行（モデルのインプット名に合わせて実行）
    const outputTensor = await model.executeAsync(normalized);

    // テンソルの形状を整理
    let rawOutput;
    if (Array.isArray(outputTensor)) {
      rawOutput = outputTensor[0];
      outputTensor.forEach(t => { if(t !== rawOutput) t.dispose(); });
    } else {
      rawOutput = outputTensor;
    }

    // YOLO v8/v11 の nms=False 時の出力形状は [1, 4 + クラス数, 8400] 
    // これを扱いやすいようにスクイーズ（[4 + クラス数, 8400]）してトランスポーズ（[8400, 4 + クラス数]）する
    const squeezed = rawOutput.squeeze();
    const transposed = squeezed.transpose([1, 0]); // 形状: [8400, 4 + クラス数]
    const data = await transposed.data();
    const shape = transposed.shape; // [8400, 4 + num_classes]

    const numBoxes = shape[0];
    const numAttributes = shape[1];
    const numClasses = numAttributes - 4;

    const boxes = [];
    const scores = [];
    const classIds = [];

    const confThreshold = 0.25; // 信頼度のしきい値（低すぎるとノイズが増えます）

    // 3. 全てのバウンディングボックスの解析
    for (let i = 0; i < numBoxes; i++) {
      const offset = i * numAttributes;
      
      // YOLOの出力座標は [cx, cy, w, h] (中心座標と縦横幅)
      const cx = data[offset];
      const cy = data[offset + 1];
      const w = data[offset + 2];
      const h = data[offset + 3];

      // 各クラスのスコアのうち、最大のものを探す
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
        // [cx, cy, w, h] を [ymin, xmin, ymax, xmax] に変換（TF.jsのNMS関数用）
        const ymin = cy - h / 2;
        const xmin = cx - w / 2;
        const ymax = cy + h / 2;
        const xmax = cx + w / 2;

        boxes.push([ymin, xmin, ymax, xmax]);
        scores.push(maxScore);
        classIds.push(classId);
      }
    }

    // 4. 【精度向上の鍵】JavaScript側で非最大値抑制（NMS）をかける
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imgElement, 0, 0);

    let count = 0;
    let maxConfidence = 0;

    if (boxes.length > 0) {
      const boxesTensor = tf.tensor2d(boxes);
      const scoresTensor = tf.tensor1d(scores);
      
      // 重複したボックスを綺麗に削除するTF.js標準関数
      const nmsIndices = await tf.image.nonMaxSuppressionAsync(
        boxesTensor,
        scoresTensor,
        100,          // 最大検出数
        0.45,         // IOUしきい値（重なり具合の許容度）
        confThreshold // 信頼度しきい値
      );

      const indices = await nmsIndices.data();

      ctx.lineWidth = 2;
      ctx.strokeStyle = 'red';
      ctx.font = '16px Arial';
      ctx.fillStyle = 'red';

      // NMSを生き残った正しいボックスだけを描画
      for (let i = 0; i < indices.length; i++) {
        const idx = indices[i];
        const [ymin, xmin, ymax, xmax] = boxes[idx];
        const score = scores[idx];
        const classId = classIds[idx];

        count++;
        maxConfidence = Math.max(maxConfidence, score);

        // レターボックス（余白）を考慮して、元の画像座標に逆変換する
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

      // メモリ解放
      tf.dispose([boxesTensor, scoresTensor, nmsIndices]);
    }

    // 残りのメモリ解放
    squeezed.dispose();
    transposed.dispose();
    rawOutput.dispose();
    tf.dispose([inputTensor, resized, padded, expanded, normalized]);

    resultDiv.textContent = `検出数: ${count} (最高信頼度: ${(maxConfidence * 100).toFixed(1)}%)`;

  } catch (error) {
    console.error(error);
    resultDiv.textContent = `エラー: ${error.message}`;
    // エラー時も念のためメモリ解放
    tf.dispose([inputTensor, resized, expanded, normalized]);
  }
}


runBtn.addEventListener('click', runInference);
loadModelList();
