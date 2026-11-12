# 評分設定（goal.json）

## 欄位

| 欄位 | 說明 |
|---|---|
| `task` | 一句話重述使用者的目標 |
| `input_dir` | 原圖資料夾，只讀不寫 |
| `output.format` | `png`（無損，預設）或 `jpg` |
| `output.mode` | `keep`、`gray` 或 `color`。最後一步會自動轉換成這個模式 |
| `checks` | proxy 評分用的檢查項目，見下方說明 |
| `primary` | `proxy` 或 `eval_cmd`，決定排行榜用哪個分數 |
| `eval_cmd` | 下游評估指令，見下方說明 |
| `eval_higher_is_better` | eval 分數越高越好時設為 `true`。如果是錯誤率之類越低越好的指標，設為 `false` |
| `min_keep_rate` | 預設 0.9。低於此比例的版本視為無效 |
| `holdout_fraction` | 預設 0.2。圖片少於 10 張時不切出 holdout |
| `max_versions` / `patience` / `min_improvement` | 停止條件。預設最多 10 版、連續 3 版沒有改善就停，改善至少要 0.5 分才算數。`min_improvement` 要符合分數的尺度：proxy 是 0–100，所以用 0.5；`primary` 是 `eval_cmd` 時必須明確填寫，例如分數在 0–1 之間就填 0.005 |

## checks

所有指標都會先把圖片縮到最長邊 512 px 再量測，所以不同尺寸的圖片可以直接比較。但比 512 小的圖不會被放大，所以 pipeline 把圖縮到 512 以下（例如 224）時，`sharpness` 和 `noise` 的量測尺度會改變，跟 baseline 的數值不能直接比較。這種情況下，這兩個 check 的門檻要依輸出尺寸重新設定，或者只看其他 check。proxy 分數的算法是：對每張保留下來的圖片計算通過 check 的比例，取全部圖片的平均，再乘以 100。

| check | 值 | 指標意義 |
|---|---|---|
| `brightness` | `[lo, hi]` | 灰階平均亮度，範圍 0–255 |
| `contrast_min` | 數值 | 灰階標準差 |
| `sharpness_min` | 數值 | Laplacian 變異數 |
| `noise_max` | 數值 | Immerkær 雜訊估計。乾淨的圖通常小於 1，明顯有雜訊時大於 3 |
| `clip_dark_max` / `clip_bright_max` | 0–1 | 過暗（≤3）或過亮（≥252）像素的比例 |
| `min_side` / `max_side` | px | 依實際輸出尺寸計算 |
| `size` | `[w, h]` | 輸出必須剛好是這個尺寸 |
| `channels` | 1 或 3 | 色彩通道數 |

## 依任務選擇起點

`inspect_dataset.py --preset photo|ocr|detect|vlm` 會把對應的起點直接寫進 `goal.draft.json`（`detect` 預設 `size [640,640]`，`vlm` 預設 `max_side 2048`，要改成實際模型的輸入尺寸）。以下只是起點，實際數值要依使用者描述調整。

- **文件/單據 OCR**：`brightness [190,250]`、`contrast_min 40`、`noise_max 1.5`、`clip_dark_max 0.05`，`output.mode gray`。白紙本來就很亮，所以不要設 `clip_bright`。
- **自然照片分類/embedding**：`brightness [80,190]`、`contrast_min 40`、`sharpness_min 100`、`noise_max 2`、兩種 clip 都 ≤ 0.05，並加上 `size` 或 `min_side`，對應模型的輸入尺寸。
- **物件偵測**：同上，但要保留長寬比，所以用 letterbox，不要直接拉伸。尺寸用 `size` 檢查。
- **VLM 輸入**：重點是方向正確、去除邊框、尺寸不超過模型上限，可用 `max_side`。不要過度銳化或過度降噪，避免產生偽影。

## eval_cmd（下游評分）

使用者有模型可以實際評估時，用它當作 `primary`，比 proxy 準確得多。

- 指令中的 `{output_dir}`、`{input_dir}`、`{list_file}` 會自動替換成實際路徑，其他大括號會原樣保留。其中 `list_file` 是每行一個輸出圖檔路徑的清單。
- 指令必須在 stdout 印出一行 JSON，且其中包含 `score`，例如 `{"score": 0.873, "n": 120}`。只會讀取最後一個含有 `score` 的 JSON 行，其他輸出會被忽略。
- 範例：`"eval_cmd": "python eval_ocr.py --images {output_dir} --labels labels.csv"`

如果使用者沒有現成的評估腳本，但有標註資料（文字、類別、bbox），可以替他寫一個簡短的評估腳本放在 `W/`：

- OCR：字元正確率，或模型的平均信心值
- 分類：accuracy
- 偵測：mAP 或 recall

寫好後要先在 baseline 上執行一次，確認數值合理。

## proxy 的盲點

proxy 只量測「看起來乾不乾淨」，不代表模型的表現。以下情況會讓分數提高，但模型表現可能變差：

- binarize 或過強的 CLAHE：contrast 大幅上升，但細節和色彩資訊流失
- 強力模糊或強力 NLM：noise 下降，但細小文字也一起被抹掉
- 過度裁切：剩下的圖片指標更好看，但遺失了內容
- 過度銳化：sharpness 上升，但會產生光暈和雜訊

只用 proxy 時，必須查看對照圖，或在回報中明確說明這些風險。
