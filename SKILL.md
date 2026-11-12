---
name: cv-preprocess-explore
description: 依使用者在對話中描述的需求，探索並比較多種圖片前處理方案（光線、雜訊、幾何校正、裁切、尺寸、色彩），用量化分數選出最佳流程，產出清理後的圖片和可重用的 pipeline，供後續模型輸入使用。用於「這批照片要清理或前處理給模型用，但不確定怎麼做最好」這類需要嘗試比較的任務；已有固定流程只需套用時不必使用。
---

# CV 前處理探索

這個 skill 固定的是評分方式、紀錄方式、停止條件和交付格式。用哪些方法、怎麼組合、參數設多少，由你根據資料和分數決定。

## 工具

`<SKILL>` 代表這份 SKILL.md 所在資料夾的絕對路徑。你的工作目錄通常不是 `<SKILL>`，所以指令一律寫完整路徑，例如 `python3 /path/to/cv-preprocess-explore/scripts/ops.py --list`。沒有 `python3` 時改用 `python`。

需要 Python 環境裝有 `opencv-python` 和 `numpy`。如果 import 失敗，告訴使用者缺少哪個套件。sandbox 可能沒有網路，所以不要自行安裝。

| 指令 | 用途 |
|---|---|
| `python3 <SKILL>/scripts/inspect_dataset.py <input_dir> --workdir W --preset photo` | 分析資料集，產生 `inspect.json`、`inspect_sheet.jpg` 和 `goal.draft.json`。`--preset` 可選 `photo`、`ocr`、`detect`、`vlm`，決定 draft 的起始 checks |
| `python3 <SKILL>/scripts/ops.py --list` | 列出所有內建 op 和參數 |
| `python3 <SKILL>/scripts/experiment.py --workdir W --baseline` | 建立 v0 基準：原圖不處理 |
| `python3 <SKILL>/scripts/experiment.py --workdir W --config W/configs/vN.json --note "..."` | 跑一個版本，自動評分、產生前後對照圖並寫入紀錄 |
| `python3 <SKILL>/scripts/experiment.py --workdir W --status` | 查看排行榜，以及是否已達停止條件 |
| `python3 <SKILL>/scripts/export_pipeline.py --workdir W [--version vN]` | 用 holdout 驗證，然後輸出交付物。只能執行一次 |

`W` 是工作目錄，預設用 `./cv_explore/<資料夾名稱>/`。所有輸出都只寫在 `W` 裡。

## 流程

1. **整理目標。** 從對話中整理出以下資訊：
   - 下游模型或用途，例如 OCR、偵測、分類、VLM、embedding
   - 照片的問題
   - 輸出規格，包括尺寸、色彩模式、格式
   - 不能被破壞的內容，例如顏色、長寬比、細小文字
   - 是否有可以實際評分的下游模型

   只有在輸出規格或評分方式完全無法推斷時才詢問使用者，其他部分用合理預設並寫明。如果沒有使用者可以回答（例如自動執行的 sandbox），不要等待：採用合理預設，把假設寫進 `goal.json` 的 `task` 欄位，然後繼續。

   接著用最接近任務的 `--preset` 跑 inspect，再依使用者需求修改 `goal.draft.json` 的數值，存成 `W/goal.json`。不要原封不動地複製 draft。欄位、各種 check 的意義和各任務的預設值請參考 [references/evaluation.md](references/evaluation.md)。

2. **看資料。** 閱讀 inspect 的終端摘要。如果你能看圖，也看 `inspect_sheet.jpg`。用幾行向使用者回報：主要問題及數量、無法讀取的檔案、疑似重複的組數。重複和損壞的檔案只回報，不要刪除。

3. **建立基準。** 執行 `--baseline`。

4. **探索。** 方向和參數起點請讀 [references/search-space.md](references/search-space.md)。
   - 每個版本寫成一個 `W/configs/vN.json`，格式是 `{"steps": [...], "filter": {...}}`，然後用 `experiment.py` 執行。`--note` 要寫出相對於哪個版本改了什麼，以及原因，例如 `"v3 + denoise h 7→12：noise_max 通過率只有 0.4"`。
   - 先廣後深。先針對每個主要問題各試一個代表方法，再深入改善有效的方向，例如調整參數、順序，或改成條件式步驟。每個版本只改一個主要變因。
   - 依照輸出中最差的 check 通過率決定下一步。如果你能看圖，也參考 `runs/vN/sheet.jpg`。
   - 輸出出現 `WARNING: proxy jumped` 時，先比對 `median after` 和前一個最佳版本的數值，說明為什麼這是真的改善。說明不了就把這個版本當成無效方向。
   - 內建 op 無法處理時，可以在 `W/custom_ops/*.py` 寫自訂函式，介面為 `fn(img, **params) -> img`，並在 config 中用 `{"op": "custom", "module": "custom_ops/x.py", "fn": "...", ...}` 呼叫。自訂函式必須是 deterministic。

5. **停止。** `experiment.py` 印出 `STOP` 時，或已經達到使用者的目標時，就停止探索。版本數用完後 `experiment.py` 會拒絕再執行；只有使用者明確同意追加時才加 `--force`。沒有使用者可問時，直接進入匯出。

6. **匯出。** 確定要交付的版本後，執行一次 `export_pipeline.py`。holdout 只評一次，所以不要匯出後再回頭修改版本。腳本會拒絕第二次匯出；只有使用者明確同意時才加 `--force`。輸出 `WARNING: holdout gain ... overfit` 時，要在回報中說明。

7. **回報。** 格式見文末。

## 規則

- 不修改、刪除或移動原始圖片。
- 探索期間不要使用 holdout，也不要為了設計流程去看 holdout 的圖片。腳本已經會把 holdout 排除在外。
- 不寫針對個別圖片的特例，例如依檔名或索引處理。需要依圖片狀況處理時，用 step 的 `"if": {"metric": "brightness", "lt": 80}`。
- 紀錄不可修改。不要手動編輯 `runs.jsonl` 或 `log.md`，也不要重用版本名稱。
- proxy 分數可能被鑽漏洞，例如 binarize 會讓 contrast 衝高、強力模糊會讓 noise 變低、裁太多會遺失內容。看到分數大幅跳升（腳本會印出 WARNING）時，先確認對照圖或說明為什麼這是真的改善。使用者有下游模型時，優先用 `eval_cmd` 評分。
- `filter` 只用來剔除真的無法使用的圖片，例如損壞或極小的圖。不能為了提高分數而剔除；`keep_rate` 低於門檻的版本會被判定為無效。
- 兩個版本的分數差距小於 `min_improvement` 時，選擇較快、較簡單的版本，並一併回報 `ms/img`。

## 交付物

`W/deliverable/` 包含：

- `cleaned/`：全部清理後的圖片
- `preprocess.py`：可在 CLI 用 `--input/--output` 執行，也可以 `from preprocess import preprocess` 直接接到模型輸入。`preprocess()` 輸出 uint8 的 BGR 或灰階圖；模型特有的轉換（BGR→RGB、除以 255、mean/std 正規化、轉成 NCHW tensor）不在 pipeline 裡，由呼叫端負責
- `ops.py`、`pipeline.json`
- `report.json`
- `custom_ops/`：有自訂 op 時才會出現

## 最終回報格式

用簡短的段落和列表回報：

1. 目標，以及評分方式（proxy 或 eval_cmd）
2. 資料狀況，一兩行即可
3. 最佳版本與完整 pipeline
4. 分數：baseline → 最佳，分別列出 explore 和 holdout，並附 `ms/img`
5. 嘗試過但無效的方向，各一行說明原因
6. 風險或注意事項，例如過擬合跡象、被剔除的圖片、proxy 指標無法反映的部分
7. 交付物位置、如何在應用程式中呼叫 `preprocess()`，以及送進模型前還需要的轉換（例如 BGR→RGB、正規化）
