# 搜尋空間：問題對應候選 op

完整參數清單請用 `python scripts/ops.py --list` 查看。以下是起點建議，不是規定。

## 建議順序

幾何 → 光線/色彩 → 雜訊 → 對比 → 銳化 → 二值化 → 尺寸/格式

順序不對常常會抵消效果：

- 先銳化再降噪，會把雜訊一起放大。
- CLAHE 放在降噪之前，會讓雜訊變明顯。
- 先 resize 再 deskew，會損失解析度。

## 問題 → 候選方法

| 問題（inspect 的 flag 或 check） | 先試 | 再試 / 調整方向 |
|---|---|---|
| 整體偏暗或偏亮（`dark`/`bright`、`brightness`） | `auto_gamma target=128`（文件可用 200） | `auto_levels low_pct=1 high_pct=99`、`gamma g=0.6~1.5` |
| 光線不均、陰影（文件、平面物） | `remove_shadow k=31` | k 試 21~61；照片場景改用 `clahe clip=2 tile=8` |
| 對比低（`low_contrast`） | `clahe clip=2 tile=8` | `clip` 1~4、`tile` 4~16；`auto_levels` |
| 色偏 | `gray_world` | 照片中有白卡或色卡時，寫 custom op 依參考區域校正 |
| 雜訊（`noisy`、`noise_max`） | `denoise_nlm h=7` | `h` 5~15；椒鹽雜訊用 `median k=3`；需要保留邊緣用 `bilateral` |
| 模糊（`blurry`、`sharpness_min`） | `unsharp amount=0.8 sigma=1` | `amount` 0.5~1.5。嚴重失焦救不回來，可考慮用 `filter` 剔除，但必須遵守 keep_rate |
| 歪斜（文件、標籤） | `deskew max_angle=10` | 範圍更大時用 `max_angle=20 step=0.5` |
| 透視變形（拍攝的紙張、螢幕、卡片） | `doc_warp min_area=0.2` | 找不到四邊形時會保留原圖；`min_area` 0.1~0.4 |
| 黑邊或白邊、掃描框 | `trim_border thresh=12` | `thresh` 8~30；`center_crop frac=0.95` |
| 尺寸不一、太小（`small`） | 依模型輸入：`letterbox width=W height=H` 或 `resize max_side=N` | 偵測任務用 letterbox 保持長寬比；分類任務可試 `resize width height` |
| 需要黑白輸出 | `binarize method=adaptive block=31 C=10` | `otsu`；之後用 `morph close k=2` 補斷筆 |
| 只有部分圖片需要處理 | 在 step 上加 `"if": {"metric": "noise", "gt": 3}` | 可用的 metric 見 `ops.py --list` |

## 何時寫 custom op

以下情況內建 op 無法處理，可以寫自訂 op：

- 反光或眩光去除
- 依特定目標（產品、標籤、人臉）裁出 ROI
- 摩爾紋
- 依參考色卡校色
- 公司特有的版面切割

自訂 op 必須遵守：

- 介面為 `fn(img, **params) -> img`，輸入輸出都是 uint8 BGR 或灰階
- deterministic，不能依檔名分支
- 只使用 `cv2`、`numpy`，或使用者確認已安裝的套件
- 寫好後先用一個版本單獨測試它，不要和其他改動混在同一個版本

## 不屬於前處理的事

- **隨機資料增強**（翻轉、隨機裁切、色彩抖動）屬於訓練階段，不放進 pipeline。
- **調整標註**：如果幾何變換（裁切、旋轉、warp）會影響使用者既有的標註（bbox、keypoint），要在回報中提醒。如有必要，改用不改變幾何的方案。
