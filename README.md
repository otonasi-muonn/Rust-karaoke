# Rust Karaoke 🎤

**AI搭載カラオケ採点デスクトップアプリ**

Rust・Tauri・ONNX Runtimeを活用した、完全ローカル動作のカラオケ採点システムです。  
GPU アクセラレーション対応で、リアルタイムピッチ解析と高精度な採点を実現します。

## 機能

- **YouTube URL解析** — URLを貼るだけで楽曲を自動ダウンロード・解析
- **AIボーカル分離** — BS-RoFormerによる高精度な音源分離
- **リアルタイムピッチ検出** — FCPE（超軽量AIモデル）による低遅延ピッチ解析
- **カラオケ採点** — Perfect/Great/Good/Missの4段階判定
- **ピッチ可視化** — Canvas APIによるリアルタイム音程グラフ
- **ボーダーレスウィンドウ** — 独自デザインのモダンUI
- **PiPモード** — 最前面オーバーレイ表示

## 使い方

### 1. アプリを起動する

```powershell
cargo tauri dev
```

### 2. 楽曲を解析する

1. アプリが開いたら、中央の入力欄に **YouTube の楽曲URL** を貼り付けます
2. **「解析開始」** ボタンをクリック（または Enter キー）
3. プログレスバーが表示され、以下の処理が自動で行われます：
   - 楽曲のダウンロード
   - AIによるボーカル分離（BS-RoFormer）
   - リファレンスピッチの抽出（FCPE）

### 3. カラオケを歌う

1. 解析が完了すると、ピッチ可視化画面に切り替わります
2. **「▶ 歌い始める」** をクリックしてカラオケ開始
3. マイクに向かって歌うと、リアルタイムで以下が表示されます：
   - **紫のバー** — 正解の音程（リファレンスピッチ）
   - **光るドット** — あなたの声の音程
   - **判定** — 完璧 / 素晴らしい / 良い / ミス
   - **スコア** — リアルタイム採点

### 4. 操作ボタン

| ボタン | 説明 |
|--------|------|
| ▶ 歌い始める | カラオケを開始（マイク入力＋伴奏再生） |
| ■ 終了する | カラオケを停止 |
| ← 曲選択に戻る | URL入力画面に戻る |
| ⊞ (タイトルバー) | PiPモード（最前面オーバーレイ表示） |

## アーキテクチャ

```
┌─────────────────────────────────────────┐
│  Frontend (HTML/CSS/JS + Canvas API)     │
├─────────────────────────────────────────┤
│  Tauri IPC Bridge                        │
├─────────────────────────────────────────┤
│  Rust Backend                            │
│  ├─ audio/     マイク入力・伴奏再生      │
│  ├─ inference/  ONNX推論 (FCPE, BS-RoFormer)│
│  ├─ pipeline/   DL・デコード・分離・ピッチ │
│  ├─ scoring.rs  採点アルゴリズム          │
│  └─ state.rs    共有状態管理              │
└─────────────────────────────────────────┘
```

## セットアップ

### 前提条件

- **Rust** (1.70+): https://rustup.rs
- **Node.js** (18+): https://nodejs.org
- **Tauri CLI**: `cargo install tauri-cli --version "^2"`
- **(任意) yt-dlp**: YouTube DLの信頼性向上用
- **(任意) ONNX Runtime + CUDA**: GPU推論用

### ONNXモデルの配置

`src-tauri/models/` ディレクトリに以下のモデルを配置:

1. **fcpe.onnx** — FCPEピッチ推定モデル (~10MB)
2. **bsroformer.onnx** — BS-RoFormerボーカル分離モデル

> モデルが無い場合でも、フォールバックのオートコレレーション法で動作します。

### 開発

```powershell
# Rust の PATH が通っていない場合（PowerShell）
$env:Path += ";$env:USERPROFILE\.cargo\bin"

# 開発モード（ホットリロード付き）
cargo tauri dev

# プロダクションビルド（exe / msi 生成）
cargo tauri build
```

ビルド成果物は `src-tauri/target/release/bundle/` に出力されます：
- **`nsis/Rust Karaoke_0.1.0_x64-setup.exe`** — インストーラー
- **`msi/Rust Karaoke_0.1.0_x64_ja-JP.msi`** — MSIパッケージ

> インストーラーを使わず直接実行したい場合は `src-tauri/target/release/rust-karaoke.exe` を利用できます。

### プロジェクト構造

```
Rust-karaoke/
├── Dock/               設計ドキュメント
├── ui/                 フロントエンド
│   ├── index.html      メインHTML
│   └── assets/
│       ├── style.css   スタイルシート
│       └── app.js      アプリロジック
├── src-tauri/          Rustバックエンド
│   ├── Cargo.toml      依存関係
│   ├── tauri.conf.json Tauri設定
│   ├── icons/          アプリアイコン
│   ├── models/         ONNXモデル配置先
│   └── src/
│       ├── main.rs     エントリポイント + IPCコマンド
│       ├── state.rs    アプリケーション状態
│       ├── scoring.rs  採点アルゴリズム
│       ├── audio/
│       │   ├── microphone.rs  マイク入力（cpal + ringbuf）
│       │   └── playback.rs    伴奏再生（rodio）
│       ├── inference/
│       │   ├── fcpe.rs        ピッチ抽出AI
│       │   └── bsroformer.rs  音源分離AI
│       └── pipeline/
│           ├── download.rs    YouTube DL
│           ├── decode.rs      音声デコード（symphonia）
│           ├── separation.rs  ボーカル分離
│           └── pitch.rs       リファレンスピッチ抽出
└── package.json
```

## 採点アルゴリズム

周波数 f(Hz) → セント値への変換:

```
cents = 1200 × log₂(f / 440) + 6900
```

| 判定     | 閾値(セント) | 意味         |
|----------|-------------|-------------|
| Perfect  | ≤ 50        | ほぼ正確     |
| Great    | ≤ 100       | 半音以内     |
| Good     | ≤ 200       | 全音以内     |
| Miss     | > 200       | 外れ         |

**オクターブ許容**: 1200セント単位のズレは剰余計算で許容  
**時間軸ズレ許容**: ±50ms以内で最適な参照ピッチを探索

## ライセンス

MIT License
