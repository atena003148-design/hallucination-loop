import os
import sys
import time
import signal
import logging
import requests
import base64
import json
import random
import subprocess

import shutil
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv

# Rich imports for progress display
from rich.console import Console
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeElapsedColumn,
    TimeRemainingColumn, SpinnerColumn, DownloadColumn,
    TransferSpeedColumn, TaskProgressColumn,
)
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich import box

# ============================================================
# HALLUCINATION LOOP — Concrete Negation Engine v2
# ============================================================

# ── Console ──────────────────────────────────────────────────
console = Console(force_terminal=True)

# ── Logging ──────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"

# .envを先に読み込む（OUTPUT_DIRなどの設定を反映するため）
load_dotenv(Path(__file__).parent / ".env")

# OUTPUT_DIR: .env の OUTPUT_DIR で変更可能（デフォルト: outputs）
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs").strip())
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUT_DIR / "hallucination_loop.log"

# Fix Windows console encoding
if sys.platform == "win32":
    import io
    # Don't wrap if already wrapped (re-run safety)
    if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != "utf-8":
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
        except Exception:
            pass

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("hallucination")

# ── Configuration ────────────────────────────────────────────
BLENDER_PATH = os.getenv("BLENDER_PATH", "blender").strip()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
COLAB_SERVER_URL = os.getenv("COLAB_SERVER_URL", "").strip().rstrip("/")
# Gemini 3 Flash for image analysis (replaces Llama 3.2 Vision)
REPLICATE_VISION_MODEL = os.getenv("REPLICATE_VISION_MODEL", "google/gemini-3-flash").strip()
# Nano Banana 2 (Gemini 3.1 Flash Image) for image generation (replaces FLUX)
REPLICATE_IMAGE_MODEL = os.getenv("REPLICATE_IMAGE_MODEL", "google/nano-banana-2").strip()
# SDS 3D generation models
REPLICATE_SDS_MODELS_STR = os.getenv(
    "REPLICATE_SDS_MODELS",
    "firtoz/trellis:e8f6c45206993f297372f5436b90350817bd9b4a0d52d2a76df50c1c8afa2b3c,"
    "aryamansital/instant_mesh:e353a25cc764e0edb0aa9033df0bf4b82318dcda6d0a0cd9f2aace90566068ac"
).strip()
REPLICATE_SDS_MODELS = [m.strip() for m in REPLICATE_SDS_MODELS_STR.split(",") if m.strip()]

# モデルごとの入力パラメータ設定（imageのみ vs image+promptが必須）
# prompt_required=True のモデルには nouns から自動生成したプロンプトを渡す
SDS_MODEL_CONFIGS = {
    # Feed-forward系 (数十秒〜数分で完了)
    # ★ Blenderで独自のクレイマテリアルを適用するため、テクスチャ/ビデオ生成はすべて無効化
    "firtoz/trellis": {
        "prompt_required": False,
        "image_key": "images",      # 配列形式で渡す
        "extra_params": {
            "generate_model": True,  # GLB生成を有効化
            "generate_color": False, # カラー動画不要
            "generate_normal": False, # ノーマル動画不要
            "texture_size": 512,     # 最小テクスチャで高速化
        },
        "output_key": "model_file",
    },
    "fishwowater/trellis2": {
        "prompt_required": False,
        "image_key": "image",
        "extra_params": {
            "generate_model": True,
            "generate_video": False,
            "pipeline_type": "512",
            "texture_size": 1024,
            "decimation_target": 100000,  # ★ API最小値（ポスト処理のデシメーション、エクスポート高速化）
            "tex_slat_steps": 1,          # ★ AIのテクスチャ生成計算自体を最小化（スキップ）
        },
        "output_key": "model_file",
    },
    "aryamansital/instant_mesh": {
        "prompt_required": False,
        "image_key": "image_path",
        "extra_params": {
            "remove_background": True,    # ★ キー名修正: do_remove_background → remove_background
            "export_video": False,        # ★ 動画エクスポート無効化（デフォルトTrue）
            "export_texmap": False,       # ★ テクスチャマップ無効化
            "sample_steps": 50,           # ★ デフォルト75→50に削減
        },
    },

}

# Replicate の環境変数を設定（replicate パッケージが自動的に参照する）
if REPLICATE_API_TOKEN:
    os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

NEGATION_FILE = Path("negation_history.txt")
HISTORY_FILE = OUTPUT_DIR / "history.json"

MAX_RETRIES = 3
RETRY_BASE_DELAY = 10
MAX_ITERATIONS = 999

# ── Global Prediction Throttle ───────────────────────────────
class PredictionThrottle:
    """Replicate APIのレートリミット制御 + コスト追跡。
    
    低クレジット時は 6 req/min (burst=1) に制限されるため、
    全てのprediction作成をこのクラス経由で行い、最低間隔を保証する。
    429エラー発生時は自動的に待機して再試行する。
    各サイクルのprediction IDを記録し、コスト集計を可能にする。
    """
    def __init__(self, min_interval=11.0, max_rate_limit_retries=3):
        self._last_call_time = 0.0
        self._min_interval = min_interval  # 6req/min = 10s間隔 + 1s安全マージン
        self._max_rate_limit_retries = max_rate_limit_retries
        self._total_calls = 0
        self._total_throttle_waits = 0
        self._total_cost = 0.0
        # サイクルごとのコスト追跡
        self._cycle_start_time = None
        self._cycle_prediction_ids = []

    def start_cycle(self):
        """新しいサイクルの開始を記録。"""
        self._cycle_start_time = datetime.now(timezone.utc)
        self._cycle_prediction_ids = []

    def _wait_if_needed(self):
        """前回のAPIコールから最低間隔が経過するまで待機。"""
        now = time.time()
        elapsed = now - self._last_call_time
        if elapsed < self._min_interval:
            wait_time = self._min_interval - elapsed
            self._total_throttle_waits += 1
            console.print(f"  [dim]⏳ レートリミット制御: {wait_time:.1f}秒待機中...[/]")
            log.info(f"[Throttle] {wait_time:.1f}秒待機 (予防的スロットル)")
            time.sleep(wait_time)

    def _handle_rate_limit(self, err_msg):
        """429エラーからリセット時間を抽出して待機。Returns True if rate limit was detected."""
        import re
        if "429" in err_msg or "throttled" in err_msg.lower() or "rate limit" in err_msg.lower():
            m = re.search(r'resets in ~(\d+)s', err_msg)
            wait_sec = int(m.group(1)) + 3 if m else 15
            console.print(f"  [cyan]⏳ レートリミット検出: {wait_sec}秒待機...[/]")
            log.warning(f"[Throttle] レートリミット検出 → {wait_sec}秒待機")
            time.sleep(wait_sec)
            return True
        return False

    def run(self, model, **kwargs):
        """replicate.run() のスロットル付きラッパー。"""
        import replicate
        for attempt in range(1, self._max_rate_limit_retries + 1):
            if _shutdown_requested:
                return None
            self._wait_if_needed()
            try:
                self._last_call_time = time.time()
                self._total_calls += 1
                result = replicate.run(model, **kwargs)
                return result
            except Exception as e:
                err_msg = str(e)
                if self._handle_rate_limit(err_msg):
                    if attempt < self._max_rate_limit_retries:
                        console.print(f"  [dim]レートリミット再試行 ({attempt}/{self._max_rate_limit_retries})[/]")
                        continue
                raise  # レートリミット以外のエラー or 最終試行 → 呼び出し元に伝搬

    def create_prediction(self, **kwargs):
        """replicate.predictions.create() のスロットル付きラッパー。prediction IDを追跡。"""
        import replicate
        for attempt in range(1, self._max_rate_limit_retries + 1):
            if _shutdown_requested:
                return None
            self._wait_if_needed()
            try:
                self._last_call_time = time.time()
                self._total_calls += 1
                result = replicate.predictions.create(**kwargs)
                # prediction IDを記録（コスト追跡用）
                if result and hasattr(result, 'id'):
                    self._cycle_prediction_ids.append(result.id)
                return result
            except Exception as e:
                err_msg = str(e)
                if self._handle_rate_limit(err_msg):
                    if attempt < self._max_rate_limit_retries:
                        console.print(f"  [dim]レートリミット再試行 ({attempt}/{self._max_rate_limit_retries})[/]")
                        continue
                raise

    # モデル別コスト推定レート（Replicateモデルページの公開価格に基づく）
    # type: "token" = トークン課金, "per_image" = 画像出力課金,
    #       "per_second" = 計算時間課金, "per_run" = 固定料金
    MODEL_COST_RATES = {
        "google/gemini-3-flash": {
            "type": "token",
            "input_per_mtok": 0.10,   # $0.10 per million input tokens
            "output_per_mtok": 0.40,  # $0.40 per million output tokens
        },
        "google/nano-banana-2": {
            "type": "per_image",
            "cost_per_image": 0.003,  # Flash-level pricing (~$0.003/image)
        },
        "firtoz/trellis": {
            "type": "per_run",
            "cost_per_run": 0.036,    # モデルページ: "~$0.036 to run"
        },
        "aryamansital/instant_mesh": {
            "type": "per_run",
            "cost_per_run": 0.18,     # モデルページ: "~$0.18 to run"
        },

    }

    def _estimate_prediction_cost(self, model_name, metrics):
        """metricsデータからモデル別の推定コストを計算。"""
        rate = self.MODEL_COST_RATES.get(model_name)
        if not rate:
            # 未知のモデルはpredict_time * 汎用GPU単価で概算
            predict_time = metrics.get("predict_time", 0.0) or 0.0
            return predict_time * 0.001375 if predict_time > 0 else None  # A40相当

        billing_type = rate["type"]

        if billing_type == "token":
            input_tokens = metrics.get("token_input_count", 0) or 0
            output_tokens = metrics.get("token_output_count", 0) or 0
            cost = (input_tokens * rate["input_per_mtok"] / 1_000_000
                    + output_tokens * rate["output_per_mtok"] / 1_000_000)
            return cost

        elif billing_type == "per_image":
            image_count = metrics.get("image_output_count", 1) or 1
            return rate["cost_per_image"] * image_count

        elif billing_type == "per_run":
            return rate["cost_per_run"]

        elif billing_type == "per_second":
            predict_time = metrics.get("predict_time", 0.0) or 0.0
            return predict_time * rate["cost_per_sec"]

        return None

    def get_cycle_costs(self):
        """サイクル中に実行されたpredictionのコスト情報を取得・推定。
        
        Replicate APIからサイクル開始以降のprediction一覧を取得し、
        各predictionのmetricsから推定コストを計算する。
        
        Returns: dict with predictions info, or None on failure.
        """
        if self._cycle_start_time is None:
            return None

        try:
            resp = requests.get(
                "https://api.replicate.com/v1/predictions",
                headers={
                    "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
                    "Content-Type": "application/json",
                },
                params={"limit": 20},  # 直近20件で十分
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            predictions_list = data.get("results", [])
        except Exception as e:
            log.warning(f"[CostTracker] prediction一覧取得失敗: {e}")
            return None

        cycle_predictions = []
        total_predict_time = 0.0
        total_cost = 0.0

        for pred in predictions_list:
            # created_at を比較して、サイクル開始前のpredictionは無視
            created_str = pred.get("created_at", "")
            if not created_str:
                continue
            try:
                created_at = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                continue

            if created_at < self._cycle_start_time:
                break  # これ以降は古いpredictionなのでスキップ

            metrics = pred.get("metrics", {}) or {}
            predict_time = metrics.get("predict_time", 0.0) or 0.0
            total_predict_time += predict_time

            # モデル名の抽出
            model_name = pred.get("model", "unknown")
            status = pred.get("status", "unknown")

            # metricsベースのコスト推定
            cost = self._estimate_prediction_cost(model_name, metrics) if status == "succeeded" else 0.0

            if cost is not None:
                total_cost += cost

            cycle_predictions.append({
                "id": pred.get("id", "?"),
                "model": model_name,
                "status": status,
                "predict_time": predict_time,
                "cost": cost,
            })

        # 累積コスト更新
        self._total_cost += total_cost

        return {
            "predictions": cycle_predictions,
            "count": len(cycle_predictions),
            "total_predict_time": total_predict_time,
            "total_cost": total_cost,
            "cumulative_cost": self._total_cost,
        }

    def get_stats(self):
        cost_str = f", 累積コスト: ~${self._total_cost:.4f}" if self._total_cost > 0 else ""
        return f"APIコール数: {self._total_calls}, スロットル待機: {self._total_throttle_waits}回{cost_str}"


# グローバルスロットルインスタンス
throttle = PredictionThrottle(min_interval=11.0)

INITIAL_NEGATIONS = [
    "抽象", "球", "立方体", "円柱", "幾何学形態",
    "塊", "滑らか", "ミニマリズム", "プリミティブ", "シンプル",
]
SEED_NOUNS = ["fish", "tree", "coral"]

# 分析出力のランダム言語プール
ANALYSIS_LANGUAGES = [
    {"code": "en", "name": "English", "instruction": "in English"},
    {"code": "hi", "name": "हिन्दी", "instruction": "in Hindi using Devanagari script"},
    {"code": "ar", "name": "العربية", "instruction": "in Arabic using Arabic script"},
    {"code": "ja", "name": "日本語", "instruction": "in Japanese using Kanji or Hiragana"},
    {"code": "zh", "name": "中文", "instruction": "in Simplified Chinese using Chinese characters"},
    {"code": "ko", "name": "한국어", "instruction": "in Korean using Hangul"},
    {"code": "th", "name": "ไทย", "instruction": "in Thai using Thai script"},
    {"code": "ru", "name": "Русский", "instruction": "in Russian using Cyrillic script"},
    {"code": "ta", "name": "தமிழ்", "instruction": "in Tamil using Tamil script"},
    {"code": "el", "name": "Ελληνικά", "instruction": "in Greek using Greek script"},
    {"code": "he", "name": "עברית", "instruction": "in Hebrew using Hebrew script"},
    {"code": "ka", "name": "ქართული", "instruction": "in Georgian using Georgian script"},
]

# ── Graceful Shutdown ────────────────────────────────────────
_shutdown_requested = False

def _signal_handler(sig, frame):
    global _shutdown_requested
    if _shutdown_requested:
        console.print("\n[bold red]強制終了します。[/]")
        sys.exit(1)
    _shutdown_requested = True
    console.print("\n[bold yellow]⚠ シャットダウン要求を受信しました。現在のサイクル完了後に終了します。[/]")
    console.print("[dim](もう一度 Ctrl+C で即座に終了)[/]")

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ── Startup Validation ───────────────────────────────────────
def validate_environment():
    """Fail fast if critical dependencies are missing."""
    errors = []

    # バックエンド判定
    using_colab   = bool(COLAB_SERVER_URL)
    using_gemini  = bool(GEMINI_API_KEY)
    using_replicate = bool(REPLICATE_API_TOKEN)

    if not using_colab and not using_replicate:
        errors.append(
            "画像生成・3D生成のバックエンドが設定されていません。\n"
            "  Colabを使う場合: .env に COLAB_SERVER_URL を設定してください。\n"
            "  Replicateを使う場合: .env に REPLICATE_API_TOKEN を設定してください。"
        )
    if not using_gemini and not using_replicate:
        errors.append(
            "画像分析のバックエンドが設定されていません。\n"
            "  Gemini直接を使う場合: .env に GEMINI_API_KEY を設定してください（無料: aistudio.google.com）\n"
            "  Replicateを使う場合: .env に REPLICATE_API_TOKEN を設定してください。"
        )

    blender = shutil.which(BLENDER_PATH) or (Path(BLENDER_PATH).exists() if BLENDER_PATH != "blender" else False)
    if not blender:
        errors.append(f"Blender が見つかりません: '{BLENDER_PATH}' — .env の BLENDER_PATH を確認してください。")

    if not Path("blender_render.py").exists():
        errors.append("blender_render.py が見つかりません。main.py と同じフォルダに配置してください。")

    if errors:
        for e in errors:
            console.print(f"[bold red]✗[/] {e}")
            log.error(e)
        sys.exit(1)

    if using_colab:
        console.print(f"[green]✓[/] 画像/3D生成: [cyan]Colab サーバー ({COLAB_SERVER_URL})[/]")
        log.info(f"Colab server: {COLAB_SERVER_URL}")
    else:
        console.print(f"[green]✓[/] 画像/3D生成: [cyan]Replicate (Nano Banana 2, {len(REPLICATE_SDS_MODELS)} SDS models)[/]")
        log.info(f"Replicate image={REPLICATE_IMAGE_MODEL}, SDS={REPLICATE_SDS_MODELS}")

    if using_gemini:
        console.print(f"[green]✓[/] 画像分析:    [cyan]Google Gemini API 直接 (無料枠)[/]")
        log.info("Vision: Gemini API direct")
    else:
        console.print(f"[green]✓[/] 画像分析:    [cyan]Replicate ({REPLICATE_VISION_MODEL})[/]")
        log.info(f"Vision: Replicate {REPLICATE_VISION_MODEL}")

    console.print(f"[green]✓[/] Blender:      [cyan]{BLENDER_PATH}[/]")
    console.print(f"[green]✓[/] Output:       [cyan]{OUTPUT_DIR.resolve()}[/]")
    log.info(f"Blender: {BLENDER_PATH}, Output: {OUTPUT_DIR.resolve()}")


# ── Negation File I/O ────────────────────────────────────────
def load_negations():
    if not NEGATION_FILE.exists():
        save_negations(INITIAL_NEGATIONS)
        return list(INITIAL_NEGATIONS)
    try:
        content = NEGATION_FILE.read_text(encoding="utf-8").strip()
        if not content:
            save_negations(INITIAL_NEGATIONS)
            return list(INITIAL_NEGATIONS)
        return [n.strip() for n in content.split(",") if n.strip()]
    except Exception as e:
        log.warning(f"否定ファイル読み込み失敗: {e} — 初期値で再作成します。")
        save_negations(INITIAL_NEGATIONS)
        return list(INITIAL_NEGATIONS)


def save_negations(negations):
    tmp = NEGATION_FILE.with_suffix(".tmp")
    try:
        tmp.write_text(", ".join(negations), encoding="utf-8")
        tmp.replace(NEGATION_FILE)
    except Exception as e:
        log.error(f"否定ファイル保存失敗: {e}")


# ── History (JSON) for Resume ────────────────────────────────
def load_history():
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            log.warning("history.json の読み込みに失敗しました。新規で開始します。")
    return []


def save_history(history):
    tmp = HISTORY_FILE.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(HISTORY_FILE)
    except Exception as e:
        log.error(f"履歴保存失敗: {e}")


# ── Retry helper ─────────────────────────────────────────────
def retry(fn, step_name, max_retries=MAX_RETRIES):
    """Execute fn(), retrying with exponential backoff on failure."""
    for attempt in range(1, max_retries + 1):
        if _shutdown_requested:
            return None
        try:
            result = fn()
            if result:
                return result
        except requests.exceptions.Timeout:
            log.warning(f"[{step_name}] タイムアウト (試行 {attempt}/{max_retries})")
            console.print(f"  [yellow]⚠ タイムアウト[/] (試行 {attempt}/{max_retries})")
        except requests.exceptions.ConnectionError:
            log.warning(f"[{step_name}] 接続エラー (試行 {attempt}/{max_retries})")
            console.print(f"  [yellow]⚠ 接続エラー[/] (試行 {attempt}/{max_retries})")
        except Exception as e:
            log.warning(f"[{step_name}] エラー: {e} (試行 {attempt}/{max_retries})")
            console.print(f"  [yellow]⚠ エラー: {e}[/] (試行 {attempt}/{max_retries})")

        if attempt < max_retries:
            delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), 120)
            console.print(f"  [dim]{delay}秒後にリトライ...[/]")
            time.sleep(delay)

    log.error(f"[{step_name}] {max_retries}回失敗しました。")
    console.print(f"  [bold red]✗ {step_name}: {max_retries}回失敗 — スキップします[/]")
    return None


# ── File Validation ──────────────────────────────────────────
def is_valid_file(path, min_bytes=500):
    """Check if a file exists and has reasonable size."""
    p = Path(path)
    return p.exists() and p.stat().st_size >= min_bytes


# ── Download with Progress ───────────────────────────────────
def download_with_progress(url, output_path, label, timeout=120, min_size=100):
    """Stream-download a file with a rich progress bar. Returns True on success."""
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))

        with Progress(
            SpinnerColumn(),
            TextColumn(f"  [cyan]{label}[/]"),
            BarColumn(bar_width=30),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(label, total=total if total > 0 else None)
            chunks = []
            downloaded = 0

            for chunk in response.iter_content(chunk_size=8192):
                if _shutdown_requested:
                    return False
                chunks.append(chunk)
                downloaded += len(chunk)
                progress.update(task, advance=len(chunk))
                if total > 0:
                    progress.update(task, completed=downloaded)

        data = b"".join(chunks)
        if len(data) < min_size:
            console.print(f"  [red]✗ ダウンロードサイズが小さすぎます ({len(data)} bytes)[/]")
            return False

        tmp_path = Path(str(output_path) + ".downloading")
        tmp_path.write_bytes(data)
        tmp_path.replace(output_path)
        console.print(f"  [green]✓[/] {label} [dim]({len(data):,} bytes)[/]")
        log.info(f"[{label}] ダウンロード完了 ({len(data):,} bytes)")
        return True

    except requests.exceptions.RequestException as e:
        console.print(f"  [red]✗ ダウンロード失敗: {e}[/]")
        log.error(f"[{label}] ダウンロード失敗: {e}")
        return False
    except (IOError, OSError) as e:
        console.print(f"  [red]✗ ファイル保存失敗: {e}[/]")
        log.error(f"[{label}] ファイル保存失敗: {e}")
        return False


# ── API Functions ────────────────────────────────────────────

def analyze_image_gemini(image_path, negations, language=None):
    """Analyze backview with Gemini 3 Flash — always returns exactly 3 new nouns."""
    import re
    import replicate
    lang_name = language["name"] if language else "English"
    console.print(f"  [bold magenta]🔍 Gemini 3 Flash 具象分析 [{lang_name}] (Replicate)...[/]")
    log.info(f"[Gemini] 具象分析中... (model={REPLICATE_VISION_MODEL}, lang={lang_name})")

    negation_lower = {n.lower().strip() for n in negations}
    # 個別単語も展開（"tree branch" → {"tree", "branch", "tree branch"}）
    negation_words = set()
    for neg in negation_lower:
        negation_words.add(neg)
        for word in neg.split():
            if len(word) >= 3:  # 短すぎる単語は無視（"a", "of" など）
                negation_words.add(word)

    def _is_negated(noun):
        """既存の否定ワードに含まれるかチェック（単語レベルでも照合）"""
        n_lower = noun.lower().strip()
        # 完全一致
        if n_lower in negation_lower:
            return True
        # 否定ワードが名詞に含まれる（部分文字列）
        if any(neg in n_lower for neg in negation_lower):
            return True
        # 名詞が否定ワードに含まれる
        if any(n_lower in neg for neg in negation_lower):
            return True
        # 名詞の個別単語が否定ワードの個別単語と一致
        noun_words = {w for w in n_lower.split() if len(w) >= 3}
        if noun_words & negation_words:
            return True
        return False

    def _parse_nouns(text):
        """テキストから名詞リストを抽出。不要な記号や説明文を除去して堅牢に。"""
        all_nouns = []
        # JSON形式の配列っぽく出力された場合の対処
        if '[' in text and ']' in text:
            try:
                import json
                start = text.find('[')
                end = text.rfind(']') + 1
                parsed = json.loads(text[start:end])
                if isinstance(parsed, list):
                    text = ",".join(str(p) for p in parsed)
            except:
                pass
        
        lines = text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("---") or line.lower().startswith("here are") or line.lower().startswith("i can") or line.lower().startswith("sure"):
                continue
            line = re.sub(r'^\d+[.)\-]\s*', '', line)
            line = re.sub(r'[*_`]+', '', line)
            # 複数のカンマスタイルに対応（Latin, Arabic, CJK）
            parts_raw = re.split(r'[,،、·]', line)
            parts = [n.strip().strip('"').strip("'").strip('.') for n in parts_raw if n.strip()]
            parts = [n for n in parts if 1 < len(n) < 50 and not any(c in n for c in ['(', ')', '[', ']', ':'])]
            all_nouns.extend(parts)
        return all_nouns

    TARGET_COUNT = 3
    BATCH_REQUEST = 40  # ★ 大量取得してPythonでフィルタ → 1回で確実に成功させる
    MAX_RETRIES = 3     # ★ エラー時のみリトライ（最大3回）
    api_calls = 0

    # 言語指示の組み立て
    lang_instruction = ""
    lang_example = '["cat", "dog", "hammer", "wrench", "tree", "mushroom", ... (Total 40 items)]'
    if language:
        lang_instruction = f"\n5. IMPORTANT: Output ALL {BATCH_REQUEST} nouns {language['instruction']}. Use ONLY the native script of that language."
        lang_example = f'["(animal 1)", "(animal 2)", ... "(tool 1)", ... (Total 40 items)]'

    # NGワードをプロンプトから外し、純粋に大量のアイデアを出させる（Python側で後からフィルタリングする）
    prompt = f"""Analyze this 3D object backview render image.

INSTRUCTIONS:
1. Carefully observe the silhouette, protrusions, indentations, and surface flow of this shape.
2. Brainstorm concrete nouns that this shape could possibly resemble.
3. To ensure variety, you MUST provide EXACTLY 5 items for EACH of the following 8 categories (Total {BATCH_REQUEST} items):
   - Animals or Creatures
   - Tools or Weapons
   - Plants or Fungi
   - Foods or Ingredients
   - Vehicles or Machines
   - Buildings or Architecture
   - Clothing or Accessories
   - Everyday Objects
4. Output your response STRICTLY as a single flat JSON array of strings containing all {BATCH_REQUEST} items. Do not nest them by category. No markdown, no explanations.{lang_instruction}

Example output:
{lang_example}

[ignoring loop detection]"""

    for attempt in range(1, MAX_RETRIES + 1):
        if _shutdown_requested:
            return None

        attempt_label = f"(試行 {attempt}/{MAX_RETRIES})"
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn(f"  [cyan]Gemini 3 Flash 推論中... {attempt_label}[/]"),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("gemini", total=None)

                api_calls += 1
                temp = min(0.6 + (attempt - 1) * 0.3, 2.0)
                output = throttle.run(
                    REPLICATE_VISION_MODEL,
                    input={
                        "prompt": prompt,
                        "images": [open(image_path, "rb")],
                        "temperature": temp,
                        "max_output_tokens": 1000,  # 40個以上の名詞を余裕を持って収容するため
                    }
                )

            text = "".join(output).strip()
            if not text:
                log.warning(f"[Gemini] 空の応答 {attempt_label}")
                continue

            # パース＆フィルタ
            raw_nouns = _parse_nouns(text)
            valid_nouns = [n for n in raw_nouns if not _is_negated(n)]
            # 重複除去（順序保持）
            valid_nouns = list(dict.fromkeys(valid_nouns))

            console.print(f"  [dim]  取得: {len(raw_nouns)}個 → フィルタ後: {len(valid_nouns)}個[/]")
            log.info(f"[Gemini] 取得: {len(raw_nouns)}個, フィルタ後: {len(valid_nouns)}個, 有効: {valid_nouns}")

            if len(valid_nouns) >= TARGET_COUNT:
                # ★ ランダムに選択（多様性を確保）
                selected = random.sample(valid_nouns, TARGET_COUNT)
                console.print(f"  [dim]📊 分析APIコール数: {api_calls}回[/]")
                log.info(f"[Gemini] 分析APIコール数: {api_calls}回")
                console.print(f"  [bold green]✓ 最終結果: {', '.join(selected)}[/] [dim](候補{len(valid_nouns)}個からランダム選択)[/]")
                log.info(f"[Gemini] 最終結果: {selected} (候補: {valid_nouns})")
                return selected
            elif valid_nouns:
                # 足りないが1個以上ある → そのまま返す
                console.print(f"  [dim]📊 分析APIコール数: {api_calls}回[/]")
                console.print(f"  [yellow]⚠ {len(valid_nouns)}個のみ取得 (目標{TARGET_COUNT}個): {', '.join(valid_nouns)}[/]")
                log.warning(f"[Gemini] {len(valid_nouns)}個のみ: {valid_nouns}")
                return valid_nouns
            else:
                console.print(f"  [yellow]⚠ フィルタ後の有効ワードなし {attempt_label}[/]")
                log.warning(f"[Gemini] フィルタ後の有効ワードなし (raw: {raw_nouns[:5]})")

        except Exception as e:
            err_msg = str(e)
            log.warning(f"[Gemini] エラー: {err_msg} {attempt_label}")
            console.print(f"  [yellow]⚠ エラー: {err_msg}[/]")

            import re
            m = re.search(r'resets in ~(\d+)s', err_msg)
            if m:
                wait_sec = int(m.group(1)) + 2
                console.print(f"  [dim]Rate limitにより {wait_sec} 秒待機します...[/]")
                time.sleep(wait_sec)
            elif "404" in err_msg or "not be found" in err_msg.lower():
                console.print(f"  [bold red]⚠ モデル名 '{REPLICATE_VISION_MODEL}' が見つかりません。[/]")
                console.print("  [dim].env の REPLICATE_VISION_MODEL を確認するか、別のモデルIDを指定してください。[/]")
                break
            else:
                time.sleep(5)
            continue

    # 全リトライ失敗
    console.print(f"  [dim]📊 分析APIコール数: {api_calls}回[/]")
    console.print("  [red]✗ 新しいワードを取得できませんでした[/]")
    log.error("[Gemini] 新しいワードを取得できませんでした")
    return None

def generate_image_nano_banana(nouns, negations, output_path, input_image_path=None):
    """Generate image via Nano Banana 2 (Gemini 3.1 Flash Image).
    
    input_image_path が指定された場合は img2img（編集）モード、
    未指定の場合は txt2img（新規生成）モードとして動作。
    """
    import replicate

    mode = "img2img" if input_image_path else "txt2img"
    console.print(f"  [bold blue]🎨 Nano Banana 2 {mode} 変容画像生成...[/]")
    log.info(f"[NanoBanana] {mode} 画像を生成中...")

    nouns_str = ", ".join(nouns)
    core_negations = [n for n in negations if n in INITIAL_NEGATIONS and n not in nouns]
    other_negations = [n for n in negations if n not in INITIAL_NEGATIONS and n not in nouns]
    neg_list = core_negations + other_negations[-40:]
    neg_str = ", ".join(neg_list)

    if input_image_path:
        # img2img: シルエットをより重視しつつ、表面や内部のディテールで変容を表現する
        prompt = (
            f"Transform this 3D object render to embody {nouns_str}. "
            f"STRONGLY maintain the exact core silhouette and foundational shape of the input image. "
            f"The transformation should primarily occur within the boundaries of the original silhouette, using new surface details, textures, and internal structural blending to express {nouns_str}. "
            f"The result must clearly preserve the original outline while mutating its contents into {nouns_str}. "
            f"Standalone sculpture on a pure solid clean studio background. "
            f"Avoid: altering the original silhouette, {neg_str}, complex background, environment, frame, border, abstract, geometric, text, label. "
            f"Bright studio lighting."
        )
    else:
        # txt2img: 新規生成
        prompt = (
            f"A standalone highly detailed photorealistic 3D sculpture on a pure solid neutral grey background. "
            f"The entire surface and structural details are a fusion of {nouns_str}. "
            f"The object appears as if {nouns_str} grew together into one solid form. "
            f"Soft natural lighting, clay-like material. "
            f"Do NOT include: complex background, environment, frame, border, particle effects, {neg_str}, abstract, geometric, text, label, simple shapes"
        )

    def _call():
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn(f"  [cyan]Nano Banana 2 {mode} 生成中...[/]"),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("nano_banana", total=None)

                api_input = {
                    "prompt": prompt,
                    "aspect_ratio": "1:1",
                    "output_format": "png",
                }

                # img2img モード: image_input に前回画像を渡す
                if input_image_path:
                    api_input["image_input"] = [open(input_image_path, "rb")]

                output = throttle.run(
                    REPLICATE_IMAGE_MODEL,
                    input=api_input,
                )

            if output is None:
                raise ValueError(f"Replicate API returned None ({mode}).")

            tmp_path = Path(str(output_path) + ".tmp")

            if isinstance(output, str) or hasattr(output, "read"):
                output_list = [output]
            else:
                output_list = list(output) if not isinstance(output, list) else output

            if not output_list:
                log.warning("[NanoBanana] 出力が空です")
                return None

            first_item = output_list[0]

            if hasattr(first_item, "read"):
                console.print(f"  [green]✓[/] Nano Banana 2 生成完了")
                log.info(f"[NanoBanana] 生成完了 (FileOutput)")
                tmp_path.write_bytes(first_item.read())
            else:
                image_url = str(first_item)
                console.print(f"  [green]✓[/] Nano Banana 2 生成完了")
                log.info(f"[NanoBanana] 生成完了: {image_url[:80]}...")

                resp = requests.get(image_url, timeout=60)
                resp.raise_for_status()
                tmp_path.write_bytes(resp.content)

            tmp_path.replace(output_path)

            if is_valid_file(output_path, min_bytes=1000):
                console.print(f"  [green]✓[/] 画像保存完了: [dim]{output_path.name}[/]")
                return True
            else:
                log.warning("[NanoBanana] 画像が不正です")
                return None

        except Exception as e:
            log.warning(f"[NanoBanana] エラー: {e}")
            console.print(f"  [yellow]⚠ Nano Banana 2 エラー: {e}[/]")
            return None

    return retry(_call, "NanoBanana") is not None


def _detect_3d_format(file_path):
    """ファイルヘッダーから実際の3Dフォーマットを判定する。
    Returns: 'glb', 'gltf', 'fbx', 'obj', 'ply' のいずれか。"""
    try:
        with open(file_path, "rb") as f:
            header = f.read(20)
        if header[:4] == b'glTF':
            return 'glb'
        elif header[:1] == b'{':
            return 'gltf'
        elif b'FBX' in header[:20]:
            return 'fbx'
        elif header[:3] == b'ply':
            return 'ply'
        else:
            # OBJ: テキスト形式（v, vn, vt, f, # で始まる行）
            return 'obj'
    except Exception:
        return 'glb'  # 判定失敗時はデフォルト


def _get_sds_model_config(model_id):
    """モデルIDをキーにSDS_MODEL_CONFIGSを検索（前方一致）。"""
    for key, cfg in SDS_MODEL_CONFIGS.items():
        if model_id.startswith(key):
            return cfg
    return {"prompt_required": False}  # デフォルト


def generate_3d_replicate(image_path, output_glb_path, model_id, nouns=None):
    """Send image to Replicate for 3D generation.
    
    nouns: サイクルの名詞リスト。prompt_required なモデルに自動渡しされる。
    Returns: 実際に保存されたファイルのPath（正しい拡張子付き）、失敗時はNone。
    """
    import replicate
    console.print(f"  [bold green]🧊 Replicate Image-to-3D 生成 [{model_id.split('/')[1].split(':')[0]}]...[/]")
    log.info(f"[Replicate 3D] Image-to-3D 開始... ({model_id})")

    import re
    from PIL import Image
    from pathlib import Path
    import requests

    cfg = _get_sds_model_config(model_id)
    prompt_required = cfg.get("prompt_required", False)

    # プロンプト文字列の組み立て（必要な場合のみ）
    if prompt_required:
        if nouns:
            prompt_str = f"A 3D object that is a fusion of {', '.join(nouns)}"
        else:
            prompt_str = "A detailed 3D object"
        console.print(f"  [dim]prompt: {prompt_str}[/]")
        log.info(f"[Replicate 3D] prompt: {prompt_str}")
    else:
        prompt_str = None

    # 画像を90度回転してから送信（ハルシネーション促進）
    rotated_path = Path(image_path).with_name("mutated_rotated.png")
    try:
        with Image.open(image_path) as img:
            rotated = img.rotate(-90, expand=True)
            rotated.save(rotated_path, format="PNG")
        console.print("  [dim]↻ 画像を90°回転して別のファイルに保存しました[/]")
        log.info("[Replicate 3D] 画像を90°回転して保存しました")
    except Exception as e:
        log.error(f"[Replicate 3D] 画像回転エラー: {e}")
        console.print(f"  [red]✗ 画像の回転に失敗しました: {e}[/]")
        return False

    def _call():
        try:
            # モデルによって画像の渡し方を切り替え
            use_data_uri = cfg.get("use_data_uri", False)
            if use_data_uri:
                # Data URI: base64エンコードでMIMEタイプを直接埋め込む
                # （hunyuan-3d-3.1等、URLの拡張子でフォーマット判定するモデル向け）
                with open(rotated_path, "rb") as f:
                    img_bytes = f.read()
                image_url = f"data:image/png;base64,{base64.b64encode(img_bytes).decode()}"
                console.print(f"  [dim]📤 画像アップロード完了 (Data URI)[/]")
                log.info(f"[Replicate 3D] Data URI作成完了 ({len(img_bytes):,} bytes)")
            else:
                # 通常: Replicateストレージに事前アップロードしてURLを取得
                uploaded_file = replicate.files.create(
                    str(rotated_path),
                    content_type="image/png",
                )
                image_url = uploaded_file.urls["get"]
                console.print(f"  [dim]📤 画像アップロード完了[/]")
                log.info(f"[Replicate 3D] 画像アップロード完了: {image_url}")

            # モデル固有の画像入力キーを使用 ("image", "images", "input_image", "image_path"など)
            image_key = cfg.get("image_key", "image")
            if image_key == "images":
                # Trellis等: 配列形式で渡す
                api_input = {"images": [image_url]}
            else:
                api_input = {image_key: image_url}
            if prompt_required and prompt_str:
                api_input["prompt"] = prompt_str
            # モデル固有の追加パラメータ（num_steps等）をマージ
            extra_params = cfg.get("extra_params", {})
            if extra_params:
                api_input.update(extra_params)
                console.print(f"  [dim]params: {extra_params}[/]")

            # 非同期prediction作成 + 手動ポーリング（HTTPタイムアウト回避）
            prediction = throttle.create_prediction(
                model=model_id if "/" in model_id and ":" not in model_id else None,
                version=model_id.split(":")[-1] if ":" in model_id else None,
                input=api_input,
            )
            if prediction is None:
                return None  # シャットダウン要求
            log.info(f"[Replicate 3D] prediction作成: {prediction.id}")

            MAX_WAIT = 300  # 5分タイムアウト
            POLL_INTERVAL = 3
            waited = 0

            with Progress(
                SpinnerColumn(),
                TextColumn(f"  [cyan]3D生成中... ({model_id.split('/')[1].split(':')[0]})[/]"),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task("replicate_3d", total=None)

                while prediction.status not in ("succeeded", "failed", "canceled"):
                    if _shutdown_requested:
                        try:
                            prediction.cancel()
                        except Exception:
                            pass
                        return None
                    if waited >= MAX_WAIT:
                        console.print(f"  [yellow]⚠ {MAX_WAIT}秒タイムアウト — スキップします[/]")
                        log.warning(f"[Replicate 3D] {MAX_WAIT}秒タイムアウト (prediction={prediction.id})")
                        try:
                            prediction.cancel()
                        except Exception:
                            pass
                        return None
                    time.sleep(POLL_INTERVAL)
                    waited += POLL_INTERVAL
                    prediction.reload()

            if prediction.status == "failed":
                error_msg = prediction.error or "Unknown error"
                raise ValueError(f"Prediction failed: {error_msg}")

            if prediction.status == "canceled":
                return None

            output = prediction.output

            if output is None:
                raise ValueError("Replicate API returned None.")

            tmp_path = Path(str(output_glb_path) + ".tmp")

            # Trellis等: 出力が辞書形式の場合、output_keyでGLBファイルを取得
            output_key = cfg.get("output_key")
            if isinstance(output, dict) and output_key:
                glb_output = output.get(output_key)
                if not glb_output:
                    raise ValueError(f"出力に '{output_key}' キーがありません: {list(output.keys())}")
                output = glb_output

            if hasattr(output, "read"):
                console.print(f"  [green]✓[/] Replicate 生成完了 (FileOutput)")
                log.info(f"[Replicate 3D] 生成完了")
                tmp_path.write_bytes(output.read())
            else:
                model_url = str(output)
                if isinstance(output, list) and len(output) > 0:
                    # .glb/.objを含むURLを優先して探す
                    mesh_urls = [str(u) for u in output if any(ext in str(u).lower() for ext in [".glb", ".obj", ".fbx", ".ply"])]
                    model_url = mesh_urls[0] if mesh_urls else str(output[0])

                console.print(f"  [green]✓[/] Replicate 生成完了")
                log.info(f"[Replicate 3D] 生成完了: {model_url[:80]}...")

                # Download 3D model
                success = download_with_progress(
                    model_url, tmp_path,
                    label="3D モデル",
                    timeout=120,
                    min_size=100,
                )
                if not success:
                    return None

            # tmp_path にダウンロード済み → ヘッダーで実際のフォーマットを判定
            if not tmp_path.exists():
                return None

            actual_format = _detect_3d_format(tmp_path)
            correct_ext = f".{actual_format}"
            final_path = output_glb_path.with_suffix(correct_ext)
            tmp_path.replace(final_path)

            if actual_format != "glb":
                console.print(f"  [dim]📦 実際のフォーマット: {actual_format.upper()} → {final_path.name}[/]")
                log.info(f"[Replicate 3D] フォーマット検出: {actual_format} → {final_path.name}")

            if is_valid_file(final_path, min_bytes=100):
                return str(final_path)
            return None

        except Exception as e:
            err_str = str(e)
            log.warning(f"[Replicate 3D] エラー: {err_str}")
            console.print(f"  [yellow]⚠ Replicate 3D エラー: {err_str}[/]")
            
            # 429 Rate Limit の場合は指定秒数待機してリトライ
            if "429" in err_str and "resets in" in err_str:
                match = re.search(r"resets in ~(\d+)s", err_str)
                if match:
                    wait_sec = int(match.group(1)) + 2
                    console.print(f"  [cyan]⏳ レートリミット制限: {wait_sec}秒待機してから再試行します...[/]")
                    time.sleep(wait_sec)
                    return "RATE_LIMIT_RETRY"
                    
            return None

    # フォールバックさせるため、この関数内での通常のリトライは行わない(max_retries=1)
    # ただしレートリミットの場合は特別に同モデルで再試行する
    while True:
        res = retry(_call, f"Replicate 3D ({model_id.split('/')[1].split(':')[0]})", max_retries=1)
        if res == "RATE_LIMIT_RETRY":
            continue
        # res は実際のファイルパス(str) or None
        return res

def analyze_image_gemini_direct(image_path, negations, language=None):
    """Google Gemini API を直接呼び出して画像分析（新SDK: google-genai）。"""
    import re
    from google import genai as google_genai
    from google.genai import types as genai_types
    from PIL import Image as PILImage

    lang_name = language["name"] if language else "English"
    console.print(f"  [bold magenta]🔍 Gemini API 直接 具象分析 [{lang_name}]...[/]")
    log.info(f"[GeminiDirect] 分析中 (lang={lang_name})")

    client = google_genai.Client(api_key=GEMINI_API_KEY)
    VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash-lite")

    negation_lower = {n.lower().strip() for n in negations}
    negation_words = set()
    for neg in negation_lower:
        negation_words.add(neg)
        for word in neg.split():
            if len(word) >= 3:
                negation_words.add(word)

    def _is_negated(noun):
        n_lower = noun.lower().strip()
        if n_lower in negation_lower:
            return True
        if any(neg in n_lower for neg in negation_lower):
            return True
        if any(n_lower in neg for neg in negation_lower):
            return True
        noun_words = {w for w in n_lower.split() if len(w) >= 3}
        if noun_words & negation_words:
            return True
        return False

    def _parse_nouns(text):
        all_nouns = []
        if '[' in text and ']' in text:
            try:
                start = text.find('[')
                end = text.rfind(']') + 1
                parsed = json.loads(text[start:end])
                if isinstance(parsed, list):
                    text = ",".join(str(p) for p in parsed)
            except Exception:
                pass
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = re.sub(r'^\d+[.)\-]\s*', '', line)
            line = re.sub(r'[*_`]+', '', line)
            parts = re.split(r'[,،、·]', line)
            parts = [n.strip().strip('"').strip("'").strip('.') for n in parts if n.strip()]
            parts = [n for n in parts if 1 < len(n) < 50 and not any(c in n for c in ['(', ')', '[', ']', ':'])]
            all_nouns.extend(parts)
        return all_nouns

    BATCH_REQUEST = 40
    TARGET_COUNT  = 3
    lang_instruction = ""
    if language:
        lang_instruction = f"\n5. IMPORTANT: Output ALL {BATCH_REQUEST} nouns {language['instruction']}."

    prompt = f"""Analyze this 3D object backview render image.

INSTRUCTIONS:
1. Carefully observe the silhouette, protrusions, indentations, and surface flow of this shape.
2. Brainstorm concrete nouns that this shape could possibly resemble.
3. To ensure variety, you MUST provide EXACTLY 5 items for EACH of the following 8 categories (Total {BATCH_REQUEST} items):
   - Animals or Creatures
   - Tools or Weapons
   - Plants or Fungi
   - Foods or Ingredients
   - Vehicles or Machines
   - Buildings or Architecture
   - Clothing or Accessories
   - Everyday Objects
4. Output your response STRICTLY as a single flat JSON array of strings. No markdown, no explanations.{lang_instruction}"""

    for attempt in range(1, 4):
        if _shutdown_requested:
            return None
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn(f"  [cyan]Gemini API 推論中... (試行 {attempt}/3)[/]"),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("gemini_direct", total=None)
                pil_img = PILImage.open(image_path)
                response = client.models.generate_content(
                    model=VISION_MODEL,
                    contents=[prompt, pil_img],
                    config=genai_types.GenerateContentConfig(
                        temperature=min(0.6 + (attempt - 1) * 0.3, 1.5),
                        max_output_tokens=1200,
                    ),
                )
            text = response.text.strip()
            if not text:
                continue
            raw_nouns = _parse_nouns(text)
            valid_nouns = list(dict.fromkeys(n for n in raw_nouns if not _is_negated(n)))
            console.print(f"  [dim]取得: {len(raw_nouns)}個 → フィルタ後: {len(valid_nouns)}個[/]")
            if len(valid_nouns) >= TARGET_COUNT:
                selected = random.sample(valid_nouns, TARGET_COUNT)
                console.print(f"  [bold green]✓ 結果: {', '.join(selected)}[/]")
                log.info(f"[GeminiDirect] 結果: {selected}")
                return selected
            elif valid_nouns:
                return valid_nouns
        except Exception as e:
            log.warning(f"[GeminiDirect] エラー: {e} (試行 {attempt}/3)")
            console.print(f"  [yellow]⚠ {e}[/]")
            time.sleep(5)

    console.print("  [red]✗ Gemini API 分析失敗[/]")
    return None


def generate_image_colab(nouns, negations, output_path, input_image_path=None):
    """Colab サーバーに FLUX.1-schnell で画像生成を依頼する。"""
    mode = "img2img" if input_image_path else "txt2img"
    console.print(f"  [bold blue]🎨 Colab FLUX.1-schnell {mode}...[/]")
    log.info(f"[ColabImage] {mode} 開始")

    nouns_str = ", ".join(nouns)
    core_neg  = [n for n in negations if n in INITIAL_NEGATIONS and n not in nouns]
    other_neg = [n for n in negations if n not in INITIAL_NEGATIONS and n not in nouns]
    neg_str   = ", ".join(core_neg + other_neg[-40:])

    if input_image_path:
        prompt = (
            f"Transform this 3D object render to embody {nouns_str}. "
            f"STRONGLY maintain the exact core silhouette of the input image. "
            f"Surface details and textures express {nouns_str}. "
            f"Standalone sculpture on pure solid studio background. "
            f"Avoid: altering silhouette, {neg_str}, abstract, geometric, text."
        )
    else:
        prompt = (
            f"A standalone highly detailed photorealistic 3D sculpture on a pure solid neutral grey background. "
            f"The entire surface is a fusion of {nouns_str}. "
            f"Soft natural lighting, clay-like material. "
            f"Do NOT include: complex background, {neg_str}, abstract, geometric, text."
        )

    def _call():
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn(f"  [cyan]Colab FLUX {mode} 生成中...[/]"),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("colab_img", total=None)
                data   = {"prompt": prompt, "num_steps": 4, "strength": 0.6}
                files  = {}
                opened = None
                if input_image_path:
                    opened = open(input_image_path, "rb")
                    files["input_image"] = opened
                resp = requests.post(
                    f"{COLAB_SERVER_URL}/generate_image",
                    data=data, files=files, timeout=180,
                )
                if opened:
                    opened.close()
            resp.raise_for_status()
            tmp = Path(str(output_path) + ".tmp")
            tmp.write_bytes(resp.content)
            tmp.replace(output_path)
            if is_valid_file(output_path, min_bytes=1000):
                console.print(f"  [green]✓[/] 画像保存完了: [dim]{output_path.name}[/]")
                return True
        except Exception as e:
            log.warning(f"[ColabImage] エラー: {e}")
            console.print(f"  [yellow]⚠ Colab 画像生成エラー: {e}[/]")
        return None

    return retry(_call, "ColabImage") is not None


def generate_3d_colab(image_path, output_path, nouns=None):
    """Colab サーバーに TripoSR で 3D 生成を依頼する。OBJ ファイルパスを返す。"""
    console.print("  [bold green]🧊 Colab TripoSR Image-to-3D...[/]")
    log.info("[Colab3D] TripoSR 開始")

    # 画像を90度回転して送信（ハルシネーション促進）
    rotated_path = Path(image_path).with_name("mutated_rotated.png")
    try:
        from PIL import Image as PILImage
        with PILImage.open(image_path) as img:
            img.rotate(-90, expand=True).save(rotated_path, format="PNG")
        console.print("  [dim]↻ 画像を90°回転して送信[/]")
    except Exception as e:
        log.error(f"[Colab3D] 画像回転エラー: {e}")
        return None

    def _call():
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("  [cyan]Colab TripoSR 3D生成中...[/]"),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("colab_3d", total=None)
                with open(rotated_path, "rb") as f:
                    resp = requests.post(
                        f"{COLAB_SERVER_URL}/generate_3d",
                        files={"image": f},
                        timeout=300,
                    )
            resp.raise_for_status()

            tmp = output_path.with_suffix(".tmp")
            tmp.write_bytes(resp.content)

            actual_format = _detect_3d_format(tmp)
            final_path    = output_path.with_suffix(f".{actual_format}")
            tmp.replace(final_path)

            if is_valid_file(final_path, min_bytes=100):
                console.print(f"  [green]✓[/] 3Dメッシュ保存完了: [dim]{final_path.name}[/]")
                log.info(f"[Colab3D] 完了: {final_path.name}")
                return str(final_path)
        except Exception as e:
            log.warning(f"[Colab3D] エラー: {e}")
            console.print(f"  [yellow]⚠ Colab 3D生成エラー: {e}[/]")
        return None

    return retry(_call, "Colab3D")


def render_blender(glb_path, output_front_path, output_back_path):
    """Render front and back views with Blender headless, showing spinner progress."""
    console.print("  [bold yellow]📷 Blender レンダリング...[/]")
    log.info("[Blender] レンダリング中...")
    blender_script = Path("blender_render.py").resolve()
    cmd = [
        BLENDER_PATH, "-b",
        "-P", str(blender_script),
        "--",
        str(Path(glb_path).resolve()),
        str(Path(output_front_path).resolve()),
        str(Path(output_back_path).resolve()),
    ]

    def _call():
        with Progress(
            SpinnerColumn(),
            TextColumn("  [cyan]Blender レンダリング実行中...[/]"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("blender", total=None)
            result = subprocess.run(
                cmd, capture_output=True, timeout=120,
                encoding="utf-8", errors="replace",
            )

        if result.returncode != 0:
            log.warning(f"[Blender] 終了コード {result.returncode}")
            stderr_tail = result.stderr[-500:] if result.stderr else "(stderr empty)"
            log.warning(f"[Blender] stderr: {stderr_tail}")
            console.print(f"  [red]✗ Blender 終了コード {result.returncode}[/]")
            return None
        if is_valid_file(output_front_path) and is_valid_file(output_back_path):
            console.print("  [green]✓[/] レンダリング完了")
            log.info("[Blender] レンダリング完了")
            return True
        console.print("  [red]✗ 出力ファイルが見つかりません[/]")
        log.warning("[Blender] 出力ファイルが見つかりません。")
        return None

    return retry(_call, "Blender", max_retries=3) is not None


# ── Dashboard ────────────────────────────────────────────────

def update_dashboard(history, negations=None):
    html_path = OUTPUT_DIR / "dashboard.html"

    cards = ""
    for entry in reversed(history):
        i = entry["iteration"]
        iter_dir = OUTPUT_DIR / f"iteration_{i:03d}"
        
        # 絶対パスで画像を参照（OUTPUT_DIRがプロジェクト外でも表示可能に）
        backview = (iter_dir / "backview.png").resolve().as_posix()
        mutated  = (iter_dir / "mutated.png").resolve().as_posix()
        rotated  = (iter_dir / "mutated_rotated.png").resolve().as_posix()
        frontview= (iter_dir / "frontview.png").resolve().as_posix()
        nouns_display = ", ".join(entry["new_nouns"])
        neg_count = entry["neg_count"]
        ts = entry.get("timestamp", "")
        model_id_disp = entry.get("model_id", "Unknown")
        lang_disp = entry.get("language", "English")
        cycle_cost = entry.get("cost")
        cycle_predict_time = entry.get("predict_time")
        cycle_pred_count = entry.get("prediction_count")

        # Build flow HTML dynamically based on file existence
        flow_html = ""
        gen_type = "txt2img" if i == 1 else "img2img"
        
        if (iter_dir / "mutated.png").exists():
            flow_html += f'''
                <div class="process-arrow">
                    <span class="arrow-label">{gen_type}</span>
                    <div class="arrow">→</div>
                </div>
                <div class="step">
                    <img src="file:///{mutated}" alt="Mutated" onerror="this.style.display='none'">
                    <p class="label">mutated (input)</p>
                </div>'''
            
        if (iter_dir / "mutated_rotated.png").exists():
            flow_html += f'''
                <div class="process-arrow">
                    <span class="arrow-label">rotate</span>
                    <div class="arrow">→</div>
                </div>
                <div class="step">
                    <img src="file:///{rotated}" alt="Rotated" onerror="this.style.display='none'">
                    <p class="label">-90° rotated</p>
                </div>'''
            
        if (iter_dir / "frontview.png").exists():
            flow_html += f'''
                <div class="process-arrow">
                    <span class="arrow-label">img23D</span>
                    <div class="arrow">→</div>
                </div>
                <div class="step">
                    <img src="file:///{frontview}" alt="Frontview" onerror="this.style.display='none'">
                    <p class="label">frontview</p>
                </div>'''
        elif (iter_dir / "mesh.glb").exists():
             flow_html += f'''
                <div class="process-arrow">
                    <span class="arrow-label">img23D</span>
                    <div class="arrow">→</div>
                </div>
                <div class="step">
                    <p class="label mesh-label">mesh.glb</p>
                </div>'''

        if (iter_dir / "backview.png").exists():
            if flow_html:
                flow_html += '''
                <div class="process-arrow">
                    <span class="arrow-label">render back</span>
                    <div class="arrow">→</div>
                </div>'''
            flow_html += f'''
                <div class="step">
                    <img src="file:///{backview}" alt="Backview" onerror="this.style.display='none'">
                    <p class="label">backview</p>
                </div>'''
            
        if flow_html:
            flow_html += '''
                <div class="process-arrow">
                    <span class="arrow-label">LLM analysis</span>
                    <div class="arrow">→</div>
                </div>'''
        flow_html += f'''
                <div class="step analysis">
                    <div class="noun-box">
                        <p class="nouns">{nouns_display}</p>
                        <p class="label">extracted nouns</p>
                    </div>
                </div>'''

        # コストバッジ
        cost_badge = ""
        if cycle_cost is not None:
            cost_badge = f'<span class="cost-badge">${cycle_cost:.4f}</span>'
        elif cycle_predict_time is not None:
            cost_badge = f'<span class="cost-badge time-only">{cycle_predict_time:.1f}s</span>'

        cards += f"""
        <div class="card">
            <div class="card-header">
                <h2>Cycle {i:03d}</h2>
                <div>
                    {cost_badge}
                    <span class="neg-badge">{neg_count} negations</span>
                    <span class="model-badge">{model_id_disp}</span>
                    <span class="lang-badge">🌐 {lang_disp}</span>
                    <span class="timestamp">{ts}</span>
                </div>
            </div>
            <div class="flow">
                {flow_html}
            </div>
        </div>"""

    # 否定ワード一覧セクション
    neg_section = ""
    if negations:
        neg_tags = "".join(f'<span class="neg-tag">{n}</span>' for n in negations)
        neg_section = f"""
        <div class="negation-panel">
            <div class="negation-header">
                <h2>Negation Words ({len(negations)})</h2>
            </div>
            <div class="neg-tags">
                {neg_tags}
            </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hallucination Loop</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans:wght@300;500;700&family=Noto+Sans+JP:wght@300;500;700&family=Noto+Sans+Arabic:wght@300;500;700&family=Noto+Sans+Devanagari:wght@300;500;700&family=Noto+Sans+KR:wght@300;500;700&family=Noto+Sans+SC:wght@300;500;700&family=Noto+Sans+Thai:wght@300;500;700&family=Noto+Sans+Tamil:wght@300;500;700&display=swap');
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Noto Sans', 'Noto Sans JP', 'Noto Sans Arabic', 'Noto Sans Devanagari', 'Noto Sans KR', 'Noto Sans SC', 'Noto Sans Thai', 'Noto Sans Tamil', sans-serif; background: #0a0a0a; color: #e0e0e0; padding: 30px; }}
        h1 {{ text-align: center; font-weight: 300; font-size: 1.8em; margin-bottom: 30px; letter-spacing: 3px; color: #fff; }}
        .card {{ background: #161616; border: 1px solid #2a2a2a; border-radius: 12px; padding: 24px; margin-bottom: 24px; }}
        .card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; border-bottom: 1px solid #2a2a2a; padding-bottom: 12px; }}
        .card-header h2 {{ font-size: 1.1em; font-weight: 500; color: #c8b89a; }}
        .neg-badge {{ background: #2a1a1a; color: #e06060; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; }}
        .model-badge {{ background: #1a2a40; color: #80a0e0; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; margin-left: 10px; }}
        .lang-badge {{ background: #1a3a20; color: #80e0a0; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; margin-left: 10px; }}
        .cost-badge {{ background: #2a2a1a; color: #e0c060; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; margin-left: 10px; font-weight: 500; }}
        .cost-badge.time-only {{ color: #a0a080; }}
        .timestamp {{ font-size: 0.75em; color: #555; margin-left: 10px; }}
        .flow {{ display: flex; align-items: center; gap: 12px; overflow-x: auto; padding: 8px 0; }}
        .step {{ text-align: center; flex-shrink: 0; }}
        .step img {{ width: 180px; height: 180px; object-fit: cover; border-radius: 8px; border: 1px solid #333; background: #111; }}
        .process-arrow {{ display: flex; flex-direction: column; align-items: center; justify-content: center; flex-shrink: 0; margin: 0 5px; }}
        .arrow-label {{ font-size: 0.65em; color: #888; margin-bottom: -5px; font-weight: 500; text-transform: uppercase; letter-spacing: 1px; }}
        .arrow {{ font-size: 1.5em; color: #555; }}
        .label {{ margin-top: 8px; font-size: 0.75em; color: #666; }}
        .analysis {{ display: flex; align-items: center; }}
        .noun-box {{ background: #1a1a2a; padding: 16px 20px; border-radius: 8px; border: 1px solid #334; min-width: 160px; }}
        .nouns {{ font-size: 1em; font-weight: 500; color: #c0d8ff; line-height: 1.6; }}
        .mesh-label {{ color: #80c080; font-weight: 500; }}
        .negation-panel {{ background: #161616; border: 1px solid #2a2a2a; border-radius: 12px; padding: 24px; margin-bottom: 30px; }}
        .negation-header {{ margin-bottom: 16px; border-bottom: 1px solid #2a2a2a; padding-bottom: 12px; }}
        .negation-header h2 {{ font-size: 1.1em; font-weight: 500; color: #e06060; }}
        .neg-tags {{ display: flex; flex-wrap: wrap; gap: 8px; }}
        .neg-tag {{ background: #1a1212; color: #cc8080; padding: 4px 12px; border-radius: 6px; font-size: 0.8em; border: 1px solid #332222; }}
    </style>
</head>
<body>
    <h1>H A L L U C I N A T I O N &nbsp; L O O P</h1>
    {neg_section}
    {cards}
</body>
</html>"""

    tmp = html_path.with_suffix(".tmp")
    try:
        tmp.write_text(html, encoding="utf-8")
        tmp.replace(html_path)
        log.info("[Dashboard] 更新完了")
    except Exception as e:
        log.error(f"[Dashboard] 書き込み失敗: {e}")


# ── Cycle Status Display ─────────────────────────────────────

def print_cycle_header(iteration, current_nouns, negation_count):
    """Print a rich panel for the cycle header."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold white")
    table.add_column(style="cyan")
    table.add_row("サイクル", f"#{iteration:03d}")
    table.add_row("キーワード", ", ".join(current_nouns))
    table.add_row("累積否定", f"{negation_count}語")

    panel = Panel(
        table,
        title=f"[bold white]CYCLE {iteration:03d}[/]",
        border_style="bright_blue",
        padding=(1, 2),
    )
    console.print()
    console.print(panel)


def print_cycle_result(iteration, nouns, success):
    """Print cycle completion status."""
    if success:
        console.print(
            Panel(
                f"[bold green]✓ 完了[/] → [bold white]{', '.join(nouns)}[/]",
                border_style="green",
                padding=(0, 2),
            )
        )
    else:
        console.print(
            Panel(
                "[bold red]✗ 失敗[/]",
                border_style="red",
                padding=(0, 2),
            )
        )


def print_pipeline_progress(step, total_steps=4):
    """Print which step of the pipeline we're on."""
    step_names = {
        1: "🎨 画像生成 (Replicate)",
        2: "🧊 3D生成 (Replicate)",
        3: "📷 レンダリング (Blender)",
        4: "🔍 具象分析 (Gemini 3 Flash)",
    }
    bar = ""
    for i in range(1, total_steps + 1):
        if i < step:
            bar += "[green]●[/] "
        elif i == step:
            bar += "[bold cyan]◉[/] "
        else:
            bar += "[dim]○[/] "

    name = step_names.get(step, f"Step {step}")
    console.print(f"\n  {bar}  [bold]{name}[/]")
    console.print(f"  {'─' * 50}")


# ── Main Loop ────────────────────────────────────────────────

def _print_cycle_costs(cost_info, iteration, success=True):
    """サイクルのコスト情報をリッチなテーブルで表示する。"""
    if cost_info is None:
        console.print("  [dim]💰 コスト情報を取得できませんでした[/]")
        return

    count = cost_info.get("count", 0)
    total_time = cost_info.get("total_predict_time", 0.0)
    total_cost = cost_info.get("total_cost")
    cumulative_cost = cost_info.get("cumulative_cost")
    predictions = cost_info.get("predictions", [])

    status_label = "[green]✓[/]" if success else "[red]✗[/]"
    cost_color = "green" if success else "red"

    # コストテーブル
    table = Table(
        show_header=True,
        header_style="bold dim",
        box=box.SIMPLE,
        padding=(0, 1),
        expand=False,
    )
    table.add_column("モデル", style="cyan", min_width=20)
    table.add_column("状態", style="dim", justify="center", min_width=6)
    table.add_column("計算時間", justify="right", min_width=8)
    table.add_column("推定コスト", justify="right", min_width=10)

    for pred in predictions:
        model = pred.get("model", "unknown")
        # モデル名を短縮表示
        if isinstance(model, str) and "/" in model:
            model = model.split("/")[-1]
        status = pred.get("status", "?")
        status_icon = {"succeeded": "✓", "failed": "✗", "canceled": "⊘"}.get(status, "⋯")
        pt = pred.get("predict_time", 0.0)
        cost = pred.get("cost")

        time_str = f"{pt:.1f}s" if pt else "-"
        cost_str = f"~${cost:.4f}" if cost is not None and cost > 0 else "-"

        table.add_row(str(model), status_icon, time_str, cost_str)

    # サマリー行
    summary = f"  {status_label} [bold]Cycle {iteration:03d}[/] 💰 [{cost_color}]~${total_cost:.4f}[/] ({count}件, {total_time:.1f}s)"

    if cumulative_cost is not None and cumulative_cost > 0:
        summary += f"  [dim]| 累積: ~${cumulative_cost:.4f}[/]"

    console.print()
    if predictions:
        console.print("  [bold dim]── サイクルコスト詳細 (推定) ──[/]")
        console.print(table)
    console.print(summary)
    console.print()

    # ログ出力
    log.info(f"[CostTracker] Cycle {iteration:03d}: ~${total_cost:.4f} ({count}件, {total_time:.1f}s, 累積: ~${cumulative_cost:.4f})")

    for pred in predictions:
        log.info(f"  [{pred.get('status', '?')}] {pred.get('model', '?')}: {pred.get('predict_time', 0):.1f}s, cost=~${pred.get('cost', 0):.4f}")


def main():
    console.print()
    console.print(Panel(
        "[bold white]H A L L U C I N A T I O N   L O O P[/]\n"
        "[dim]Concrete Negation Engine v2[/]",
        border_style="bright_blue",
        padding=(1, 4),
    ), justify="center")
    console.print()

    validate_environment()

    negations = load_negations()
    history = load_history()

    # Find resume point based on history (not directory scan)
    if history:
        iteration = history[-1]["iteration"] + 1
    else:
        iteration = 1

    # Determine starting nouns
    if history and history[-1].get("new_nouns"):
        current_nouns = history[-1]["new_nouns"]
        console.print(f"[cyan]↻[/] 前回のサイクルから再開します (Cycle {iteration:03d})")
        console.print(f"  継続キーワード: [bold]{', '.join(current_nouns)}[/]")
        log.info(f"前回のサイクルから再開します。(Cycle {iteration:03d})")
    else:
        current_nouns = SEED_NOUNS
        # シードワードも否定リストに追加（最初の分析で再出力されないように）
        for noun in current_nouns:
            if noun not in negations:
                negations.append(noun)
        save_negations(negations)
        console.print(f"[cyan]▶[/] 新規開始します")
        console.print(f"  シード: [bold]{', '.join(current_nouns)}[/]")
        console.print(f"  [dim](シードワードを否定リストに追加済み)[/]")
        log.info(f"新規開始します。シード: {', '.join(current_nouns)}")

    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 5

    while iteration <= MAX_ITERATIONS:
        if _shutdown_requested:
            console.print("\n[yellow]シャットダウン中... 状態を保存しています。[/]")
            log.info("シャットダウン中... 状態を保存しています。")
            save_history(history)
            save_negations(negations)
            console.print("[green]✓ 正常に終了しました。[/]")
            log.info("正常に終了しました。")
            break

        iter_dir = OUTPUT_DIR / f"iteration_{iteration:03d}"
        iter_dir.mkdir(exist_ok=True)
        frontview_path= iter_dir / "frontview.png"
        backview_path = iter_dir / "backview.png"
        mutated_path  = iter_dir / "mutated.png"
        glb_path      = iter_dir / "mesh.glb"

        print_cycle_header(iteration, current_nouns, len(negations))
        console.print(f"  [dim]📊 {throttle.get_stats()}[/]")
        log.info(f"CYCLE {iteration:03d} | キーワード: {', '.join(current_nouns)} | 否定: {len(negations)}語")

        # サイクルコスト追跡開始
        throttle.start_cycle()

        cycle_ok = True

        # ── Step 1: Generate mutated concept image ──
        print_pipeline_progress(1)
        img_generated = False

        # ★ 部分成功キャッシュ
        if is_valid_file(mutated_path, min_bytes=1000):
            console.print("  [green]✓[/] 既存の mutated.png を再利用します")
            log.info("[Step 1] 既存の mutated.png を再利用")
            img_generated = True
        else:
            prev_backview_path = OUTPUT_DIR / f"iteration_{iteration-1:03d}" / "backview.png"
            if iteration > 1 and prev_backview_path.exists():
                if COLAB_SERVER_URL:
                    img_generated = generate_image_colab(
                        current_nouns, negations, mutated_path,
                        input_image_path=prev_backview_path,
                    )
                else:
                    img_generated = generate_image_nano_banana(
                        current_nouns, negations, mutated_path,
                        input_image_path=prev_backview_path,
                    )
                if not img_generated:
                    console.print("  [yellow]⚠ img2img 失敗。サイクルをリトライします[/]")
                    log.warning("img2img 失敗")
            else:
                if COLAB_SERVER_URL:
                    img_generated = generate_image_colab(current_nouns, negations, mutated_path)
                else:
                    img_generated = generate_image_nano_banana(current_nouns, negations, mutated_path)

        if not img_generated:
            log.error("Step 1 (画像生成) 失敗")
            cycle_ok = False

        # ── Step 2: 3D generation via Replicate ──
        if cycle_ok:
            print_pipeline_progress(2)
            
            # ★ 部分成功キャッシュ: 既存の3Dモデルファイルをチェック
            existing_mesh = None
            for ext in [".glb", ".gltf", ".obj", ".fbx", ".ply"]:
                candidate = iter_dir / f"mesh{ext}"
                if is_valid_file(candidate, min_bytes=100):
                    existing_mesh = str(candidate)
                    break
            
            if existing_mesh:
                console.print(f"  [green]✓[/] 既存の {Path(existing_mesh).name} を再利用します (APIコール節約)")
                log.info(f"[Step 2] 既存の {Path(existing_mesh).name} を再利用")
                result = existing_mesh
                actual_model_path = Path(existing_mesh)
                current_model_id = "cached"
            elif not is_valid_file(mutated_path):
                log.error("Step 2: 入力画像が見つかりません")
                console.print("  [red]✗ 入力画像 (mutated.png) が見つかりません[/]")
                cycle_ok = False
            else:
                if COLAB_SERVER_URL:
                    current_model_id = "colab/triposr"
                    result = generate_3d_colab(mutated_path, glb_path, nouns=current_nouns)
                    if result:
                        actual_model_path = Path(result)
                    else:
                        log.error("Step 2 (3D生成) Colab TripoSR 失敗")
                        cycle_ok = False
                else:
                    sds_models = REPLICATE_SDS_MODELS
                    if not sds_models:
                        sds_models = ["firtoz/trellis:e8f6c45206993f297372f5436b90350817bd9b4a0d52d2a76df50c1c8afa2b3c"]

                    result = None
                    for attempt_idx in range(len(sds_models)):
                        current_model_id = sds_models[(iteration - 1 + attempt_idx) % len(sds_models)]
                        short_name = current_model_id.split("/")[1].split(":")[0]
                        console.print(f"  [dim]🔄 {short_name} (候補 {attempt_idx+1}/{len(sds_models)})[/]")
                        result = generate_3d_replicate(
                            mutated_path, glb_path,
                            model_id=current_model_id,
                            nouns=current_nouns,
                        )
                        if result:
                            actual_model_path = Path(result)
                            break
                        console.print(f"  [yellow]⚠ {short_name} 失敗。次のモデルを試します...[/]")

                    if not result:
                        log.error("Step 2 (3D生成) 全モデルで失敗")
                        cycle_ok = False

        # ── Step 3: Render backview ──
        if cycle_ok:
            print_pipeline_progress(3)
            if not is_valid_file(actual_model_path, min_bytes=100):
                log.error(f"Step 3: 3Dモデルファイルが見つかりません ({actual_model_path.name})")
                console.print(f"  [red]✗ 3Dモデルファイル ({actual_model_path.name}) が見つかりません[/]")
                cycle_ok = False
            elif not render_blender(actual_model_path, frontview_path, backview_path):
                log.error("Step 3 (レンダリング) 失敗")
                cycle_ok = False

        # ── Step 4: Gemini 3 Flash analysis (Replicate) ──
        cycle_language = None
        if cycle_ok:
            print_pipeline_progress(4)
            cycle_language = random.choice(ANALYSIS_LANGUAGES)
            console.print(f"  [bold magenta]🌐 出力言語: {cycle_language['name']}[/]")
            log.info(f"[Language] 出力言語: {cycle_language['name']}")
            if not is_valid_file(backview_path):
                log.error("Step 4: backview画像が見つかりません")
                console.print("  [red]✗ backview画像が見つかりません[/]")
                new_nouns = None
            else:
                if GEMINI_API_KEY:
                    new_nouns = analyze_image_gemini_direct(backview_path, negations, language=cycle_language)
                else:
                    new_nouns = analyze_image_gemini(backview_path, negations, language=cycle_language)
        else:
            new_nouns = None

        if not cycle_ok or not new_nouns:
            consecutive_failures += 1
            log.error(f"Cycle {iteration:03d} 失敗 (連続失敗: {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})")
            print_cycle_result(iteration, [], False)
            # 失敗サイクルのコストも表示
            _print_cycle_costs(throttle.get_cycle_costs(), iteration, success=False)
            console.print(f"  [dim]連続失敗: {consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}[/]")

            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                console.print(f"\n[bold red]✗ {MAX_CONSECUTIVE_FAILURES}回連続で失敗しました。安全のため停止します。[/]")
                console.print("[dim]outputs/hallucination_loop.log を確認してください。[/]")
                log.error(f"{MAX_CONSECUTIVE_FAILURES}回連続で失敗しました。安全のため停止します。")
                break

            # ★ 部分成功ファイルは保持（次回のAPIコール節約のため）
            # mutated.png や mesh.glb が生成済みなら削除しない
            for partial_file in ["frontview.png", "backview.png"]:
                partial_path = iter_dir / partial_file
                if partial_path.exists():
                    partial_path.unlink(missing_ok=True)
            backoff_delay = min(RETRY_BASE_DELAY * (2 ** (consecutive_failures - 1)), 120)
            console.print(f"  [dim]{backoff_delay}秒後にリトライ...[/]")
            time.sleep(backoff_delay)
            continue

        # ── Success ──
        consecutive_failures = 0

        # サイクルコスト取得
        cycle_cost_info = throttle.get_cycle_costs()
        cycle_cost_value = cycle_cost_info["total_cost"] if cycle_cost_info else None

        history.append({
            "iteration": iteration,
            "new_nouns": new_nouns,
            "neg_count": len(negations),
            "model_id": current_model_id,
            "language": cycle_language["name"] if cycle_language else "English",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "cost": cycle_cost_value,
            "predict_time": cycle_cost_info["total_predict_time"] if cycle_cost_info else None,
            "prediction_count": cycle_cost_info["count"] if cycle_cost_info else None,
        })

        negations.extend(new_nouns)
        negations = list(dict.fromkeys(negations))  # Deduplicate, preserve order
        save_negations(negations)
        save_history(history)
        update_dashboard(history, negations)

        print_cycle_result(iteration, new_nouns, True)
        _print_cycle_costs(cycle_cost_info, iteration, success=True)
        log.info(f"Cycle {iteration:03d} 完了 → {', '.join(new_nouns)}")

        current_nouns = new_nouns
        iteration += 1

    console.print()
    console.print(Panel(
        "[bold white]HALLUCINATION LOOP 終了[/]",
        border_style="bright_blue",
        padding=(0, 4),
    ), justify="center")
    console.print()
    log.info("HALLUCINATION LOOP 終了。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Ctrl+C で中断されました。[/]")
        log.info("Ctrl+C で中断されました。")
        sys.exit(0)
    except Exception as e:
        log.critical(f"予期しないエラーでクラッシュしました: {e}", exc_info=True)
        console.print(f"\n[bold red]✗ 予期しないエラー: {e}[/]")
        console.print("[dim]outputs/hallucination_loop.log を確認してください。[/]")
        sys.exit(1)
