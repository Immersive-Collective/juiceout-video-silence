#!/usr/bin/env python3
# webcam_captions_gpt.py — left-anchored, two-line live captions from macOS webcam/mic (avfoundation)
# Requires: ffmpeg/ffplay on PATH, pip install faster-whisper numpy

import argparse, os, sys, shutil, subprocess, time, signal, logging, tempfile, glob, atexit, textwrap, re, math, datetime
from typing import Optional, List
import numpy as np

# ===== knobs =====
AUDIO_RATE = 16000
WIN_SEC = 6.0
CHUNK_SEC = 1.0
UPDATE_SEC = 0.75
CAPTION_WRITE_DELAY = 0.05

WRAP_COLS = 42
MAX_LINES = 2

PREBUFFER_SEC = 0.7
RMS_WINDOW_SEC = 0.6
RMS_SILENCE_LEVEL = 0.008

SILENCE_COMMIT_SEC = 1.3
SILENCE_CLEAR_SEC  = 1.6
TAIL_SEC = 3.0

MIN_CHARS = 8
MIN_WORDS_COMPLETE = 6
MIN_WORDS_PARTIAL  = 7

# transcript de-dup threshold
MIN_LOG_GROWTH_CHARS = 6
MIN_LOG_GROWTH_WORDS = 1

TERMINAL_PUNCT = (".","!","?")
DEFAULT_MODEL_NAME = "tiny"
COMPUTE_TYPE = "auto"

SHOW_ELLIPSIS = False
LEFT_MARGIN = 80
BOTTOM_MARGIN = 160

# Partials must be monotonic-append of prior partial
APPEND_ONLY_PARTIALS = True

DEFAULT_FPS = 30
DEFAULT_SIZE = "1280x720"

FONT_DIRS = [
    "/Library/Fonts","/System/Library/Fonts","/System/Library/Fonts/Supplemental",
    os.path.expanduser("~/Library/Fonts"),
]
FONT_PREF = ["Arial Unicode MS","Arial Unicode","Arial","Helvetica","San Francisco","SFNS"]

player_proc: Optional[subprocess.Popen] = None
ffmpeg_proc: Optional[subprocess.Popen] = None

DEVICE_LINE_RE = re.compile(r'\[\s*(\d+)\s*\]\s*(.+)')

# ---------- time & files ----------
def ts_for_filename():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def human_ts():
    return datetime.datetime.now().strftime("%H:%M:%S")

def write_line(path: str, line: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ---------- ANSI & logging ----------
ANSI_RED = "\033[31m"
ANSI_YEL = "\033[33m"
ANSI_RESET = "\033[0m"

class HitExitColorFilter(logging.Filter):
    """Colorize only HIT:/EXIT: on console."""
    def filter(self, record):
        if isinstance(record.msg, str):
            if record.msg.startswith("HIT: "):
                record.msg = f"{ANSI_RED}{record.msg}{ANSI_RESET}"
            elif record.msg.startswith("EXIT: "):
                record.msg = f"{ANSI_YEL}{record.msg}{ANSI_RESET}"
        return True

def log_setup(file_path: Optional[str], color_console: bool=True):
    fmt = "%(asctime)s.%(msecs)03d %(levelname)s | %(message)s"
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    if file_path:
        fh = logging.FileHandler(file_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
        root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    if color_console:
        ch.addFilter(HitExitColorFilter())
    root.addHandler(ch)

def emit_colored_line(prefix: str, s: str, vu_active: bool, color: str):
    if vu_active:
        print()  # newline before colored message to avoid VU overwrite
    print(f"{color}{prefix}: {s}{ANSI_RESET}", flush=True)

# ---------- ffmpeg plumbing ----------
def esc_drawtext(p: str) -> str:
    return p.replace("\\","\\\\").replace(":","\\:").replace("'","\\'").replace("%","\\%")

def list_fonts() -> List[str]:
    out=[]
    for d in FONT_DIRS:
        if not os.path.isdir(d): continue
        for pat in ("*.ttf","*.otf","*.ttc","*.dfont"):
            out += glob.glob(os.path.join(d, pat))
    return sorted(set(out))

def pick_font_file() -> Optional[str]:
    fonts = list_fonts()
    for pref in FONT_PREF:
        for f in fonts:
            if pref.lower().replace(" ","") in os.path.basename(f).lower().replace(" ",""):
                logging.info("Overlay font: %s", f); return f
    if fonts:
        logging.info("Overlay font (fallback): %s", fonts[0]); return fonts[0]
    logging.warning("No fonts found; ffmpeg default will be used.")
    return None

def build_drawtext(caption_path: str, font_file: Optional[str]) -> str:
    parts = ["drawtext="]
    if font_file: parts.append(f"fontfile={esc_drawtext(font_file)}:")
    parts.append(f"textfile={esc_drawtext(caption_path)}:reload=1")
    parts.append(":fontcolor=white:fontsize=48")
    parts.append(f":x={LEFT_MARGIN}:y=h-{BOTTOM_MARGIN}")
    parts.append(":box=1:boxcolor=black@0.45:boxborderw=20")
    parts.append(":line_spacing=6")
    parts.append(":fix_bounds=1")
    return "".join(parts)

def list_avfoundation_devices_raw():
    cmd = ["ffmpeg","-hide_banner","-f","avfoundation","-list_devices","true","-i",""]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.stderr or p.stdout

def parse_devices(raw):
    video, audio = [], []
    section = None
    for line in (raw or "").splitlines():
        if "AVFoundation video devices" in line:
            section = "video"; continue
        if "AVFoundation audio devices" in line:
            section = "audio"; continue
        m = DEVICE_LINE_RE.search(line)
        if m and section in ("video","audio"):
            idx = int(m.group(1)); name = m.group(2).strip()
            (video if section=="video" else audio).append((idx, name))
    return {"video": video, "audio": audio}

def start_ffplay_webcam(caption_path: str, font_file: Optional[str],
                        video_idx: int, audio_idx: int, fps: int, size: str):
    global player_proc
    drawtext = build_drawtext(caption_path, font_file)
    av_in = f"{video_idx}:{audio_idx}"
    cmd = [
        "ffplay","-hide_banner","-loglevel","warning",
        "-f","avfoundation","-framerate", str(fps), "-video_size", size,
        "-i", av_in, "-threads","2", "-vf", f"{drawtext},fps=24", "-bufsize","1024k"
    ]
    logging.info("Starting ffplay (webcam)")
    player_proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL, close_fds=True
    )
    return player_proc

def start_ffmpeg_audio_pipe_mic(audio_idx: int):
    global ffmpeg_proc
    av_in = f":{audio_idx}"
    cmd = [
        "ffmpeg","-hide_banner","-loglevel","warning","-nostdin",
        "-f","avfoundation","-i", av_in,
        "-ac","1","-ar",str(AUDIO_RATE),
        "-af","dynaudnorm=f=150:g=7",
        "-f","s16le","-"
    ]
    logging.info("Starting ffmpeg (mic→PCM pipe) via avfoundation, audio=%s", audio_idx)
    ffmpeg_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    return ffmpeg_proc

def write_caption(path: str, text: str):
    payload = text if text else ("…" if SHOW_ELLIPSIS else " ")
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, path)
        time.sleep(CAPTION_WRITE_DELAY)
    except Exception as e:
        logging.error("Failed to write caption: %s", e)

def tail_stream(tag: str, stream, level=logging.INFO):
    try:
        if stream and not stream.closed:
            avail = stream.peek() if hasattr(stream,"peek") else b""
            if avail:
                data = stream.read(len(avail))
                if data:
                    for line in data.decode(errors="replace").splitlines():
                        logging.log(level, "%s: %s", tag, line)
    except Exception:
        pass

def stop_children():
    for name,proc in (("ffmpeg",ffmpeg_proc),("ffplay",player_proc)):
        if proc and proc.poll() is None:
            logging.info("Stopping %s…", name)
            try:
                proc.terminate(); proc.wait(timeout=2)
            except:
                try: proc.kill()
                except: pass

def handle_signal(sig, frm):
    logging.warning("Signal %s — shutting down", sig)
    stop_children()
    sys.exit(130 if sig==signal.SIGINT else 143)

atexit.register(stop_children)
signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

# ---------- text helpers ----------
def is_silent(audio_f32: np.ndarray) -> bool:
    if audio_f32.size == 0: return True
    rms = np.sqrt(np.mean(np.square(audio_f32)))
    return rms < RMS_SILENCE_LEVEL

def last_seconds_i16(full_i16: np.ndarray, sec: float) -> np.ndarray:
    n = int(AUDIO_RATE * sec)
    if n <= 0 or full_i16.size == 0: return full_i16[-0:]
    if full_i16.size <= n: return full_i16
    return full_i16[-n:]

def wrap_two_lines(s: str) -> str:
    s = " ".join(s.split())
    lines = textwrap.wrap(s, width=WRAP_COLS, break_long_words=False, break_on_hyphens=False)
    if not lines: return s
    if len(lines) <= MAX_LINES: return "\n".join(lines)
    return "\n".join(lines[:MAX_LINES])

def _clean_text(s: str) -> str:
    return " ".join(s.split())

def _join_tail_segments(segments, tail_sec: float) -> str:
    if not segments: return ""
    ends = [getattr(s,"end",None) for s in segments if getattr(s,"end",None) is not None]
    if not ends: return ""
    last_end = max(ends)
    cutoff = last_end - tail_sec
    tail = [s for s in segments
            if getattr(s,"end",None) is not None and s.end >= cutoff and getattr(s,"text","").strip()]
    if not tail: return ""
    tail.sort(key=lambda s: (getattr(s,"start",0.0) or 0.0))
    return _clean_text(" ".join(_clean_text(getattr(s,"text","")) for s in tail))

def latest_completed_phrase(segments) -> str:
    if not segments: return ""
    last_i = -1
    for i,s in enumerate(segments):
        t = _clean_text(getattr(s,"text",""))
        if t and t.endswith(TERMINAL_PUNCT): last_i = i
    if last_i == -1: return ""
    joined = _join_tail_segments(segments[:last_i+1], TAIL_SEC) or _clean_text(getattr(segments[last_i],"text",""))
    if len(joined.split()) < MIN_WORDS_COMPLETE or len(joined) < MIN_CHARS: return ""
    return joined

def best_partial_phrase(segments) -> str:
    if not segments: return ""
    last_done = -1
    for i,s in enumerate(segments):
        t = _clean_text(getattr(s,"text",""))
        if t.endswith(TERMINAL_PUNCT): last_done = i
    tail = segments[last_done+1:] if last_done + 1 < len(segments) else segments
    if not tail: return ""
    joined = _join_tail_segments(tail, TAIL_SEC)
    if not joined:
        non_empty = [_clean_text(getattr(s,"text","")) for s in tail if getattr(s,"text","").strip()]
        joined = non_empty[-1] if non_empty else ""
    if len(joined.split()) < MIN_WORDS_PARTIAL or len(joined) < MIN_CHARS: return ""
    return joined

# ---------- audio metrics ----------
def rms_dbfs_from_i16(x: np.ndarray) -> float:
    if x.size == 0: return -120.0
    rms = max(1e-9, float(np.sqrt(np.mean(np.square(x.astype(np.float32))))))
    return 20.0 * math.log10(rms / 32768.0)

def vu_bar(db, width=40):
    span = 60.0
    v = max(0.0, min(1.0, 1.0 - (-(db))/span))
    n = int(round(v * width))
    return "[" + "#"*n + "-"*(width-n) + f"]  {db:7.1f} dBFS"

# ---------- ASR loop ----------
def transcribe_loop(audio_idx: int, caption_path: str, force_lang: Optional[str],
                    model_name: str, show_vu: bool, kw_regex: Optional[re.Pattern],
                    txt_path: str, srt_path: Optional[str], completed_only: bool,
                    exit_regex: Optional[re.Pattern], exit_delay: float):
    try:
        from faster_whisper import WhisperModel
    except ModuleNotFoundError:
        logging.error("Missing 'faster_whisper'. Install:\n  %s -m pip install faster-whisper numpy", sys.executable)
        sys.exit(1)

    logging.info("Loading Faster-Whisper: %s (compute=%s)", model_name, COMPUTE_TYPE)
    t0 = time.time()
    model = WhisperModel(model_name, compute_type=COMPUTE_TYPE)
    logging.info("Model loaded in %.2fs", time.time()-t0)

    ff = start_ffmpeg_audio_pipe_mic(audio_idx)
    ring = b""
    bps = AUDIO_RATE * 2
    max_bytes = int(bps * WIN_SEC)

    last_commit_at = 0.0
    last_voice_at = time.time()
    first_audio_at = None

    onscreen_raw = ""
    hold_full = False

    # transcript / srt state
    session_t0 = time.time()
    seg_start_t = session_t0
    srt_index = 1
    last_flat_for_transcript = ""

    write_caption(caption_path, "")
    logging.info("Caption file: %s", caption_path)
    write_line(txt_path, f"# Transcript started {datetime.datetime.now().isoformat(timespec='seconds')}")

    try:
        while True:
            tail_stream("ffmpeg.stderr", ff.stderr, level=logging.INFO)

            chunk = ff.stdout.read(int(bps * CHUNK_SEC)) if ff.stdout else b""
            if not chunk:
                if ff.poll() is not None:
                    logging.info("ffmpeg exited code=%s", ff.returncode)
                    break
                time.sleep(0.02)
                continue

            if first_audio_at is None:
                first_audio_at = time.time()

            ring = (ring + chunk)[-max_bytes:]
            i16 = np.frombuffer(ring, dtype=np.int16)

            # VU meter (100ms of newest audio)
            if show_vu:
                tail_i16_vu = i16[-int(AUDIO_RATE*0.1):] if i16.size else i16
                db = rms_dbfs_from_i16(tail_i16_vu)
                print("\rVU  " + vu_bar(db), end="", flush=True)
            else:
                db = None

            # silence tracking
            tail_i16 = last_seconds_i16(i16, RMS_WINDOW_SEC)
            tail_f32 = tail_i16.astype(np.float32) / 32768.0
            if not is_silent(tail_f32):
                last_voice_at = time.time()

            now = time.time()
            if now - last_commit_at < UPDATE_SEC: continue
            if first_audio_at and (now - first_audio_at) < PREBUFFER_SEC: continue

            f32 = i16.astype(np.float32) / 32768.0
            try:
                segs_iter, _ = model.transcribe(
                    f32,
                    language=force_lang,
                    vad_filter=False,
                    beam_size=1,
                    condition_on_previous_text=False,
                )
                segs = list(segs_iter)
            except Exception as e:
                logging.error("Transcription error: %s", e)
                continue

            completed = latest_completed_phrase(segs)
            partial   = "" if completed else best_partial_phrase(segs)
            silent_for = now - last_voice_at

            candidate = ""
            if completed:
                hold_full = False
                onscreen_raw = completed
                candidate = completed
            elif partial and not completed_only and silent_for >= SILENCE_COMMIT_SEC:
                if APPEND_ONLY_PARTIALS:
                    if (not onscreen_raw) or partial.startswith(onscreen_raw):
                        if not hold_full:
                            onscreen_raw = partial
                            candidate = partial
                else:
                    if not hold_full:
                        onscreen_raw = partial
                        candidate = partial
            else:
                if silent_for >= SILENCE_CLEAR_SEC and onscreen_raw:
                    onscreen_raw = ""
                    candidate = ""

            if candidate == "" and onscreen_raw == "":
                blank = "" if not SHOW_ELLIPSIS else "…"
                write_caption(caption_path, blank)
                last_commit_at = now
                continue

            wrapped = wrap_two_lines(onscreen_raw)
            lines = wrapped.split("\n")
            if len(lines) >= MAX_LINES and not completed:
                hold_full = True

            flat = wrapped.replace("\n"," ").strip()
            if not flat:
                last_commit_at = now
                continue

            # ----- EXIT phrase check (on the flat, printable text)
            if exit_regex and exit_regex.search(flat):
                emit_colored_line("EXIT", flat, show_vu, ANSI_YEL)
                logging.warning("EXIT: %s", flat)
                write_caption(caption_path, "Goodbye…")
                time.sleep(exit_delay)
                break

            # ----- Transcript/log write with strict de-dup
            meaningful = False
            if completed:
                meaningful = (flat != last_flat_for_transcript)
            else:
                # only log partials if they meaningfully extend previous
                if flat != last_flat_for_transcript:
                    # growth must be forward extension, not a reorder/regression
                    if flat.startswith(last_flat_for_transcript):
                        growth_chars = len(flat) - len(last_flat_for_transcript)
                        growth_words = max(0, len(flat.split()) - len(last_flat_for_transcript.split()))
                        meaningful = (growth_chars >= MIN_LOG_GROWTH_CHARS and growth_words >= MIN_LOG_GROWTH_WORDS)
                    else:
                        # regression or rewrite — skip to avoid spam
                        meaningful = False

            if meaningful:
                write_caption(caption_path, wrapped)

                loud = f" | {db:5.1f} dBFS" if db is not None else ""
                write_line(txt_path, f"[{human_ts()}] {flat}{loud}")

                if srt_path and completed:
                    t_start = seg_start_t - session_t0
                    t_end = time.time() - session_t0
                    def fmt(t):
                        h = int(t//3600); m = int((t%3600)//60); s = int(t%60); ms = int((t - int(t))*1000)
                        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
                    with open(srt_path, "a", encoding="utf-8") as sf:
                        sf.write(f"{srt_index}\n{fmt(t_start)} --> {fmt(t_end)}\n{wrapped}\n\n")
                    srt_index += 1
                    seg_start_t = time.time()

                logging.info("Committed | '%s%s'%s",
                             flat[:60], "…" if len(flat) > 60 else "",
                             f" | {db:5.1f} dBFS" if db is not None else "")

                if kw_regex and kw_regex.search(flat):
                    emit_colored_line("HIT", flat, show_vu, ANSI_RED)
                    logging.warning("HIT: %s", flat)

                last_flat_for_transcript = flat

                if completed:
                    seg_start_t = time.time()

            last_commit_at = now

    finally:
        logging.info("==== Transcription loop ended ====")
        # Clean up handled in main

# ---------- auto-pick audio ----------
def rms_probe(idx: int, seconds: float = 1.2) -> float:
    p = start_ffmpeg_audio_pipe_mic(idx)
    bps = AUDIO_RATE * 2
    chunk = int(bps * 0.1)
    t_end = time.time() + seconds
    vals = []
    try:
        while time.time() < t_end:
            data = p.stdout.read(chunk) if p.stdout else b""
            if not data:
                if p.poll() is not None: break
                time.sleep(0.02); continue
            x = np.frombuffer(data, dtype=np.int16)
            vals.append(rms_dbfs_from_i16(x))
    finally:
        try: p.terminate(); p.wait(timeout=1)
        except Exception:
            try: p.kill()
            except Exception: pass
    return (sum(vals)/len(vals)) if vals else -120.0

def pick_loudest_audio_idx() -> Optional[int]:
    raw = list_avoundation_devices_raw()
    devs = parse_devices(raw)
    candidates = [i for (i, _) in devs["audio"]]
    if not candidates: return None
    scores = []
    for idx in candidates:
        db = rms_probe(idx, seconds=1.0)
        logging.info("Audio idx %d avg level: %.1f dBFS", idx, db)
        scores.append((db, idx))
    scores.sort(reverse=True)
    return scores[0][1]

# ---------- CLI helpers ----------
def build_kw_regex(kw: Optional[str]) -> Optional[re.Pattern]:
    if not kw: return None
    parts = [p.strip() for p in kw.split(",") if p.strip()]
    if not parts: return None
    pats = []
    for p in parts:
        if re.fullmatch(r"[0-9A-Za-zÀ-ž_]+", p):
            pats.append(rf"\b{re.escape(p)}\b")
        else:
            pats.append(re.escape(p))
    try:
        return re.compile("(" + "|".join(pats) + ")", re.IGNORECASE)
    except re.error:
        logging.warning("Invalid --kw; disabling keyword highlighting.")
        return None

def build_exit_regex(exit_phrase: Optional[str]) -> Optional[re.Pattern]:
    if not exit_phrase: return None
    p = exit_phrase.strip()
    if not p: return None
    if re.fullmatch(r"[0-9A-Za-zÀ-ž_ ]+", p):
        pat = r"\b" + re.escape(p) + r"\b"
    else:
        pat = re.escape(p)
    return re.compile(pat, re.IGNORECASE)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Live captions over webcam (macOS avfoundation)")
    ap.add_argument("--list", action="store_true", help="List avfoundation devices and exit")
    ap.add_argument("--video", type=int, default=0, help="Video device index (default: 0)")
    ap.add_argument("--audio", default="0", help='Audio device index (e.g. "2") or "auto"')
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS, help=f"Webcam framerate (default: {DEFAULT_FPS})")
    ap.add_argument("--size", default=DEFAULT_SIZE, help=f"Webcam size WxH (default: {DEFAULT_SIZE})")
    ap.add_argument("--lang", default=None, help='Force ASR language (e.g., "en"); default: auto-detect')
    ap.add_argument("--model", default=DEFAULT_MODEL_NAME, help=f'Whisper model name (default: "{DEFAULT_MODEL_NAME}")')
    ap.add_argument("--vu", action="store_true", help="Show terminal VU meter while running")
    ap.add_argument("--kw", default=None, help='Comma-separated keywords to highlight (console red + log HIT)')
    ap.add_argument("--srt", action="store_true", help="Also save coarse SRT for completed sentences")
    ap.add_argument("--completed-only", action="store_true", help="Transcript only completed sentences (ignore partials)")
    ap.add_argument("--exit-phrase", default=None, help='Say this phrase to end (e.g. "GOOD BYE")')
    ap.add_argument("--exit-delay", type=float, default=1.0, help="Seconds to wait before closing after exit phrase")
    args = ap.parse_args()

    if args.list:
        print(list_avfoundation_devices_raw()); return

    if not shutil.which("ffmpeg") or not shutil.which("ffplay"):
        sys.exit("Install ffmpeg (with ffplay) and ensure it's on PATH")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    stamp = ts_for_filename()
    log_path = os.path.join(base_dir, f"webcam_captions-{stamp}.log")
    txt_path = os.path.join(base_dir, f"webcam_captions-{stamp}.txt")
    srt_path = os.path.join(base_dir, f"webcam_captions-{stamp}.srt") if args.srt else None

    log_setup(log_path, color_console=True)
    kw_regex = build_kw_regex(args.kw)
    exit_regex = build_exit_regex(args.exit_phrase)

    # resolve audio
    if str(args.audio).lower() == "auto":
        sel = pick_loudest_audio_idx()
        if sel is None: sys.exit("No audio devices found.")
        audio_idx = sel
    else:
        try:
            audio_idx = int(args.audio)
        except:
            sys.exit("Invalid --audio value. Use an integer index or 'auto'.")

    logging.info("==== Live Captions (webcam) starting ====")
    logging.info("Session files: log=%s, transcript=%s%s",
                 log_path, txt_path, f", srt={srt_path}" if srt_path else "")
    logging.info("Devices: video=%d audio=%d fps=%d size=%s", args.video, audio_idx, args.fps, args.size)
    logging.info("Params: model=%s compute=%s win=%.1fs chunk=%.1fs update=%.2fs",
                 args.model, COMPUTE_TYPE, WIN_SEC, CHUNK_SEC, UPDATE_SEC)
    if args.lang: logging.info("Forced ASR language: %s", args.lang)
    if args.kw:   logging.info("Keyword highlight: %s", args.kw)
    if args.completed_only: logging.info("Transcript mode: completed sentences only")
    if args.exit_phrase: logging.info("Exit phrase: %s (delay %.2fs)", args.exit_phrase, args.exit_delay)

    caption_path = os.path.join(tempfile.gettempdir(), "live_caption_webcam.txt")
    write_caption(caption_path, "")

    font_file = pick_font_file()
    start_ffplay_webcam(caption_path, font_file, args.video, audio_idx, args.fps, args.size)

    try:
        transcribe_loop(audio_idx, caption_path, args.lang, args.model, args.vu,
                        kw_regex, txt_path, srt_path, args.completed_only,
                        exit_regex, args.exit_delay)
    except Exception as e:
        logging.exception("Transcription crashed: %s", e)

    if ffmpeg_proc:
        tail_stream("ffmpeg.stderr", ffmpeg_proc.stderr, level=logging.INFO)
    stop_children()
    try: os.remove(caption_path)
    except: pass
    logging.info("==== Live Captions finished ====")

if __name__ == "__main__":
    main()
