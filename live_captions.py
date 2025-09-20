#!/usr/bin/env python3
# live_captions.py — stable, append-only, two-line captions (no left ellipsis)
# Run: python3 live_captions.py /path/to/video.mp4
#
# Requires: ffmpeg/ffplay on PATH, pip install faster-whisper numpy

import argparse, os, sys, shutil, subprocess, time, signal, logging, tempfile, glob, atexit, textwrap
from typing import Optional, List
import numpy as np

# ===== knobs =====
AUDIO_RATE = 16000
WIN_SEC = 6.0
CHUNK_SEC = 1.0
UPDATE_SEC = 0.75
CAPTION_WRITE_DELAY = 0.05

# wrapping
WRAP_COLS = 42
MAX_LINES = 2 # hard cap: never more than two lines on screen

# audio preproc / silence gating
PREBUFFER_SEC = 0.7
RMS_WINDOW_SEC = 0.6
RMS_SILENCE_LEVEL = 0.008

# phrase gating
SILENCE_COMMIT_SEC = 1.3 # show partial after brief pause
SILENCE_CLEAR_SEC = 1.6 # clear after longer pause
TAIL_SEC = 3.0 # join only segments that ended within this window

# text length guards
MIN_CHARS = 8
MIN_WORDS_COMPLETE = 6
MIN_WORDS_PARTIAL = 7

# ASR model
TERMINAL_PUNCT = (".","!","?")
MODEL_NAME = "tiny" # try "base.en" for steadier English
COMPUTE_TYPE = "auto"

# UI
SHOW_ELLIPSIS = False
LEFT_MARGIN = 80
BOTTOM_MARGIN = 160

# behavior
APPEND_ONLY_PARTIALS = False # CHANGED: Set to False to prevent appending

FONT_DIRS = [
    "/Library/Fonts","/System/Library/Fonts","/System/Library/Fonts/Supplemental",
    os.path.expanduser("~/Library/Fonts"),
]
FONT_PREF = ["Arial Unicode MS","Arial Unicode","Arial","Helvetica","San Francisco","SFNS"]
# =================

player_proc: Optional[subprocess.Popen] = None
ffmpeg_proc: Optional[subprocess.Popen] = None

# ---------- plumbing ----------
def log_setup():
    fmt = "%(asctime)s.%(msecs)03d %(levelname)s | %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%H:%M:%S")

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

def start_ffplay(video_path: str, caption_path: str, font_file: Optional[str]):
    global player_proc
    drawtext = build_drawtext(caption_path, font_file)
    cmd = [
        "ffplay","-hide_banner","-loglevel","warning",
        "-threads","2",
        "-vf",f"{drawtext},fps=24",
        "-bufsize","1024k",
        video_path
    ]
    logging.info("Starting ffplay")
    # Avoid pipe back-pressure: send stdio to DEVNULL
    player_proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL, close_fds=True
    )
    return player_proc

def start_ffmpeg_audio_pipe(video_path: str):
    global ffmpeg_proc
    cmd = [
        "ffmpeg","-hide_banner","-loglevel","warning","-nostdin","-re",
        "-i",video_path,"-vn","-ac","1","-ar",str(AUDIO_RATE),
        "-af","dynaudnorm=f=150:g=7","-f","s16le","-"
    ]
    logging.info("Starting ffmpeg audio (realtime -re, 16k mono + dynaudnorm)")
    ffmpeg_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    return ffmpeg_proc

def write_caption(path: str, text: str):
    # Keep newlines — real multi-line rendering.
    # Modified to handle empty text payload for clearing the screen
    payload = text if text else ("…" if SHOW_ELLIPSIS else " ") if SHOW_ELLIPSIS else " "
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, path) # atomic swap
        time.sleep(CAPTION_WRITE_DELAY) # give ffplay time to reload
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
                proc.terminate()
                proc.wait(timeout=2)
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

# ---------- helpers ----------
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
    """
    Wrap to WRAP_COLS, then clamp to MAX_LINES lines (no left-ellipsis).
    If text grows past two lines, we *hold* at two lines (don’t drop leading words).
    """
    s = " ".join(s.split())
    lines = textwrap.wrap(s, width=WRAP_COLS, break_long_words=False, break_on_hyphens=False)
    if not lines:
        return s
    if len(lines) <= MAX_LINES:
        return "\n".join(lines)
    # clamp to first MAX_LINES lines; no left chopping, no ellipsis
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
    return joined # full sentence (we’ll wrap but never chop the left)

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

# ---------- main loop ----------
def transcribe_loop(video_path: str, caption_path: str):
    try:
        from faster_whisper import WhisperModel
    except ModuleNotFoundError:
        logging.error("Missing 'faster_whisper'. Install:\n %s -m pip install faster-whisper numpy", sys.executable)
        sys.exit(1)

    logging.info("Loading Faster-Whisper: %s (compute=%s)", MODEL_NAME, COMPUTE_TYPE)
    t0 = time.time()
    model = WhisperModel(MODEL_NAME, compute_type=COMPUTE_TYPE)
    logging.info("Model loaded in %.2fs", time.time()-t0)

    ff = start_ffmpeg_audio_pipe(video_path)
    ring = b""
    bps = AUDIO_RATE * 2
    max_bytes = int(bps * WIN_SEC)

    last_commit_at = 0.0
    last_voice_at = time.time()
    first_audio_at = None

    onscreen_raw = "" # unwrapped text we *intend* to show (for append-only check)
    hold_full = False # true when two-line area is full and we’re waiting for sentence end

    write_caption(caption_path, "")
    logging.info("Caption file: %s", caption_path)

    updates = 0
    t_start = time.time()

    try:
        while True:
            tail_stream("ffmpeg.stderr", ff.stderr, level=logging.INFO)

            chunk = ff.stdout.read(int(bps * CHUNK_SEC)) if ff.stdout else b""
            if not chunk:
                if ff.poll() is not None:
                    logging.info("ffmpeg exited code=%s", ff.returncode)
                    break
                time.sleep(0.03)
                continue

            if first_audio_at is None:
                first_audio_at = time.time()

            ring = (ring + chunk)[-max_bytes:]
            i16 = np.frombuffer(ring, dtype=np.int16)

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
                    language=None, # set to "en" for steadier English if desired
                    vad_filter=False,
                    beam_size=1,
                    # condition_on_previous_text=False, # NOTE: We keep this False for non-appending behavior
                )
                segs = list(segs_iter)
            except Exception as e:
                logging.error("Transcription error: %s", e)
                continue

            completed = latest_completed_phrase(segs)
            partial = "" if completed else best_partial_phrase(segs)
            silent_for = now - last_voice_at

            candidate = ""
            if completed:
                # Reset any hold and show the full completed phrase (wrapped)
                hold_full = False
                onscreen_raw = completed
                candidate = completed
            elif partial and silent_for >= SILENCE_COMMIT_SEC:
                # The core fix: always replace old partial with new one
                if not hold_full:
                    onscreen_raw = partial
                    candidate = partial
            else:
                # maybe clear if long silence
                if silent_for >= SILENCE_CLEAR_SEC and onscreen_raw:
                    onscreen_raw = ""
                    candidate = "" # will clear below

            # If nothing to write, continue
            if candidate == "" and onscreen_raw == "":
                blank = "" if not SHOW_ELLIPSIS else "…"
                write_caption(caption_path, blank)
                last_commit_at = now
                continue

            # Wrap and clamp to two lines.
            wrapped = wrap_two_lines(onscreen_raw)

            # If the wrapped text already uses two lines and adding more words would overflow,
            # we enter "hold_full" to freeze until the sentence completes.
            lines = wrapped.split("\n")
            if len(lines) >= MAX_LINES and not completed:
                hold_full = True

            # Only write if different AND not too short
            if wrapped and len(wrapped.replace("\n", " ").strip()) >= MIN_CHARS:
                write_caption(caption_path, wrapped)
                updates += 1
                logging.info("Committed #%d | '%s%s'",
                             updates, onscreen_raw[:60], "…" if len(onscreen_raw) > 60 else "")
            else:
                # Clear the display if the candidate text is too short after wrapping
                write_caption(caption_path, "")
            last_commit_at = now

    finally:
        logging.info("Loop end | updates=%d | elapsed=%.2fs", updates, time.time()-t_start)

# ---------- entry ----------
def main():
    log_setup()
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("video", nargs=1)
    args = ap.parse_args()
    video = args.video[0]

    logging.info("==== Live Captions starting ====")
    logging.info("Video: %s", video)
    logging.info("Params: model=%s compute=%s win=%.1fs chunk=%.1fs update=%.2fs",
                 MODEL_NAME, COMPUTE_TYPE, WIN_SEC, CHUNK_SEC, UPDATE_SEC)

    if not os.path.isfile(video): sys.exit(f"Video not found: {video}")
    if not shutil.which("ffmpeg") or not shutil.which("ffplay"):
        sys.exit("Install ffmpeg (with ffplay) and ensure it's on PATH")

    caption_path = os.path.join(tempfile.gettempdir(), "live_caption.txt")
    write_caption(caption_path, "")

    font_file = pick_font_file()
    start_ffplay(video, caption_path, font_file)

    try:
        transcribe_loop(video, caption_path)
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