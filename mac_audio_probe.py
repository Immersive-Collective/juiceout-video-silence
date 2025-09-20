#!/usr/bin/env python3
# mac_audio_probe.py — find which AVFoundation audio device actually has signal (macOS)
# Usage:
#   python3 mac_audio_probe.py --list
#   python3 mac_audio_probe.py --probe                 # probe all audio devices
#   python3 mac_audio_probe.py --probe --seconds 3.0   # probe longer
#
# Requires: ffmpeg on PATH, Python 3.8+, numpy

import subprocess, sys, re, time, argparse, shutil, math
import numpy as np

AUDIO_RATE = 16000
CHANNELS = 1
FMT = "s16le"

DEVICE_LINE_RE = re.compile(r'\[\s*(\d+)\s*\]\s*(.+)')

def have_ffmpeg():
    return bool(shutil.which("ffmpeg"))

def list_avfoundation_devices_raw():
    cmd = ["ffmpeg","-hide_banner","-f","avfoundation","-list_devices","true","-i",""]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.stderr or p.stdout

def parse_devices(raw):
    """
    Returns: dict with 'video': [(idx, name), ...], 'audio': [(idx, name), ...]
    """
    video, audio = [], []
    section = None
    for line in (raw or "").splitlines():
        if "AVFoundation video devices" in line:
            section = "video"; continue
        if "AVFoundation audio devices" in line:
            section = "audio"; continue
        m = DEVICE_LINE_RE.search(line)
        if m and section in ("video","audio"):
            idx = int(m.group(1))
            name = m.group(2).strip()
            (video if section=="video" else audio).append((idx, name))
    return {"video": video, "audio": audio}

def rms_dbfs(x):
    if x.size == 0: return -120.0
    rms = max(1e-9, float(np.sqrt(np.mean(np.square(x)))))
    # assuming int16 full-scale +/-32768
    db = 20.0 * math.log10(rms / 32768.0)
    return db

def start_ffmpeg_audio_pipe(audio_idx):
    av_in = f":{audio_idx}"
    cmd = [
        "ffmpeg","-hide_banner","-loglevel","warning","-nostdin",
        "-f","avfoundation","-i", av_in,
        "-ac", str(CHANNELS), "-ar", str(AUDIO_RATE),
        "-f", FMT, "-"  # raw PCM
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)

def vu_bar(db, width=40):
    # scale db from -60..0 to 0..width
    span = 60.0
    v = max(0.0, min(1.0, 1.0 - (-(db))/span))
    n = int(round(v * width))
    return "[" + "#"*n + "-"*(width-n) + "]"

def probe_device(idx, seconds=2.0):
    p = start_ffmpeg_audio_pipe(idx)
    bps = AUDIO_RATE * 2  # s16le mono
    chunk = int(bps * 0.1)  # 100ms windows
    t_end = time.time() + seconds
    all_rms = []
    peak_db = -120.0
    try:
        while time.time() < t_end:
            data = p.stdout.read(chunk) if p.stdout else b""
            if not data:
                if p.poll() is not None: break
                time.sleep(0.02); continue
            x = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            db = rms_dbfs(x)
            all_rms.append(db)
            if db > peak_db: peak_db = db
            # live bar
            print(f"\r  VU  {vu_bar(db)}  {db:6.1f} dBFS", end="", flush=True)
        print()
    finally:
        try:
            p.terminate(); p.wait(timeout=1)
        except Exception:
            try: p.kill()
            except Exception: pass

    if all_rms:
        avg_db = sum(all_rms)/len(all_rms)
    else:
        avg_db = -120.0
    return avg_db, peak_db

def main():
    ap = argparse.ArgumentParser(description="Probe AVFoundation audio devices on macOS")
    ap.add_argument("--list", action="store_true", help="List devices and exit")
    ap.add_argument("--probe", action="store_true", help="Probe all audio devices")
    ap.add_argument("--seconds", type=float, default=2.0, help="Seconds to probe each device (default 2.0)")
    args = ap.parse_args()

    if not have_ffmpeg():
        sys.exit("ffmpeg not found on PATH. Install ffmpeg first.")

    raw = list_avfoundation_devices_raw()
    devs = parse_devices(raw)

    print("== AVFoundation devices ==")
    print("Video:")
    for i, name in devs["video"]:
        print(f"  [{i}] {name}")
    print("Audio:")
    for i, name in devs["audio"]:
        print(f"  [{i}] {name}")

    if args.list and not args.probe:
        return

    if not devs["audio"]:
        print("\nNo audio devices found.")
        return

    if args.probe:
        print("\n== Probing audio devices ==")
        results = []
        for i, name in devs["audio"]:
            print(f"\nDevice [{i}] {name}")
            avg, peak = probe_device(i, args.seconds)
            print(f"  avg: {avg:6.1f} dBFS   peak: {peak:6.1f} dBFS")
            results.append((i, name, avg, peak))

        # rank by avg dB
        results.sort(key=lambda t: t[2], reverse=True)
        print("\n== Ranking by average level ==")
        for i, name, avg, peak in results:
            print(f"  [{i}] {name:40s}  avg {avg:6.1f} dBFS   peak {peak:6.1f} dBFS")

        if results:
            best = results[0]
            print(f"\nSuggested audio index: [{best[0]}] {best[1]} (avg {best[2]:.1f} dBFS)")

if __name__ == "__main__":
    main()
