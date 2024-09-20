import subprocess
import sys

def speedup_video(input_video, output_video, speed_factor):
    if speed_factor <= 0:
        print("Speed factor must be greater than 0.")
        return

    # FFmpeg command to speed up video and keep the same pitch for audio
    video_speed_filter = f"setpts={1/speed_factor}*PTS"
    audio_speed_filter = f"atempo={min(speed_factor, 2)}"
    
    # For speeds > 2x, we chain the `atempo` filter
    if speed_factor > 2:
        audio_speed_filter = f"atempo=2,atempo={speed_factor/2}"

    command = [
        'ffmpeg', '-i', input_video,
        '-filter_complex', f"[0:v]{video_speed_filter}[v];[0:a]{audio_speed_filter}[a]",
        '-map', '[v]', '-map', '[a]',
        '-y', output_video
    ]
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"Video successfully sped up to {speed_factor}x speed and saved to {output_video}.")
    else:
        print(f"Error speeding up video: {result.stderr}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python speedup_video.py <input_video> <output_video> <speed_factor>")
        sys.exit(1)
    
    input_video = sys.argv[1]
    output_video = sys.argv[2]
    speed_factor = float(sys.argv[3])

    speedup_video(input_video, output_video, speed_factor)
