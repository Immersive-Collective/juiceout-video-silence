import subprocess
import numpy as np
import librosa
import os
import sys

def analyze_silence(input_video, silence_threshold=-40.0, min_silence_duration=2.0):
    audio_file = "temp_audio.wav"
    
    # Extract audio using FFmpeg
    command = ['ffmpeg', '-i', input_video, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', audio_file, '-y']
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Check if the audio extraction succeeded
    if result.returncode != 0:
        print(f"FFmpeg failed: {result.stderr}")
        return []
    
    # Ensure the audio file was created
    if not os.path.exists(audio_file):
        print("Audio extraction failed. 'temp_audio.wav' was not created.")
        return []

    # Load the extracted audio
    audio, sr = librosa.load(audio_file, sr=44100)
    
    # Compute the Root Mean Square (RMS) energy
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=512)

    silence_times = []
    silence_start = None

    for t, value in zip(times, rms):
        if librosa.amplitude_to_db(value, ref=np.max(rms)) < silence_threshold:
            if silence_start is None:
                silence_start = t
        else:
            if silence_start is not None:
                if t - silence_start >= min_silence_duration:
                    silence_times.append((silence_start, t))
                silence_start = None

    if silence_start is not None and len(audio)/sr - silence_start >= min_silence_duration:
        silence_times.append((silence_start, len(audio)/sr))

    # Cleanup
    os.remove(audio_file)
    
    return silence_times, len(audio)/sr

def chop_segments(input_video, output_video, silence_segments, total_duration, mode="talk"):
    if not silence_segments:
        print("No segments found or analysis failed.")
        return
    
    input_streams = ['-i', input_video]
    filter_complex = []

    if mode == "talk":
        # Generate non-silence (talk) segments
        non_silent_segments = []
        if silence_segments[0][0] > 0:  # If the video starts with a talking part
            non_silent_segments.append((0, silence_segments[0][0]))
        
        for i in range(1, len(silence_segments)):
            non_silent_segments.append((silence_segments[i-1][1], silence_segments[i][0]))
        
        if silence_segments[-1][1] < total_duration:  # If the video ends with a talking part
            non_silent_segments.append((silence_segments[-1][1], total_duration))

        for i, (start, end) in enumerate(non_silent_segments):
            filter_complex.append(f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{i}];"
                                  f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}]")
    else:
        # Use silence segments directly
        for i, (start, end) in enumerate(silence_segments):
            filter_complex.append(f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{i}];"
                                  f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}]")

    filter_complex_str = ';'.join(filter_complex) + ';' + ''.join(f"[v{i}][a{i}]" for i in range(len(filter_complex))) + f"concat=n={len(filter_complex)}:v=1:a=1[v][a]"

    # Run FFmpeg to chop the segments
    subprocess.call(['ffmpeg', *input_streams, '-filter_complex', filter_complex_str, '-map', '[v]', '-map', '[a]', output_video, '-y'])

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python chop_silence.py <input_video> <output_video> <silence_threshold> <min_silence_duration> <mode>")
        print("Mode options: 'talk' (keep talking parts) or 'silence' (keep silent parts)")
        sys.exit(1)
    
    input_video = sys.argv[1]
    output_video = sys.argv[2]
    silence_threshold = float(sys.argv[3]) if len(sys.argv) > 3 else -40.0
    min_silence_duration = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
    mode = sys.argv[5] if len(sys.argv) > 5 else "talk"

    # Analyze silence segments
    silence_segments, total_duration = analyze_silence(input_video, silence_threshold, min_silence_duration)

    # Chop out either silence or non-silence segments from the video
    chop_segments(input_video, output_video, silence_segments, total_duration, mode=mode)
