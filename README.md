# Video Processing Scripts

This repository contains two Python scripts for video processing using FFmpeg:
1. **Silence Chopping Script**: Removes or retains silent or talking segments from a video.
2. **Video Speedup Script**: Speeds up a video without changing the pitch of the audio.

## Demo

### Original Video

Watch the original, unprocessed video:

[Original Video on YouTube](https://youtu.be/RkpzS6JO_Fs?feature=shared)

### Processed Video (`-40 0.4 talk`)

This version has been processed to remove silent parts with the parameters `-40 dB` silence threshold and `0.4 seconds` minimum silence duration, keeping only the talking segments:

[Processed Video (talking segments only)](https://github.com/user-attachments/assets/bbbfcbcc-fe59-468b-8d38-6d95ee890d2e)

### Sped-Up Video (1.5x)

This is the same video, sped up 1.5 times without changing the pitch of the audio:

[Sped-Up Video (1.5x)](https://github.com/user-attachments/assets/7c40a71f-d36b-47fa-b73d-4c8201460470)

## Requirements

- **Python 3.x**
- **FFmpeg** installed on your system and available in your `PATH`
- **Virtual Environment** (optional but recommended)

## Installation

### 1. Clone the Repository

First, clone this repository to your local machine:

```bash
git clone git@github.com:Immersive-Collective/juiceout-video-silence.git
cd juiceout-video-silence
```

### 2. Set Up a Virtual Environment (Optional but recommended)

To keep the dependencies isolated, you can use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # For macOS/Linux
# For Windows: venv\Scripts\activate
```


### 3.1 Install Dependencies from requirements

You can install all required dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### or 3.2 Install Dependencies

Install the required Python libraries using `pip`:

```bash
pip install numpy librosa
```

Ensure that **FFmpeg** is installed and available. You can check this by running:

```bash
ffmpeg -version
```

If FFmpeg is not installed, follow these instructions based on your operating system:

- **macOS** (via Homebrew):
  ```bash
  brew install ffmpeg
  ```
  
- **Ubuntu/Linux**:
  ```bash
  sudo apt update
  sudo apt install ffmpeg
  ```

- **Windows**: Download FFmpeg from [here](https://ffmpeg.org/download.html) and add it to your system `PATH`.

## Usage

### 1. Silence Chopping Script

This script allows you to either remove or retain silent or non-silent (talking) parts from a video.

#### Run the Script

```bash
python chop_silence.py <input_video> <output_video> <silence_threshold> <min_silence_duration> <mode>
```

#### Arguments

- `input_video`: Path to the input video file.
- `output_video`: Path where the output video will be saved.
- `silence_threshold`: Silence threshold in dB (e.g., `-30` for -30 dB). Default: `-40.0`.
- `min_silence_duration`: Minimum duration of silence in seconds. Default: `2.0`.
- `mode`: The mode for processing:
  - `talk`: Retain only the talking parts (non-silent segments).
  - `silence`: Retain only the silent parts.

#### Example

To retain only the talking parts and remove silent segments with a threshold of `-30 dB` and a minimum silence duration of `1.2 seconds`:

```bash
python chop_silence.py input.mp4 output_talk.mp4 -30 1.2 talk
```

To retain only the silent parts:

```bash
python chop_silence.py input.mp4 output_silence.mp4 -30 1.2 silence
```

### 2. Video Speedup Script

This script speeds up a video without altering the pitch of the audio.

#### Run the Script

```bash
python speedup_video.py <input_video> <output_video> <speed_factor>
```

#### Arguments

- `input_video`: Path to the input video file.
- `output_video`: Path where the output video will be saved.
- `speed_factor`: The factor by which the video should be sped up (e.g., `2.0` for 2x speed, `1.5` for 1.5x speed).

#### Example

To speed up the video by 1.5x:

```bash
python speedup_video.py input.mp4 output_1.5x.mp4 1.5
```

To speed up the video by 2.0x:

```bash
python speedup_video.py input.mp4 output_2x.mp4 2.0
```

## License

This project is licensed under the MIT License.
