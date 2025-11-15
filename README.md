# Digital Voicing of Silent Speech

## Data

The necessary data can be downloaded from https://doi.org/10.5281/zenodo.4064408.

## Codex Setup For Whisper Transcription
### Step 1: Download FFmpeg for Windows
- Go to: https://ffmpeg.org/download.html
- Choose Windows → Click through to a build provider like gyan.dev
- Download the "release full" zip (e.g., ffmpeg-release-full.7z or .zip)
### Step 2: Extract and Add to PATH
- Extract the archive (e.g., to C:\ffmpeg)
- Inside C:\ffmpeg\bin, you’ll find ffmpeg.exe
- Add C:\ffmpeg\bin to your system PATH:
- Open Start → search “Environment Variables”
- Edit the Path variable under your user or system variables
- Add: C:\ffmpeg\bin
- Restart your terminal or IDE
### Step 3: Verify Installation
In a new terminal:
```
ffmpeg -version
```

You should see version info printed.
Steps produced by copilot.

## Environment Setup

uses `pip` for libraries. I have this running on `python 3.10`, but later versions should work.

Run `Train_emg.bat` to install requirements and begin training the model.

`requirements.txt` have all of the necessary libraries to run the code.


## Running

To train an EMG to speech feature transduction model, use
```
call .venv\Scripts\activate.bat
pip install -r requirements.txt 
python transduction_model.py --output_directory ./models/transduction_model/ --voiced_data_directories ./emg_data/voiced_parallel_data,./emg_data/nonparallel_data --silent_data_directories ./emg_data/silent_parallel_data
```
Or run `train_emg.bat`
At the end of training, an ASR evaluation will be run on the validation set.

To skip training and use a premade model, use
```
call .venv\Scripts\activate.bat
pip install -r requirements.txt 
python transduction_model.py --output_directory ./models/transduction_model/ --start_training_from ./models/transduction_model/model.pt --voiced_data_directories ./emg_data/voiced_parallel_data,./emg_data/nonparallel_data --silent_data_directories ./emg_data/silent_parallel_data --n_epochs 0
```
Or run generate_audio.bat
This will also automatically evaluate

To evaluate a model on the test set, use
```
call .venv\Scripts\activate.bat
python evaluate.py --models transduction_model --output_directory ./models/transduction_model/ --voiced_data_directories ./emg_data/voiced_parallel_data,./emg_data/nonparallel_data --silent_data_directories ./emg_data/silent_parallel_data
```
