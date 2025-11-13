call .venv\Scripts\activate.bat
pip install -r requirements.txt 
python transduction_model.py --pretrained_wavenet_model "./models/wavenet_model/wavenet_model.pt" --output_directory "./models/transduction_model/" --voiced_data_directories "./emg_data/voiced_parallel_data,./emg_data/nonparallel_data" --silent_data_directories "./emg_data/silent_parallel_data"