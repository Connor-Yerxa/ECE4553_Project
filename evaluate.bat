call .venv\Scripts\activate.bat
python evaluate.py --models transduction_model --output_directory ./models/transduction_model/ --voiced_data_directories ./emg_data/voiced_parallel_data,./emg_data/nonparallel_data --silent_data_directories ./emg_data/silent_parallel_data

