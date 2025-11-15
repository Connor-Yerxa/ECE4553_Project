if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
pip install -r requirements.txt 
python transduction_model.py --output_directory ./models/transduction_model/ --load_model True --voiced_data_directories ./emg_data/voiced_parallel_data,./emg_data/nonparallel_data --silent_data_directories ./emg_data/silent_parallel_data --n_epochs 0