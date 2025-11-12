# Digital Voicing of Silent Speech

## Data

The necessary data can be downloaded from https://doi.org/10.5281/zenodo.4064408.

## Environment Setup

uses `pip` for libraries. I have this running on `python 3.10`, but later versions should work.

Run `Train_emg.bat` to install requirements and begin training the model.

`requirements.txt` have all of the necessary libraries to run the code.


## Running

To train an EMG to speech feature transduction model, use
```
python transduction_model.py --pretrained_wavenet_model "./models/wavenet_model/wavenet_model.pt" --output_directory "./models/transduction_model/" --voiced_data_directories "./emg_data/voiced_parallel_data,./emg_data/nonparallel_data" --silent_data_directories "./emg_data/silent_parallel_data"
```
At the end of training, an ASR evaluation will be run on the validation set.

Finally, to evaluate a model on the test set, use
```
python evaluate.py --models transduction_model --pretrained_wavenet_model ./models/wavenet_model/wavenet_model.pt --silent_data_directories ./emg_data/silent_parallel_data --voiced_data_directories ./emg_data/voiced_parallel_data --output_directory evaluation_output
```
