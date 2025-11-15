import os

import whisper
import jiwer
import soundfile as sf
import numpy as np
from tqdm import tqdm
from unidecode import unidecode

def evaluate(testset, audio_directory, model_size="base"):
    model = whisper.load_model(model_size)
    predictions = []
    targets = []
    # for i, datapoint in enumerate(testset):
    for i, datapoint in enumerate(tqdm(testset, desc="Transcribing with Whisper")):
        audio_path = os.path.join(audio_directory,f'example_output_{i}.wav')
        results = model.transcribe(audio_path)
        text = results["text"]
        # audio, rate = sf.read(os.path.join(audio_directory,f'example_output_{i}.wav'))
        # assert rate == model.sampleRate(), 'wrong sample rate'
        # audio_int16 = (audio*(2**15)).astype(np.int16)
        # text = model.stt(audio_int16)
        predictions.append(text)
        target_text = unidecode(datapoint['text'])
        targets.append(target_text)

    transformation = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
    targets = transformation(targets)
    predictions = transformation(predictions)
    print('targets:', targets)
    print('predictions:', predictions)
    print('wer:', jiwer.wer(targets, predictions))
