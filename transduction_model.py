import os
import sys
from time import time, sleep

import numpy as np

# To run Training, copy ".\New_Train_emg.bat" in the terminal
# to bypass train for save_output() only, Run ".\Bypass_Training.bat"

import pyworld
import soundfile as sf

from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F

from read_emg import EMGDataset
# from wavenet_model import save_output as save_wavenet_output
from align import get_cca_transform, get_all_alignments
from asr import evaluate

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('model_size', 1024, 'number of hidden dimensions')
flags.DEFINE_integer('num_layers', 3, 'number of layers')
flags.DEFINE_float('dropout', 0.5, 'dropout')
flags.DEFINE_integer('batch_size', 16, 'training batch size')
flags.DEFINE_integer('learning_rate_patience', 5, 'learning rate decay patience')
flags.DEFINE_float('alignment_audio_weight', 10.0, 'weight of audio feature distance for alignments')
flags.DEFINE_boolean('no_audio_alignment', False, "don't use predicted audio to refine alignment between silent and voiced")
flags.DEFINE_boolean('no_cca', False, "don't use CCA to refine alignments")
flags.DEFINE_boolean('load_model', False, 'start training from a model')
flags.DEFINE_float('data_size_fraction', 1.0, 'fraction of training data to use')
flags.DEFINE_boolean('no_session_embed', False, "don't use a session embedding")

flags.DEFINE_string('output_directory', './models/transduction_model/', 'Directory to save model outputs')
# flags.DEFINE_list('voiced_data_directories', [], 'List of directories with voiced EMG data')
# flags.DEFINE_list('silent_data_directories', [], 'List of directories with silent EMG data')
flags.DEFINE_boolean('debug', False, 'Run in debug mode (forces CPU)')

flags.DEFINE_integer('n_epochs', 1, 'Changes number of epochs')

class Model(nn.Module):
    def __init__(self, num_ins, num_outs, num_sessions):
        super().__init__()

        if FLAGS.no_session_embed:
            lstm_in_size = num_ins
        else:
            emb_size = 32
            self.session_emb = nn.Embedding(num_sessions, emb_size)
            lstm_in_size = num_ins+emb_size
        self.lstm = nn.LSTM(lstm_in_size, FLAGS.model_size, batch_first=True, bidirectional=True, num_layers=FLAGS.num_layers, dropout=FLAGS.dropout)
        self.w1 = nn.Linear(FLAGS.model_size*2, num_outs)

    def forward(self, x, session_ids):
        # x shape is (batch, time, electrode)
        if not FLAGS.no_session_embed:
            emb = self.session_emb(session_ids)
            x = torch.cat([x, emb], -1)
        x = F.dropout(x, FLAGS.dropout, training=self.training)
        x, _ = self.lstm(x)
        x = F.dropout(x, FLAGS.dropout, training=self.training)
        return self.w1(x)

def test(model, testset, device):
    model.eval()

    dataloader = torch.utils.data.DataLoader(testset, batch_size=1)
    losses = []
    all_distances = []
    with torch.no_grad():
        for example in dataloader:
            X = example['emg'].to(device)
            y = example['audio_features'].to(device)
            sess = example['session_ids'].to(device)

            pred = model(X, sess)

            loss = F.mse_loss(pred, y)
            losses.append(loss.item())

            pred = pred.squeeze(0).cpu().detach().numpy()
            y = y.squeeze(0).cpu().numpy()
            pred_orig_scale = testset.mfcc_norm.inverse(pred)
            target_orig_scale = testset.mfcc_norm.inverse(y)
            diff = pred_orig_scale-target_orig_scale
            distances = np.sqrt((diff[:,1:]*diff[:,1:]).sum(-1))
            all_distances.append(distances)
    model.train()
    return np.mean(losses), np.concatenate(all_distances).mean()

# def save_output(model, datapoint, filename, device, gold_mfcc=False):
#     model.eval()
#     emg = datapoint['emg']
#     if gold_mfcc:
#         y = datapoint['audio_features']
#     else:
#         sess = torch.tensor(datapoint['session_ids'], device=device).unsqueeze(0)
#         X = torch.tensor(emg, dtype=torch.float32, device=device).unsqueeze(0)
#         y = model(X, sess).squeeze(0)
#         y = y.cpu().detach().numpy()
#
#     # wavenet_model = WavenetModel(y.shape[1]).to(device)
#     # assert FLAGS.pretrained_wavenet_model is not None
#     # Skipped loading pretrained WaveNet — not used in this setup
#     # wavenet_model.load_state_dict(torch.load(FLAGS.pretrained_wavenet_model))
#     # save_wavenet_output(wavenet_model, y, filename, device)
#     model.train()
def save_output(model, datapoint, filename, device, mfcc_norm, gold_mfcc=False):
    # print(f"Saving output to", filename)
    model.eval()
    emg = datapoint['emg']
    sess = torch.tensor(datapoint['session_ids'], device=device).unsqueeze(0)
    X = torch.tensor(emg, dtype=torch.float32, device=device).unsqueeze(0)
    y = model(X, sess).squeeze(0).cpu().detach().numpy()

    # Inverse normalization
    y = mfcc_norm.inverse(y)

    # Split into f0 and mcep
    f0 = np.ascontiguousarray(y[:, 0].reshape(-1).astype(np.float64))  # Ensure 1D float64
    mcep = np.ascontiguousarray(y[:, 1:].astype(np.float64))  # Ensure 2D float64

    # Decode spectral envelope
    sp = pyworld.decode_spectral_envelope(mcep, 16000, fft_size=512)
    sp = np.ascontiguousarray(sp.astype(np.float64))  # Just in case

    # Generate aperiodicity
    ap = np.ascontiguousarray(np.zeros_like(sp, dtype=np.float64))  # Ensure float64 and contiguous

    # Synthesize waveform
    wav = pyworld.synthesize(f0, sp, ap, 16000)

    sf.write(filename, wav, 16000)
    model.train()

def get_emg_alignment_features(example):
    return example['emg'], example['parallel_voiced_emg']

def create_audio_alignment_feature_function(model, device='cpu'):
    def feature_function(example):
        X = torch.tensor(example['emg'], dtype=torch.float32, device=device).unsqueeze(0)
        sess = torch.tensor(example['session_ids'], device=device).unsqueeze(0)
        y = model(X, sess).squeeze(0)
        y = y.cpu().detach().numpy()

        return y, example['parallel_voiced_audio_features']
    return feature_function

def train_model(trainset, devset, device, save_sound_outputs=True, n_epochs=50):
    global selected_model
    if FLAGS.data_size_fraction >= 1:
        training_subset = trainset
    else:
        training_subset = torch.utils.data.Subset(trainset, list(range(int(len(trainset)*FLAGS.data_size_fraction))))
    dataloader = torch.utils.data.DataLoader(training_subset, batch_size=FLAGS.batch_size, shuffle=True, pin_memory=(device=='cuda'), collate_fn=devset.collate_fixed_length)
    model = Model(devset.num_features, devset.num_speech_features, devset.num_sessions).to(device)

    if FLAGS.load_model:
        list_models = []
        for file in os.listdir(FLAGS.output_directory):
            if file.endswith(".pt"):
                list_models.append(os.path.join(FLAGS.output_directory, file))
                print(len(list_models)-1, list_models[len(list_models)-1])
        if len(list_models)>1:
            model_index = input("Enter desired model index: ")
            selected_model = list_models[int(model_index)]
            print("Loading model:", selected_model)
            state_dict = torch.load(selected_model)
        else:
            selected_model = list_models[0]
            print("Loading model:", selected_model)
            state_dict = torch.load(selected_model)
        del state_dict['session_emb.weight']
        model.load_state_dict(state_dict, strict=False)

    optim = torch.optim.Adam(model.parameters())
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', 0.5, patience=FLAGS.learning_rate_patience)
    best_validation = float('inf')

    os.makedirs(FLAGS.output_directory, exist_ok=True)

    if len(FLAGS.silent_data_directories) > 0:
        silent_trainset = trainset.silent_subset()
        if FLAGS.no_cca:
            emg_alignment_func = get_emg_alignment_features
        else:
            emg_alignment_func = get_cca_transform(silent_trainset, get_emg_alignment_features)

        alignments = get_all_alignments(silent_trainset, [emg_alignment_func])
        trainset.set_silent_alignments(silent_trainset, alignments)
        alignments = get_all_alignments(devset, [emg_alignment_func])
        devset.set_silent_alignments(devset, alignments)

    for epoch_idx in range(n_epochs):
        if not FLAGS.no_audio_alignment and len(FLAGS.silent_data_directories) > 0 and (epoch_idx+1)%5==0:
            audio_feature_func = create_audio_alignment_feature_function(model, device)
            alignments = get_all_alignments(silent_trainset, [emg_alignment_func, audio_feature_func], [1., FLAGS.alignment_audio_weight])
            trainset.set_silent_alignments(silent_trainset, alignments)
            alignments = get_all_alignments(devset, [emg_alignment_func, audio_feature_func], [1., FLAGS.alignment_audio_weight])
            devset.set_silent_alignments(devset, alignments)

        losses = []
        # for example in dataloader:
        for example in tqdm(dataloader, desc=f"Epoch {epoch_idx + 1}", leave=False):
            optim.zero_grad()
            X = example['emg'].to(device)
            y = example['audio_features'].to(device)
            sess = example['session_ids'].to(device)

            pred = model(X, sess)
            loss = F.mse_loss(pred, y)
            losses.append(loss.item())

            loss.backward()
            optim.step()
        train_loss = np.mean(losses)
        val, _ = test(model, devset, device)
        lr_sched.step(val)
        print(f'finished epoch {epoch_idx+1} with validation loss {val:.4f} and training loss {train_loss:.4f}')
        if val < best_validation:
            torch.save(model.state_dict(), selected_model)
            best_validation = val
        if save_sound_outputs:
            save_output(model, devset[0], os.path.join(FLAGS.output_directory, f'epoch_{epoch_idx}_output.wav'), device, devset.mfcc_norm)

    print("model.load_state_dict")
    model.load_state_dict(torch.load(selected_model)) # re-load best parameters

    if save_sound_outputs:
        # for i, datapoint in enumerate(devset):
        for i, datapoint in enumerate(tqdm(devset, desc="Synthesizing audio outputs")):
            save_output(model, datapoint, os.path.join(FLAGS.output_directory, f'example_output_{i}.wav'), device, devset.mfcc_norm)

    evaluate(devset, FLAGS.output_directory)

    return model

def main():
    t = time()
    trainset = EMGDataset(dev=False,test=False)
    devset = EMGDataset(dev=True)
    print('output example:', devset.example_indices[0])
    print('train / dev split:',len(trainset),len(devset))

    # device = 'cuda' #forces cuda cores
    device = 'cuda' if torch.cuda.is_available() and not FLAGS.debug else 'cpu'
    print('device:', device)

    model = train_model(trainset, devset, device, save_sound_outputs=True, n_epochs=FLAGS.n_epochs)
    input(f"Total Compute Time: {time()-t}\nPress enter to exit:")

if __name__ == '__main__':
    print(sys.argv)
    FLAGS(sys.argv)
    selected_model = os.path.join(FLAGS.output_directory, 'model.pt')
    main()
