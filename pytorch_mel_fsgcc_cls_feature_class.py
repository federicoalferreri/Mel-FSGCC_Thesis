# Contains routines for labels creation, features extraction and normalization
#

from cls_vid_features import VideoFeatures
from PIL import Image
import os
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal.windows import hann, boxcar
from scipy.stats import linregress
from sklearn import preprocessing
import joblib
from IPython import embed
import matplotlib.pyplot as plot
import librosa
plot.switch_backend('agg')
import shutil
import math
import wave
import contextlib
import torch
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import time
# import cv2



def nCr(n, r):
    return math.factorial(n) // math.factorial(r) // math.factorial(n-r)

def enframe_center(x, frame_len, hop_len, NFFT):
  ''' Divide an input array into overlapping frames (librosa style) '''

  # librosa centering
  pad_signal = np.pad(x, int(NFFT // 2), mode='reflect')

  signal_length = len(pad_signal)
  num_frames = int(np.floor(float(np.abs(signal_length - frame_len)) / hop_len))

  indices = np.tile(np.arange(0, frame_len), (num_frames, 1)).T + np.tile(np.arange(0, num_frames * hop_len, hop_len), (frame_len, 1))
  x_frames = pad_signal[indices.astype(np.int32)] # indices must have int type
  return x_frames


def Hz_to_Mel(f_hz):
  ''' Frequency in Hz to Mels '''
  mel = 2595 * np.log10(1 + (f_hz/700))
  return mel

def Mel_to_Hz(f_mel):
  ''' Mels to frequency in Hz '''
  hz = (700 * (10**(f_mel/ 2595) - 1))
  return hz

def Mel_bins(nfilt, f_min, f_max, sr, NFFT):
  ''' Defines FFT bins defining limits of a Mel filterbank '''
  low_freq_mel = Hz_to_Mel(f_min)
  high_freq_mel = Hz_to_Mel(f_max)

  mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt+2)
  hz_points = Mel_to_Hz(mel_points)

  bin_sep = sr/NFFT
  return np.round(hz_points/bin_sep).astype(int)

def Mel_filters(nfilt, f_min, f_max, sr, NFFT):
  ''' Creates a Mel filterbank matrix '''

  # get transition points in original FFT
  bin = Mel_bins(nfilt, f_min, f_max, sr, NFFT)

  # Initialize filterbank
  fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))  # nfilt x nbins
  for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1])   # left
    f_m = int(bin[m])             # center
    f_m_plus = int(bin[m + 1])    # right

    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

  return fbank

def extend_spectrogram(stft):

    num_frames, num_freqs, num_channels = stft.shape

    dc_component = stft[:, :1, :]
    positive_freqs = stft[:, 1:num_freqs - 1, :]
    nyquist_freq = stft[:, num_freqs - 1:num_freqs, :]
    mirrored_freqs = np.flip(np.conj(positive_freqs), axis=1)

    nyquist_freq = np.real(nyquist_freq) + 0j

    extended_stft = np.concatenate([dc_component, positive_freqs, nyquist_freq, mirrored_freqs], axis=1)

    return extended_stft


class FeatureClass:
    def __init__(self, params, is_eval=False):
        """

        :param params: parameters dictionary
        :param is_eval: if True, does not load dataset labels.
        """

        # Input directories
        self._feat_label_dir = params['feat_label_dir']
        self._dataset_dir = params['dataset_dir']
        self._dataset_combination = '{}_{}'.format(params['dataset'], 'eval' if is_eval else 'dev')
        #self._dataset_combination = params['dataset']
        self._aud_dir = os.path.join(self._dataset_dir, self._dataset_combination)

        self._desc_dir = None if is_eval else os.path.join(self._dataset_dir, 'metadata_dev')
        #self._desc_dir = None if is_eval else os.path.join(self._dataset_dir, 'labels')

        self._vid_dir = os.path.join(self._dataset_dir, 'video_{}'.format('eval' if is_eval else 'dev'))
        # Output directories
        self._label_dir = None
        self._feat_dir = None
        self._feat_dir_norm = None
        self._vid_feat_dir = None

        # Local parameters
        self._is_eval = is_eval

        self._fs = params['fs']
        self._hop_len_s = params['hop_len_s']
        self._hop_len = int(self._fs * self._hop_len_s)

        self._label_hop_len_s = params['label_hop_len_s']
        self._label_hop_len = int(self._fs * self._label_hop_len_s)
        self._label_frame_res = self._fs / float(self._label_hop_len)
        self._nb_label_frames_1s = int(self._label_frame_res)

        self._win_len = 2 * self._hop_len
        self._nfft = 2048
        #self._nfft = self._next_greater_power_of_2(self._win_len)

        self._dataset = params['dataset']
        self._eps = 1e-8
        self._nb_channels = 4
        self._unique_classes = params['unique_classes']

        self._multi_accdoa = params['multi_accdoa']
        self._use_salsalite = params['use_salsalite']
        if self._use_salsalite and self._dataset=='mic':
            # Initialize the spatial feature constants
            self._lower_bin = np.int(np.floor(params['fmin_doa_salsalite'] * self._nfft / np.float(self._fs)))
            self._lower_bin = np.max((1, self._lower_bin))
            self._upper_bin = np.int(np.floor(np.min((params['fmax_doa_salsalite'], self._fs//2)) * self._nfft / np.float(self._fs)))


            # Normalization factor for salsalite
            c = 343
            self._delta = 2 * np.pi * self._fs / (self._nfft * c)
            self._freq_vector = np.arange(self._nfft//2 + 1)
            self._freq_vector[0] = 1
            self._freq_vector = self._freq_vector[None, :, None]  # 1 x n_bins x 1

            # Initialize spectral feature constants
            self._cutoff_bin = np.int(np.floor(params['fmax_spectra_salsalite'] * self._nfft / np.float(self._fs)))
            assert self._upper_bin <= self._cutoff_bin, 'Upper bin for doa featurei {} is higher than cutoff bin for spectrogram {}!'.format()
            self._nb_mel_bins = self._cutoff_bin - self._lower_bin
        else:
            self._nb_mel_bins = params['nb_mel_bins']
            self._mel_wts = librosa.filters.mel(sr=self._fs, n_fft=self._nfft, n_mels=self._nb_mel_bins).T
        # Sound event classes dictionary
        self._nb_unique_classes = params['unique_classes']

        self._filewise_frames = {}

    def get_frame_stats(self):

        if len(self._filewise_frames) != 0:
            return

        print('Computing frame stats:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tfeat_dir {}'.format(
            self._aud_dir, self._desc_dir, self._feat_dir))
        for sub_folder in os.listdir(self._aud_dir):
            loc_aud_folder = os.path.join(self._aud_dir, sub_folder)
            for file_cnt, file_name in enumerate(os.listdir(loc_aud_folder)):
                wav_filename = '{}.wav'.format(file_name.split('.')[0])
                with contextlib.closing(wave.open(os.path.join(loc_aud_folder, wav_filename), 'r')) as f:
                    audio_len = f.getnframes()
                nb_feat_frames = int(audio_len / float(self._hop_len))
                nb_label_frames = int(audio_len / float(self._label_hop_len))
                self._filewise_frames[file_name.split('.')[0]] = [nb_feat_frames, nb_label_frames]
        return

    def _load_audio(self, audio_path):
        fs, audio = wav.read(audio_path)
        audio = audio / 2**15
        return audio, fs

    # INPUT FEATURES
    @staticmethod
    def _next_greater_power_of_2(x):
        return 2 ** (x - 1).bit_length()

    def _spectrogram(self, audio_input):
        _nb_ch = audio_input.shape[1]
        frame_length = self._win_len
        winfunc = np.hanning(frame_length)
        winfunc = np.expand_dims(winfunc, 1)
        apply_hann = True
        spectra = []
        for ch_cnt in range(_nb_ch):
            frames = enframe_center(audio_input[:, ch_cnt], self._win_len, self._hop_len, self._nfft)
            if apply_hann == True:
                spectra.append(frames * winfunc)
        spectra = np.moveaxis(np.asarray(spectra), 0, 2)
        _Spectra = []
        for ch_cnt in range(_nb_ch):
            _Spectra.append(np.fft.fft(spectra[:, :, ch_cnt], self._nfft, axis=0))
        _Spectra = np.moveaxis(np.asarray(_Spectra), 0, 2)
        return _Spectra

    def _spectrogram_gcc(self, audio_input, _nb_frames):
        _nb_ch = audio_input.shape[1]
        nb_bins = self._nfft // 2
        spectra = []
        for ch_cnt in range(_nb_ch):
            stft_ch = librosa.core.stft(np.asfortranarray(audio_input[:, ch_cnt]), n_fft=self._nfft, hop_length=self._hop_len,
                                        win_length=self._win_len, window='hann')
            spectra.append(stft_ch[:, :_nb_frames])
        return np.array(spectra).T


    def _get_mel_spectrogram(self, linear_spectra):
        Mel_fbank = Mel_filters(self._nb_mel_bins, 0, self._fs/2, self._fs, self._nfft)
        Nbins = int(np.floor(self._nfft / 2) + 1)
        powframes = np.abs(linear_spectra[:Nbins, :, :]) ** 2
        Mel_sp = []
        for c in range(linear_spectra.shape[-1]):
            mel_specgram = np.dot(Mel_fbank, powframes[:, :, c])
            mel_specgram = np.where(mel_specgram == 0, np.finfo(float).eps, mel_specgram)  # Numerical Stability
            Mel_sp.append(10 * np.log10(mel_specgram))
        Mel_sp = np.moveaxis(np.asarray(Mel_sp), 0, 2)
        return Mel_sp

    def _get_mel_spectrogram_gcc(self, linear_spectra):
        mel_feat = np.zeros((linear_spectra.shape[0], self._nb_mel_bins, linear_spectra.shape[-1]))
        for ch_cnt in range(linear_spectra.shape[-1]):
            mag_spectra = np.abs(linear_spectra[:, :, ch_cnt])**2
            mel_spectra = np.dot(mag_spectra, self._mel_wts)
            log_mel_spectra = librosa.power_to_db(mel_spectra)
            mel_feat[:, :, ch_cnt] = log_mel_spectra
        mel_feat = mel_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))
        return mel_feat

    def _get_foa_intensity_vectors(self, linear_spectra):
        W = linear_spectra[:, :, 0]
        I = np.real(np.conj(W)[:, :, np.newaxis] * linear_spectra[:, :, 1:])
        E = self._eps + (np.abs(W)**2 + ((np.abs(linear_spectra[:, :, 1:])**2).sum(-1)) / 3.0)

        I_norm = I / E[:, :, np.newaxis]
        I_norm_mel = np.transpose(np.dot(np.transpose(I_norm, (0, 2, 1)), self._mel_wts), (0, 2, 1))
        foa_iv = I_norm_mel.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], self._nb_mel_bins * 3))
        if np.isnan(foa_iv).any():
            print('Feature extraction is generating nan outputs')
            exit()
        return foa_iv

    def _get_gcc(self, linear_spectra):

        gcc_channels = nCr(linear_spectra.shape[-1], 2)
        gcc_feat = np.zeros((linear_spectra.shape[0], self._nb_mel_bins, gcc_channels))
        cnt = 0
        for m in range(linear_spectra.shape[-1]):
            for n in range(m+1, linear_spectra.shape[-1]):
                R = np.conj(linear_spectra[:, :, m]) * linear_spectra[:, :, n]
                cc = np.fft.irfft(np.exp(1.j*np.angle(R)))
                cc = np.concatenate((cc[:, -self._nb_mel_bins//2:], cc[:, :self._nb_mel_bins//2]), axis=-1)
                gcc_feat[:, :, cnt] = cc
                cnt += 1
        return gcc_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))

    def _get_salsalite(self, linear_spectra):
        # Adapted from the official SALSA repo- https://github.com/thomeou/SALSA
        # spatial features
        phase_vector = np.angle(linear_spectra[:, :, 1:] * np.conj(linear_spectra[:, :, 0, None]))
        phase_vector = phase_vector / (self._delta * self._freq_vector)
        phase_vector = phase_vector[:, self._lower_bin:self._cutoff_bin, :]
        phase_vector[:, self._upper_bin:, :] = 0
        phase_vector = phase_vector.transpose((0, 2, 1)).reshape((phase_vector.shape[0], -1))

        # spectral features
        linear_spectra = np.abs(linear_spectra)**2
        for ch_cnt in range(linear_spectra.shape[-1]):
            linear_spectra[:, :, ch_cnt] = librosa.power_to_db(linear_spectra[:, :, ch_cnt], ref=1.0, amin=1e-10, top_db=None)
        linear_spectra = linear_spectra[:, self._lower_bin:self._cutoff_bin, :]
        linear_spectra = linear_spectra.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))

        return np.concatenate((linear_spectra, phase_vector), axis=-1)

    def _get_spectrogram_for_file(self, audio_filename):
        audio_in, fs = self._load_audio(audio_filename)

        nb_feat_frames = int(len(audio_in) / float(self._hop_len))
        nb_label_frames = int(len(audio_in) / float(self._label_hop_len))
        self._filewise_frames[os.path.basename(audio_filename).split('.')[0]] = [nb_feat_frames, nb_label_frames]

        audio_spec = self._spectrogram_gcc(audio_in, nb_feat_frames)
        return audio_spec

    # OUTPUT LABELS
    def get_labels_for_file(self, _desc_file, _nb_label_frames):
        """
        Reads description file and returns classification based SED labels and regression based DOA labels

        :param _desc_file: metadata description file
        :return: label_mat: of dimension [nb_frames, 3*max_classes], max_classes each for x, y, z axis,
        """

        # If using Hungarian net set default DOA value to a fixed value greater than 1 for all axis. We are choosing a fixed value of 10
        # If not using Hungarian net use a deafult DOA, which is a unit vector. We are choosing (x, y, z) = (0, 0, 1)
        se_label = np.zeros((_nb_label_frames, self._nb_unique_classes))
        x_label = np.zeros((_nb_label_frames, self._nb_unique_classes))
        y_label = np.zeros((_nb_label_frames, self._nb_unique_classes))
        z_label = np.zeros((_nb_label_frames, self._nb_unique_classes))
        dist_label = np.zeros((_nb_label_frames, self._nb_unique_classes))

        for frame_ind, active_event_list in _desc_file.items():
            if frame_ind < _nb_label_frames:
                for active_event in active_event_list:
                    #print(active_event)
                    se_label[frame_ind, active_event[0]] = 1
                    x_label[frame_ind, active_event[0]] = active_event[2]
                    y_label[frame_ind, active_event[0]] = active_event[3]
                    z_label[frame_ind, active_event[0]] = active_event[4]
                    dist_label[frame_ind, active_event[0]] = active_event[5]

        label_mat = np.concatenate((se_label, x_label, y_label, z_label, dist_label), axis=1)
        return label_mat

    # OUTPUT LABELS
    def get_adpit_labels_for_file(self, _desc_file, _nb_label_frames):
        """
        Reads description file and returns classification based SED labels and regression based DOA labels
        for multi-ACCDOA with Auxiliary Duplicating Permutation Invariant Training (ADPIT)

        :param _desc_file: metadata description file
        :return: label_mat: of dimension [nb_frames, 6, 4(=act+XYZ), max_classes]
        """

        se_label = np.zeros((_nb_label_frames, 6, self._nb_unique_classes))  # [nb_frames, 6, max_classes]
        x_label = np.zeros((_nb_label_frames, 6, self._nb_unique_classes))
        y_label = np.zeros((_nb_label_frames, 6, self._nb_unique_classes))
        z_label = np.zeros((_nb_label_frames, 6, self._nb_unique_classes))
        dist_label = np.zeros((_nb_label_frames, 6, self._nb_unique_classes))

        for frame_ind, active_event_list in _desc_file.items():
            if frame_ind < _nb_label_frames:
                active_event_list.sort(key=lambda x: x[0])  # sort for ov from the same class
                active_event_list_per_class = []
                for i, active_event in enumerate(active_event_list):
                    active_event_list_per_class.append(active_event)
                    if i == len(active_event_list) - 1:  # if the last
                        if len(active_event_list_per_class) == 1:  # if no ov from the same class
                            # a0----
                            active_event_a0 = active_event_list_per_class[0]
                            se_label[frame_ind, 0, active_event_a0[0]] = 1
                            x_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[2]
                            y_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[3]
                            z_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[4]
                            dist_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[5]/100.
                        elif len(active_event_list_per_class) == 2:  # if ov with 2 sources from the same class
                            # --b0--
                            active_event_b0 = active_event_list_per_class[0]
                            se_label[frame_ind, 1, active_event_b0[0]] = 1
                            x_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[2]
                            y_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[3]
                            z_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[4]
                            dist_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[5]/100.
                            # --b1--
                            active_event_b1 = active_event_list_per_class[1]
                            se_label[frame_ind, 2, active_event_b1[0]] = 1
                            x_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[2]
                            y_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[3]
                            z_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[4]
                            dist_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[5]/100.
                        else:  # if ov with more than 2 sources from the same class
                            # ----c0
                            active_event_c0 = active_event_list_per_class[0]
                            se_label[frame_ind, 3, active_event_c0[0]] = 1
                            x_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[2]
                            y_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[3]
                            z_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[4]
                            dist_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[5]/100.
                            # ----c1
                            active_event_c1 = active_event_list_per_class[1]
                            se_label[frame_ind, 4, active_event_c1[0]] = 1
                            x_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[2]
                            y_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[3]
                            z_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[4]
                            dist_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[5]/100.
                            # ----c2
                            active_event_c2 = active_event_list_per_class[2]
                            se_label[frame_ind, 5, active_event_c2[0]] = 1
                            x_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[2]
                            y_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[3]
                            z_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[4]
                            dist_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[5]/100.

                    elif active_event[0] != active_event_list[i + 1][0]:  # if the next is not the same class
                        if len(active_event_list_per_class) == 1:  # if no ov from the same class
                            # a0----
                            active_event_a0 = active_event_list_per_class[0]
                            se_label[frame_ind, 0, active_event_a0[0]] = 1
                            x_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[2]
                            y_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[3]
                            z_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[4]
                            dist_label[frame_ind, 0, active_event_a0[0]] = active_event_a0[5]/100.
                        elif len(active_event_list_per_class) == 2:  # if ov with 2 sources from the same class
                            # --b0--
                            active_event_b0 = active_event_list_per_class[0]
                            se_label[frame_ind, 1, active_event_b0[0]] = 1
                            x_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[2]
                            y_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[3]
                            z_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[4]
                            dist_label[frame_ind, 1, active_event_b0[0]] = active_event_b0[5]/100.
                            # --b1--
                            active_event_b1 = active_event_list_per_class[1]
                            se_label[frame_ind, 2, active_event_b1[0]] = 1
                            x_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[2]
                            y_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[3]
                            z_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[4]
                            dist_label[frame_ind, 2, active_event_b1[0]] = active_event_b1[5]/100.
                        else:  # if ov with more than 2 sources from the same class
                            # ----c0
                            active_event_c0 = active_event_list_per_class[0]
                            se_label[frame_ind, 3, active_event_c0[0]] = 1
                            x_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[2]
                            y_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[3]
                            z_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[4]
                            dist_label[frame_ind, 3, active_event_c0[0]] = active_event_c0[5]/100.
                            # ----c1
                            active_event_c1 = active_event_list_per_class[1]
                            se_label[frame_ind, 4, active_event_c1[0]] = 1
                            x_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[2]
                            y_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[3]
                            z_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[4]
                            dist_label[frame_ind, 4, active_event_c1[0]] = active_event_c1[5]/100.
                            # ----c2
                            active_event_c2 = active_event_list_per_class[2]
                            se_label[frame_ind, 5, active_event_c2[0]] = 1
                            x_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[2]
                            y_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[3]
                            z_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[4]
                            dist_label[frame_ind, 5, active_event_c2[0]] = active_event_c2[5]/100.
                        active_event_list_per_class = []

        label_mat = np.stack((se_label, x_label, y_label, z_label, dist_label), axis=2)  # [nb_frames, 6, 5(=act+XYZ+dist), max_classes]
        return label_mat

    # ------------------------------- EXTRACT FEATURE AND PREPROCESS IT -------------------------------

    def extract_file_feature(self, _arg_in):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _file_cnt, _wav_path, _feat_path = _arg_in
        spect = self._get_spectrogram_for_file(_wav_path)
        print('STFT shape: {}'.format(spect.shape))

        spect_mics = np.zeros((spect.shape[0], spect.shape[1], 4), dtype=np.complex128)
        spect_mics[:, :, 0] = spect[:, :, 14]
        spect_mics[:, :, 1] = spect[:, :, 15]
        spect_mics[:, :, 2] = spect[:, :, 16]
        spect_mics[:, :, 3] = spect[:, :, 17]

        # extract mel
        if not self._use_salsalite:
            mel_spect = self._get_mel_spectrogram_gcc(spect_mics)
            print('Mel spectorgram shape: {}'.format(mel_spect.shape))

        extended_spect = extend_spectrogram(spect_mics)
        extended_spect = np.transpose(extended_spect, (1, 0, 2))
        print("Shape estesa:", extended_spect.shape, "dtype:", extended_spect.dtype)

        feats = None

        if self._dataset == 'foa':
            # extract intensity vectors
            foa_iv = self._get_foa_intensity_vectors(spect)
            feats = np.concatenate((mel_spect, foa_iv), axis=-1)

        elif self._dataset == 'mic':
            if self._use_salsalite:
                feats = self._get_salsalite(spect)
            else:
                Nframes = extended_spect.shape[1]
                # lag corresponding to each index of the GCC
                lags = torch.arange(-(self._nfft / 2), self._nfft / 2, dtype=torch.float64, device=device)
                # maximum lag expected according to microphone separation
                max_lag = torch.round(torch.tensor(2 * 1.5 / 343) * self._fs).int()
                #pairs = list(itertools.combinations(range(self._nb_channels), 2))
                pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

                # Precalcolo del filtro Mel per tutte le bande
                #k_lims = Mel_bins(self._nb_mel_bins, 0, self._fs / 2, self._fs, self._nfft)  # Shape: [nbands+2]
                mel_bins_edges_hz = librosa.mel_frequencies(n_mels=self._nb_mel_bins + 2, fmin=0, fmax=self._fs / 2)

                k_lims = np.round(mel_bins_edges_hz / self._fs * self._nfft).astype(int)
                win = 'boxcar'

                maxlag_ind1 = self._nfft // 2 - max_lag
                maxlag_ind2 = self._nfft // 2 + max_lag

                lagmask = torch.zeros(self._nfft, dtype=torch.float64, device=device)
                lagmask[maxlag_ind1:maxlag_ind2 + 1] = 1

                start_time = time.time()
                Xframes = torch.tensor(extended_spect, device=device)

                Meltde = torch.zeros((self._nb_mel_bins, Nframes, len(pairs)), dtype=torch.float64, device='cpu')
                Melmde = torch.zeros((self._nb_mel_bins, Nframes, len(pairs)), dtype=torch.float64, device='cpu')
                Melstd = torch.zeros((self._nb_mel_bins, Nframes, len(pairs)), dtype=torch.float64, device='cpu')
                Melavg = torch.zeros((self._nb_mel_bins, Nframes, len(pairs)), dtype=torch.float64, device='cpu')
                #MelGCClag = torch.zeros((self._nb_mel_bins, Nframes, len(pairs), maxlag_ind2 - maxlag_ind1 + 1), dtype=torch.complex64, device='cpu')
                # print(torch.cuda.memory_summary())
                batch_size = 100


                for idx, (p1, p2) in enumerate(pairs):
                    X1 = Xframes[:, :, p1]
                    X2 = Xframes[:, :, p2]

                    GCC = torch.exp(1j * torch.angle(X2 * torch.conj(X1)))

                    # Elaborazione in batch
                    for start in range(0, Nframes, batch_size):
                        end = min(start + batch_size, Nframes)

                        # Calcolo temporaneo su GPU
                        batch_GCCm = torch.zeros((self._nb_mel_bins, end - start, self._nfft), dtype=torch.complex128, device=device)

                        for k in range(self._nb_mel_bins):
                            BW = k_lims[k + 2] - k_lims[k] + 1
                            BW = BW + BW % 2

                            if win == 'hann':
                                wind = torch.hann_window(BW, device=device)
                            else:
                                wind = torch.ones(BW, dtype=torch.float64, device=device)

                            windmask = torch.zeros(self._nfft, dtype=torch.complex128, device=device)
                            windmask[:BW // 2] = wind[BW // 2:]
                            windmask[-BW // 2:] = wind[:BW // 2]

                            GCCd = torch.roll(GCC[:, start:end], shifts=k_lims[k + 1].item(), dims=0)
                            GCCd = GCCd * windmask[:, None]
                            aux = (1 / BW) * torch.fft.fftshift(torch.fft.ifft(GCCd, dim=0), dim=0) * lagmask[:, None]

                            batch_GCCm[k, :, :] = aux.T  # Trasposta per ottenere [n_frames, Nfft] per ogni banda
                            abs_aux = torch.abs(aux)

                            max_ind = torch.argmax(abs_aux, dim=0)  # Su Nfft, shape: (n_frames,)
                            Melmde[k, start:end, idx] = abs_aux[max_ind, torch.arange(end - start)].to('cpu')
                            Meltde[k, start:end, idx] = lags[max_ind].to('cpu')

                            auxpdf = abs_aux / abs_aux.sum(dim=0, keepdim=True)
                            Melavg[k, start:end, idx] = (lags[:, None] * auxpdf).sum(dim=0).to('cpu')
                            Melstd[k, start:end, idx] = torch.sqrt(((lags[:, None] - Melavg[k, start:end, idx].to(device)) ** 2 * auxpdf).sum(dim=0)).to('cpu')


                        #MelGCClag[:, start:end, idx, :] = batch_GCCm[:, :, maxlag_ind1:maxlag_ind2 + 1].to('cpu')

                print(f'Time: {time.time() - start_time:.2f} s')

                Meltde = Meltde / max_lag
                Melmde = Melmde / (0.5 * (1 / self._nfft))
                Melstd = Melstd / max_lag
                Melavg = Melavg / (0.5 * max_lag)

                Meltde = Meltde.permute(1, 2, 0).reshape(Nframes, -1)
                Melmde = Melmde.permute(1, 2, 0).reshape(Nframes, -1)
                Melstd = Melstd.permute(1, 2, 0).reshape(Nframes, -1)
                Melavg = Melavg.permute(1, 2, 0).reshape(Nframes, -1)

                feats = torch.cat((Meltde, Melmde, Melstd, Melavg), dim=-1)
                feats = feats.numpy()
                feats = np.concatenate((mel_spect, feats), axis=-1)

        else:
            print('ERROR: Unknown dataset format {}'.format(self._dataset))
            exit()

        if feats is not None:
            print('{}: {}, {}'.format(_file_cnt, os.path.basename(_wav_path), feats.shape))
            np.save(_feat_path, feats)


    def extract_all_feature(self):
        # setting up folders
        self._feat_dir = self.get_unnormalized_feat_dir()
        create_folder(self._feat_dir)
        from multiprocessing import Pool
        import time
        start_s = time.time()
        # extraction starts
        print('Extracting spectrogram:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tfeat_dir {}'.format(
            self._aud_dir, self._desc_dir, self._feat_dir))
        #arg_list = []
        for sub_folder in os.listdir(self._aud_dir):
            loc_aud_folder = os.path.join(self._aud_dir, sub_folder)
            for file_cnt, file_name in enumerate(os.listdir(loc_aud_folder)):
                wav_filename = '{}.wav'.format(file_name.split('.')[0])
                wav_path = os.path.join(loc_aud_folder, wav_filename)
                feat_path = os.path.join(self._feat_dir, '{}.npy'.format(wav_filename.split('.')[0]))
                #if not os.path.exists(feat_path):
                print("Processed file: {}".format(wav_path))
                self.extract_file_feature((file_cnt, wav_path, feat_path))
                #arg_list.append((file_cnt, wav_path, feat_path_tde))
#        with Pool() as pool:
#            result = pool.map(self.extract_file_feature, iterable=arg_list)
#            pool.close()
#            pool.join()
        print(time.time()-start_s)

    def preprocess_features(self):
        # Setting up folders and filenames
        self._feat_dir = self.get_unnormalized_feat_dir()
        self._feat_dir_norm = self.get_normalized_feat_dir()
        create_folder(self._feat_dir_norm)
        normalized_features_wts_file = self.get_normalized_wts_file()
        spec_scaler = None

        # pre-processing starts
        if self._is_eval:
            spec_scaler = joblib.load(normalized_features_wts_file)
            print('Normalized_features_wts_file: {}. Loaded.'.format(normalized_features_wts_file))

        else:
            print('Estimating weights for normalizing feature files:')
            print('\t\tfeat_dir: {}'.format(self._feat_dir))

            spec_scaler = preprocessing.StandardScaler()
            for file_cnt, file_name in enumerate(os.listdir(self._feat_dir)):
                print('{}: {}'.format(file_cnt, file_name))
                feat_file = np.load(os.path.join(self._feat_dir, file_name))
                #feat_file = feat_file.transpose((0, 2, 1)).reshape((feat_file.shape[0], -1))
                spec_scaler.partial_fit(feat_file)
                del feat_file
            joblib.dump(
                spec_scaler,
                normalized_features_wts_file
            )
            print('Normalized_features_wts_file: {}. Saved.'.format(normalized_features_wts_file))

        print('Normalizing feature files:')
        print('\t\tfeat_dir_norm {}'.format(self._feat_dir_norm))
        for file_cnt, file_name in enumerate(os.listdir(self._feat_dir)):
            print('{}: {}'.format(file_cnt, file_name))
            feat_file = np.load(os.path.join(self._feat_dir, file_name))
            #feat_file = feat_file.transpose((0, 2, 1)).reshape((feat_file.shape[0], -1))
            feat_file = spec_scaler.transform(feat_file)
            np.save(
                os.path.join(self._feat_dir_norm, file_name),
                feat_file
            )
            del feat_file

        print('normalized files written to {}'.format(self._feat_dir_norm))


    # ------------------------------- EXTRACT LABELS AND PREPROCESS IT -------------------------------
    def extract_all_labels(self):
        self.get_frame_stats()
        self._label_dir = self.get_label_dir()

        print('Extracting labels:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tlabel_dir {}'.format(
            self._aud_dir, self._desc_dir, self._label_dir))
        create_folder(self._label_dir)
        for sub_folder in os.listdir(self._desc_dir):
            loc_desc_folder = os.path.join(self._desc_dir, sub_folder)
            for file_cnt, file_name in enumerate(os.listdir(loc_desc_folder)):
                wav_filename = '{}.wav'.format(file_name.split('.')[0])
                nb_label_frames = self._filewise_frames[file_name.split('.')[0]][1]
                desc_file_polar = self.load_output_format_file(os.path.join(loc_desc_folder, file_name))
                desc_file = self.convert_output_format_polar_to_cartesian(desc_file_polar)
                if self._multi_accdoa:
                    label_mat = self.get_adpit_labels_for_file(desc_file, nb_label_frames)
                else:
                    label_mat = self.get_labels_for_file(desc_file, nb_label_frames)
                print('{}: {}, {}'.format(file_cnt, file_name, label_mat.shape))
                np.save(os.path.join(self._label_dir, '{}.npy'.format(wav_filename.split('.')[0])), label_mat)

    # -------------------------------  DCASE OUTPUT  FORMAT FUNCTIONS -------------------------------
    def load_output_format_file(self, _output_format_file, cm2m=False):  # TODO: Reconsider cm2m conversion
        """
        Loads DCASE output format csv file and returns it in dictionary format

        :param _output_format_file: DCASE output format CSV
        :return: _output_dict: dictionary
        """
        _output_dict = {}
        _fid = open(_output_format_file, 'r')
        # next(_fid)
        _words = []     # For empty files
        for _line in _fid:
            _words = _line.strip().split(',')
            _frame_ind = int(_words[0])
            #print(f"Line parsed: {_line.strip()} -> {_words}")
            if _frame_ind not in _output_dict:
                _output_dict[_frame_ind] = []
            if len(_words) == 4:  # frame, class idx,  polar coordinates(2) # no distance data, for example in eval pred
                _output_dict[_frame_ind].append([int(_words[1]), 0, float(_words[2]), float(_words[3])])
            if len(_words) == 5:  # frame, class idx, source_id, polar coordinates(2) # no distance data, for example in synthetic data fold 1 and 2
                _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4])])
            if len(_words) == 6: # frame, class idx, source_id, polar coordinates(2), distance
                _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4]), float(_words[5])/100 if cm2m else float(_words[5])])
            elif len(_words) == 7: # frame, class idx, source_id, cartesian coordinates(3), distance
                _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4]), float(_words[5]), float(_words[6])/100 if cm2m else float(_words[6])])
            #print(f"Adding to frame {_frame_ind}: {int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4]), float(_words[5]) / 100 if cm2m else float(_words[5])}")
        _fid.close()
        if len(_words) == 7:
            _output_dict = self.convert_output_format_cartesian_to_polar(_output_dict)
        #if not _output_dict:
            #print(f"Warning: _output_dict is empty for file {_output_format_file}")
        #else:
            #print(f"_output_dict successfully populated for file {_output_format_file}.")
        return _output_dict

    def write_output_format_file(self, _output_format_file, _output_format_dict):
        """
        Writes DCASE output format csv file, given output format dictionary

        :param _output_format_file:
        :param _output_format_dict:
        :return:
        """
        _fid = open(_output_format_file, 'w')
        # _fid.write('{},{},{},{}\n'.format('frame number with 20ms hop (int)', 'class index (int)', 'azimuth angle (int)', 'elevation angle (int)'))
        for _frame_ind in _output_format_dict.keys():
            for _value in _output_format_dict[_frame_ind]:
                # Write Cartesian format output. Since baseline does not estimate track count and distance we use fixed values.
                _fid.write('{},{},{},{},{},{},{}\n'.format(int(_frame_ind), int(_value[0]), 0, float(_value[1]), float(_value[2]), float(_value[3]), float(_value[4])))
                # TODO: What if our system estimates track cound and distence (or only one of them)
        _fid.close()

    def segment_labels(self, _pred_dict, _max_frames):
        '''
            Collects class-wise sound event location information in segments of length 1s from reference dataset
        :param _pred_dict: Dictionary containing frame-wise sound event time and location information. Output of SELD method
        :param _max_frames: Total number of frames in the recording
        :return: Dictionary containing class-wise sound event location information in each segment of audio
                dictionary_name[segment-index][class-index] = list(frame-cnt-within-segment, azimuth, elevation)
        '''
        nb_blocks = int(np.ceil(_max_frames / float(self._nb_label_frames_1s)))
        output_dict = {x: {} for x in range(nb_blocks)}
        for frame_cnt in range(0, _max_frames, self._nb_label_frames_1s):

            # Collect class-wise information for each block
            # [class][frame] = <list of doa values>
            # Data structure supports multi-instance occurence of same class
            block_cnt = frame_cnt // self._nb_label_frames_1s
            loc_dict = {}
            for audio_frame in range(frame_cnt, frame_cnt + self._nb_label_frames_1s):
                if audio_frame not in _pred_dict:
                    continue
                for value in _pred_dict[audio_frame]:
                    if value[0] not in loc_dict:
                        loc_dict[value[0]] = {}

                    block_frame = audio_frame - frame_cnt
                    if block_frame not in loc_dict[value[0]]:
                        loc_dict[value[0]][block_frame] = []
                    loc_dict[value[0]][block_frame].append(value[1:])

            # Update the block wise details collected above in a global structure
            for class_cnt in loc_dict:
                if class_cnt not in output_dict[block_cnt]:
                    output_dict[block_cnt][class_cnt] = []

                keys = [k for k in loc_dict[class_cnt]]
                values = [loc_dict[class_cnt][k] for k in loc_dict[class_cnt]]

                output_dict[block_cnt][class_cnt].append([keys, values])

        return output_dict

    def organize_labels(self, _pred_dict, _max_frames):
        '''
            Collects class-wise sound event location information in every frame, similar to segment_labels but at frame level
        :param _pred_dict: Dictionary containing frame-wise sound event time and location information. Output of SELD method
        :param _max_frames: Total number of frames in the recording
        :return: Dictionary containing class-wise sound event location information in each frame
                dictionary_name[frame-index][class-index][track-index] = [azimuth, elevation, (distance)] or
                                                                         [x, y, z, (distance)]
        '''
        nb_frames = _max_frames
        output_dict = {x: {} for x in range(nb_frames)}
        for frame_idx in range(0, _max_frames):
            if frame_idx not in _pred_dict:
                continue
            for [class_idx, track_idx, *localization] in _pred_dict[frame_idx]:
                if class_idx not in output_dict[frame_idx]:
                    output_dict[frame_idx][class_idx] = {}

                if track_idx not in output_dict[frame_idx][class_idx]:
                    output_dict[frame_idx][class_idx][track_idx] = localization
                else:
                    # Repeated track_idx for the same class_idx in the same frame_idx, the model is not estimating
                    # track IDs, so track_idx is set to a negative value to distinguish it from a proper track ID
                    min_track_idx = np.min(np.array(list(output_dict[frame_idx][class_idx].keys())))
                    new_track_idx = min_track_idx - 1 if min_track_idx < 0 else -1
                    output_dict[frame_idx][class_idx][new_track_idx] = localization

        return output_dict

    def regression_label_format_to_output_format(self, _sed_labels, _doa_labels):
        """
        Converts the sed (classification) and doa labels predicted in regression format to dcase output format.

        :param _sed_labels: SED labels matrix [nb_frames, nb_classes]
        :param _doa_labels: DOA labels matrix [nb_frames, 2*nb_classes] or [nb_frames, 3*nb_classes]
        :return: _output_dict: returns a dict containing dcase output format
        """

        _nb_classes = self._nb_unique_classes
        _is_polar = _doa_labels.shape[-1] == 2*_nb_classes
        _azi_labels, _ele_labels = None, None
        _x, _y, _z = None, None, None
        if _is_polar:
            _azi_labels = _doa_labels[:, :_nb_classes]
            _ele_labels = _doa_labels[:, _nb_classes:]
        else:
            _x = _doa_labels[:, :_nb_classes]
            _y = _doa_labels[:, _nb_classes:2*_nb_classes]
            _z = _doa_labels[:, 2*_nb_classes:]

        _output_dict = {}
        for _frame_ind in range(_sed_labels.shape[0]):
            _tmp_ind = np.where(_sed_labels[_frame_ind, :])
            if len(_tmp_ind[0]):
                _output_dict[_frame_ind] = []
                for _tmp_class in _tmp_ind[0]:
                    if _is_polar:
                        _output_dict[_frame_ind].append([_tmp_class, _azi_labels[_frame_ind, _tmp_class], _ele_labels[_frame_ind, _tmp_class]])
                    else:
                        _output_dict[_frame_ind].append([_tmp_class, _x[_frame_ind, _tmp_class], _y[_frame_ind, _tmp_class], _z[_frame_ind, _tmp_class]])
        return _output_dict

    def convert_output_format_polar_to_cartesian(self, in_dict):
        out_dict = {}
        for frame_cnt in in_dict.keys():
            if frame_cnt not in out_dict:
                out_dict[frame_cnt] = []
                for tmp_val in in_dict[frame_cnt]:
                    ele_rad = tmp_val[3]*np.pi/180.
                    azi_rad = tmp_val[2]*np.pi/180.

                    tmp_label = np.cos(ele_rad)
                    x = np.cos(azi_rad) * tmp_label
                    y = np.sin(azi_rad) * tmp_label
                    z = np.sin(ele_rad)
                    out_dict[frame_cnt].append(tmp_val[0:2] + [x, y, z] + tmp_val[4:])
        return out_dict

    def convert_output_format_cartesian_to_polar(self, in_dict):
        out_dict = {}
        for frame_cnt in in_dict.keys():
            if frame_cnt not in out_dict:
                out_dict[frame_cnt] = []
                for tmp_val in in_dict[frame_cnt]:
                    x, y, z = tmp_val[2], tmp_val[3], tmp_val[4]

                    # in degrees
                    azimuth = np.arctan2(y, x) * 180 / np.pi
                    elevation = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
                    r = np.sqrt(x**2 + y**2 + z**2)
                    out_dict[frame_cnt].append(tmp_val[0:2] + [azimuth, elevation] + tmp_val[5:])
        return out_dict

    # ------------------------------- Misc public functions -------------------------------

    def get_normalized_feat_dir(self):
        return os.path.join(
            self._feat_label_dir,
            '{}_norm'.format('{}_salsa'.format(self._dataset_combination) if (self._dataset=='mic' and self._use_salsalite) else self._dataset_combination)
        )


    def get_unnormalized_feat_dir(self):
        return os.path.join(
            self._feat_label_dir,
            '{}'.format('{}_salsa'.format(self._dataset_combination) if (self._dataset=='mic' and self._use_salsalite) else self._dataset_combination)
        )


    def get_label_dir(self):
        if self._is_eval:
            return None
        else:
            return os.path.join(
                self._feat_label_dir,
               '{}_label'.format('{}_adpit'.format(self._dataset_combination) if self._multi_accdoa else self._dataset_combination)
        )



    def get_normalized_wts_file(self):
        return os.path.join(
            self._feat_label_dir,
            '{}_wts'.format(self._dataset)
        )



    def get_vid_feat_dir(self):
        return os.path.join(self._feat_label_dir, 'video_{}'.format('eval' if self._is_eval else 'dev'))

    def get_nb_channels(self):
        return self._nb_channels

    def get_nb_classes(self):
        return self._nb_unique_classes

    def nb_frames_1s(self):
        return self._nb_label_frames_1s

    def get_hop_len_sec(self):
        return self._hop_len_s

    def get_nb_mel_bins(self):
        return self._nb_mel_bins

    def get_classes(self):
        return self._unique_classes


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        print('{} folder does not exist, creating it.'.format(folder_name))
        os.makedirs(folder_name)


def delete_and_create_folder(folder_name):
    if os.path.exists(folder_name) and os.path.isdir(folder_name):
        shutil.rmtree(folder_name)
    os.makedirs(folder_name, exist_ok=True)

