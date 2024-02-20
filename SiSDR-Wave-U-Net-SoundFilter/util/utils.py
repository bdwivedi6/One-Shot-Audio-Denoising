import importlib
import time
import os
import random
import torch
from pesq import pesq
import numpy as np
from pystoi.stoi import stoi
#from asteroid.losses import PITLossWrapper, sdr
from torchmetrics.audio import SignalDistortionRatio, ScaleInvariantSignalDistortionRatio, SignalNoiseRatio, ScaleInvariantSignalNoiseRatio



def load_checkpoint(checkpoint_path, device):
    _, ext = os.path.splitext(os.path.basename(checkpoint_path))
    assert ext in (".pth", ".tar"), "Only support ext and tar extensions of model checkpoint."
    model_checkpoint = torch.load(checkpoint_path, map_location=device)

    if ext == ".pth":
        print(f"Loading {checkpoint_path}.")
        return model_checkpoint
    else:  # tar
        print(f"Loading {checkpoint_path}, epoch = {model_checkpoint['epoch']}.")
        return model_checkpoint["model"]


def prepare_empty_dir(dirs, resume=False):
    """
    if resume experiment, assert the dirs exist,
    if not resume experiment, make dirs.

    Args:
        dirs (list): directors list
        resume (bool): whether to resume experiment, default is False
    """
    for dir_path in dirs:
        if resume:
            assert dir_path.exists()
        else:
            dir_path.mkdir(parents=True, exist_ok=True)


class ExecutionTime:
    """
    Usage:
        timer = ExecutionTime()
        <Something...>
        print(f'Finished in {timer.duration()} seconds.')
    """

    def __init__(self):
        self.start_time = time.time()

    def duration(self):
        return int(time.time() - self.start_time)


def initialize_config(module_cfg, pass_args=True):
    """According to config items, load specific module dynamically with params.
    e.g., Config items as follow：
        module_cfg = {
            "module": "model.model",
            "main": "Model",
            "args": {...}
        }
    1. Load the module corresponding to the "module" param.
    2. Call function (or instantiate class) corresponding to the "main" param.
    3. Send the param (in "args") into the function (or class) when calling ( or instantiating)
    """
    module = importlib.import_module(module_cfg["module"])

    if pass_args:
        return getattr(module, module_cfg["main"])(**module_cfg["args"])
    else:
        return getattr(module, module_cfg["main"])



def compute_PESQ(clean_signal, noisy_signal, sr=16000):
    if(sr == 16000):
        band = "wb"
    else:
        band = "nb"
    return pesq(sr, clean_signal, noisy_signal, band)


def z_score(m):
    mean = np.mean(m)
    std_var = np.std(m)
    return (m - mean) / std_var, mean, std_var


def reverse_z_score(m, mean, std_var):
    return m * std_var + mean


def min_max(m):
    m_max = np.max(m)
    m_min = np.min(m)

    return (m - m_min) / (m_max - m_min), m_max, m_min


def reverse_min_max(m, m_max, m_min):
    return m * (m_max - m_min) + m_min


def sample_fixed_length_data_aligned_train(data_a, data_b, sample_length): #(mix, clean, sample_length)
    """sample with fixed length from two dataset
    """
    assert len(data_a) == len(data_b), "Inconsistent dataset length, unable to sampling"
    # assert len(data_a) >= sample_length, f"len(data_a) is {len(data_a)}, sample_length is {sample_length}."


    if(len(data_b) < 2 * sample_length):
        padded_length =  2 * sample_length - len(data_b)
        #print("*************", data_b.shape)
        data_b = np.concatenate((data_b, np.zeros((padded_length,))), axis=-1)
        data_a = np.concatenate((data_a, np.zeros((padded_length,))), axis=-1)

    #print("****************",len(data_b))
    # To keep target audio and conditioning audio crops disjoint
    p = random.choice([0, 1])
    frames_total = len(data_a)

    if(p == 0):
        start = np.random.randint(int(np.ceil(frames_total/2)) - sample_length + 1)
        end = start + sample_length
        start_conditioning1 = np.random.randint(end,frames_total - sample_length + 1)
        end_conditioning1 = start_conditioning1 + sample_length
        start_conditioning2 = np.random.randint(end,frames_total - sample_length + 1)
        end_conditioning2 = start_conditioning2 + sample_length

    else:
        start = np.random.randint(int(np.ceil(frames_total/2)),frames_total - sample_length + 1)
        end = start + sample_length
        start_conditioning1 = np.random.randint(start-sample_length+1)
        end_conditioning1 = start_conditioning1 + sample_length
        start_conditioning2 = np.random.randint(start-sample_length+1)
        end_conditioning2 = start_conditioning2 + sample_length

    return data_a[start:end], data_b[start:end], data_b[start_conditioning1:end_conditioning1], data_b[start_conditioning2:end_conditioning2]

def sample_fixed_length_data_aligned_val(data_a, data_b, sample_length): #(mix, clean, sample_length)
    """sample with fixed length from two dataset
    """
    #assert len(data_a) == len(data_b), "Inconsistent dataset length, unable to sampling"
    #assert len(data_a) >= sample_length, f"len(data_a) is {len(data_a)}, sample_length is {sample_length}."

    ###################################################

    if(len(data_b) < int(2 * sample_length)):
        padded_length = int(2 * sample_length) - len(data_b)
        data_b = np.concatenate((data_b, np.zeros((padded_length,))), axis=-1)
        data_a = np.concatenate((data_a, np.zeros((padded_length,))), axis=-1)

    frames_total = len(data_a)

    # start = np.random.randint(frames_total - sample_length + 1)
    # print(f"Random crop from: {start}")
    # end = start + sample_length
    start_conditioning1 = np.random.randint(frames_total - sample_length + 1)
    end_conditioning1 = start_conditioning1 + sample_length
    start_conditioning2 = np.random.randint(frames_total - sample_length + 1)
    end_conditioning2 = start_conditioning2 + sample_length
    ####################################################

    return data_b[start_conditioning1:end_conditioning1], data_b[start_conditioning2:end_conditioning2] 


#def compute_STOI(clean_signal, noisy_signal, sr=16000):
#    return stoi(clean_signal, noisy_signal, sr, extended=False)


def compute_SISDR(clean_signal, noisy_signal):
    sisdr_loss_fn = ScaleInvariantSignalDistortionRatio()
    sisdr_loss = sisdr_loss_fn(clean_signal,noisy_signal)
    return sisdr_loss

def compute_SDR(clean_signal, noisy_signal):
    sdr_loss_fn = SignalDistortionRatio()
    sdr_loss = sdr_loss_fn(clean_signal,noisy_signal)
    return sdr_loss

def compute_SNR(clean_signal, noisy_signal):
    snr_loss_fn = SignalNoiseRatio()
    snr_loss = snr_loss_fn(clean_signal,noisy_signal)
    return snr_loss

def compute_SISNR(clean_signal, noisy_signal):
    sisnr_loss_fn = ScaleInvariantSignalNoiseRatio()
    sisnr_loss = sisnr_loss_fn(clean_signal,noisy_signal)
    return sisnr_loss

#def compute_SISDR(clea  n_signal, noisy_signal):
    # print("*****", clean_signal.shape, noisy_signal.shape) #(119884,) (119884,)
#    sisdr_loss_fn = PITLossWrapper(sdr.PairwiseNegSDR("sisdr"),pit_from='pw_mtx')
#    sisdr_loss = sisdr_loss_fn(clean_signal,noisy_signal)
#    return sisdr_loss

#def compute_SNR(clean_signal, noisy_signal):
    # print("*****", clean_signal.shape, noisy_signal.shape) #(119884,) (119884,)
#    snr_loss_fn = PITLossWrapper(sdr.PairwiseNegSDR("snr"),pit_from='pw_mtx')
#   snr_loss = snr_loss_fn(clean_signal,noisy_signal)
#    return snr_loss

#def compute_SDR(clean_signal, noisy_signal):
    # print("*****", clean_signal.shape, noisy_signal.shape) #(119884,) (119884,)
#    sdr_loss_fn = PITLossWrapper(sdr.PairwiseNegSDR("sdsdr"),pit_from='pw_mtx')
#    sdr_loss = sdr_loss_fn(clean_signal,noisy_signal)
#    return sdr_loss

def print_tensor_info(tensor, flag="Tensor"):
    floor_tensor = lambda float_tensor: int(float(float_tensor) * 1000) / 1000
    print(flag)
    print(
        f"\tmax: {floor_tensor(torch.max(tensor))}, min: {float(torch.min(tensor))}, mean: {floor_tensor(torch.mean(tensor))}, std: {floor_tensor(torch.std(tensor))}")
