import os

import librosa
from torch.utils import data

from util.utils import sample_fixed_length_data_aligned_train,sample_fixed_length_data_aligned_val



class Dataset(data.Dataset):
    def __init__(self,
                 dataset,
                 limit=None,
                 offset=0,
                 sample_length=8000,
                 mode="train"):
        """Construct dataset for training and validation.
        Args:
            dataset (str): *.txt, the path of the dataset list file. See "Notes."
            limit (int): Return at most limit files in the list. If None, all files are returned.
            offset (int): Return files starting at an offset within the list. Use negative values to offset from the end of the list.
            sample_length(int): The model only supports fixed-length input. Use sample_length to specify the feature size of the input.
            mode(str): If mode is "train", return fixed-length signals. If mode is "validation", return original-length signals.

        Notes:
            dataset list file：
            <noisy_1_path><space><clean_1_path>
            <noisy_2_path><space><clean_2_path>
            ...
            <noisy_n_path><space><clean_n_path>

            e.g.
            /train/noisy/a.wav /train/clean/a.wav
            /train/noisy/b.wav /train/clean/b.wav
            ...

        Return:
            (mixture signals, clean signals, filename)
        """
        super(Dataset, self).__init__()
        dataset_list = [line.rstrip('\n') for line in open(os.path.abspath(os.path.expanduser(dataset)), "r")]

        dataset_list = dataset_list[offset:]
        if limit:
            dataset_list = dataset_list[:limit]

        assert mode in ("train", "validation"), "Mode must be one of 'train' or 'validation'."

        self.length = len(dataset_list)
        self.dataset_list = dataset_list
        self.sample_length = sample_length
        self.mode = mode

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        mixture_path, clean_path = self.dataset_list[item].split(" ")
        filename = os.path.splitext(os.path.basename(mixture_path))[0]
        mixture, _ = librosa.load(os.path.abspath(os.path.expanduser(mixture_path)), sr=8000) #Load an audio file as a floating point time series.Audio will be automatically resampled to the given rate (default sr=22050).
        clean, _ = librosa.load(os.path.abspath(os.path.expanduser(clean_path)), sr=8000)

        if self.mode == "train":
            # The input of model should be fixed-length in the training.
            mixture, clean, condition1, condition2 = sample_fixed_length_data_aligned_train(mixture, clean, self.sample_length)
            return mixture.reshape(1, -1), clean.reshape(1, -1), condition1.reshape(1,-1), condition2.reshape(1,-1),filename
        else:
            condition1, condition2 = sample_fixed_length_data_aligned_val(mixture, clean, self.sample_length)
            return mixture.reshape(1, -1), clean.reshape(1, -1),condition1.reshape(1,-1), condition2.reshape(1,-1), filename
