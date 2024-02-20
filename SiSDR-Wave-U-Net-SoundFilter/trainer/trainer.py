# FUSS: Format:  All audio clips are provided as uncompressed PCM 16 bit, 16 kHz, mono audio files.
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from trainer.base_trainer import BaseTrainer
from util.utils import compute_SISDR, compute_SNR, compute_SDR, compute_SISNR
plt.switch_backend('agg')


class Trainer(BaseTrainer):
    def __init__(
            self,
            config,
            resume: bool,
            model,
            loss_function,
            optimizer,
            train_dataloader,
            validation_dataloader,
    ):
        super(Trainer, self).__init__(config, resume, model, loss_function, optimizer)
        self.train_data_loader = train_dataloader
        self.validation_data_loader = validation_dataloader
        self.loss_function = self.loss_function.cuda()

    def _train_epoch(self, epoch):
        loss_total = 0.0
        self.model = self.model.double()
        for i, (mixture, clean, condition1, condition2, name) in enumerate(tqdm(self.train_data_loader)):
            mixture = mixture.to(self.device)
            clean = clean.to(self.device)

            self.optimizer.zero_grad()
            enhanced = self.model(mixture,condition1.cuda(), condition2.cuda())
            clean = clean.double()
            loss = self.loss_function(clean.cuda(),enhanced.cuda()).cuda()
            loss.backward()
            self.optimizer.step()

            loss_total += loss.item()

        dl_len = len(self.train_data_loader)
        print("Train loss:", epoch,":",loss_total/dl_len)
        self.writer.add_scalar(f"Train/Loss", loss_total / dl_len, epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        val_loss_total = 0.0 
        visualize_audio_limit = self.validation_custom_config["visualize_audio_limit"]
        visualize_waveform_limit = self.validation_custom_config["visualize_waveform_limit"]
        visualize_spectrogram_limit = self.validation_custom_config["visualize_spectrogram_limit"]

        sample_length = self.validation_custom_config["sample_length"]

        stoi_c_n = []  # clean and noisy
        stoi_c_e = []  # clean and enhanced
        sisdr_c_n = [] # clean and noisy
        sisdr_c_e = [] # clean and enhanced
        snr_c_n = [] # clean and noisy
        snr_c_e = [] # clean and enhanced
        sisnr_c_n = [] # clean and noisy
        sisnr_c_e = [] # clean and enhanced
        sdr_c_n = [] # clean and noisy
        sdr_c_e = [] # clean and enhanced
        # pesq_c_n = []
        # pesq_c_e = []

        for i, (mixture, clean, condition1, condition2, name) in enumerate(tqdm(self.validation_data_loader)):
            assert len(name) == 1, "Only support batch size is 1 in enhancement stage."
            name = name[0]
            padded_length = 0

            mixture = mixture.to(self.device)  # [1, 1, T]

            # The input of the model should be fixed length.
            if mixture.size(-1) % sample_length != 0:
                padded_length = sample_length - (mixture.size(-1) % sample_length)
                mixture = torch.cat([mixture, torch.zeros(1, 1, padded_length, device=self.device)], dim=-1) 
                #if mixture is smaller than sample_length -> padded to sample_length
                #if mixture is longer than sample_length -> padded to n*sample_length
                

            assert mixture.size(-1) % sample_length == 0 and mixture.dim() == 3
            # split n*sample_length into chunks, feed each chunk to model one-by-one 
            # and concatenate the output chunks
            mixture_chunks = list(torch.split(mixture, sample_length, dim=-1))

            enhanced_chunks = []
            
            for chunk in mixture_chunks:
                enhanced_chunks.append(self.model(chunk,condition1.cuda(), condition2.cuda()).detach())

            enhanced = torch.cat(enhanced_chunks, dim=-1)  # [1, 1, T]
            enhanced = enhanced if padded_length == 0 else enhanced[:, :, :-padded_length]
            mixture = mixture if padded_length == 0 else mixture[:, :, :-padded_length]

            val_loss = self.loss_function(clean.cuda(),enhanced.cuda()).cuda()
            val_loss_total += val_loss.item()

            #For Calculating SI-SDR METRICS
            sisdr_clean = torch.reshape(clean, (1,1,-1)).cpu()
            sisdr_enhanced = torch.reshape(enhanced, (1,1,-1)).cpu()
            sisdr_mixture = torch.reshape(mixture, (1,1,-1)).cpu()

            enhanced = enhanced.cpu().reshape(-1).numpy()
            clean = clean.numpy().reshape(-1)
            mixture = mixture.cpu().numpy().reshape(-1)
            assert len(mixture) == len(enhanced) == len(clean)

            # Visualize audio
            if i <= visualize_audio_limit:
                self.writer.add_audio(f"Speech/{name}_Noisy", mixture, epoch, sample_rate=8000)
                self.writer.add_audio(f"Speech/{name}_Enhanced", enhanced, epoch, sample_rate=8000)
                self.writer.add_audio(f"Speech/{name}_Clean", clean, epoch, sample_rate=8000)

            # Visualize waveform
            if i <= visualize_waveform_limit:
                fig, ax = plt.subplots(3, 1)
                for j, y in enumerate([mixture, enhanced, clean]):
                    ax[j].set_title("mean: {:.3f}, std: {:.3f}, max: {:.3f}, min: {:.3f}".format(
                        np.mean(y),
                        np.std(y),
                        np.max(y),
                        np.min(y)
                    ))
                    librosa.display.waveshow(y, sr=8000, ax=ax[j])
                plt.tight_layout()
                self.writer.add_figure(f"Waveform/{name}", fig, epoch)

            # Visualize spectrogram
            noisy_mag, _ = librosa.magphase(librosa.stft(mixture, n_fft=320, hop_length=160, win_length=320))
            enhanced_mag, _ = librosa.magphase(librosa.stft(enhanced, n_fft=320, hop_length=160, win_length=320))
            clean_mag, _ = librosa.magphase(librosa.stft(clean, n_fft=320, hop_length=160, win_length=320))

            if i <= visualize_spectrogram_limit:
                fig, axes = plt.subplots(3, 1, figsize=(6, 6))
                for k, mag in enumerate([
                    noisy_mag,
                    enhanced_mag,
                    clean_mag,
                ]):
                    axes[k].set_title(f"mean: {np.mean(mag):.3f}, "
                                      f"std: {np.std(mag):.3f}, "
                                      f"max: {np.max(mag):.3f}, "
                                      f"min: {np.min(mag):.3f}")
                    librosa.display.specshow(librosa.amplitude_to_db(mag), cmap="magma", y_axis="linear", ax=axes[k], sr=8000)
                plt.tight_layout()
                self.writer.add_figure(f"Spectrogram/{name}", fig, epoch)

            # Metric
            #stoi_c_n.append(compute_STOI(clean, mixture, sr=8000))
            #stoi_c_e.append(compute_STOI(clean, enhanced, sr=8000))
            sisdr_c_n.append(compute_SISDR(sisdr_clean, sisdr_mixture))
            sisdr_c_e.append(compute_SISDR(sisdr_clean, sisdr_enhanced))
            snr_c_n.append(compute_SNR(sisdr_clean, sisdr_mixture))
            snr_c_e.append(compute_SNR(sisdr_clean, sisdr_enhanced))
            sdr_c_n.append(compute_SDR(sisdr_clean, sisdr_mixture))
            sdr_c_e.append(compute_SDR(sisdr_clean, sisdr_enhanced))
            sisnr_c_n.append(compute_SISNR(sisdr_clean, sisdr_mixture))
            sisnr_c_e.append(compute_SISNR(sisdr_clean, sisdr_enhanced))

            # pesq_c_n.append(compute_PESQ(clean, mixture, sr=8000))
            # pesq_c_e.append(compute_PESQ(clean, enhanced, sr=8000))

        val_dl_len = len(self.validation_data_loader)
        self.writer.add_scalar(f"Val/Loss",val_loss_total/val_dl_len ,epoch)
        print("Validation loss:", epoch,":",val_loss_total/val_dl_len)
        get_metrics_ave = lambda metrics: np.sum(metrics) / len(metrics)
        #self.writer.add_scalars(f"Metric/STOI", {
        #    "Clean and noisy": get_metrics_ave(stoi_c_n),
        #    "Clean and enhanced": get_metrics_ave(stoi_c_e)
        #}, epoch)

        self.writer.add_scalars(f"Metric/SI-SDR", {
            "Clean and noisy": get_metrics_ave(sisdr_c_n),
            "Clean and enhanced": get_metrics_ave(sisdr_c_e)
        }, epoch)

        self.writer.add_scalars(f"Metric/SI-SNR", {
            "Clean and noisy": get_metrics_ave(sisnr_c_n),
            "Clean and enhanced": get_metrics_ave(sisnr_c_e)
        }, epoch)

        self.writer.add_scalars(f"Metric/SNR", {
            "Clean and noisy": get_metrics_ave(snr_c_n),
            "Clean and enhanced": get_metrics_ave(snr_c_e)
        }, epoch)

        self.writer.add_scalars(f"Metric/SDR", {
            "Clean and noisy": get_metrics_ave(sdr_c_n),
            "Clean and enhanced": get_metrics_ave(sdr_c_e)
        }, epoch)

        # self.writer.add_scalars(f"Metric/PESQ", {
        #     "Clean and noisy": get_metrics_ave(pesq_c_n),
        #     "Clean and enhanced": get_metrics_ave(pesq_c_e)
        # }, epoch)

        # score = (get_metrics_ave(stoi_c_e) + self._transform_pesq_range(get_metrics_ave(pesq_c_e))) / 2
        score = get_metrics_ave(sisdr_c_e) #Updated score for non-speech data
        return score
