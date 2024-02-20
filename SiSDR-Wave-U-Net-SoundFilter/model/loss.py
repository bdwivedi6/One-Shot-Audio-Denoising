import torch
#from asteroid.losses import PITLossWrapper, sdr
from torchmetrics import ScaleInvariantSignalDistortionRatio

def mse_loss():
    return torch.nn.MSELoss()

def l1_loss():
    return torch.nn.L1Loss()

def sisdr_loss():
    si_sdr = ScaleInvariantSignalDistortionRatio()
    return -si_sdr


def soft_clip_sisdr_loss(source, estimate, clip_threshold=10, alpha=1):
    """
    Calculate the SI-SDR loss with soft clipping using the logarithmic function.
    
    Args:
        source (torch.Tensor): Ground truth source signals (batch_size, num_channels, num_samples).
        estimate (torch.Tensor): Estimated source signals (batch_size, num_channels, num_samples).
        clip_threshold (float): Threshold for soft clipping (default: 10).
        alpha (float): Shape parameter for soft clipping (default: 1).
    
    Returns:
        torch.Tensor: Soft clipped SI-SDR loss.
    """
    # Ensure the tensors have the same shape
    assert source.shape == estimate.shape
    
    # Compute the numerator and denominator of SI-SDR
    target_energy = torch.sum(source ** 2, dim=2, keepdim=True)
    projection = torch.sum(source * estimate, dim=2, keepdim=True) * source / target_energy
    interference = estimate - projection
    numerator = torch.sum(projection ** 2, dim=2)
    denominator = torch.sum(interference ** 2, dim=2)
    
    # Avoid division by zero
    epsilon = 1e-10
    numerator = torch.where(numerator < epsilon, epsilon, numerator)
    denominator = torch.where(denominator < epsilon, epsilon, denominator)
    
    # Compute SI-SDR
    sisdr = 10 * torch.log10(numerator / denominator)
    
    # Apply soft clipping
    soft_clip_sisdr = torch.where(sisdr > clip_threshold, 
                                  clip_threshold + (1 / alpha) * torch.log(1 + alpha * (sisdr - clip_threshold)), 
                                  sisdr)
    
    # Return negative mean soft clipped SI-SDR as loss
    return -torch.mean(soft_clip_sisdr)


#def sisdr_loss_asteroid(): #https://asteroid.readthedocs.io/en/v0.3.2/_modules/asteroid/losses/sdr.html
#    sisdr_loss_fn = PITLossWrapper(sdr.PairwiseNegSDR("sisdr"),pit_from='pw_mtx')
    # sisdr_loss = sisdr_loss_fn(noisy_signal,clean_signal)
#    return -sisdr_loss_fn



#def snr_loss(): #https://asteroid.readthedocs.io/en/v0.3.2/_modules/asteroid/losses/sdr.html
#    snr_loss_fn = PITLossWrapper(sdr.PairwiseNegSDR("snr"),pit_from='pw_mtx')
#    # sisdr_loss = sisdr_loss_fn(noisy_signal,clean_signal)
#    return snr_loss_fn

# def sisdr_loss_aura(): #https://github.com/csteinmetz1/auraloss
#     import auraloss
#     sisdr = auraloss.time.SISDRLoss()
#     # input = torch.rand(8, 1, 44100)
#     # target = torch.rand(8, 1, 44100)
#     loss = sisdr(input, target)
#     return loss

# def snr_loss_aura():
#     import auraloss
#     snr =     auraloss.time.SNRLoss()
#     # input = torch.rand(8, 1, 44100)
#     # target = torch.rand(8, 1, 44100)
#     loss = snr(input, target)
#     return loss
