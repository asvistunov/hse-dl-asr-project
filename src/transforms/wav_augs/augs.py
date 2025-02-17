import torch_audiomentations
from torch import Tensor, nn


class HighPassFilter(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.HighPassFilter(sample_rate=16000, *args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)

class PitchShift(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.PitchShift(sample_rate=16000, *args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)

class PolarityInversion(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.PolarityInversion(sample_rate=16000, *args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)


class TimeInversion(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = torch_audiomentations.TimeInversion(sample_rate=16000, *args, **kwargs)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)