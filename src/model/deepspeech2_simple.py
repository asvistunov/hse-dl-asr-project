import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNActivationBlock(nn.Module):
    """
    A convolutional block consisting of Conv2D, BatchNorm, and HardTanh activation,
    commonly used in the early layers of DeepSpeech2.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the convolutional block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, Freq, Time).

        Returns:
            torch.Tensor: Output tensor after Conv2D, BatchNorm, and HardTanh.
        """
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.hardtanh(x, min_val=0.0, max_val=20.0)
        return x


class SimpleDeepSpeech2(nn.Module):
    """
    A simplified implementation of the DeepSpeech2 model,
    adapted for input shape (B, n_mels, time).
    
    Steps:
        1) Add a dummy channel (unsqueeze).
        2) Pass through Conv2D blocks (Conv + BN + HardTanh).
        3) Transpose axes to feed into the RNN along the time axis.
        4) Apply LSTM layers.
        5) Apply a linear layer to generate logits.
        6) Return tensor in the shape (T, B, n_tokens) for CTC loss.
    """

    def __init__(
        self,
        num_mels: int = 80,
        n_tokens: int = 29,
        rnn_hidden_size: int = 512,
        num_rnn_layers: int = 5,
        bidirectional: bool = False,
        dropout_p: float = 0.1
    ):
        super().__init__()

        # --- Convolutional layers ---
        self.conv1 = ConvBNActivationBlock(
            in_channels=1,
            out_channels=32,
            kernel_size=(11, 41),
            stride=(2, 2),
            padding=(5, 20)
        )
        self.conv2 = ConvBNActivationBlock(
            in_channels=32,
            out_channels=32,
            kernel_size=(11, 21),
            stride=(1, 2),
            padding=(5, 10)
        )

        # Estimate the RNN input size:
        # After two convolutional layers, (n_mels, time) is approximately reduced by half.
        rnn_input_dim = 32 * (num_mels // 2)

        # --- RNN ---
        self.rnn = nn.LSTM(
            input_size=rnn_input_dim,
            hidden_size=rnn_hidden_size,
            num_layers=num_rnn_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_p
        )

        rnn_output_dim = rnn_hidden_size * 2 if bidirectional else rnn_hidden_size

        # --- Fully connected layer for classification ---
        self.fc = nn.Linear(rnn_output_dim, n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Forward pass for DeepSpeech2.

        Args:
            spectrogram (torch.Tensor): Input tensor of shape (B, n_mels, time).
            spectrogram_length (torch.Tensor): Lengths of each spectrogram in the batch.

        Returns:
            dict: A dictionary with log probabilities and their lengths.
        """
        # 1) Add a dummy channel dimension: (B, 1, n_mels, time)
        x = spectrogram.unsqueeze(1)

        # 2) Apply convolutional blocks
        x = self.conv1(x)  # Shape: (B, 32, Freq, Time)
        x = self.conv2(x)  # Shape: (B, 32, Freq, Time)

        # 3) Transpose axes so that time becomes the second dimension
        x = x.transpose(2, 3)  # Shape: (B, 32, Time, Freq)

        # Flatten the frequency and channel dimensions into one
        batch_size, channels, time_steps, freq_dim = x.size()
        x = x.transpose(1, 2).contiguous().view(batch_size, time_steps, channels * freq_dim)

        # 4) Apply RNN
        x, _ = self.rnn(x)

        # 5) Apply the fully connected layer to get logits
        logits = self.fc(x)  # Shape: (B, T, n_tokens)

        # 6) Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs_length = self.compute_output_lengths(spectrogram_length)
        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def compute_output_lengths(self, input_lengths):
        """
        Computes the output temporal lengths after convolutional layers.

        Args:
            input_lengths (torch.Tensor): Original input lengths.

        Returns:
            torch.Tensor: Output lengths after compression by convolutional layers.
        """
        return input_lengths // 4
