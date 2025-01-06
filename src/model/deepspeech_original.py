import torch
import torch.nn as nn
import torch.nn.functional as F

# Source: https://github.com/SeanNaren/deepspeech.pytorch/blob/master/deepspeech_pytorch/model.py
# This code was adapted and rewritten for improved readability, with additional docstrings and comments added.

class MaskConv(nn.Module):
    """
    Applies a sequence of convolutional layers and masks the output sequences
    to ensure the results remain consistent when batch sizes vary.

    Args:
        seq_module (nn.Sequential): A sequential container with convolutional layers.
    """

    def __init__(self, seq_module):
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        """
        Forward pass of MaskConv with masking.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, D, T).
            lengths (torch.Tensor): Actual sequence lengths before padding.

        Returns:
            tuple: Masked output tensor and updated sequence lengths.
        """
        for module in self.seq_module:
            x = module(x)
            mask = torch.zeros_like(x, dtype=torch.bool)
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(lengths):
                length = length.item()
                if mask[i].size(2) > length:
                    mask[i][:, :, length:].fill_(True)
            x = x.masked_fill(mask, 0)
        return x, lengths


class SequenceWise(nn.Module):
    """
    Applies a given module on a per-time-step basis by collapsing the time and batch dimensions,
    applying the module, and restoring the original dimensions.

    Args:
        module (nn.Module): The module to be applied to each time step.
    """

    def __init__(self, module):
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        """
        Forward pass of the SequenceWise module.

        Args:
            x (torch.Tensor): Input tensor of shape (T, N, H).

        Returns:
            torch.Tensor: Output tensor of shape (T, N, H) after applying the module.
        """
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        return x.view(t, n, -1)

    def __repr__(self):
        return f"{self.__class__.__name__}(\n  {self.module}\n)"



class BatchRNN(nn.Module):
    """
    A recurrent neural network layer with optional batch normalization
    and support for bidirectional RNNs.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of hidden units in the RNN.
        rnn_type (nn.Module): Type of RNN to use (default: nn.LSTM).
        bidirectional (bool): Whether to use a bidirectional RNN.
        batch_norm (bool): Whether to apply batch normalization.
    """

    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=True)

    def flatten_parameters(self):
        """
        Flattens RNN parameters for faster execution.
        """
        self.rnn.flatten_parameters()

    def forward(self, x, output_lengths, h=None):
        """
        Forward pass of the BatchRNN.

        Args:
            x (torch.Tensor): Input tensor of shape (T, N, H).
            output_lengths (torch.Tensor): Lengths of sequences before padding.
            h (torch.Tensor, optional): Initial hidden state.

        Returns:
            tuple: Output tensor of shape (T, N, H) and hidden state.
        """
        if self.batch_norm:
            x = self.batch_norm(x)
        packed_x = nn.utils.rnn.pack_padded_sequence(x, output_lengths, enforce_sorted=False)
        packed_out, h = self.rnn(packed_x, h)
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_out)
        
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(dim=2)
        return x, h


class Lookahead(nn.Module):
    """
    Implements a lookahead convolution layer for unidirectional RNNs to
    provide context beyond the current time step.

    Args:
        n_features (int): Number of input features.
        context (int): Context window size.
    """

    def __init__(self, n_features, context):
        super(Lookahead, self).__init__()
        assert context > 0, "Context size must be positive."
        self.context = context
        self.pad = (0, context - 1)
        self.conv = nn.Conv1d(
            n_features, n_features,
            kernel_size=context, stride=1,
            groups=n_features, bias=False
        )

    def forward(self, x):
        """
        Forward pass of Lookahead.

        Args:
            x (torch.Tensor): Input tensor of shape (T, N, H).

        Returns:
            torch.Tensor: Output tensor with lookahead context applied.
        """
        x = x.transpose(0, 1).transpose(1, 2)
        x = F.pad(x, self.pad, value=0)
        x = self.conv(x)
        return x.transpose(1, 2).transpose(0, 1).contiguous()

    def __repr__(self):
        return f"{self.__class__.__name__}(n_features={self.conv.in_channels}, context={self.context})"


class DeepSpeech(nn.Module):
    """
    DeepSpeech-style model for end-to-end speech recognition.

    Args:
        n_tokens (int): Number of output tokens.
        hidden_size (int): Number of hidden units in RNN layers.
        hidden_layers (int): Number of RNN layers.
        lookahead_context (int): Context size for the lookahead layer.
        pad_id (int, optional): Padding index for token sequences (default: 0).
    """

    def __init__(self, n_tokens, hidden_size, hidden_layers, lookahead_context, pad_id=0):
        super(DeepSpeech, self).__init__()
        self.bidirectional = True

        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        ))

        rnn_input_size = 1024
        self.rnns = nn.Sequential(
            BatchRNN(rnn_input_size, hidden_size, bidirectional=self.bidirectional, batch_norm=False),
            *(BatchRNN(hidden_size, hidden_size, bidirectional=self.bidirectional) for _ in range(hidden_layers - 1))
        )

        self.lookahead = nn.Sequential(
            Lookahead(hidden_size, lookahead_context),
            nn.Hardtanh(0, 20, inplace=True)
        ) if not self.bidirectional else None

        self.fc = nn.Sequential(
            SequenceWise(nn.Sequential(
                nn.BatchNorm1d(hidden_size),
                nn.Linear(hidden_size, n_tokens, bias=False)
            ))
        )

    def forward(self, spectrogram, spectrogram_length, hs=None, **batch):
        """
        Forward pass of the DeepSpeech model.

        Args:
            spectrogram (torch.Tensor): Input spectrogram of shape (B, D, T).
            spectrogram_length (torch.Tensor): Lengths of each spectrogram in the batch.
            hs (list, optional): Initial hidden states for the RNN layers.

        Returns:
            dict: A dictionary containing:
                - logits (torch.Tensor): Output logits of shape (T, B, n_tokens).
                - log_probs (torch.Tensor): Log probabilities after applying log softmax.
                - log_probs_length (torch.Tensor): Lengths of log probabilities.
        """
        lengths = spectrogram_length.cpu().int()
        output_lengths = self.get_seq_lens(lengths)
        x, _ = self.conv(spectrogram.unsqueeze(1), output_lengths)

        B, C, D, T = x.size()
        x = x.view(B, C * D, T).transpose(1, 2).transpose(0, 1).contiguous()

        if hs is None:
            hs = [None] * len(self.rnns)

        new_hs = []
        for i, rnn in enumerate(self.rnns):
            x, h = rnn(x, output_lengths, hs[i])
            new_hs.append(h)

        if not self.bidirectional:
            x = self.lookahead(x)

        x = self.fc(x).transpose(0, 1)
        return {"logits": x, "log_probs": F.log_softmax(x, dim=-1), "log_probs_length": output_lengths}

    def get_seq_lens(self, input_length):
        """
        Computes the output sequence lengths after convolutional layers.

        Args:
            input_length (torch.Tensor): Input sequence lengths.

        Returns:
            torch.Tensor: Output sequence lengths after convolutional compression.
        """
        for layer in self.conv.modules():
            if isinstance(layer, nn.Conv2d):
                input_length = ((input_length + 2 * layer.padding[1] - layer.dilation[1] * (layer.kernel_size[1] - 1) - 1) // layer.stride[1]) + 1
        return input_length.int()
