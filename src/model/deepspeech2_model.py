import torch
import torch.nn as nn
import torch.nn.functional as F

class DS2ConvBlock(nn.Module):
    """
    Блок свёртки + BatchNorm + HardTanh,
    используемый в ранних слоях DeepSpeech2.
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
        x ожидается в формате (B, C, Freq, Time).
        На входе к первым слоям C может быть 1.
        """
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.hardtanh(x, min_val=0.0, max_val=20.0)
        return x


class DeepSpeech2(nn.Module):
    """
    Упрощённая реализация DeepSpeech2, 
    адаптированная к входу вида (B, n_mels, time).
    
    Шаги:
      1) Добавляем фиктивный канал (unsqueeze).
      2) Пропускаем через 2D Conv-блоки (Conv+BN+HardTanh).
      3) Транспонируем оси так, чтобы RNN шла по временной оси.
      4) Пропускаем через LSTM.
      5) Линейный слой -> logits.
      6) Для CTC возвращаем тензор формата (T, B, n_tokens).
    """
    def __init__(
        self,
        num_mels: int = 80,
        n_tokens: int = 29,
        rnn_hidden_size: int = 512,
        num_rnn_layers: int = 5,
        bidirectional: bool = False
    ):
        super().__init__()

        # --- Свёрточная часть ---
        self.conv_block1 = DS2ConvBlock(
            in_channels=1,
            out_channels=32,
            kernel_size=(11, 41),
            stride=(2, 2),
            padding=(5, 20)
        )
        self.conv_block2 = DS2ConvBlock(
            in_channels=32,
            out_channels=32,
            kernel_size=(11, 21),
            stride=(1, 2),
            padding=(5, 10)
        )

        # Приблизительно считаем размер входа в RNN:
        # в DS2 обычно после 2 свёрток пространство (n_mels, time) уменьшается.
        # Для примера берём упрощённую оценку:
        #   rnn_input_size = 32 * (num_mels // 4)
        # Но на практике вычисляйте точнее (зависит от stride/padding).
        rnn_input_size = 32 * (num_mels // 2)

        # --- RNN ---
        self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=num_rnn_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        rnn_output_size = rnn_hidden_size * 2 if bidirectional else rnn_hidden_size

        # --- Последний линейный слой (классификатор для CTC) ---
        self.fc = nn.Linear(rnn_output_size, n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        x: (B, n_mels, time)
           Входная спектрограмма без канального измерения.
        input_lengths: (B,) — (опционально) длины по временной оси до паддинга.

        return: (T, B, n_tokens) — для CTC.
        """
        # 1) Добавляем фиктивный канал: (B, 1, n_mels, time)
        x = spectrogram.unsqueeze(1)  #  torch.Size([2, 1, 128, 352])

        # 2) Свёрточные блоки
        x = self.conv_block1(x)   # torch.Size([2, 32, 64, 176])
        x = self.conv_block2(x)   # torch.Size([2, 32, 64, 88])

        # 3) Меняем оси так, чтобы "time" стал вторым измерением 
        #    и шёл в RNN по временной оси.
        # Сейчас x: (B, 32, freq, time).
        # Сначала поменяем местами freq и time:
        x = x.transpose(2, 3)  # torch.Size([2, 32, 25, 40])

        # Теперь x: (B, C=32, T, Freq).
        # RNN обычно ожидает (B, T, Feature). 
        # Сдвигаем каналы и частотную ось в одно измерение:
        B, C, T, Freq = x.size()
        x = x.transpose(1, 2)       # (B, T, C, Freq) -> меняем местами ось 1 и 2 -> torch.Size([2, 25, 32, 40])
        x = x.contiguous().view(B, T, C * Freq)  # (B, T, C*Freq) -> torch.Size([2, 25, 1280])

        # 4) RNN
        x, _ = self.rnn(x)          # (B, T, rnn_hidden_size * num_directions)

        # 5) Линейный слой -> логиты
        x = self.fc(x)              # (B, T, n_tokens)

        log_probs = nn.functional.log_softmax(x, dim=-1)
        log_probs_length = self.transform_input_lengths(spectrogram_length)
        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        return input_lengths // 4



# Пример использования
if __name__ == "__main__":
    # Допустим, у нас мел-спектрограмма  (batch_size=2, n_mels=80, time=100)
    batch_size = 2
    time_steps = 100
    num_mels = 80
    dummy_input = torch.randn(batch_size, num_mels, time_steps)

    model = DeepSpeech2(num_mels=num_mels, n_tokens=29)
    logits = model(dummy_input)  # shape: (time_out, B, n_tokens)
    print("Shape of output logits:", logits.shape)
