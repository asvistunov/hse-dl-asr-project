import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    audios = [item["audio"] for item in dataset_items]
    audio_paths = [item["audio_path"] for item in dataset_items]
    texts = [item["text"] for item in dataset_items]

    spectrograms = [item["spectrogram"] for item in dataset_items]
    spectrogram_lengths = [spec.shape[-1] for spec in spectrograms]

    text_encoded_list = [item["text_encoded"] for item in dataset_items]
    text_encoded_lengths = [enc.shape[-1] for enc in text_encoded_list]

    max_time = max(spec.shape[-1] for spec in spectrograms)

    # Сформируем список падженых спектрограмм
    padded_spectrograms = []
    for spec in spectrograms:
        # spec имеет форму (1, n_mels, time)
        # Нужно допадить по последней оси (time) до max_time
        current_time = spec.shape[-1]
        pad_size = max_time - current_time
        if pad_size > 0:
            # pad принимает аргументы в порядке (left, right, top, bottom, ...)
            # здесь паддим только по последней размерности (time)
            spec_padded = F.pad(spec, (0, pad_size), mode="constant", value=0.0)
        else:
            spec_padded = spec

        padded_spectrograms.append(spec_padded)

    spectrograms_batched = torch.stack(padded_spectrograms, dim=0)

    text_encoded_padded = pad_sequence(
        [enc.squeeze(0) for enc in text_encoded_list],
        batch_first=True,
        padding_value=0
    )

    spectrogram_lengths = torch.tensor(spectrogram_lengths, dtype=torch.long)
    text_encoded_lengths = torch.tensor(text_encoded_lengths, dtype=torch.long)


    batch = {
        "audio": audios,  # список необработанных аудио (можете здесь тоже паддить, если нужно)
        "audio_path": audio_paths,
        "text": texts,   # список истинных строк
        # Спектрограммы + длины
        "spectrogram": spectrograms_batched.squeeze(1),                # (B, 1, n_mels, max_time)
        "spectrogram_length": spectrogram_lengths,          # (B,)
        # Тексты в виде индексных последовательностей + длины
        "text_encoded": text_encoded_padded,                # (B, max_text_len)
        "text_encoded_length": text_encoded_lengths         # (B,)
    }

    return batch
