import torch
import torch.nn.functional as F
from typing import List, Dict

def collate_fn(dataset_items: List[Dict], padding_values: Dict[str, int] = None):
    """
    Collates and pads fields from a list of dataset items into a batch suitable for model input.

    This function handles both tensor-based fields (e.g., 'audio', 'spectrogram', 'text_encoded') 
    and non-tensor fields (e.g., 'text', 'audio_path'). Tensor fields are padded to the length 
    of the longest item in the batch, while non-tensor fields are collected as lists.

    Args:
        dataset_items (List[Dict]): A list of dataset items, where each item is a dictionary 
            containing fields to be collated.
        padding_values (Dict[str, int], optional): A dictionary specifying padding values 
            for specific fields. If a field is not specified, it defaults to padding with 0.

    Returns:
        Dict[str, torch.Tensor or List]: A dictionary containing collated and padded fields.
    """

    if padding_values is None:
        padding_values = {}

    batch = {}

    for key in ('audio', 'spectrogram', 'text_encoded', 'text', 'audio_path'):

        if key in ('text', 'audio_path'):
            batch[key] = [el[key] for el in dataset_items]
            continue

        tensors = [el[key] for el in dataset_items]
        lengths = [tensor.shape[-1] for tensor in tensors]

        batch[key + '_length'] = torch.tensor(lengths)

        fill_with = padding_values.get(key, 0)

        max_length = max(lengths)
        padded_tensors = [F.pad(tensor, (0, max_length - tensor.shape[-1]), value=fill_with) for tensor in tensors]
        batch[key] = torch.cat(padded_tensors)
    
    return batch