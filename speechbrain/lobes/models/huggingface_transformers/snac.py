"""This lobe enables the integration of huggingface pretrained SNAC.

Mimi codec is a state-of-the-art audio neural codec, developed by Kyutai.
It combines semantic and acoustic information into audio tokens running at 12Hz and a bitrate of 1.1kbps.

Note that you need to install `snac` to use this module.

Repository: https://huggingface.co/kyutai/mimi
Paper: https://kyutai.org/Moshi.pdf

Authors
 * Pooneh Mousavi 2024
"""

import torch
import torch.nn as nn

from speechbrain.dataio.dataio import clean_padding_, length_to_mask
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class SNAC(nn.Module):
    """This lobe enables the integration of HuggingFace pretrained Mimi model.
    Mimi codec is a state-of-the-art audio neural codec, developed by Kyutai.
    It combines semantic and acoustic information into audio tokens running at 12Hz and a bitrate of 1.1kbps.

    Source paper:
       https://kyutai.org/Moshi.pdf

    snac library needs to be installed:
        pip install snac

    The code is adapted from the official HF Kyutai repository:
        https://huggingface.co/kyutai/mimi

    Arguments
    ---------
    source : str
        A HuggingFace repository identifier or a path
    save_path : str
        The location where the pretrained model will be saved
    sample_rate : int (default: 24000)
        The audio sampling rate
    freeze : bool
        whether the model will be frozen (e.g. not trainable if used as part of training another model)
    num_codebooks : int (default: 8)
        Number of qunatizer. It could be [2,3,4,5,6,7,8]

    Example
    -------
    >>> model_hub = "hubertsiuzdak/snac_24khz"
    >>> save_path = "savedir"
    >>> model = SNAC(model_hub, save_path)
    >>> audio = torch.randn(4, 48000)
    >>> length = torch.tensor([1.0, .5, .75, 1.0])
    >>> tokens, emb = model.encode(audio, length)
    >>> tokens.shape
    torch.Size([4, 8, 25])
    >>> emb.shape
    torch.Size([4, 8, 25, 256])
    >>> rec = model.decode(tokens, length)
    >>> rec.shape
    torch.Size([4, 1, 48000])
    """

    def __init__(
        self,
        source,
        save_path=None,
        sample_rate=24000,
        freeze=True,
        num_codebooks=8,
    ):
    # Lazy import to avoid circular dependency issues
        try:
            from snac import SNAC


            self.SNAC = SNAC
        except ImportError:
            raise ImportError(
                "Please install the SNAC module using: "
                "pip install snac`"
            )
        super().__init__()
        self.num_codebooks = num_codebooks
        self.sampling_rate = sample_rate
        self.model = self.SNAC.from_pretrained(source,cache_dir=save_path).eval()

    def encode(self, inputs, length):
        """Encodes the input audio as tokens and embeddings

        Arguments
        ---------
        inputs : torch.Tensor
            A (Batch x Samples) or (Batch x Channel x Samples)
            tensor of audio
        length : torch.Tensor
            A tensor of relative lengths
        Returns
        -------
        tokens : torch.Tensor
            A (Batch x num_codebooks x Length) tensor of audio tokens
        """
        return self.model.encode(inputs)

    def decode(self, tokens, length=None):
        """Decodes audio from tokens

        Arguments
        ---------
        tokens : torch.Tensor
            A (Batch x num_codebooks x Length) tensor of audio tokens
        length : torch.Tensor
            A 1-D tensor of relative lengths

        Returns
        -------
        audio : torch.Tensor
            the reconstructed audio
        """
        return self.model.decode(tokens)


model_hub = "hubertsiuzdak/snac_24khz"
save_path = "savedir"
model = SNAC(model_hub, save_path)
audio = torch.randn(4, 48000)
length = torch.tensor([1.0, .5, .75, 1.0])
tokens = model.encode(audio, length)
rec = model.decode(tokens, length)
