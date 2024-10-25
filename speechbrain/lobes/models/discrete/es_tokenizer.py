"""This lobe enables the integration of pretrained discrete SSL (hubert,wavlm,wav2vec) with RVQ for training enhanced semantic Tokenizer.
   The code is adopted from official encodec github repo: https://github.com/descriptinc/descript-audio-codec

Author
 * Pooneh Mousavi 2024
"""

from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm



def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class VectorQuantize(nn.Module):
    """
    Implementation of VQ similar to Karpathy's repo:
    https://github.com/karpathy/deep-vector-quantization
    Additionally uses following tricks from Improved VQGAN
    (https://arxiv.org/pdf/2110.04627.pdf):
        1. Factorized codes: Perform nearest neighbor lookup in low-dimensional space
            for improved codebook usage
        2. l2-normalized codes: Converts euclidean distance to cosine similarity which
            improves training stability
    """

    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z):
        """Quantized the input tensor using a fixed codebook and returns
        the corresponding codebook vectors

        Parameters
        ----------
        z : Tensor[B x D x T]

        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        Tensor[1]
            Commitment loss to train encoder to predict vectors closer to codebook
            entries
        Tensor[1]
            Codebook loss to update the codebook
        Tensor[B x T]
            Codebook indices (quantized discrete representation of input)
        Tensor[B x D x T]
            Projected latents (continuous representation of input before quantization)
        """

        # Factorized codes (ViT-VQGAN) Project input into low-dimensional space
        z_e = self.in_proj(z)  # z_e : (B x D x T)
        z_q, indices = self.decode_latents(z_e)

        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        z_q = (
            z_e + (z_q - z_e).detach()
        )  # noop in forward pass, straight-through gradient estimator in backward pass

        z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        encodings = rearrange(latents, "b d t -> (b t) d")
        codebook = self.codebook.weight  # codebook: (N x D)

        # L2 normalize encodings and codebook (ViT-VQGAN)
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # Compute euclidean distance with codebook
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = rearrange((-dist).max(1)[1], "(b t) -> b t", b=latents.size(0))
        z_q = self.decode_code(indices)
        return z_q, indices


class ResidualVectorQuantizer(nn.Module):
    """
    Introduced in SoundStream: An end2end neural audio codec
    https://arxiv.org/abs/2107.03312
    Parameters
    ----------
    n_codebooks : should be == ssl-layers
    
    Example
    -------
    >>> import torch
    >>> from speechbrain.lobes.models.huggingface_transformers.hubert import (HuBERT)
    >>> rvq = ResidualVectorQuantizer(quantizer_dropout=True, input_dim=1024, n_codebooks=25)
    >>> inputs = torch.rand([3, 2000])
    >>> model_hub = "facebook/hubert-large-ll60k"
    >>> save_path = "savedir"
    >>> sample_rate= 16000
    >>> ssl_model = HuBERT(model_hub, save_path,output_all_hiddens=True)

    >>> x = ssl_model(inputs)
    >>> y = rvq(rearrange(x,"n b t d -> n b d t"))
    >>> print(y['embeddings'].shape)
    torch.Size([3, 25, 1024, 6])
    """
    

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(input_dim, codebook_size, codebook_dim[i])
                for i in range(n_codebooks)
            ]
        )
        self.quantizer_dropout = quantizer_dropout

    def forward(self, z, n_quantizers: int = None):
        """Quantized the input tensor using a fixed set of `n` codebooks and returns
        the corresponding codebook vectors
        Parameters
        ----------
        z : Tensor[B x D x T]
        n_quantizers : int, optional
            No. of quantizers to use
            (n_quantizers < self.n_codebooks ex: for quantizer dropout)
            Note: if `self.quantizer_dropout` is True, this argument is ignored
                when in training mode, and a random number of quantizers is used.
        Returns
        -------
        dict
            A dictionary with the following keys:

            "z" : Tensor[B x D x T]
                Quantized continuous representation of input
            "codes" : Tensor[B x N x T]
                Codebook indices for each codebook
                (quantized discrete representation of input)
            "latents" : Tensor[B x N*D x T]
                Projected latents (continuous representation of input before quantization)
            "vq/commitment_loss" : Tensor[1]
                Commitment loss to train encoder to predict vectors closer to codebook
                entries
            "vq/codebook_loss" : Tensor[1]
                Codebook loss to update the codebook
        """

        N, B, D, T = z.shape
        z_q = torch.zeros(B, D, T, device=z.device)
        commitment_loss = 0
        codebook_loss = 0

        codebook_indices = []
        codebook_embeddings= []
        latents = []
        
        if n_quantizers is None:
            n_quantizers = [[i for i in range(1, N)] for _ in range(B)]
        elif isinstance(n_quantizers, list) and all(not isinstance(i, list) for i in n_quantizers):
            assert all(item < N for item  in n_quantizers), "All elements in n_quantizers must be less than n_codebooks."
            # Expand the single element to a list of B elements with the same value
            n_quantizers = [sorted(n_quantizers)] * B

        # assert n_quantizers <=  z.shape[0], " No. of quantizers should be less or equal to the No. of SSL layers(the input)"
        if self.training:
            # Random generation of layers, num_layers, and selected_samples
            all_layers = torch.stack([torch.randperm(N-1, device=z.device)+1 for _ in range(B)], dim=0)
            num_layers = torch.randint(1, N , (B,), device=z.device)
            selected_samples = torch.rand(B, device=z.device) < self.quantizer_dropout

            # Create a mask to determine which samples to select partially
            mask = selected_samples.unsqueeze(1)  # Shape (B, 1)

            # Create an index tensor based on num_layers for each sample
            range_tensor = torch.arange(1,N, device=z.device).expand(B, N-1)

            # Mask for the number of layers for selected samples
            num_layers_mask = range_tensor < num_layers.unsqueeze(1)

            # Combine both masks: select partial layers for masked samples, otherwise all layers
            combined_mask = torch.where(mask, num_layers_mask, torch.ones_like(num_layers_mask, dtype=torch.bool))

            # Apply the mask to all_layers to get the output
            output_layers = torch.where(combined_mask, all_layers, torch.tensor(-1, device=all_layers.device))
            output_layers_sorted, _ = torch.sort(output_layers, dim=1)

            # Generate n_quantizers based on the sorted output layers, excluding -1 values
            n_quantizers = [output_layers_sorted[i, output_layers_sorted[i] >= 0].tolist() if selected_samples[i] 
                            else list(range(1,N)) for i in range(B)]

        # Precompute masks for all quantizers based on n_quantizers
        mask = torch.zeros((N, B), device=z.device)
        for i in range(B):
            mask[n_quantizers[i], i] = 1
        mask[0, :] = 1
        for i, quantizer in enumerate(self.quantizers):
            if self.training is False and mask[i].sum() == 0:
                codebook_embeddings.append(torch.zeros(B, D, T))
                codebook_indices.append(torch.zeros(B, T))
                continue

            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(z[i]-z_q)

            # Apply the mask efficiently
            mask_i = mask[i].view(B, 1, 1)
            z_q =  z_q_i* mask_i +  z_q* (1-mask_i)
            # z_q .append(z[i] - z_q_i* mask_i)  # Update residual

            commitment_loss += (commitment_loss_i * mask_i.squeeze()).mean()
            codebook_loss += (codebook_loss_i * mask_i.squeeze()).mean()

            codebook_embeddings.append(z_q_i* mask_i)
            codebook_indices.append(indices_i* mask_i.squeeze(-1))
            latents.append(z_e_i)

        codebook_embeddings = torch.stack(codebook_embeddings, dim=1)
        codes = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)

        return {
            "embeddings": codebook_embeddings,
            "codes": codes,
            "latents": latents,
            "vq/commitment_loss": commitment_loss,
            "vq/codebook_loss": codebook_loss
        }
    def from_codes(self, codes: torch.Tensor):
        """Given the quantized codes, reconstruct the continuous representation
        Parameters
        ----------
        codes : Tensor[B x N x T]
            Quantized discrete representation of input
        Returns
        -------
        Tensor[B x D x T]
            Quantized continuous representation of input
        """
        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[1]
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
        return z_q, torch.cat(z_p, dim=1), codes

    def from_latents(self, latents: torch.Tensor):
        """Given the unquantized latents, reconstruct the
        continuous representation after quantization.

        Parameters
        ----------
        latents : Tensor[B x N x T]
            Continuous representation of input after projection

        Returns
        -------
        Tensor[B x D x T]
            Quantized representation of full-projected space
        Tensor[B x D x T]
            Quantized representation of latent space
        """
        z_q = 0
        z_p = []
        codes = []
        dims = np.cumsum([0] + [q.codebook_dim for q in self.quantizers])

        n_codebooks = np.where(dims <= latents.shape[1])[0].max(axis=0, keepdims=True)[
            0
        ]
        for i in range(n_codebooks):
            j, k = dims[i], dims[i + 1]
            z_p_i, codes_i = self.quantizers[i].decode_latents(latents[:, j:k, :])
            z_p.append(z_p_i)
            codes.append(codes_i)

            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i

        return z_q, torch.cat(z_p, dim=1), torch.stack(codes, dim=1)

