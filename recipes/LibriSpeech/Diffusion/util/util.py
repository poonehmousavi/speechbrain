from model.Diffusion_LM import  CrossAttention_Diffusion_LM

from diffusion_util.respace import SpacedDiffusion, space_timesteps
import logging


logger = logging.getLogger(__name__)





def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}