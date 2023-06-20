"""This lobe enables the integration of huggingface pretrained whisper model.

Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Adel Moumen 2022
 * Titouan Parcollet 2022
 * Luca Della Libera 2022
"""

import torch
import logging
from torch import nn

try:
    from transformers import BertTokenizer, BertConfig

except ImportError:
    MSG = "Please install transformers from HuggingFace \n"
    MSG += "E.G. run: pip install transformers"
    raise ImportError(MSG)

logger = logging.getLogger(__name__)


class HuggingFace_LM(nn.Module):
    def __init__(
            self,
            source,
            timestep,
            num_steps,
            save_path,
            from_scratch=False,load_step=-1
        ):
        super().__init__()
        self.source= source
        if timestep in ['none', 'token']:
            from model.modeling_bert import BertForMaskedLM
        elif timestep == 'layerwise':
            from model.modeling_bert_new_timestep import BertForMaskedLM
        else:
            raise NotImplementedError
        if source in ['bert-base-uncased', 'bert-large-uncased']:
            model_cls = BertForMaskedLM
            cfg_cls = BertConfig
            tok_cls = BertTokenizer
        else:
            raise NotImplementedError

        
        self.tokenizer = tok_cls.from_pretrained(source, cache_dir=save_path)
        # TODO: Task1:  Load from ckeckpoint
        # if args.load_step > 0:
        #     ckpt = torch.load(os.path.join(save_path, f'{args.load_step}.th'))
        cfg = cfg_cls.from_pretrained(source)
        cfg.overall_timestep = num_steps

        if from_scratch:
            self.model = model_cls(cfg)
        elif load_step <= 0:
            self.model = model_cls.from_pretrained(source, config=cfg)
    
    def forward(self, **kwarg):
        return self.model(**kwarg)

        # TODO: Task1:  Load from ckeckpoint
        # else:
        #     self.model = model_cls(cfg).to(device)
        #     model.load_state_dict(ckpt['model'])

