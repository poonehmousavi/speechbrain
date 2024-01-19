import copy
from functools import partial
import os
import sys
import torch
from torch import nn
from torch.serialization import default_restore_location
import logging
from tqdm.contrib import tqdm
import collections
import numpy as np
from transformers import (
    get_linear_schedule_with_warmup,
    AutoTokenizer,
    AdamW,
)
from jiwer import wer,cer
import speechbrain as sb
from torch.utils.data import DataLoader
from speechbrain.dataio.dataloader import LoopedLoader
from speechbrain.utils.distributed import run_on_main, if_main_process
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
from util.util import (
    create_gaussian_diffusion,
)
from diffusion_util.resample import create_named_schedule_sampler, LossAwareSampler, UniformSampler
INITIAL_LOG_LOSS_SCALE = 20.0
CheckpointState = collections.namedtuple("CheckpointState",
                                                     ['model_dict', 'optimizer_dict', 'scheduler_dict', 'offset'])


def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    logger.info('Reading saved model from %s', model_file)
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    logger.info('model_state_dict keys %s', state_dict.keys())
    return CheckpointState(**state_dict)


'''
rounding
'''
def denoised_fn_round(model, text_emb,t):

    down_proj_emb = model.weight  # input_embs
    # print(t)
    old_shape = text_emb.shape
    old_device = text_emb.device

    def get_efficient_knn(down_proj_emb, text_emb, dist='l2'):
        if dist == 'l2':
            emb_norm = (down_proj_emb ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = torch.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            # print(emb_norm.shape, arr_norm.shape)
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * torch.mm(down_proj_emb,
                                                                     text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            dist = torch.clamp(dist, 0.0, np.inf)
            # print(dist.shape)
        topk_out = torch.topk(-dist, k=1, dim=0)
        #     adjacency = down_proj_emb.unsqueeze(1).expand(-1, text_emb.size(0), -1) - text_emb.unsqueeze(0).expand(
        #         down_proj_emb.size(0), -1, -1)
        #     adjacency = -th.norm(adjacency, dim=-1)
        # topk_out = th.topk(adjacency, k=1, dim=0)
        # print(topk_out1.indices == topk_out.indices)
        # assert th.all(topk_out1.indices == topk_out.indices)
        return topk_out.values, topk_out.indices

    def get_knn(down_proj_emb, text_emb, dist='l2'):
        if dist == 'l2':
            adjacency = down_proj_emb.unsqueeze(1).expand(-1, text_emb.size(0), -1) - text_emb.unsqueeze(0).expand(
                down_proj_emb.size(0), -1, -1)
            adjacency = -torch.norm(adjacency, dim=-1)
        topk_out = torch.topk(adjacency, k=1, dim=0)
        return topk_out.values, topk_out.indices

    dist = 'l2'
    if len(text_emb.shape) > 2:
        text_emb = text_emb.reshape(-1, text_emb.size(-1))
    else:
        text_emb = text_emb
    # val, indices = get_knn(down_proj_emb,
    #                        text_emb.to(down_proj_emb.device), dist=dist)
    val, indices = get_efficient_knn(down_proj_emb,
                                     text_emb.to(down_proj_emb.device), dist=dist)
    rounded_tokens = indices[0]
    # print(rounded_tokens.shape)
    new_embeds = model(rounded_tokens).view(old_shape).to(old_device)
    return new_embeds


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]

    # We get the tokenizer as we need it to encode the labels when creating
    # mini-batches.
    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "src_input_ids", "tgt_input_ids"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode(wrd, add_special_tokens=True,
                                        max_length=hparams['maxlength'], truncation=True,
                                       padding='max_length')
        src_input_ids = torch.LongTensor((tokens_list))
        yield src_input_ids
        tgt_input_ids = torch.LongTensor(tokens_list)
        yield tgt_input_ids

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd",  "src_input_ids", "tgt_input_ids"],
    )

    return (
        train_data,
        valid_data,
        test_datasets,
    )

def clean(sentence):
    sentence = sentence.replace('[CLS]', '')
    sentence = sentence.replace('[SEP]', '')
    sentence = sentence.replace('[PAD]', '')
    sentence = sentence.replace('[UNK]', 'unk')
    return sentence.strip()

logger = logging.getLogger(__name__)
if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing Librispeech)
    from librispeech_prepare import prepare_librispeech  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    hparams["tokenizer"] = tokenizer

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets = dataio_prepare(
        hparams
    )

    model = hparams['model']
    diffusion = create_gaussian_diffusion(**hparams["diff_args"])
    hparams['diffusion'] = diffusion

    # Load pretrained model
    if os.path.isfile(hparams['pretrain_model_path']):
        print("load model ckpt at :", hparams['pretrain_model_path'])
        saved_state = load_states_from_checkpoint(hparams['pretrain_model_path'])
        model.load_state_dict(saved_state.model_dict, strict=False)
    model.to(run_opts['device'])
    model.eval()

    sample_fn = (
        diffusion.p_sample_loop
    )

    # bert tokenizer
    logger.info("-------------------------------------------------------------")
    logger.info("start generate query from dev dataset, for every passage, we generate ", hparams['num_samples'], " querys...")
    logger.info("-------------------------------------------------------------")

    Path(hparams['generate_path']).mkdir(parents=True, exist_ok=True)
    epoch_num = 0
    emb_model = model.word_embedding

    for epoch in range(hparams['num_samples'] - epoch_num):
        each_sample_list = []
        tgt_sample =[]
        logger.info("-------------------------------------------------------------")
        logger.info("start sample ", epoch+1+epoch_num, " epoch...")
        logger.info("-------------------------------------------------------------")
        test_dataset= test_datasets['test-clean']
        if not (
        isinstance(test_dataset, DataLoader) or isinstance(test_dataset, LoopedLoader)
        ):
            test_dataset= sb.dataio.dataloader.make_dataloader(
            test_dataset, **hparams["test_dataloader_opts"])


        with tqdm(test_dataset, dynamic_ncols=True,) as t:
            for batch in t:
                '''
                for s2s
                '''
                src_input_ids, _ = batch['src_input_ids']
                input_shape = (src_input_ids.shape[0],hparams['maxlength'], hparams['in_channel'])
                tgt_input_ids, _ = batch['tgt_input_ids']
                wavs, wav_lens = batch['sig']
                audio_feats = hparams['wavlm'](wavs, wav_lens).to(run_opts['device'])
                # print(p_input_ids.shape)
                src_attention_mask=  (src_input_ids != 0).long()
                model_kwargs = {'src_input_ids' : src_input_ids.to(run_opts['device']), 'src_attention_mask': src_attention_mask.to(run_opts['device']),'audio_inputs':audio_feats}

                sample = sample_fn(
                    model,
                    input_shape,
                    clip_denoised=False,
                    denoised_fn=partial(denoised_fn_round, emb_model.cuda()),
                    model_kwargs=model_kwargs,
                    top_p=-1.0,
                    interval_step=hparams['interval_step'],
                )

                print("sample result shape: ", sample.shape)
                print('decoding for e2e... ')

                logits = model.get_logits(sample)
                cands = torch.topk(logits, k=1, dim=-1)
                sample_id_list = cands.indices
                #print("decode id list example :", type(sample_id_list[0]), "  ", sample_id_list[0])

                '''
                for s2s
                '''
                # print("src text: ", tokenizer.decode(src_input_ids.squeeze()))
                # print("tgt text: ", tokenizer.decode(tgt_input_ids.squeeze()))

                print("sample control generate query: ")
                for sample_id in sample_id_list:
                    sentence = tokenizer.decode(sample_id.squeeze())
                    each_sample_list.append(clean(sentence))
                    # print(sentence)
                for sentence in tgt_input_ids:
                    sentence = tokenizer.decode(sentence.squeeze())
                    tgt_sample.append(clean(sentence))


        # total_sample_list.append(each_sample_list)
        out_path = os.path.join(hparams['generate_path'], "_gen_seed_101" +
                                "_num" + str(hparams['num_samples']) + "_epoch" + str(epoch + 1 + epoch_num) + ".txt")
        # total_sample_list.append(each_sample_list)
        tgt_out_path = os.path.join(hparams['generate_path'], "_tgt_seed_101" +
                                "_num" + str(hparams['num_samples']) + "_epoch" + str(epoch + 1 + epoch_num) + ".txt")
            
        wer_score = wer(tgt_sample,each_sample_list)*100
        cer_score= cer(tgt_sample,each_sample_list)*100
        logger.info(f"Testset : WER: {wer_score} , CER : {cer_score}")
        with open(out_path, 'w') as f:
            for sentence in each_sample_list:
                f.write(sentence + '\n')
        with open(tgt_out_path, 'w') as f:
            for sentence in tgt_sample:
                f.write(sentence + '\n')

