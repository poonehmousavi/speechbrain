import copy
import os
import sys
import torch
from torch import nn
from torch.serialization import default_restore_location
import logging
from functools import partial
from tqdm.contrib import tqdm
import collections
from transformers import (
    get_linear_schedule_with_warmup,
    AutoTokenizer,
    AdamW,
)
import numpy as np
import speechbrain as sb
from torch.utils.data import DataLoader
from speechbrain.dataio.dataloader import LoopedLoader
from speechbrain.utils.distributed import run_on_main, if_main_process
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
from diffusion_util.resample import create_named_schedule_sampler, LossAwareSampler, UniformSampler
from diffusion_util import gaussian_diffusion as gd
from diffusion_util.gaussian_diffusion import GaussianDiffusion

class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        
        wavs, wav_lens = batch.sig
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        tgt_input_ids, _ = batch.tgt_input_ids
        tgt_input_ids = tgt_input_ids.to(self.device)
        audio_feats = self.modules.enc(wavs, wav_lens)
        t, weights = hparams['schedule_sampler'].sample(wavs.shape[0], self.device)
        x_start_mean = self.modules.model.get_embeds(tgt_input_ids)
        std = gd._extract_into_tensor(hparams['diffusion'].sqrt_one_minus_alphas_cumprod,
                                   torch.tensor([0]).to(self.device),
                                   x_start_mean.shape)
        x_start = hparams['diffusion'].get_x_start(x_start_mean, std)

        # Padding masks for source and targets (use padding_mas)
        src_key_padding_mask = self.hparams.padding_mask(wavs,  pad_idx=0)
        # tgt_key_padding_mask = self.hparams.padding_mask(tgt_input_ids,  pad_idx=0)
        noise = None
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = hparams['diffusion'].q_sample(x_start, t, noise=noise)  # reparametrization trick.
        # get_logits = self.modules.model.get_logits
        model_output = self.modules.model(x_t, hparams['diffusion']._scale_timesteps(t), src_input_ids=audio_feats, src_attention_mask=src_key_padding_mask,audio_inputs=audio_feats)
        return model_output,x_start,x_t,x_start_mean,noise
    
    def compute_objectives(self, predictions, batch, stage):
        predictions,x_start,x_t,x_start_mean,noise =predictions
        tgt_input_ids, _ = batch.tgt_input_ids
        tgt_input_ids = tgt_input_ids.to(self.device)
        t, weights = hparams['schedule_sampler'].sample(tgt_input_ids.shape[0], self.device)
        get_logits = self.modules.model.get_logits
        terms = {}
        target = {
            gd.ModelMeanType.PREVIOUS_X: hparams['diffusion'].q_posterior_mean_variance(
                x_start=x_start, x_t=x_t, t=t
            )[0],
            gd.ModelMeanType.START_X: x_start,
            gd.ModelMeanType.EPSILON: noise,
            }[hparams['diffusion'].model_mean_type]
        terms["mse"] = gd.mean_flat((target - predictions) ** 2)
        # print( terms["mse"])
        model_out_x_start = hparams['diffusion'].x0_helper(predictions, x_t, t)['pred_xstart']
        t0_mask = (t == 0)
        t0_loss = gd.mean_flat((x_start_mean - model_out_x_start) ** 2)

        terms["t0_loss"] = t0_loss
        terms["mse_pre"] = terms["mse"]
        terms["mse"] = torch.where(t0_mask, t0_loss, terms["mse"])

        # tT_mask = (t == self.num_timesteps - 1)
        out_mean, _, _ =  hparams['diffusion'].q_mean_variance(x_start, torch.LongTensor([hparams['diffusion'].num_timesteps - 1]).to(x_start.device))
        tT_loss = gd.mean_flat(out_mean ** 2)
        terms["tT_loss"] = tT_loss

        decoder_nll =  hparams['diffusion'].token_discrete_loss(x_start, get_logits, tgt_input_ids)
        terms["decoder_nll"] = decoder_nll

        # assert (model.lm_head.weight == model.word_embedding.weight).all()

        if "vb" in terms:
            terms["loss"] = terms["mse"] + terms["vb"]
        else:
            # KEY
            terms["loss"] = terms["mse"] + (decoder_nll + tT_loss)

        loss = (terms["loss"] * weights).mean()

        if (stage == sb.Stage.TEST) or (stage == sb.Stage.VALID and hparams['epoch_counter'].current % hparams['valid_search_interval'] == 0):
            self.generate(batch,stage)
        
        return loss

    def generate(self,batch,stage):
        sample_fn = (
            hparams['diffusion'].p_sample_loop)
        emb_model = self.modules.model.word_embedding
        
        each_sample_list = []
        tgt_sample =[]
        

        ids = batch.id
        wavs, wav_lens = batch.sig
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        tgt_input_ids, _ = batch.tgt_input_ids
        tgt_input_ids = tgt_input_ids.to(self.device)
        audio_feats = self.modules.enc(wavs, wav_lens)
        src_key_padding_mask = self.hparams.padding_mask(wavs,  pad_idx=0)
        model_kwargs = {'src_input_ids' : audio_feats, 'src_attention_mask': src_key_padding_mask,'audio_inputs':audio_feats}
        input_shape = (wavs.shape[0],hparams['maxlength'], hparams['in_channel'])
        sample = sample_fn(
                        self.modules.model,
                        input_shape,
                        clip_denoised=False,
                        denoised_fn=partial(self.denoised_fn_round, emb_model),
                        model_kwargs=model_kwargs,
                        top_p=-1.0,
                        interval_step=hparams['interval_step'],
                    )
        
        logger.info(f"sample result shape: {sample.shape}", )
        logger.info('decoding for e2e... ')
        logits = self.modules.model.get_logits(sample)
        cands = torch.topk(logits, k=1, dim=-1)
        sample_id_list = cands.indices
        predicted_words = tokenizer.batch_decode(
            sample_id_list.squeeze(dim=-1),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            )
        target_words = tokenizer.batch_decode(
            tgt_input_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            )
        

        self.wer_metric.append(ids, predicted_words, target_words)
        self.cer_metric.append(ids, predicted_words, target_words)

    '''
    rounding
    '''
    def denoised_fn_round(self,model, text_emb,t):

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
    
    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        elif (stage == sb.Stage.TEST) or (stage == sb.Stage.VALID and hparams['epoch_counter'].current % hparams['valid_search_interval'] == 0):
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr_model, new_lr_model = self.hparams.lr_annealing_model(
                stage_stats["loss"]
            )
            old_lr_enc, new_lr_enc = self.hparams.lr_annealing_enc(
                stage_stats["loss"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.model_optimizer, new_lr_model
            )
            sb.nnet.schedulers.update_learning_rate(
                self.enc_optimizer, new_lr_enc
            )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_model": old_lr_model,
                    "lr_enc": old_lr_enc,
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            if hparams['epoch_counter'].current % hparams['valid_search_interval'] == 0:
                self.checkpointer.save_and_keep_only(
                    meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
                )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(self.hparams.test_wer_file, "w") as w:
                    self.wer_metric.write_stats(w)

    def init_optimizers(self):
        "Initializes the encoder optimizer and model optimizer"

        # HuggingFace pretrained model
        self.enc_optimizer = self.hparams.enc_opt_class(
                self.modules.enc.parameters()
        )

        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )

        # save the optimizers in a dictionary
        # the key will be used in `freeze_optimizers()`
        self.optimizers_dict = {
            "model_optimizer": self.model_optimizer,
        }
        if not self.hparams.freeze_enc:
            self.optimizers_dict["enc_optimizer"] = self.enc_optimizer

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "enc_opt_class", self.enc_optimizer
            )
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)
        



INITIAL_LOG_LOSS_SCALE = 20.0
CheckpointState = collections.namedtuple("CheckpointState",
                                                     ['model_dict', 'optimizer_dict', 'scheduler_dict', 'offset'])

def train(train_set,valid_set, hparams,epoch):
    train_losses=[]
    

    logger.info("***** Running training *****")
    logger.info("  Max steps = %d", hparams['lr_anneal_steps'])
    logger.info("  Instantaneous batch size per GPU = %d",  hparams['batch_size'])
    logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d",
             hparams['batch_size']
            * hparams['gradient_accumulation_steps']
            ,
        )
    logger.info("  Gradient Accumulation steps = %d", hparams['gradient_accumulation_steps'])
    hparams['model'].zero_grad()
    hparams['model'].train()
    


    if not (
        isinstance(train_set, DataLoader) or isinstance(train_set, LoopedLoader)
    ):
        train_set = sb.dataio.dataloader.make_dataloader(
            train_set, **hparams["train_dataloader_opts"])

    with tqdm(train_set, dynamic_ncols=True,) as t:
        for batch in t:
            loss = train_batch(batch)
            train_losses.append(loss.item())
            
    
            # if hparams['global_step'] % hparams['eval_interval'] == 0:
            #     logger.info('eval on validation set...')
            #     if not (
            #         isinstance(valid_set, DataLoader) or isinstance(valid_set, LoopedLoader)
            #     ):
            #         valid_set = sb.dataio.dataloader.make_dataloader(
            #             valid_set, **hparams["valid_dataloader_opts"])

            #     valid_losses=[]
            #     with tqdm(valid_set, dynamic_ncols=True,) as t:
                
            #         for step, batch in enumerate(t):
            #             loss = forward_only(batch)
            #             valid_losses.extend(loss['loss'])
            #             # if step > 10:
            #             #     break
                
            #     logger.info(f"validation loss: {sum(valid_losses).item()/len(valid_losses)}, Train Loss: {sum(train_losses)/len(train_losses)}")

    logger.info('eval on validation set...')
    if not (
            isinstance(valid_set, DataLoader) or isinstance(valid_set, LoopedLoader)
            ):
        valid_set = sb.dataio.dataloader.make_dataloader(
            valid_set, **hparams["valid_dataloader_opts"])

    valid_losses=[]
    with tqdm(valid_set, dynamic_ncols=True,) as t:   
        for step, batch in enumerate(t):
            loss = forward_only(batch)
            valid_losses.extend(loss['loss'])
                        # if step > 10:
                        #     break
                
    logger.info(f"Epoch:{epoch}, validation loss: {sum(valid_losses).item()/len(valid_losses)}, Train Loss: {sum(train_losses)/len(train_losses)}")
    # save loss stats
    log_file = open(hparams['train_log'], "a")
    log_file.write(f"Epoch:{epoch}, validation loss: {sum(valid_losses).item()/len(valid_losses)}, Train Loss: {sum(train_losses)/len(train_losses)}\n")
    log_file.close()
    save()
                

def save():

    def save_checkpoint(rate, ema_params):
        model_to_save = get_model_obj(hparams['model'])
        if not rate:
            model_state_dict = model_to_save.state_dict()
        else:
            model_state_dict = model_to_save.state_dict()
            for i, (name, _value) in enumerate(model_to_save.named_parameters()):
                assert name in model_state_dict
                model_state_dict[name] = ema_params[i]

        opt_state_dict = hparams['optimizer'].state_dict()
        sch_state_dict = hparams['scheduler'].state_dict()
        offset = hparams['global_step']
        state = CheckpointState(model_state_dict,
                                    opt_state_dict,
                                    sch_state_dict,
                                    offset,
                                    )
        Path(hparams['save_folder']).mkdir(parents=True, exist_ok=True)
        if not rate:
            ckpt_path = os.path.join(hparams['save_folder'], 'model_checkpoint-' + str(offset))
        else:
            ckpt_path = os.path.join(hparams['save_folder'], 'ema_' + str(rate) + '_checkpoint-' + str(offset))

        torch.save(state._asdict(), ckpt_path)
        logger.info('Saved checkpoint at %s', ckpt_path)


    save_checkpoint(0, None)
    for rate, params in zip(hparams['ema_rate'], hparams['ema_params']):
        save_checkpoint(rate, params)

def train_batch(batch):
    hparams['model'].train()
    # forward loss
    loss = forward_backward(batch)
    if hparams['precision'] == 'fp16':
         pass
    else:
        # gradient clip
        if hparams['gradient_clipping'] > 0:
            grad_clip()
        hparams['optimizer'].step()
        # lr scheduler
        hparams['scheduler'].step()
        hparams['model'].zero_grad()
         # ema
        for rate, params in zip(hparams['ema_rate'], hparams['ema_params']):
            update_ema(params, hparams['master_params'], rate=rate)
    hparams['global_step'] += 1
    return loss

def get_model_obj(model: nn.Module):
    return model.module if hasattr(model, 'module') else model

def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        # print("target_params:", targ.device)
        # print("source_params:", src.device)
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

def load_states_from_checkpoint(model_file: str) -> CheckpointState:
    logger.info('Reading saved model from %s', model_file)
    state_dict = torch.load(model_file, map_location=lambda s, l: default_restore_location(s, 'cpu'))
    logger.info('model_state_dict keys %s', state_dict.keys())
    return CheckpointState(**state_dict)

def forward_backward(batch):
    src_input_ids, _ = batch['src_input_ids']

    t, weights = hparams['schedule_sampler'].sample(src_input_ids.shape[0], run_opts['device'])
    losses = hparams['diffusion'].training_losses(hparams['model'], batch, t, hparams['wavlm'])

    loss = (losses["loss"] * weights).mean()
    if hparams['precision'] == 'fp16':
        loss_scale = 2 ** hparams['lg_loss_scale']
        (loss * loss_scale).backward()
    else:
        loss.backward()
    return loss

def forward_only( batch):
    with torch.no_grad():
        hparams['model'].zero_grad()
        '''
        for s2s
        '''
        src_input_ids, _ = batch['src_input_ids']
        t, weights = hparams['schedule_sampler'].sample(src_input_ids.shape[0], run_opts['device'])
        losses = hparams['diffusion'].training_losses(hparams['model'], batch, t,hparams['wavlm'])
    return losses
def grad_clip():
    # print('doing gradient clipping')
    max_grad_norm=hparams['gradient_clipping'] #3.0
    if hasattr(hparams['optimizer'], "clip_grad_norm"):
         # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
        hparams['optimizer'].clip_grad_norm(max_grad_norm)
        # else:
        #     assert False
        # elif hasattr(self.model, "clip_grad_norm_"):
        #     # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
        #     self.model.clip_grad_norm_(args.max_grad_norm)
    else:
            # Revert to normal clipping otherwise, handling Apex or full precision
        torch.nn.utils.clip_grad_norm_(
            hparams['model'].parameters(), #amp.master_params(self.opt) if self.use_apex else
            max_grad_norm,
        )


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
'''
create diffusion process
'''
def create_gaussian_diffusion(
    steps=1000,
    noise_schedule="cosine",
    rescale_timesteps=False,
):

    # β , Determine according to the maximum T and variance schedule
    logger.info(f"noise_schedule: {noise_schedule}")
    logger.info(f"Diffusion Steps: {steps}" )

    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    logger.info(f"betas: {betas}")

    return GaussianDiffusion(
        betas=betas,
        model_mean_type=gd.ModelMeanType.START_X,
        model_var_type= gd.ModelVarType.FIXED_LARGE,
        loss_type=gd.LossType.MSE,
        rescale_timesteps=rescale_timesteps,


    )

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

    # # We load the pretrained wav2vec2 model
    if "pretrainer" in hparams.keys():
        run_on_main(hparams["pretrainer"].collect_files)
        hparams["pretrainer"].load_collected()

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )





    
    diffusion = create_gaussian_diffusion(**hparams["diff_args"])
    hparams['diffusion'] = diffusion

    # time step schedule sampler
    hparams['schedule_sampler'] = create_named_schedule_sampler(hparams['schedule_sampler'], diffusion)

    # # Load pretrained model
    # if os.path.isfile(hparams['pretrain_model_path']):
    # print("load model ckpt at :", hparams['pretrain_model_path'])
    # saved_state = load_states_from_checkpoint(hparams['pretrain_model_path'])
    # model.load_state_dict(torch.load("pre-trained/model.ckpt", map_location='cpu'), strict=False)
    # torch.save(saved_state.model_dict,"results/diff_asr/1986/save/model.ckpt" )
    # torch.save(saved_state.scheduler_dict,"results/diff_asr/1986/save/scheduler_model.ckpt")
    # torch.save(saved_state.optimizer_dict,"results/diff_asr/1986/save/optimizer_model.ckpt")
    # model.load_state_dict(torch.load("pre-trained/model.ckpt", map_location='cpu'), strict=False)
    # model.to(run_opts['device'])

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Testing
    if not os.path.exists(hparams["output_wer_folder"]):
        os.makedirs(hparams["output_wer_folder"])

    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        asr_brain.hparams.test_wer_file = os.path.join(
            hparams["output_wer_folder"], f"wer_{k}.txt"
        )
        asr_brain.evaluate(
            test_datasets[k],
            min_key="WER",
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )




    
    # hparams['ema_rate'] = (
    #         [hparams['ema_rate']]
    #         if isinstance(hparams['ema_rate'], float)
    #         else [float(x) for x in hparams['ema_rate'].split(",")]
    #     )
    # hparams['ema_params'] = [
    #             copy.deepcopy(hparams['master_params']) for _ in range(len(hparams['ema_rate']))
    #         ]
    # hparams['lg_loss_scale'] = INITIAL_LOG_LOSS_SCALE
    # hparams['global_step'] = 0
    # for i in  range(hparams['epoch_counter']):
    #     train(train_data,valid_data,hparams,i)
