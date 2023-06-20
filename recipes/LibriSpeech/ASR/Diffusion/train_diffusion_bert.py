#!/usr/bin/env python3
"""Recipe for training a Diffusion ASR system with librispeech.
The system employs Diffusion_BERT follwoing this paper:  https://arxiv.org/abs/2211.15029

To run this recipe, do the following:
> python train_diffusion_bert.py hparams/train_with_diffusion_bert.yaml


Authors
 * Pooneh Mousavi, 2023
"""

import os
import sys
import torch
import logging
from pathlib import Path
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        input_ids, _ = batch.tokens
        attention_mask, _ = batch.attention_mask
        word_freq_logits, _ = batch.word_freq_logits
        input_ids, attention_mask, word_freq_logits = input_ids.to(self.device), attention_mask.to(self.device), word_freq_logits.to(self.device)

        # forward modules
        if stage == sb.Stage.TRAIN:
            metrics = diffusion_word_freq.compute_kl_reverse_process(
                input_ids,
                hparams['diffusion_instance'].sample_t(),
                denoise_fn= hparams['denoise_fn'],
                diffusion=hparams['diffusion_instance'],
                target_mask=attention_mask,
                hybrid_lambda=hparams['hybrid_lambda'],
                predict_x0=hparams['predict_x0'],
                word_freq_logits=word_freq_logits
            )
        else:
            metrics = diffusion_word_freq.discrete_diffusion_elbo(
                input_ids,
                denoise_fn= hparams['denoise_fn'],
                diffusion=hparams['diffusion_instance'],
                target_mask=attention_mask,
                normalize_without_padding=True,
                eval_step_size=hparams['eval_step_size'],
                word_freq_logits=word_freq_logits,
                device=self.device
            )
    

        return metrics

    def compute_objectives(self, predictions, batch, stage):
        input_ids, _ = batch.tokens
        attention_mask, _ = batch.attention_mask
        word_freq_logits, _ = batch.word_freq_logits
        input_ids, attention_mask, word_freq_logits = input_ids.to(self.device), attention_mask.to(self.device), word_freq_logits.to(self.device)


        if stage == sb.Stage.TRAIN:
            loss = predictions['loss'] / input_ids.shape[0]
        else:
            loss = predictions['elbo']/ input_ids.shape[0]

        return loss

    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint averge if needed"""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key, min_key=min_key
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model", device=self.device
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()
        print("Loaded the average")

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        with torch.no_grad():
            predictions = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    # def on_stage_start(self, stage, epoch):
    #     """Gets called at the beginning of each epoch"""
    #     if stage != sb.Stage.TRAIN:
    # #         self.acc_metric = self.hparams.acc_computer()
    #         self.bleu_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        
        else:
            # show_process = True if stage == sb.Stage.TEST else False
            show_process=False
            file_name = "sample_test.txt" if stage == sb.Stage.TEST else f'sample_epoch_{self.hparams.epoch_counter.current}.txt'
            dataset= test_datasets['test-clean'] if stage == sb.Stage.TEST else valid_data
            sample_file = os.path.join(hparams['sample_folder'],file_name)
            generate_sample(sample_file, n_samples=hparams['n_samples'], seq_len=hparams['seq_len'],temperature= hparams['temperature'],topk=hparams['topk'], show_process=show_process, device=run_opts['device'])
            self_bleu, bleu, dist1, div4 = calculate_metric(sample_file,dataset)
            stage_stats["self_bleu"] = self_bleu
            stage_stats["bleu"] = bleu
            stage_stats["dist1"] = dist1
            stage_stats["div4"] = div4
        

        # else:
        #       stage_stats["BLEU"] = self.bleu_metric.summarize()
        #     stage_stats["ACC"] = self.acc_metric.summarize()
        #     current_epoch = self.hparams.epoch_counter.current
        #     valid_search_interval = self.hparams.valid_search_interval
        #     if (
        #         current_epoch % valid_search_interval == 0
        #         or stage == sb.Stage.TEST
        #     ):
        #         stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():

            lr = self.hparams.noam_annealing.current_lr
            steps = self.optimizer_step
            optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_stats["loss"], "epoch": epoch},
                max_keys=["loss"],
                num_to_keep=5,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            # with open(self.hparams.wer_file, "w") as w:
            #     self.wer_metric.write_stats(w)

            # save the averaged checkpoint at the end of the evaluation stage
            # delete the rest of the intermediate checkpoints
            # ACC is set to 1.1 so checkpointer only keeps the averaged checkpoint
            self.checkpointer.save_and_keep_only(
                meta={"loss": 1.1, "epoch": epoch},
                max_keys=["loss"],
                num_to_keep=1,
            )

    def fit_batch(self, batch):

        should_step = self.step % self.grad_accumulation_factor == 0
        # Managing automatic mixed precision
        if self.auto_mix_prec:
            with torch.autocast(torch.device(self.device).type):
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)

            # Losses are excluded from mixed precision to avoid instabilities
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            with self.no_sync(not should_step):
                self.scaler.scale(
                    loss / self.grad_accumulation_factor
                ).backward()
            if should_step:
                self.scaler.unscale_(self.optimizer)
                if self.check_gradients(loss):
                    self.scaler.step(self.optimizer)
                self.scaler.update()
                self.zero_grad()
                self.optimizer_step += 1
                self.hparams.noam_annealing(self.optimizer)
        else:
            if self.bfloat16_mix_prec:
                with torch.autocast(
                    device_type=torch.device(self.device).type,
                    dtype=torch.bfloat16,
                ):
                    outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                    loss = self.compute_objectives(
                        outputs, batch, sb.Stage.TRAIN
                    )
            else:
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            with self.no_sync(not should_step):
                (loss / self.grad_accumulation_factor).backward()
            if should_step:
                if self.check_gradients(loss):
                    self.optimizer.step()
                self.zero_grad()
                self.optimizer_step += 1
                self.hparams.noam_annealing(self.optimizer)

        self.on_fit_batch_end(batch, outputs, loss, should_step)
        return loss.detach().cpu()

def process_fn_in_collate(wf):
    return wf - wf.mean()

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
    valtest_datasets = [valid_data] + [i for k, i in test_datasets.items()]

    # We get the tokenizer as we need it to encode the labels when creating
    # mini-batches.
    tokenizer = hparams["tokenizer"]
    word_freq =  hparams['word_freq']


    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(valtest_datasets, audio_pipeline)

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline_train(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item([train_data], audio_pipeline_train)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list","attention_mask", "tokens_bos", "tokens_eos", "tokens","word_freq_logits"
    )
    def text_pipeline(wrd):
        yield wrd
        input_encodings = tokenizer.encode_plus(wrd, max_length=hparams['max_length'], truncation=True, add_special_tokens=False)
        tokens_list = input_encodings['input_ids']
        yield tokens_list
        attention_mask = torch.LongTensor(input_encodings['attention_mask'])
        yield attention_mask
        tokens_bos = torch.LongTensor([tokenizer.cls_token_id] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [tokenizer.sep_token_id])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens
        word_freq_logits =process_fn_in_collate(word_freq.gather(0, torch.tensor(tokens_list)))
        yield word_freq_logits

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens","attention_mask", "word_freq_logits"],
    )

    # 5. If Dynamic Batching is used, we instantiate the needed samplers.
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams = hparams["dynamic_batch_sampler"]
        num_buckets = dynamic_hparams["num_buckets"]

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            dynamic_hparams["max_batch_len"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"],
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
            max_batch_ex=dynamic_hparams["max_batch_ex"],
        )

        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            dynamic_hparams["max_batch_len_val"],
            num_buckets=num_buckets,
            length_func=lambda x: x["duration"],
            shuffle=dynamic_hparams["shuffle_ex"],
            batch_ordering=dynamic_hparams["batch_ordering"],
        )

    return (
        train_data,
        valid_data,
        test_datasets,
        tokenizer,
        train_batch_sampler,
        valid_batch_sampler,
    )
    
def word_freq_preprocess_fn(wf):
        wf = wf + 1
        wf = wf.log()
        wf = wf / wf.max()

        # range: 0 - 1
        return wf

def generate_sample(output_path, n_samples=10, seq_len=32,temperature =0.1,topk=30, show_process=False, device='cuda'):
    with open(output_path, 'a+') as fdata:
        sentences = []

        state = diffusion_word_freq.discrete_diffusion_predict_fn(
                shape= torch.Size([n_samples, seq_len]),
                denoise_fn=hparams['denoise_fn'],
                diffusion=hparams['diffusion_instance'],
                tokenizer= hparams['tokenizer'],
                predict_x0=hparams['predict_x0'],
                sample_cls=hparams['sample_cls'],
                step_size=hparams['eval_step_size'],
                topk=topk,
                target_mask=torch.ones(torch.Size([n_samples, seq_len]), device=device),
                show_process=show_process,
                temperature=temperature,
                device=device
                        # word_freq=True
                        # context_fn=context_fn
                )['final_state']
        sentence = hparams['tokenizer'].batch_decode(state)
        sentences.extend(sentence)
        # print(sentence)
        for s in sentence:
            print(s, file=fdata, flush=True)

import compute_metric 
def calculate_metric(sample_file, refernce_dataset):
    self_bleu = compute_metric.self_bleu_for_unconditional_generation(sample_file)
    dist1 = compute_metric.dist1(sample_file)
    div4 = compute_metric.div4(sample_file)
    bleu= compute_metric.compute_quality_in_unconditional_gen(refernce_dataset,sample_file)
    return self_bleu, bleu, dist1, div4

if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If --distributed_launch then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)



    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    if not os.path.isdir(hparams['sample_folder']):
       os.makedirs(hparams['sample_folder'])
    
    # 1.  # Dataset prep (parsing Librispeech)
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

    tokenizer =  hparams['lm_model'].tokenizer
    hparams["tokenizer"] = tokenizer
    
    # 4. #generate diffusion instance for handling adding/removing noise
    import  diffusion_word_freq
    diffusion_schedule = diffusion_word_freq.create_discrete_diffusion_schedule(hparams['schedule'], num_steps=hparams['num_steps'])
    diffusion_instance = diffusion_word_freq.MaskDiffusion(
        dim=tokenizer.vocab_size,
        schedule=diffusion_schedule,
        tokenizer=tokenizer,
        sample_cls=hparams['sample_cls'],
        word_freq_lambda=hparams['word_freq_lambda'],
        device=run_opts['device'],
    )
    hparams['diffusion_instance'] = diffusion_instance
    hparams['diffusion_schedule'] = diffusion_schedule



    # 3.  # calculate word-freq for tokens in training data ti be used in Spindle noise schedule
    from word_freq import prepare_word_frequencies  # noqa
    run_on_main(
        prepare_word_frequencies,
        kwargs={
            "data_file": hparams["train_csv"],
            "save_folder": hparams["output_folder"],
            "tokenizer": hparams["tokenizer"]
        },
    )
    word_freq = torch.load(os.path.join(hparams["output_folder"], "word_freq.pt"))
    if not( word_freq.size(0) == tokenizer.vocab_size):
        logger.error("Word frequency file and tokenizer don't match.!!!")
    
    word_freq = word_freq_preprocess_fn(word_freq)
    word_freq[tokenizer.pad_token_id] = 0.  # stable training
    hparams['word_freq']= word_freq

    cls = torch.full((1, 1), fill_value=tokenizer.cls_token_id, device=run_opts['device'])
    sep = torch.full((1, 1), fill_value=tokenizer.sep_token_id, device=run_opts['device'])

    att_ones = torch.ones((1, 1), device=run_opts['device'])
    att_zeros = torch.zeros((1, 1), device=run_opts['device'])

    # define denoise funstion based on the how time-step is incorporated in the model
    if  hparams['timestep'] == 'none':
        def denoise_fn(targets, timestep, attention_mask):
            assert len(targets.size()) == 2  # bsz * seqlen
            bsz = targets.size(0)
            targets = torch.cat((cls.repeat(bsz, 1), targets, sep.repeat(bsz, 1)), dim=1)
            attention_mask = torch.cat((att_ones.repeat(bsz, 1), attention_mask, att_zeros.repeat(bsz, 1)), dim=1)
            return hparams['lm_model'](input_ids=targets, timestep=timestep - 1, attention_mask=attention_mask)['logits'][:, 1:-1, :]
    elif hparams['timestep'] == 'token':

        def denoise_fn(targets, timestep, attention_mask):
            assert len(targets.size()) == 2  # bsz * seqlen
            bsz = targets.size(0)
            targets = torch.cat((
                cls.repeat(bsz, 1),
                torch.full((bsz, 1), fill_value=timestep.item() + 110, device=run_opts['device']),
                targets,
                sep.repeat(bsz, 1)
            ), dim=1)
            attention_mask = torch.cat((att_ones.repeat(bsz, 2), attention_mask, att_zeros.repeat(bsz, 1)), dim=1)
            return hparams['lm_model'](input_ids=targets, timestep=timestep - 1, attention_mask=attention_mask)['logits'][:, 2:-1, :]
    elif hparams['timestep'] == 'layerwise':
        def denoise_fn(targets, timestep, attention_mask):
            assert len(targets.size()) == 2  # bsz * seqlen
            bsz = targets.size(0)
            targets = torch.cat((
                cls.repeat(bsz, 1),
                targets,
                sep.repeat(bsz, 1)
            ), dim=1)
            attention_mask = torch.cat((att_ones.repeat(bsz, 1), attention_mask, att_zeros.repeat(bsz, 1)), dim=1)
            return hparams['lm_model'](input_ids=targets, timestep=timestep - 1, attention_mask=attention_mask)['logits'][:, 1:-1, :]
    else:
        raise NotImplementedError
    hparams['denoise_fn'] = denoise_fn

    hparams['diffusion_instance'].word_freq = word_freq.to(run_opts['device'])


    # here we create the datasets objects as well as tokenization and encoding
    (
        train_data,
        valid_data,
        test_datasets,
        tokenizer,
        train_bsampler,
        valid_bsampler,
    ) = dataio_prepare(hparams)


    # # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # # the path given in the YAML file). The tokenizer is loaded at the same time.
    # run_on_main(hparams["pretrainer"].collect_files)
    # hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # adding objects to trainer:
    asr_brain.tokenizer = hparams["tokenizer"]
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    if train_bsampler is not None:
        train_dataloader_opts = {
            "batch_sampler": train_bsampler,
            "num_workers": hparams["num_workers"],
        }

    if valid_bsampler is not None:
        valid_dataloader_opts = {"batch_sampler": valid_bsampler}

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=train_dataloader_opts,
        valid_loader_kwargs=valid_dataloader_opts,
    )
    # Testing
    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        asr_brain.evaluate(
            test_datasets[k],
            max_key="loss",
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )
    
    # sample_file = os.path.join(hparams['sample_folder'],'sample_test.txt')
    # generate_sample(sample_file, n_samples=hparams['n_samples'], seq_len=hparams['seq_len'],temperature= hparams['temperature'],topk=hparams['topk'], show_process=True, device=run_opts['device'])
    # calculate_metric(sample_file, test_datasets['test-clean'])

 