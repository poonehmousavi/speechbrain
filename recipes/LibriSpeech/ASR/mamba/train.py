#!/usr/bin/env/python3
"""Recipe for training a wav2vec-based ctc ASR system with librispeech.
The system employs wav2vec as its encoder. Decoding is performed with
ctc greedy decoder during validation and a beam search with an optional
language model during test. The test searcher can be chosen from the following
options: CTCBeamSearcher, CTCPrefixBeamSearcher, TorchAudioCTCPrefixBeamSearcher.

To run this recipe, do the following:
> python train_with_wav2vec.py hparams/train_{hf,sb}_wav2vec.yaml
The neural network is trained on CTC likelihood target and character units
are used as basic recognition tokens.

Authors
 * Pooneh Mousavi 2024
"""
import os
import sys
import torch
import torchaudio
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main, if_main_process
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
from speechbrain.dataio.batch import PaddedBatch

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        input_ids, input_ids_lens = batch.input_ids
        input_ids, input_ids_lens = input_ids.to(self.device), input_ids_lens.to(self.device)

        # # Downsample the inputs if specified
        # if hasattr(self.modules, "downsampler"):
        #     wavs = self.modules.downsampler(wavs)

        # # Add waveform augmentation if specified.
        # if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
        #     wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)

                # Forward Pass
        padding_mask = ~self.hparams.padding_mask(
            input_ids, pad_idx=self.tokenizer.unk_token_id
        )
        outputs = self.modules.mamba(
            input_ids, padding_mask
        ).logits

        return  outputs


    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        batch = batch.to(self.device)
        ids = batch.id
        lm_labels, labels_lens = batch.lm_labels
        audio_bos, audio_lens = batch.audio_bos
        tokens_eos, tokens_eos_lens = batch.text_tokens_eos


        loss = self.hparams.ce_loss(
            predictions.flatten(end_dim=-2), lm_labels.flatten()
        )
        if stage != sb.Stage.TRAIN:
            # hyps = None
            # current_epoch = self.hparams.epoch_counter.current
            # if current_epoch % self.hparams.valid_search_interval == 0:
            # history_bos = torch.LongTensor([hparams["bos_index"]] + (history_bos))
            padding_mask = ~self.hparams.padding_mask(
                audio_bos, pad_idx=self.tokenizer.unk_token_id
            )
            hyps = self.modules.mamba.generate(
                audio_bos.detach(),
                padding_mask.detach(),
            )
        # elif stage == sb.Stage.TEST:
        #     padding_mask = ~self.hparams.padding_mask(
        #         audio_bos, pad_idx=self.tokenizer.unk_token_id
        #     )
        #     hyps = self.modules.mamba.generate(
        #         audio_bos.detach(),
        #         padding_mask.detach(),
        #         "beam",
        #     )

        if stage != sb.Stage.TRAIN:
            text_truncated = [
                tokens_eos[i][
                    : int(tokens_eos_lens[i].item() * tokens_eos.shape[1] - 1)
                ].detach()
                for i in range(tokens_eos_lens.shape[0])
            ]
            predicted_words = self.tokenizer.batch_decode(
                hyps[:, audio_bos.shape[1] :],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            target_words = self.tokenizer.batch_decode(
                text_truncated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

        if stage == sb.Stage.TEST:
            if hasattr(self.hparams, "rescorer"):
                self.hparams.rescorer.move_rescorers_to_device()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.model_optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]},
                min_keys=["WER"],
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
        "Initializes the wav2vec2 optimizer and model optimizer"
        self.model_optimizer = self.hparams.mamba_opt_class(
            self.hparams.mamba_model.parameters()
        )

        # save the optimizers in a dictionary
        # the key will be used in `freeze_optimizers()`
        self.optimizers_dict = {
            "model_optimizer": self.model_optimizer,
        }

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)


def dataio_prepare(hparams,tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"],
        replacements={"data_root": data_folder},
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
        csv_path=hparams["valid_csv"],
        replacements={"data_root": data_folder},
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
    # convert special tokens to their ids
    bos, eos, audio, text = tokenizer.convert_tokens_to_ids(
        hparams["special_tokens"]
    )
    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]
    
    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig","codec_tokens","audio_tokens","audio_input","audio_bos")
    def audio_pipeline(wav):
        info = torchaudio.info(wav)
        sig = sb.dataio.dataio.read_audio(wav)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate,
            hparams["sample_rate"],)(sig)
        yield resampled 
        with torch.no_grad():
            audio_codec = hparams['codec'].eval()
            codec_tokens =audio_codec.encode(resampled.to(run_opts['device']).unsqueeze(0), torch.tensor([1.0],device=run_opts['device']))[0]
            codec_tokens += torch.arange(
                0,
                hparams['num_codebooks'] * hparams['vocab_size'],
                hparams['vocab_size'],
                device=run_opts['device']
            )
            codec_tokens = codec_tokens.squeeze(0).flatten(0,1)
            yield codec_tokens
            audio_tokens = codec_tokens.to('cpu').apply_(lambda x: tokenizer.get_added_vocab()[f"<AT-{x}>"])
            yield audio_tokens
            audio_input = torch.cat((torch.tensor([audio]), audio_tokens))
            yield audio_input
            audio_bos = torch.cat((torch.tensor([bos]), audio_input))
            yield audio_bos

            


    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)


    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "text_tokens","text_tokens_eos"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode(wrd)
        yield tokens_list
        tokens = torch.LongTensor(tokens_list)
        yield tokens
        tokens_eos = torch.cat((tokens,torch.tensor([eos])))
        yield tokens_eos

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # Define input_and_token_type_pipeline
    @sb.utils.data_pipeline.takes(
        "audio_input","audio_bos","text_tokens", "text_tokens_eos"
    )
    @sb.utils.data_pipeline.provides("input_ids", "lm_labels")
    def input_pipeline(
         audio_input,audio_bos, text_tokens,text_tokens_eos
    ):
        # yield "[AD] " + item + " [/AD] [TR] "

        # put history and reply together
        # N.B. input_sequence = history_ids + reply_ids, we don't have eos in the input
        input_ids = torch.cat((audio_bos, torch.tensor([text]), text_tokens), -1)
        yield input_ids

        lm_labels = (
            [hparams["ignore_index"]] * audio_input.shape[0]
            + [hparams["ignore_index"]]
            + text_tokens_eos.tolist()
        )
        lm_labels = torch.LongTensor(lm_labels)
        yield lm_labels

    sb.dataio.dataset.add_dynamic_item(datasets, input_pipeline)


    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "wrd","text_tokens", "audio_bos", "text_tokens_eos" , "input_ids", "lm_labels"],
    )

    return train_data, valid_data, test_datasets

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

    
    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )


    # We dynamically add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for the LM!!
    asr_brain.tokenizer = hparams['mamba_model'].tokenizer
    # check if the tokens are already in the vocabulary
    new_tokens = [f"<AT-{i}>"for i in range(hparams['vocab_size']*hparams["num_codebooks"])]
    new_tokens = set(new_tokens) - set(asr_brain.tokenizer.vocab.keys())
    # add the tokens to the tokenizer vocabulary
    asr_brain.tokenizer.add_tokens(list(new_tokens))
    # add new, random embeddings for the new tokens
    hparams['mamba_model'].model.resize_token_embeddings(len(asr_brain.tokenizer))

    num_added_tokens = asr_brain.tokenizer.add_special_tokens(hparams['attr_to_special_tokens'])
    hparams['mamba_model'].model.resize_token_embeddings(len(asr_brain.tokenizer))




    class CustomPaddedBatch(PaddedBatch):
            """PaddedBatch with custom padding values.

            See the documentation of `speechbrain.dataio.batch.PaddedBatch`.

            """ 

            def __init__(self, examples, *args, **kwargs):
                _, _, _, text = asr_brain.tokenizer.convert_tokens_to_ids(
                    hparams["special_tokens"]
                )
                for k in [
                    "input_ids",
                    "audio_bos",
                    "lm_labels",
                ]:
                    max_len = max([len(x[k]) for x in examples])
                    pad_value = 0
                    if k in [
                        "input_ids",
                        "audio_bos",
                    ]:
                        pad_value = asr_brain.tokenizer.unk_token_id
                    elif k == "lm_labels":
                        pad_value = hparams["ignore_index"]
                    for example in examples:
                        x = example[k]
                        if k in ["audio_bos"]:
                            x = torch.cat(
                                (example[k], torch.LongTensor([text])), -1
                            )
                            example[k] = torch.nn.functional.pad(
                                x, [max_len - len(x), 0], value=pad_value
                            )
                        else:
                            example[k] = torch.nn.functional.pad(
                                x, [0, max_len - len(x)], value=pad_value
                            )
                super().__init__(examples, *args, **kwargs)

    hparams["train_dataloader_opts"]["collate_fn"] = CustomPaddedBatch
    hparams["valid_dataloader_opts"]["collate_fn"] = CustomPaddedBatch
    hparams["test_dataloader_opts"]["collate_fn"] = CustomPaddedBatch

    
    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets = dataio_prepare(
        hparams,hparams['mamba_model'].tokenizer
    )





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
            test_loader_kwargs=hparams["test_dataloader_opts"],
            min_key="WER",
        )
