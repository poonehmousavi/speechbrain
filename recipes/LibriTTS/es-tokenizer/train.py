#!/usr/bin/env python3
"""Recipe for training a hifi-gan vocoder.
For more details about hifi-gan: https://arxiv.org/pdf/2010.05646.pdf

To run this recipe, do the following:
> python train.py hparams/train.yaml --data_folder /path/to/LibriTTS

Authors
 * Pooneh Mousavi 2024
"""

import copy
import pathlib as pl
import random
import sys

import numpy as np
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from einops import rearrange
import speechbrain as sb
from speechbrain.utils.data_utils import scalarize
from speechbrain.tokenizers.SentencePiece import SentencePiece


class ESTBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """The forward function, generates synthesized waveforms,
        calculates the scores and the features of the discriminator
        for synthesized waveforms and real waveforms.

        Arguments
        ---------
        batch: str
            a single batch
        stage: speechbrain.Stage
            the training stage

        Returns
        -------
        y_g_hat : torch.Tensor
        scores_fake : torch.Tensor
        feats_fake : torch.Tensor
        scores_real : torch.Tensor
        feats_real : torch.Tensor
        """
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        y, _ = batch.sig
        y=y.unsqueeze(1)

        # generate synthesized waveforms
        with torch.set_grad_enabled(not self.hparams.freeze_ssl):
            feats = self.modules.ssl_model(wavs, wav_lens)
        x = self.modules.codec( rearrange(feats,"n b t d -> n b d t"))
        y_g_hat,(log_dur_pred, log_dur) = self.modules.generator(rearrange(x['embeddings'],"b n d t -> b t n d").contiguous())
        y_g_hat = y_g_hat[:,:,:y.shape[2]]
        # get scores and features from discriminator for real and synthesized waveforms
        scores_fake, feats_fake = self.modules.discriminator(y_g_hat.detach())
        scores_real, feats_real = self.modules.discriminator(y[:,:,:y_g_hat.shape[-1]])
        vq_los={"vq/commitment_loss":x["vq/commitment_loss"], "vq/codebook_loss":x["vq/codebook_loss"]}

        return (y_g_hat, scores_fake, feats_fake, scores_real, feats_real,log_dur_pred,log_dur, vq_los)

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.
        Arguments
        ---------
        predictions : torch.Tensor
            The model generated spectrograms and other metrics from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        batch = batch.to(self.device)
        y, _ = batch.sig
        y=y.unsqueeze(1)

        # Hold on to the batch for the inference sample. This is needed because
        # the inference sample is run from on_stage_end only, where
        # batch information is not available
        self.last_batch = (y, y)

        y_hat, scores_fake, feats_fake, scores_real, feats_real, log_dur_pred,log_dur ,vq_los= predictions
        loss_g = self.hparams.generator_loss(
            stage, y_hat, y[:,:,:y_hat.shape[-1]], scores_fake, feats_fake, feats_real,log_dur_pred,log_dur
        )
        loss_d = self.hparams.discriminator_loss(scores_fake, scores_real)
        loss = {**loss_g, **loss_d, **vq_los}
        self.last_loss_stats[stage] = scalarize(loss)
        return loss

    def fit_batch(self, batch):
        """Fits a single batch.
        Arguments
        ---------
        batch: tuple
            a training batch
        Returns
        -------
        loss: torch.Tensor
            detached loss
        """
        batch = batch.to(self.device)
        y, _ = batch.sig
        y=y.unsqueeze(1)
        outputs = self.compute_forward(batch, sb.core.Stage.TRAIN)
        (y_g_hat, scores_fake, feats_fake, scores_real, feats_real, log_dur_pred,log_dur, vq_los) = outputs
        # calculate discriminator loss with the latest updated generator
        loss_d = self.compute_objectives(outputs, batch, sb.core.Stage.TRAIN)[
            "D_loss"
        ]
        # First train the discriminator
        self.optimizer_d.zero_grad()
        loss_d.backward()
        self.optimizer_d.step()

        # calculate generator loss with the latest updated discriminator
        scores_fake, feats_fake = self.modules.discriminator(y_g_hat)
        scores_real, feats_real = self.modules.discriminator(y[:,:,:y_g_hat.shape[-1]])
        outputs = (y_g_hat, scores_fake, feats_fake, scores_real, feats_real, log_dur_pred,log_dur, vq_los)
        loss_g = self.compute_objectives(outputs, batch, sb.core.Stage.TRAIN)[
            "G_loss"
        ]
        # Then train the generator
        self.optimizer_g.zero_grad()
        loss_g.backward()
        self.optimizer_g.step()

        return loss_g.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Evaluate one batch.

        Arguments
        ---------
        batch : list of torch.Tensors
            Batch of data to use for evaluation. Default implementation assumes
            this batch has two elements: inputs and targets.
        stage : Stage
            The stage of the experiment: Stage.VALID, Stage.TEST

        Returns
        -------
        detached loss
        """
        out = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(out, batch, stage=stage)
        loss_g = loss["G_loss"]
        return loss_g.detach().cpu()

    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp and initializes statistics.
        """
        self.last_epoch = 0
        self.last_batch = None
        self.last_loss_stats = {}
        return super().on_fit_start()

    def init_optimizers(self):
        """Called during ``on_fit_start()``, initialize optimizers
        after parameters are fully configured (e.g. DDP, jit).
        """
        if self.opt_class is not None:
            (
                opt_g_class,
                opt_d_class,
                sch_g_class,
                sch_d_class,
            ) = self.opt_class

            self.optimizer_g = opt_g_class(self.modules.generator.parameters())
            self.optimizer_d = opt_d_class(
                self.modules.discriminator.parameters()
            )
            self.optimizers_dict = {
                "optimizer_g": self.optimizer_g,
                "optimizer_d": self.optimizer_d,
            }

            self.scheduler_g = sch_g_class(self.optimizer_g)
            self.scheduler_d = sch_d_class(self.optimizer_d)

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable(
                    "optimizer_g", self.optimizer_g
                )
                self.checkpointer.add_recoverable(
                    "optimizer_d", self.optimizer_d
                )
                self.checkpointer.add_recoverable(
                    "scheduler_g", self.scheduler_d
                )
                self.checkpointer.add_recoverable(
                    "scheduler_d", self.scheduler_d
                )

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        if stage == sb.Stage.VALID:
            # Update learning rate
            self.scheduler_g.step()
            self.scheduler_d.step()
            lr_g = self.optimizer_g.param_groups[-1]["lr"]
            lr_d = self.optimizer_d.param_groups[-1]["lr"]

            stats = {
                **self.last_loss_stats[sb.Stage.VALID],
            }

            self.hparams.train_logger.log_stats(  # 1#2#
                stats_meta={"Epoch": epoch, "lr_g": lr_g, "lr_d": lr_d},
                train_stats=self.last_loss_stats[sb.Stage.TRAIN],
                valid_stats=stats,
            )
            # The tensorboard_logger writes a summary to stdout and to the logfile.
            if self.hparams.use_tensorboard:
                self.tensorboard_logger.log_stats(
                    stats_meta={"Epoch": epoch, "lr_g": lr_g, "lr_d": lr_d},
                    train_stats=self.last_loss_stats[sb.Stage.TRAIN],
                    valid_stats=stats,
                )

            # Save the current checkpoint and delete previous checkpoints.
            epoch_metadata = {
                **{"epoch": epoch},
                **self.last_loss_stats[sb.Stage.VALID],
            }
            if self.checkpointer is not None:
                self.checkpointer.save_and_keep_only(
                    meta=epoch_metadata,
                    end_of_epoch=True,
                    min_keys=["loss"],
                    ckpt_predicate=(
                        (
                            lambda ckpt: (
                                ckpt.meta["epoch"]
                                % self.hparams.keep_checkpoint_interval
                                != 0
                            )
                        )
                        if self.hparams.keep_checkpoint_interval is not None
                        else None
                    ),
                )

            self.run_inference_sample("Valid", epoch)

        # We also write statistics about test data to stdout and to the TensorboardLogger.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(  # 1#2#
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=self.last_loss_stats[sb.Stage.TEST],
            )
            if self.hparams.use_tensorboard:
                self.tensorboard_logger.log_stats(
                    {"Epoch loaded": self.hparams.epoch_counter.current},
                    test_stats=self.last_loss_stats[sb.Stage.TEST],
                )
            self.run_inference_sample("Test", epoch)

    def run_inference_sample(self, name, epoch):
        """Produces a sample in inference mode.
        This is called when producing samples.

        Arguments
        ---------
        name: str
            the name of the saved audio folder
        epoch: int or str
            the epoch number (used in file path calculations)
            or "test" for test stage

        Returns
        -------
        None
        """
        with torch.no_grad():
            if self.last_batch is None:
                return
            x, y = self.last_batch
            feats = self.hparams.ssl_model(x.squeeze(1))
            x = self.hparams.codec( rearrange(feats,"n b t d -> n b d t"))
            # Preparing model for inference by removing weight norm
            inference_generator = copy.deepcopy(self.hparams.generator)
            inference_generator.remove_weight_norm()
            if inference_generator.duration_predictor:
                x = torch.unique_consecutive(x, dim=1)
            sig_out = inference_generator.inference(rearrange(x['embeddings'],"b n d t -> b t n d"))
            spec_out = self.hparams.mel_spectogram(
                audio=sig_out.squeeze(0).cpu()
            )
        if self.hparams.use_tensorboard:
            self.tensorboard_logger.log_audio(
                f"{name}/audio_target", y.squeeze(0), self.hparams.sample_rate
            )
            self.tensorboard_logger.log_audio(
                f"{name}/audio_pred",
                sig_out.squeeze(0),
                self.hparams.sample_rate,
            )
            self.tensorboard_logger.log_figure(f"{name}/mel_target", x)
            self.tensorboard_logger.log_figure(f"{name}/mel_pred", spec_out)
        else:
            # folder name is the current epoch for validation and "test" for test
            folder = (
                self.hparams.epoch_counter.current
                if name == "Valid"
                else "test"
            )
            self.save_audio("target", y.squeeze(0), folder)
            self.save_audio("synthesized", sig_out.squeeze(0), folder)

    def save_audio(self, name, data, epoch):
        """Saves a single wav file.

        Arguments
        ---------
        name: str
            the name of the saved audio
        data: torch.Tensor
            the  wave data to save
        epoch: int or str
            the epoch number (used in file path calculations)
            or "test" for test stage
        """
        target_path = pl.Path(self.hparams.progress_sample_path) / str(epoch)
        target_path.mkdir(parents=True, exist_ok=True)
        file_name = target_path / f"{name}.wav"
        torchaudio.save(file_name.as_posix(), data.cpu(), 16000)


def sample_interval(seqs, segment_size):
    "This function sample an interval of audio and code according to segment size."
    N = max([v.shape[-1] for v in seqs])
    seq_len = segment_size if segment_size > 0 else N
    hops = [N // v.shape[-1] for v in seqs]
    lcm = np.lcm.reduce(hops)
    interval_start = 0
    interval_end = N // lcm - seq_len // lcm
    start_step = random.randint(interval_start, interval_end)

    new_seqs = []
    for i, v in enumerate(seqs):
        start = start_step * (lcm // hops[i])
        end = (start_step + seq_len // lcm) * (lcm // hops[i])
        new_seqs += [v[..., start:end]]

    return new_seqs


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    # Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "segment")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav,segment):
        segment_size = hparams["segment_size"]
        info = torchaudio.info(wav)
        audio = sb.dataio.dataio.read_audio(wav)
        audio = torchaudio.transforms.Resample(
            info.sample_rate,
            hparams["sample_rate"],
        )(audio)
        while audio.shape[0] < segment_size:
            audio = torch.hstack([audio, audio])

        if segment:
            audio = sample_interval([audio], segment_size)[0]
        return audio
    # Define text pipeline:
    @sb.utils.data_pipeline.takes("label")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens"
    )
    def text_pipeline(label):
        yield label
        tokens_list = hparams['text_tokenizer'].sp.encode_as_ids(label)
        yield tokens_list
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    datasets = {}
    data_info = {
        "train": hparams["train_json"],
        "valid": hparams["valid_json"],
        "test": hparams["test_json"],
    }
    for dataset in hparams["splits"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline,text_pipeline ],
            output_keys=["id", "sig","tokens"],
        )

    return datasets


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
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

    # Dataset prep (parsing LibriTTS)
    from libritts_prepare import prepare_libritts

    sb.utils.distributed.run_on_main(
        prepare_libritts,
        kwargs={
            "data_folder": hparams["data_folder"],
            "save_json_train": hparams["train_json"],
            "save_json_valid": hparams["valid_json"],
            "save_json_test": hparams["test_json"],
            "sample_rate": hparams["sample_rate"],
            "split_ratio": hparams["split_ratio"],
            "libritts_subsets": hparams["libritts_subsets"],
            "train_split": hparams["train_split"],
            "valid_split": hparams["valid_split"],
            "test_split": hparams["test_split"],
            "model_name": "HiFi-GAN",
            "skip_prep": hparams["skip_prep"],
        },
    )
    # Defining tokenizer and loading it
    text_tokenizer = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["output_neurons"],
        annotation_train=hparams["train_json"],
        annotation_read="label",
        model_type=hparams["token_type"],
        character_coverage=hparams["character_coverage"],
        annotation_format="json",
    )
    hparams['text_tokenizer'] = text_tokenizer
    # here we create the datasets objects as well as tokenization and encoding
    datasets = dataio_prepare(hparams)

    # Brain class initialization
    est_brain = ESTBrain(
        modules=hparams["modules"],
        opt_class=[
            hparams["opt_class_generator"],
            hparams["opt_class_discriminator"],
            hparams["sch_class_generator"],
            hparams["sch_class_discriminator"],
        ],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    est_brain.fit(
        est_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Test
    if "test" in datasets:
        est_brain.evaluate(
            datasets["test"],
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )
