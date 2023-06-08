"""
Inspired and partly copied from https://github.com/google-research/google-research/blob/ad2d81983e4c717f477a232f625d0da2808b15aa/d3pm/text/diffusion.py
"""

import os
import torch
import abc
import numpy as np
from typing import Any
import utils
# from transformers import AutoTokenizer, top_k_top_p_filtering
# from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

class DiffusionSchedule:
    """A wrapper around a simple schedule function."""

    def __init__(self, schedule_fn, num_steps, is_constant=False):
        self._schedule_fn = schedule_fn
        self.num_steps = num_steps
        self.is_constant = is_constant

    def __call__(self, step):
        return self._schedule_fn(step)

    def __repr__(self):
        return f"DiffusionSchedule(steps: {self.num_steps}, is_constant: {self.is_constant})"

class DiscreteDiffusionBase(abc.ABC):
    num_steps: int
    dim: int
    tokenizer: Any

    @abc.abstractmethod
    def stationary_probs(self, shape):
        """Returns probs for the stationary distribution."""

    @abc.abstractmethod
    def sample_stationary(self, shape):
        """Draws a sample from the stationary distribution (q(x_T))."""

    def sample_t(self, size=(1,)):
        """Samples batches of time steps to use."""
        return torch.randint(low=0, high=self.num_steps, size=size, device=self.device)

    def supports_efficient_get(self):
        """Returns true if get() is implemented/efficient."""
        return False

    def supports_efficient_inference(self):
        """Returns true if custom_product_fn is implemented.
        The ontology of efficient_get and efficient_inference is this:
          * if efficient_inference is enabled, it is used to return q(x_t | x_0)
            without computing expensive products.
          * if efficient_get is enabled, get(...) is used to get the posterior of
            q(x_{t-1} | x_t, x_0). If not, get_q_given_q0 is called to get
            q(x_{t+1} | x_0), and qt_reverse is called to get the q(x_{t+1} | x_0).
        """
        return False

    @abc.abstractmethod
    def get_qt_given_q0(self,
                        q0,
                        t,
                        return_logits=False,
                        make_one_hot=False,
                        word_freq_logits=None,
                        epsilon=1e-20):
        """Get q(x_t), the n-step posterior.
        For example, for t = 0, it returns q0 unchanged.
        Args:
          q0: an array of floats specifying a distribution over p(x_0).
          t: t in q(x_t | x_0).
          return_logits: if True, return the output logits
          make_one_hot: if True, will convert q0 to floats if needed.
          epsilon: a small number to normalize logits conversion with, if needed.
        Returns:
          q(x_t | x_0).
        """

    @abc.abstractmethod
    def sample_and_compute_posterior_q(self,
                                       x_0,
                                       t,
                                       samples=None,
                                       transition_probs=None,
                                       return_logits=True,
                                       return_transition_probs=False,
                                       transition_probs_in_logits=True,
                                       make_one_hot=True,
                                       epsilon=1e-20,
                                       step_size=1,
                                       word_freq_logits=None,
                                       ):
        """Samples from q(x_{t+1} | x_0), then computes q(x_t | x_{t+1}, x_0).
        Args:
          x_0: an array containing x_0 samples. These are expected to be integral
            unless make_one_hot is False (in which case probabilities can be
            provided).
          t: the timestep to compute (as an int or integer array with shape that
            matches x_0.
          samples: if not None, use these samples to compute the posterior.
          transition_probs: precomputed transition probabilities.
          return_logits: if True, returns the (noisy) log of the probabilities.
          return_transition_probs: if true, returns the transition probs as well.
          transition_probs_in_logits: include transition probs in logits.
          make_one_hot: if True, will convert the input to a one_hot vector.
          epsilon: a small amount of noise to add to logits if needed.
          step_size: if provided, computes q(x_{t + step_size} | x_0), etc. This is
            used to sample fewer steps for ELBO evaluation on a longer trained
            model.
        Returns:
          a list of samples with the same shape as x_0 and the associated posterior
          probabilities (or logits).
        """


class DiscreteDiffusionMatrixBase(DiscreteDiffusionBase):
    """Base class for all matrix-noise schedulers."""
    num_steps: int
    dim: int
    tokenizer: Any

    def get(self, t):
        """Returns the transition matrix q(x_{t+1} | x_t)."""
        raise NotImplementedError

    def custom_product_fn(self, t):
        """Returns q(x_t | x_0), the product of the first t matrices."""
        raise NotImplementedError

    def qt_reverse(self,
                   qt_plus_1,
                   t,
                   return_logits=False,
                   make_one_hot=False,
                   epsilon=1e-20):
        """Get q(x_{t+1} | x_t), the one-step posterior efficiently.
        Args:
          qt_plus_1: an array of floats specifying a distribution over p(x_0).
          t: t in q(x_{t+1} | x_t).
          return_logits: if True, return the output logits
          epsilon: a small number to normalize logits conversion with, if needed.
        Returns:
          q(x_{t+1} | x_t).
        """
        raise NotImplementedError

    def get_qt_matrix(self, t):
        """Returns the matrix Q = q(x_t | x_0) materialized over all x_0."""
        if self.supports_efficient_inference():
            return self.custom_product_fn(t)

        logger.warn("WARNING: using inefficient matrix product.")

        # otherwise, multiply by the ith matrix in a for-loop.
        def product_fn(i, state):
            return torch.matmul(self.get(i), state)

        final_product = utils.fori_loop(0, t, product_fn, torch.eye(self.dim))

        return final_product

    def get_qt_given_q0(self,
                        q0,
                        t,
                        return_logits=False,
                        make_one_hot=False,
                        word_freq_logits=None,
                        epsilon=1e-20):
        """Get q(x_t), the n-step posterior.
        For example, for t = 0, it returns q0 unchanged.
        Args:
          q0: an array of floats specifying a distribution over p(x_0).
          t: t in q(x_t | x_0).
          return_logits: if True, return the output logits
          make_one_hot: if True, will convert q0 to floats if needed.
          epsilon: a small number to normalize logits conversion with, if needed.
        Returns:
          q(x_t).
        """
        if make_one_hot:
            q0 = torch.nn.functional.one_hot(q0, num_classes=self.dim)
        if self.supports_efficient_inference():
            prob_at_time_t = torch.einsum("ij,...j", self.get_qt_matrix(t), q0)
            if return_logits:
                return torch.log(prob_at_time_t + epsilon)
            else:
                return prob_at_time_t
        else:
            raise NotImplementedError

    def sample_and_compute_posterior_q(self,
                                       x_0,
                                       t,
                                       samples=None,
                                       transition_probs=None,
                                       return_logits=True,
                                       return_transition_probs=False,
                                       transition_probs_in_logits=True,
                                       make_one_hot=True,
                                       epsilon=1e-20,
                                       step_size=1,
                                       word_freq_logits=None,
                                       ):
        """Samples from q(x_{t+1} | x_0), then computes q(x_t | x_{t+1}, x_0).
        Args:
          x_0: an array containing x_0 samples. These are expected to be integral
            unless make_one_hot is False (in which case probabilities can be
            provided).
          t: the timestep to compute (as an int or integer array with shape that
            matches x_0.
          samples: if not None, use these samples to compute the posterior.
          transition_probs: precomputed transition probabilities.
          return_logits: if True, returns the (noisy) log of the probabilities.
          return_transition_probs: if true, returns the transition probs as well.
          transition_probs_in_logits: include transition probs in logits.
          make_one_hot: if True, will convert the input to a one_hot vector.
          epsilon: a small amount of noise to add to logits if needed.
          step_size: if provided, computes q(x_{t + step_size} | x_0), etc. This is
            used to sample fewer steps for ELBO evaluation on a longer trained
            model.
        Returns:
          a list of samples with the same shape as x_0 and the associated posterior
          probabilities (or logits).
        """
        dim = self.dim
        if make_one_hot:
            x_0 = torch.nn.functional.one_hot(x_0, dim).reshape(x_0.shape + (dim,))

        prob_at_time_t = self.get_qt_given_q0(q0=x_0, t=t, word_freq_logits=word_freq_logits)

        if self.supports_efficient_get():
            if step_size > 1:
                transition_matrix = torch.eye(self.dim)

                for i in range(step_size):
                    transition_matrix = self.get(t + i) @ transition_matrix

            else:
                transition_matrix = self.get(t)

            prob_at_time_t_plus_one = torch.einsum(
                "ij,...j->...i",
                transition_matrix,
                prob_at_time_t,
            )
        else:
            prob_at_time_t_plus_one = self.get_qt_given_q0(q0=x_0, t=t + step_size, word_freq_logits=word_freq_logits)

        if samples is None and transition_probs is not None:
            raise ValueError("samples were not provided but transition_probs were.")

        if samples is None:
            logits = torch.log(prob_at_time_t_plus_one + epsilon)
            samples = self.sample_cls.sample(logits, x_0)

        if transition_probs is None:
            if self.supports_efficient_get():
                transition_probs = transition_matrix[samples]
            else:
                if step_size > 1:
                    transition_probs = torch.nn.functional.one_hot(samples, self.dim)
                    for i in range(step_size):
                        transition_probs = self.qt_reverse(
                            qt_plus_1=transition_probs,
                            make_one_hot=False,
                            t=t + step_size - 1 - i)
                else:
                    transition_probs = self.qt_reverse(qt_plus_1=samples, make_one_hot=True, t=t)

        if not transition_probs_in_logits and not return_logits:
            raise ValueError(
                "Cannot exclude transition probs from logits if return_logits is false."
            )

        if return_logits:
            # for numerical stability, we can compute log(a*b) = log(a) + log(b)
            posterior_logits = torch.log(prob_at_time_t + epsilon)

            if transition_probs_in_logits:
                posterior_logits = posterior_logits + torch.log(transition_probs + epsilon)

            if return_transition_probs:
                return posterior_logits, samples, transition_probs
            else:
                return posterior_logits, samples
        else:
            # here we hope this never actually sums to zero. There's a chance
            # this will produce NaN gradients, but that's OK because they'll be
            # skipped.

            posterior = transition_probs * prob_at_time_t
            denominator = posterior.sum(dim=-1, keepdims=True)
            posterior = posterior / denominator
            if return_transition_probs:
                return posterior, samples, transition_probs
            else:
                return posterior, samples



class MaskDiffusion(DiscreteDiffusionMatrixBase):
    def __init__(self,
                 dim,
                 schedule,
                 tokenizer,
                 use_fast_inference=True,
                 sample_cls=None,
                 word_freq=None,
                 word_freq_lambda=0.,
                 device=None,
                 ):
        """A simple scheduler for masking policies.
        Args:
          dim: int, the dimensionality of the state space.
          schedule: a DiffusionSchedule object for scheduling rates.
        """

        self.num_steps = schedule.num_steps
        self.sample_cls = sample_cls
        self.schedule = schedule
        self.use_fast_inference = use_fast_inference
        self.dim = dim  # allow mask
        self.tokenizer = tokenizer
        self.device = device
        # self.mask = torch.nn.functional.one_hot(torch.tensor(self.tokenizer.mask_token_id, device=device), num_classes=self.dim).unsqueeze(1).repeat(1, self.dim).float()
        self.mask = None
        self.state = self._create_state()
        self.word_freq = word_freq
        import math
        self.word_freq_lambda = word_freq_lambda * torch.sin(torch.arange(schedule.num_steps + 1, device=self.device) / schedule.num_steps * math.pi)

    def _create_state(self):
        """Initializes values used by the get function."""
        betas = torch.cat((torch.tensor([0.0], device=self.device), self.schedule(torch.arange(self.num_steps, device=self.device)))).double()
        alphas = 1 - betas
        state = torch.cumprod(alphas, dim=0)
        state[-1] = 0.0

        return state.to(torch.float32)

    def noise_fn(self, q0, t, word_freq_logits):
        p = self.state[t]
        if word_freq_logits is None:
            word_freq_logits = self.word_freq.repeat(q0.size(0), 1).gather(1, q0.argmax(-1))
            word_freq_logits = word_freq_logits - word_freq_logits.mean(-1, keepdims=True)

        word_freq_probs = word_freq_logits.unsqueeze(-1) * self.word_freq_lambda[t]

        p = torch.clip(p + word_freq_probs, 0., .999)

        non_mask_prob = p * q0

        mask_prob = 1 - non_mask_prob.sum(-1, keepdims=True) + non_mask_prob[
            ..., self.tokenizer.mask_token_id].unsqueeze(-1)

        prob_at_time_t = torch.cat((
            non_mask_prob[..., :self.tokenizer.mask_token_id], mask_prob,
            non_mask_prob[..., self.tokenizer.mask_token_id + 1:]
        ), dim=-1)
        return prob_at_time_t


    def supports_efficient_inference(self):
        return self.use_fast_inference

    def stationary_probs(self, size):
        stationary = torch.zeros(size=size + (self.dim,), device=self.device)
        stationary[..., self.tokenizer.mask_token_id] = 1
        return stationary

    def sample_stationary(self, size):
        return torch.full(size=size, fill_value=self.tokenizer.mask_token_id, device=self.device)

    def custom_product_fn(self, t):
        """Returns product of first n matrices. Only supported for beta constant."""
        dim = self.dim

        if self.schedule.is_constant:
            beta = self.schedule(0)
            one_minus_beta_t_sq = (1 - beta) ** t
            return one_minus_beta_t_sq * torch.eye(dim) + (1 - one_minus_beta_t_sq) * self._get_mask()

        else:
            p = self.state[t]
            return p * torch.eye(dim) + (1 - p) * self._get_mask()

    def _get_mask(self):
        return self.mask

    def get(self, t):
        beta = self.schedule(t)

        return (1 - beta) * torch.eye(self.dim) + beta * self._get_mask()

    def qt_reverse(self,
                   qt_plus_1,
                   t,
                   return_logits=False,
                   make_one_hot=False,
                   epsilon=1e-20
                   ):
        """Get q(x_{t+1} | x_t), the one-step posterior efficiently.
        Args:
          qt_plus_1: an array of floats specifying a distribution over p(x_0).
          t: t in q(x_{t+1} | x_t).
          return_logits: if True, return the output logits
          make_one_hot: if True, will convert q0 to floats if needed.
          epsilon: a small number to normalize logits conversion with, if needed.
        Returns:
          q(x_{t+1} | x_t).
        """
        if make_one_hot:
            assert qt_plus_1.dtype == torch.int64
            qt_plus_1 = torch.nn.functional.one_hot(qt_plus_1, num_classes=self.dim)

        beta = self.schedule(t)
        qtpls1_at_mask = qt_plus_1[Ellipsis, self.tokenizer.mask_token_id: self.tokenizer.mask_token_id + 1]
        non_mask_prob0 = (1 - beta) * qt_plus_1[Ellipsis, :self.tokenizer.mask_token_id] + beta * qtpls1_at_mask
        non_mask_prob1 = (1 - beta) * qt_plus_1[Ellipsis, self.tokenizer.mask_token_id + 1:] + beta * qtpls1_at_mask
        prob_at_time_t = torch.cat((non_mask_prob0, qtpls1_at_mask, non_mask_prob1), dim=-1)

        if return_logits:
            return torch.log(prob_at_time_t + epsilon)
        else:
            return prob_at_time_t

    def get_qt_given_q0(self,
                        q0,
                        t,
                        return_logits=False,
                        make_one_hot=False,
                        epsilon=1e-20,
                        word_freq_logits=None):
        """Get q(x_t), the n-step posterior.
        Can do efficiently for masks.
        For example, for t = 0, it returns q0 unchanged.
        Args:
          q0: an array of floats specifying a distribution over p(x_0).
          t: t in q(x_t | x_0).
          return_logits: if True, return the output logits
          epsilon: a small number to normalize logits conversion with, if needed.
        Returns:
          q(x_t | x_0).
        """
        if not self.supports_efficient_inference():
            return super().get_qt_given_q0(
                q0,
                t,
                return_logits=return_logits,
                epsilon=epsilon)

        if make_one_hot:
            assert q0.dtype == torch.int64
            q0 = torch.nn.functional.one_hot(q0, num_classes=self.dim)

        prob_at_time_t = q0 if t == 0 else self.noise_fn(q0, t, word_freq_logits)

        if return_logits:
            return torch.log(prob_at_time_t + epsilon)
        else:
            return prob_at_time_t

    def supports_efficient_get(self):
        return not self.use_fast_inference


def create_discrete_diffusion_schedule(
        kind="linear",
        beta_min=1e-3,
        beta_max=1e-1,
        num_steps=100,
        scale=1.0,
        s=0.008
):
    """Creates a callable schedule object to use for diffusion rates.
    Args:
    kind: str, one of 'standard', 'linear', 'cosine', 'mutual_information'. If
      standard, performs standard binomial diffusion taken from Sohl-Dicksteein
      et al, ignoring betas. Otherwise, linear schedule between beta_min and
      beta_max.
    beta_min: the minimum beta. Ignored if kind == standard.
    beta_max: the maximum beta.
    num_steps: int, the number of steps to take.
    scale: for standard schedule, rescales num_steps by this amount.
    Returns:
    a DiffusionSchedule object.
    """

    assert beta_min <= beta_max
    assert num_steps > 0
    assert scale >= 1

    if kind == "mutual":
        logger.info(f"using standard schedule with num_steps: {num_steps}.")

        def schedule_fn(step):
            return 1 / (num_steps - step)

        return DiffusionSchedule(schedule_fn, num_steps, is_constant=False)

    elif kind == "linear":
        logger.info(f"using provided beta_min {beta_min} and beta_max {beta_max}.")

        is_constant = beta_min == beta_max

        schedule_fn = utils.create_learning_rate_scheduler(
            "constant * linear_warmup_from",
            warmup_steps=num_steps,
            min_learning_rate=beta_min,
            base_learning_rate=beta_max
        )

        return DiffusionSchedule(schedule_fn, num_steps, is_constant=is_constant)
    elif kind == "cosine":
        logger.info("using cosine schedule inspired by OpenAI I-DDPM paper.")

        def cosine_fn(step):
            return torch.cos((step / num_steps + s) / (1 + s) * np.pi / 2)

        def schedule_fn(step):
            return torch.clip(1 - (cosine_fn(step + 1) / cosine_fn(step)), 0, 0.999)

        return DiffusionSchedule(schedule_fn, num_steps, is_constant=False)
    else:
        raise ValueError(f"kind {kind} is not supported.")
