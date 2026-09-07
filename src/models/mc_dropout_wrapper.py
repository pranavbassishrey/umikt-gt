"""
Monte Carlo Dropout uncertainty estimation wrapper.

Why MC-Dropout over alternatives:
  - Deep Ensembles: require training N independent models; computationally
    prohibitive for the hardware budget and dataset size of this project.
  - Evidential Deep Learning: changes the loss function significantly and
    requires additional target formulation; higher implementation complexity
    for a marginal improvement at this scale.
  - Bayesian neural networks (variational): requires significant architectural
    changes (reparameterization trick, variational objectives).
  - MC-Dropout (Gal & Ghahramani 2016): uses the same model and training procedure,
    adds only N forward passes at inference. Reasonable uncertainty estimates at
    minimal cost. The main limitation is that it approximates the posterior
    (not the exact Bayesian posterior). This is documented in ARCHITECTURE.md.

Usage:
  1. Enable dropout during inference: model.train()  # or wrapper.enable_mc()
  2. Run N forward passes and collect predictions.
  3. Compute predictive mean and variance.
  4. Variance = epistemic + aleatoric uncertainty (not separated here).

Reference:
  Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation:
  Representing Model Uncertainty in Deep Learning. ICML 2016.
"""

import torch
from torch import Tensor


class MCDropoutWrapper:
    """
    Monte Carlo Dropout uncertainty estimator.

    Wraps any PyTorch model that uses nn.Dropout layers and runs N forward
    passes with dropout active to estimate predictive uncertainty.

    Args:
        model: The PyTorch model to wrap. Must have nn.Dropout layers.
        n_samples: Number of MC forward passes.
        device: Device to run inference on.
    """

    def __init__(self, model: torch.nn.Module, n_samples: int = 20, device: str = "cpu"):
        self.model = model
        self.n_samples = n_samples
        self.device = device

    def enable_mc_dropout(self):
        """
        Set model to training mode to enable dropout.

        Note: This also enables BatchNorm training statistics update if the model
        uses BatchNorm. Our model uses LayerNorm, which is unaffected.
        """
        self.model.train()

    def disable_mc_dropout(self):
        """Set model to eval mode (standard deterministic inference)."""
        self.model.eval()

    @torch.no_grad()
    def sample_predictions(
        self,
        input_fn,
        **kwargs,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Run N forward passes and return mean, variance, and all samples.

        Args:
            input_fn: Callable that takes **kwargs and returns a scalar or
                      vector prediction tensor. Typically a partial forward pass
                      that returns just the mastery head output.
            **kwargs: Arguments passed to input_fn.

        Returns:
            Tuple of:
              - mean: Predictive mean. Same shape as a single forward pass output.
              - variance: Predictive variance (uncertainty estimate).
              - samples: All N samples stacked. Shape: (N, *output_shape)
        """
        self.enable_mc_dropout()

        samples = []
        for _ in range(self.n_samples):
            pred = input_fn(**kwargs)
            samples.append(pred.detach().cpu())

        self.disable_mc_dropout()

        samples_t = torch.stack(samples, dim=0)  # (N, *output_shape)
        mean = samples_t.mean(dim=0)
        variance = samples_t.var(dim=0)

        return mean, variance, samples_t

    def get_uncertainty(
        self,
        input_fn,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """
        Convenience method returning just mean and variance.

        Returns:
            (mean, variance) tensors.
        """
        mean, variance, _ = self.sample_predictions(input_fn, **kwargs)
        return mean, variance
