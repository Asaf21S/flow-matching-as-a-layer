import torch
from torch import Tensor, nn


class ProbeBank:
    """A stack of frozen linear probes addressed per sample by a fold id.

    With one probe the bank behaves exactly like that probe. With several it lets a
    sample be scored by a probe that never saw it, which is what gives the rolled
    classification objectives a loss that is not already saturated.
    """

    def __init__(self, probes: list[nn.Linear]):
        self.weight = torch.stack([probe.weight.detach() for probe in probes])
        self.bias = torch.stack([probe.bias.detach() for probe in probes])

    @property
    def num_folds(self) -> int:
        """Number of probes held in the bank."""
        return self.weight.shape[0]

    def logits(self, features: Tensor, folds: Tensor | None = None) -> Tensor:
        """Score features with each sample's own probe.

        Args:
            features: Batch of features, shape ``(batch, dim)``.
            folds: Per-sample probe index; ignored when the bank holds one probe.

        Returns:
            Logits of shape ``(batch, num_classes)``.
        """
        if self.num_folds == 1 or folds is None:
            return features @ self.weight[0].T + self.bias[0]
        # (folds, batch, classes) is tiny for the fold counts we use.
        stacked = torch.einsum("bd,fcd->fbc", features, self.weight) + self.bias[:, None, :]
        return stacked[folds, torch.arange(features.shape[0], device=features.device)]

    def rows(self, class_ids: Tensor, folds: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Gather the weight row and bias of one class per sample.

        Args:
            class_ids: Class index per sample, shape ``(batch,)``.
            folds: Per-sample probe index; ignored when the bank holds one probe.

        Returns:
            Weight rows of shape ``(batch, dim)`` and biases of shape ``(batch,)``.
        """
        if self.num_folds == 1 or folds is None:
            return self.weight[0][class_ids], self.bias[0][class_ids]
        return self.weight[folds, class_ids], self.bias[folds, class_ids]
