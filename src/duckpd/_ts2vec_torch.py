"""Pinned TS2Vec encoder, pooling, and training primitives.

Derived from TS2Vec revision b0088e14a99706c05451316dc6db8d3da9351163.

MIT License

Copyright (c) 2022 Zhihan Yue

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor, nn


class SamePadConv(nn.Module):
    """One-dimensional convolution preserving the temporal length."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        receptive_field = (kernel_size - 1) * dilation + 1
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=receptive_field // 2,
            dilation=dilation,
        )
        self.remove = 1 if receptive_field % 2 == 0 else 0

    def forward(self, values: Tensor) -> Tensor:
        output = self.conv(values)
        return output[:, :, : -self.remove] if self.remove else output


class ConvBlock(nn.Module):
    """TS2Vec residual block with two same-padded dilated convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        *,
        final: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = SamePadConv(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )
        self.conv2 = SamePadConv(
            out_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )
        self.projector = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels or final
            else None
        )

    def forward(self, values: Tensor) -> Tensor:
        residual = values if self.projector is None else self.projector(values)
        values = self.conv1(functional.gelu(values))
        values = self.conv2(functional.gelu(values))
        return values + residual


class DilatedConvEncoder(nn.Module):
    """Exponentially dilated residual convolution stack."""

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_size: int,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            *(
                ConvBlock(
                    channels[index - 1] if index else in_channels,
                    out_channels,
                    kernel_size,
                    dilation=2**index,
                    final=index == len(channels) - 1,
                )
                for index, out_channels in enumerate(channels)
            )
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.net(values)


class TSEncoder(nn.Module):
    """TS2Vec timestamp encoder with joint channel projection."""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        hidden_dims: int,
        depth: int,
        *,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size,
        )
        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, values: Tensor, mask: str | None = None) -> Tensor:
        finite_rows = ~values.isnan().any(dim=-1)
        values = torch.where(finite_rows.unsqueeze(-1), values, 0.0)
        values = self.input_fc(values)
        selected_mask = mask or ("binomial" if self.training else "all_true")
        if selected_mask == "binomial":
            active = (
                torch.rand(
                    values.shape[:2],
                    device=values.device,
                )
                < 0.5
            )
        elif selected_mask == "all_true":
            active = torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
        else:
            raise ValueError(f"unsupported TS2Vec mask mode: {selected_mask!r}")
        active &= finite_rows
        values = torch.where(active.unsqueeze(-1), values, 0.0)
        values = values.transpose(1, 2)
        values = self.repr_dropout(self.feature_extractor(values))
        return values.transpose(1, 2)


def build_encoder(architecture: Mapping[str, object]) -> TSEncoder:
    """Construct the exact encoder declared by a verified bundle."""
    return TSEncoder(
        input_dims=cast("int", architecture["input_dims"]),
        output_dims=cast("int", architecture["output_dims"]),
        hidden_dims=cast("int", architecture["hidden_dims"]),
        depth=cast("int", architecture["depth"]),
        kernel_size=cast("int", architecture["kernel_size"]),
    )


def full_series_encode(model: TSEncoder, values: Tensor) -> Tensor:
    """Apply deterministic all-observed inference and full-series max pooling."""
    return torch.amax(model(values, mask="all_true"), dim=1)


def hierarchical_contrastive_loss(
    first: Tensor,
    second: Tensor,
    *,
    alpha: float = 0.5,
    temporal_unit: int = 0,
) -> Tensor:
    """Reproduce TS2Vec's hierarchical instance and temporal objective."""
    loss = first.new_tensor(0.0)
    depth = 0
    while first.size(1) > 1:
        if alpha:
            loss = loss + alpha * _instance_contrastive_loss(first, second)
        if depth >= temporal_unit and alpha != 1:
            loss = loss + (1 - alpha) * _temporal_contrastive_loss(first, second)
        depth += 1
        first = functional.max_pool1d(first.transpose(1, 2), 2).transpose(1, 2)
        second = functional.max_pool1d(second.transpose(1, 2), 2).transpose(1, 2)
    if first.size(1) == 1 and alpha:
        loss = loss + alpha * _instance_contrastive_loss(first, second)
        depth += 1
    return loss / depth


def _instance_contrastive_loss(first: Tensor, second: Tensor) -> Tensor:
    batch_size = first.size(0)
    if batch_size == 1:
        return first.new_tensor(0.0)
    combined = torch.cat((first, second), dim=0).transpose(0, 1)
    similarities = torch.matmul(combined, combined.transpose(1, 2))
    logits = torch.tril(similarities, diagonal=-1)[:, :, :-1]
    logits = logits + torch.triu(similarities, diagonal=1)[:, :, 1:]
    logits = -functional.log_softmax(logits, dim=-1)
    indexes = torch.arange(batch_size, device=first.device)
    return (
        logits[:, indexes, batch_size + indexes - 1].mean()
        + logits[:, batch_size + indexes, indexes].mean()
    ) / 2


def _temporal_contrastive_loss(first: Tensor, second: Tensor) -> Tensor:
    timestamps = first.size(1)
    if timestamps == 1:
        return first.new_tensor(0.0)
    combined = torch.cat((first, second), dim=1)
    similarities = torch.matmul(combined, combined.transpose(1, 2))
    logits = torch.tril(similarities, diagonal=-1)[:, :, :-1]
    logits = logits + torch.triu(similarities, diagonal=1)[:, :, 1:]
    logits = -functional.log_softmax(logits, dim=-1)
    indexes = torch.arange(timestamps, device=first.device)
    return (
        logits[:, indexes, timestamps + indexes - 1].mean()
        + logits[:, timestamps + indexes, indexes].mean()
    ) / 2


def _take_per_row(
    values: Tensor, starts: np.ndarray[Any, np.dtype[np.int64]], length: int
) -> Tensor:
    rows = torch.arange(values.size(0), device=values.device)[:, None]
    offsets = torch.as_tensor(starts, dtype=torch.long, device=values.device)[:, None]
    columns = offsets + torch.arange(length, device=values.device)[None, :]
    return values[rows, columns]


def contrastive_train_step(
    model: TSEncoder,
    averaged_model: Any,
    optimizer: torch.optim.Optimizer,
    values: Tensor,
    random: np.random.Generator,
    *,
    temporal_unit: int,
) -> float:
    """Run one bounded reproduction of the upstream random-crop training step."""
    timestamp_count = values.size(1)
    crop_length = int(random.integers(2 ** (temporal_unit + 1), timestamp_count + 1))
    crop_left = int(random.integers(timestamp_count - crop_length + 1))
    crop_right = crop_left + crop_length
    extended_left = int(random.integers(crop_left + 1))
    extended_right = int(random.integers(crop_right, timestamp_count + 1))
    crop_offset = random.integers(
        -extended_left,
        timestamp_count - extended_right + 1,
        size=values.size(0),
        dtype=np.int64,
    )

    optimizer.zero_grad(set_to_none=True)
    first = model(
        _take_per_row(
            values,
            crop_offset + extended_left,
            crop_right - extended_left,
        )
    )[:, -crop_length:]
    second = model(
        _take_per_row(
            values,
            crop_offset + crop_left,
            extended_right - crop_left,
        )
    )[:, :crop_length]
    loss = hierarchical_contrastive_loss(
        first,
        second,
        temporal_unit=temporal_unit,
    )
    cast("Any", loss).backward()
    optimizer.step()
    averaged_model.update_parameters(model)
    return float(loss.detach().cpu().item())
