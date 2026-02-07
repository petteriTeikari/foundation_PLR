import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for sequence models.

    Generates fixed positional encodings using sine and cosine functions
    at different frequencies, as described in "Attention Is All You Need".

    Parameters
    ----------
    d_model : int
        Dimension of the model (embedding dimension).
    max_len : int, optional
        Maximum sequence length to support, by default 5000.
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retrieve positional embeddings for the input sequence length.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, ...).

        Returns
        -------
        torch.Tensor
            Positional embeddings of shape (1, seq_len, d_model).
        """
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    """Token embedding using 1D convolution.

    Projects input features to model dimension using a 1D convolution
    with circular padding and kernel size 3.

    Parameters
    ----------
    c_in : int
        Number of input channels (features per timestep).
    d_model : int
        Dimension of the model (output embedding dimension).
    """

    def __init__(self, c_in: int, d_model: int) -> None:
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply token embedding via 1D convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, c_in).

        Returns
        -------
        torch.Tensor
            Embedded tensor of shape (batch_size, seq_len, d_model).
        """
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    """Fixed sinusoidal embedding using nn.Embedding with frozen weights.

    Creates a lookup table with sinusoidal embeddings that are not trainable,
    suitable for temporal feature encoding.

    Parameters
    ----------
    c_in : int
        Number of distinct input values (vocabulary size).
    d_model : int
        Dimension of the embedding vectors.
    """

    def __init__(self, c_in: int, d_model: int) -> None:
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Look up fixed embeddings for input indices.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of integer indices.

        Returns
        -------
        torch.Tensor
            Embedded tensor with gradients detached.
        """
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """Temporal embedding for datetime features.

    Embeds temporal features (minute, hour, weekday, day, month) separately
    and combines them additively.

    Parameters
    ----------
    d_model : int
        Dimension of the embedding vectors.
    embed_type : str, optional
        Type of embedding: "fixed" for sinusoidal or learnable otherwise,
        by default "fixed".
    freq : str, optional
        Frequency of the time series: "t" for minutely (includes minute embed),
        "h" for hourly, etc., by default "h".
    """

    def __init__(
        self, d_model: int, embed_type: str = "fixed", freq: str = "h"
    ) -> None:
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed temporal features and sum them.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, num_features) where
            features are ordered as [month, day, weekday, hour, minute].

        Returns
        -------
        torch.Tensor
            Combined temporal embedding of shape (batch_size, seq_len, d_model).
        """
        x = x.long()
        minute_x: Union[torch.Tensor, float] = (
            self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        )
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    """Time feature embedding using a linear projection.

    Projects continuous time features to model dimension using a linear layer.

    Parameters
    ----------
    d_model : int
        Dimension of the output embedding.
    embed_type : str, optional
        Type identifier (unused, for API compatibility), by default "timeF".
    freq : str, optional
        Frequency of the time series, determines input dimension:
        "h" (hourly) -> 4, "t" (minutely) -> 5, "s" (secondly) -> 6,
        "m" (monthly) -> 1, "a" (annually) -> 1, "w" (weekly) -> 2,
        "d" (daily) -> 3, "b" (business daily) -> 3, by default "h".
    """

    def __init__(
        self, d_model: int, embed_type: str = "timeF", freq: str = "h"
    ) -> None:
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project time features to embedding dimension.

        Parameters
        ----------
        x : torch.Tensor
            Input time features of shape (batch_size, seq_len, num_features).

        Returns
        -------
        torch.Tensor
            Embedded tensor of shape (batch_size, seq_len, d_model).
        """
        return self.embed(x)


class DataEmbedding(nn.Module):
    """Combined data embedding with value, positional, and temporal components.

    Combines token embedding, positional embedding, and temporal embedding
    with dropout for time series data.

    Parameters
    ----------
    c_in : int
        Number of input channels (features per timestep).
    d_model : int
        Dimension of the model (embedding dimension).
    embed_type : str, optional
        Type of temporal embedding: "fixed" for sinusoidal, "timeF" for
        linear projection, by default "fixed".
    freq : str, optional
        Frequency of the time series, by default "h".
    dropout : float, optional
        Dropout probability, by default 0.1.
    """

    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed_type: str = "fixed",
        freq: str = "h",
        dropout: float = 0.1,
    ) -> None:
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, x_mark: Optional[torch.Tensor]) -> torch.Tensor:
        """Embed input data with positional and optional temporal encoding.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, c_in).
        x_mark : torch.Tensor or None
            Optional temporal features of shape (batch_size, seq_len, num_time_features).
            If None, only value and positional embeddings are used.

        Returns
        -------
        torch.Tensor
            Embedded tensor of shape (batch_size, seq_len, d_model) with dropout applied.
        """
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = (
                self.value_embedding(x)
                + self.temporal_embedding(x_mark)
                + self.position_embedding(x)
            )
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    """Inverted data embedding that treats time as features and variates as tokens.

    Projects the time dimension to model dimension, inverting the typical
    treatment of sequences. Useful for channel-independent modeling.

    Parameters
    ----------
    c_in : int
        Number of input time steps (sequence length).
    d_model : int
        Dimension of the model (embedding dimension).
    embed_type : str, optional
        Unused, kept for API compatibility, by default "fixed".
    freq : str, optional
        Unused, kept for API compatibility, by default "h".
    dropout : float, optional
        Dropout probability, by default 0.1.
    """

    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed_type: str = "fixed",
        freq: str = "h",
        dropout: float = 0.1,
    ) -> None:
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, x_mark: Optional[torch.Tensor]) -> torch.Tensor:
        """Embed input with inverted dimensions (variates as sequence).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, n_variates).
        x_mark : torch.Tensor or None
            Optional temporal features of shape (batch_size, seq_len, num_time_features).
            If provided, concatenated with x along the variate dimension.

        Returns
        -------
        torch.Tensor
            Embedded tensor of shape (batch_size, n_variates, d_model) with dropout applied.
        """
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    """Data embedding without positional encoding in forward pass.

    Similar to DataEmbedding but only uses value and temporal embeddings,
    omitting positional encoding during forward. Position embedding is
    still created but not used.

    Parameters
    ----------
    c_in : int
        Number of input channels (features per timestep).
    d_model : int
        Dimension of the model (embedding dimension).
    embed_type : str, optional
        Type of temporal embedding: "fixed" for sinusoidal, "timeF" for
        linear projection, by default "fixed".
    freq : str, optional
        Frequency of the time series, by default "h".
    dropout : float, optional
        Dropout probability, by default 0.1.
    """

    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed_type: str = "fixed",
        freq: str = "h",
        dropout: float = 0.1,
    ) -> None:
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, x_mark: Optional[torch.Tensor]) -> torch.Tensor:
        """Embed input without positional encoding.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, c_in).
        x_mark : torch.Tensor or None
            Optional temporal features of shape (batch_size, seq_len, num_time_features).
            If None, only value embedding is used.

        Returns
        -------
        torch.Tensor
            Embedded tensor of shape (batch_size, seq_len, d_model) with dropout applied.
        """
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    """Patch embedding for time series using sliding windows.

    Segments the input sequence into patches using a sliding window,
    then projects each patch to the model dimension with positional encoding.

    Parameters
    ----------
    d_model : int
        Dimension of the model (embedding dimension).
    patch_len : int
        Length of each patch (window size).
    stride : int
        Stride between consecutive patches.
    padding : int
        Amount of padding to add at the end of the sequence.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self, d_model: int, patch_len: int, stride: int, padding: int, dropout: float
    ) -> None:
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Create patch embeddings from input sequence.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_vars, seq_len).

        Returns
        -------
        Tuple[torch.Tensor, int]
            - Embedded patches of shape (batch_size * n_vars, n_patches, d_model)
              with dropout applied.
            - Number of variables (n_vars) for later reshaping.
        """
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars
