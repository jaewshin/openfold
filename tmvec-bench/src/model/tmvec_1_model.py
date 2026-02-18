import inspect
from typing import Union

import lightning as L
import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from transformers import PretrainedConfig


class TransformerEncoderModuleConfig(PretrainedConfig):
    def __init__(self,
                 d_model=1024,
                 nhead=4,
                 num_layers=2,
                 dim_feedforward=2048,
                 out_dim=512,
                 dropout=0.2,
                 activation='gelu',
                 lr0=0.0001,
                 warmup_steps=300,
                 **kwargs):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.out_dim = out_dim
        self.dropout = dropout
        self.activation = activation
        self.lr0 = lr0
        self.warmup_steps = warmup_steps
        super().__init__(**kwargs)

    def build(self):
        return TransformerEncoderModule(self)


class TransformerEncoderModule(L.LightningModule, PyTorchModelHubMixin):
    """
    TransformerEncoderLayer with preset parameters followed by global pooling and dropout
    """
    def __init__(self,
                 config: Union[TransformerEncoderModuleConfig, dict],
                 random_seed: int = 42):
        """
        Initialize the TransformerEncoderModule.

        Args:
            config: TransformerEncoderModuleConfig instance or a dictionary
                containing the configuration parameters.

        Examples:
            >>> # load model locally
            >>> config = TransformerEncoderModuleConfig()
            >>> model = TransformerEncoderModule(config)
            >>> # load checkpoint
            >>> state_dict = torch.load('checkpoint.ckpt')['state_dict']
            >>> model.load_state_dict(state_dict)
            >>> # load model from HuggingFace Hub
            >>> model = TransformerEncoderModule.from_pretrained('scikit-bio/tmvec')

        """

        super().__init__()

        # loading parameters from the internet model
        torch.manual_seed(random_seed)

        if isinstance(config, TransformerEncoderModuleConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = TransformerEncoderModuleConfig(**config)
        else:
            raise ValueError("Invalid config type")

        # build encoder
        encoder_args = {
            k: getattr(self.config, k)
            for k in inspect.signature(nn.TransformerEncoderLayer).parameters
            if hasattr(self.config, k)
        }

        num_layers = self.config.num_layers

        encoder_layer = nn.TransformerEncoderLayer(batch_first=True,
                                                   **encoder_args)
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=num_layers)

        self.dropout = nn.Dropout(self.config.dropout)
        self.mlp = nn.Linear(self.config.d_model, self.config.out_dim)

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.l1_loss = nn.L1Loss(reduction='mean')

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor,
                src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the TransformerEncoderModule.

        Args:
            x (torch.Tensor): Input tensor
            src_mask (torch.Tensor): Source mask tensor
            src_key_padding_mask (torch.Tensor): Source key padding mask tensor

        Returns:
            torch.Tensor: Output tensor
        """

        x = self.encoder(x,
                         mask=src_mask,
                         src_key_padding_mask=src_key_padding_mask)
        lens = torch.logical_not(src_key_padding_mask).sum(dim=1)
        x = x.sum(dim=1) / lens.unsqueeze(1)
        x = self.dropout(x)
        x = self.mlp(x)
        return x

    def distance_loss_euclidean(self, output_seq1: torch.Tensor,
                                output_seq2: torch.Tensor,
                                tm_score: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Normalized Euclidean distance loss.

        Args:
            output_seq1 (torch.Tensor): Output sequence 1 tensor
            output_seq2 (torch.Tensor): Output sequence 2 tensor
            tm_score (torch.Tensor): TM score tensor

        Returns:
            torch.Tensor: Distance loss tensor
        """

        pdist_seq = nn.PairwiseDistance(p=2)
        dist_seq = pdist_seq(output_seq1, output_seq2)
        dist_tm = torch.cdist(dist_seq.float().unsqueeze(0),
                              tm_score.float().unsqueeze(0),
                              p=2)
        return dist_tm

    def distance_loss_sigmoid(self, output_seq1: torch.Tensor,
                              output_seq2: torch.Tensor,
                              tm_score: torch.Tensor) -> torch.Tensor:
        """
        Calculate the sigmoid distance loss.

        Args:
            output_seq1 (torch.Tensor): Output sequence 1 tensor
            output_seq2 (torch.Tensor): Output sequence 2 tensor
            tm_score (torch.Tensor): TM score tensor

        Returns:
            torch.Tensor: Distance loss tensor
        """

        dist_seq = output_seq1 - output_seq2
        dist_seq = torch.sigmoid(dist_seq).mean(1)
        dist_tm = torch.cdist(dist_seq.float().unsqueeze(0),
                              tm_score.float().unsqueeze(0),
                              p=2)
        return dist_tm

    def distance_loss(self, output_seq1: torch.Tensor,
                      output_seq2: torch.Tensor,
                      tm_score: torch.Tensor) -> torch.Tensor:
        """
        Calculate the cosine similarity distance loss.
        Args:
            output_seq1 (torch.Tensor): Output sequence 1 tensor
            output_seq2 (torch.Tensor): Output sequence 2 tensor
            tm_score (torch.Tensor): TM score tensor

        Returns:
            torch.Tensor: Distance loss tensor
        """

        dist_seq = self.cos(output_seq1, output_seq2)
        dist_tm = self.l1_loss(dist_seq.unsqueeze(0),
                               tm_score.float().unsqueeze(0))
        return dist_tm

    def training_step(self, train_batch: tuple,
                      batch_idx: int) -> torch.Tensor:
        """
        Training step.

        Args:
            train_batch (tuple): Training batch
            batch_idx (int): Batch index

        Returns:
            torch.Tensor: Loss tensor
        """

        sequence_1, sequence_2, pad_mask_1, pad_mask_2, tm_score = train_batch
        out_seq1 = self.forward(sequence_1,
                                src_mask=None,
                                src_key_padding_mask=pad_mask_1)
        out_seq2 = self.forward(sequence_2,
                                src_mask=None,
                                src_key_padding_mask=pad_mask_2)
        loss = self.distance_loss(out_seq1, out_seq2, tm_score)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, val_batch: tuple, batch_idx: int) -> None:
        """
        Validation step.

        Args:
            val_batch (tuple): Validation batch
            batch_idx (int): Batch index
        """

        sequence_1, sequence_2, pad_mask_1, pad_mask_2, tm_score = val_batch
        out_seq1 = self.forward(sequence_1,
                                src_mask=None,
                                src_key_padding_mask=pad_mask_1)
        out_seq2 = self.forward(sequence_2,
                                src_mask=None,
                                src_key_padding_mask=pad_mask_2)
        loss = self.distance_loss(out_seq1, out_seq2, tm_score)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)

    def configure_optimizers(self) -> tuple:
        """
        Configure the optimizer and scheduler.

        Returns:
            list: List of optimizers
            list: List of schedulers
        """
        optimizer = torch.optim.AdamW(self.parameters(),
                                      betas=(0.99, 0.98),
                                      lr=self.config.lr0)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                  T_max=10)
        return [optimizer], [lr_scheduler]