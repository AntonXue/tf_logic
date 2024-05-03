from dataclasses import dataclass
from typing import Optional
import torch
from transformers import GPT2Config, GPT2ForSequenceClassification
from transformers.utils import ModelOutput
from ..common import *


@dataclass
class MyGPT2Config:
    input_dim: int
    num_vars: int
    embed_dim: int
    ffwd_dim: Optional[int] = None
    num_heads: int = 1
    num_layers: int = 1
    problem_type: str = "multi_label_classification"
    use_positional_embedding: bool = False
    max_seq_len: int = 1024 # We need this even if we don't have positional encoding
    use_pretrained: bool = False # This overrides embed_dim, num_layers, num_heads, and ffwd_dim
    pad_token_id: int = -1 # Maybe change to something smarter


@dataclass
class MyGPT2Output(ModelOutput):
    loss: Optional[float] = None,
    logits: Optional[float] = None
    hidden_states: Optional[float] = None
    attentions: Optional[float] = None


class MyGPT2SeqClsModel(SeqClsModel):
    def __init__(self, config: MyGPT2Config):
        super().__init__()
        self.config = config

        if config.use_pretrained:
            self.gpt2s = GPT2ForSequenceClassification.from_pretrained(
                "gpt2",
                num_labels = config.num_vars,
                problem_type = config.problem_type,
                pad_token_id = config.pad_token_id,
            )

        else:
            self.gpt2s = GPT2ForSequenceClassification._from_config(GPT2Config(
                n_embd = config.embed_dim,
                n_inner = config.ffwd_dim,
                n_head = config.num_heads,
                n_layer = config.num_layers,
                num_labels = config.num_vars,
                n_positions = config.max_seq_len,
                problem_type = config.problem_type,
                pad_token_id = config.pad_token_id
            ))

            # Disable positional embedding
            if not self.config.use_positional_embedding:
                self.gpt2s.transformer.wpe.requires_grad_(False)
                self.gpt2s.transformer.wpe.weight.fill_(0)

        self.model_config = self.gpt2s.config

        if config.input_dim is not None:
            self.embed_fn = nn.Linear(config.input_dim, config.embed_dim)

    @property
    def model_name(self):
        return "gpt2"

    @property
    def input_dim(self):
        return self.config.input_dim

    @property
    def embed_dim(self):
        return self.config.embed_dim

    @property
    def num_labels(self):
        return self.config.num_vars

    @property
    def propblem_type(self):
        return self.config.problem_type

    def forward(
        self,
        x: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        """
            x: (batch_size, seq_len, input_dim)
            input_ids: (batch_size, seq_len)
        """
        assert not (x is None and input_ids is None)
        assert not (x is not None and input_ids is not None)

        if self.config.problem_type == "multi_label_classification" and labels is not None:
            labels = labels.float()

        if input_ids is None:
            x = self.embed_fn(x)
            out = self.gpt2s(inputs_embeds=x, attention_mask=attention_mask, labels=labels, **kwargs)
        else:
            out = self.gpt2s(input_ids=x, attention_mask=attention_mask, labels=labels, **kwargs)

        return MyGPT2Output(
            loss = out.loss,
            logits = out.logits,
            hidden_states = out.hidden_states,
            attentions = out.attentions
        )


