from typing import Optional, Union
from dataclasses import dataclass
import torch
import transformers
from transformers import AutoConfig, GPT2Config, AutoModelForSequenceClassification, \
        GPT2ForSequenceClassification, BertForSequenceClassification, \
        RobertaForSequenceClassification, LlamaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from .common import *

transformers.logging.set_verbosity_error()  # Be quiet!!

""" Hugging Face models for sequence classification """

@dataclass
class HFSeqClsConfig:
    model_name: str
    embed_dim: Optional[int] = None
    num_heads: Optional[int] = None
    num_layers: Optional[int] = None
    num_labels: Optional[int] = None
    problem_type: Optional[str] = "multi_label_classification"
    max_seq_len: Optional[int] = None
    overwriting_config_kwargs: Optional[dict] = None

    def __post_init__(self):
        if self.problem_type == "regression":
            assert self.num_labels == 1
        elif self.problem_type in ["single_label_classification", "multi_label_classification"]:
            assert self.num_labels > 1
        else:
            raise ValueError(f"Unrecognized problem type {self.problem_type}")

        if self.model_name == "gpt2":
            config_kwargs = {
                "n_embd": self.embed_dim,
                "n_head" : self.num_heads,
                "n_layer" : self.num_layers,
                "num_labels" : self.num_labels,
                "problem_type" : self.problem_type,
                "n_positions" : self.max_seq_len,
                "pad_token_id" : GPT2Config().eos_token_id
            }

        elif self.model_name == "bert":
            config_kwargs = {
                "hidden_size": self.embed_dim,
                "num_attention_heads": self.num_heads,
                "num_hidden_layers" : self.num_layers,
                "num_labels" : self.num_labels,
                "problem_type" : self.problem_type,
                "max_position_embeddings" : self.max_seq_len,
            }

        elif self.model_name == "roberta":
            config_kwargs = {
                "hidden_size": self.embed_dim,
                "num_hidden_layers": self.num_layers,
                "num_attention_heads": self.num_heads,
                "num_labels" : self.num_labels,
                "problem_type": self.problem_type,
                "max_position_embeddings": self.max_seq_len
            }

        elif self.model_name == "code_llama":
            config_kwargs = {
                "hidden_size": self.embed_dim,
                "num_hidden_layers": self.num_layers,
                "num_attention_heads": self.num_heads,
                "num_labels" : self.num_labels,
                "problem_type": self.problem_type,
                "max_position_embeddings": self.max_seq_len
            }

        else:
            raise ValueError(f"Unsupported model {self.model_name}")

        # The RHS of the OR (|) overwrites the LHS
        kwargs = config_kwargs | default(self.overwriting_config_kwargs, {})

        # Delete entries where the keys is None (i.e., were not specified)
        for k in list(kwargs.keys()):
            if kwargs[k] is None:
                del kwargs[k]

        self.__setattr__("model_config_kwargs", kwargs)


class HFSeqClsModel(nn.Module):
    """ Simple wrapper around HF models for sequence classification """
    def __init__(self, config: HFSeqClsConfig):
        super().__init__()
        self.config = config
        self.model_config = AutoConfig.for_model(config.model_name, **config.model_config_kwargs)
        self.model = AutoModelForSequenceClassification.from_config(self.model_config)

    @property
    def model_name(self):
        return self.config.model_name

    @property
    def embed_dim(self):
        if isinstance(self.model, GPT2ForSequenceClassification):
            return self.model.transformer.embed_dim
        elif isinstance(self.model, BertForSequenceClassification):
            return self.model.bert.embeddings.word_embeddings.embedding_dim
        elif isinstance(self.model, RobertaForSequenceClassification):
            return self.model.roberta.embeddings.word_embeddings.embedding_dim
        elif isinstance(self.model, LlamaForSequenceClassification):
            return self.model.model.embed_tokens.embedding_dim
        else:
            raise ValueError()

    @property
    def num_labels(self):
        return self.model.num_labels

    def forward_iter_batch(
            self,
            x: torch.FloatTensor,
            labels: Optional[torch.LongTensor],
            **kwargs):
        """ Some models (e.g., GPT-2) can't handle batch_size > 1 when using inputs_embeds """
        all_outs = []
        for i, xi in enumerate(x):
            li = None if labels is None else labels[i:i+1]
            all_outs.append(self.model(inputs_embeds=xi[None,...], labels=li, **kwargs))

        loss = None
        if labels is not None:
            loss = torch.stack([out.loss for out in all_outs]).mean(dim=0)

        logits = torch.cat([out.logits for out in all_outs], dim=0)

        hidden_states = None
        if "output_hidden_states" in kwargs and all_outs[0].hidden_states is not None:
            num_hidden_states = len(all_outs[0].hidden_states)
            hidden_states = ()
            for k in range(num_hidden_states):
                hidden_states += (torch.cat([out.hidden_states[k] for out in all_outs], dim=0),)
            
        attentions = None
        if "output_attentions" in kwargs and all_outs[0].attentions is not None:
            num_attentions = len(all_outs[0].attentions)
            attentions = ()
            for k in range(num_attentions):
                attentions += (torch.cat([out.attentions[k] for out in all_outs], dim=0),)

        return SequenceClassifierOutput(
            loss = loss,
            logits = logits,
            hidden_states = hidden_states,
            attentions = attentions)

    def forward(
        self,
        x: Union[torch.FloatTensor, torch.LongTensor],
        labels: Optional[torch.LongTensor] = None,
        use_input_ids: Optional[bool] = None,
        **kwargs
    ):
        """ use_inputs_ids == True:  x: (batch_size, seq_len)
            use_inputs_ids == False: x: (batch_size, seq_len, embed_dim)
        """
        # When using input ids, the result is simpler
        if use_input_ids:
            return self.model(input_ids=x, labels=labels, **kwargs)

        # Otherwise we assme inputs_embeds
        if self.config.problem_type == "multi_label_classification" and labels is not None:
            labels = labels.float()

        # GPT-2 can only handle batch size == 1 when using inputs_embeds
        if isinstance(self.model, GPT2ForSequenceClassification) or \
            isinstance(self.model, LlamaForSequenceClassification):
            out = self.forward_iter_batch(x, labels, **kwargs)

        else:
            out = self.model(inputs_embeds=x, labels=labels, **kwargs)

        return out


