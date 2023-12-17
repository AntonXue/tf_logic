from typing import Optional, Union
import torch
import transformers
from transformers import AutoConfig, GPT2Config, AutoModelForSequenceClassification, \
        GPT2ForSequenceClassification, BertForSequenceClassification, \
        RobertaForSequenceClassification, LlamaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from .common import *

transformers.logging.set_verbosity_error()  # Be quiet!!

""" Hugging Face models for sequence classification """

class HFSeqClsConfig:
    def __init__(
        self,
        model_name: str,
        embed_dim: Optional[int] = None,
        ffwd_dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        num_layers: Optional[int] = None,
        num_labels: Optional[int] = None,
        problem_type: Optional[str] = "multi_label_classification",
        max_seq_len: Optional[int] = None,
        overwriting_config_kwargs: Optional[dict] = None,
        use_pretrained: Optional[bool] = None,
        pretrained_kwargs: Optional[dict] = None,
    ):
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_labels = num_labels
        self.problem_type = problem_type
        self.max_seq_len = max_seq_len
        self.overwriting_config_kwargs = overwriting_config_kwargs
        self.use_pretrained = use_pretrained

        # transformers.GPT2Config
        if model_name == "gpt2":
            config_kwargs = {
                "n_embd": embed_dim,
                "n_inner": ffwd_dim,
                "n_head": num_heads,
                "n_layer": num_layers,
                "num_labels": num_labels,
                "problem_type": problem_type,
                "n_positions": max_seq_len,
                "pad_token_id": GPT2Config().eos_token_id
            }

            self.pretrained_model_name = "gpt2"

        # transformers.BertConfig
        elif model_name == "bert":
            config_kwargs = {
                "hidden_size": embed_dim,
                "intermediate_size": ffwd_dim,
                "num_attention_heads": num_heads,
                "num_hidden_layers": num_layers,
                "num_labels": num_labels,
                "problem_type": problem_type,
                "max_position_embeddings": max_seq_len,
            }

            self.pretrained_model_name = "bert-base-uncased"

        # transformers.RobertaConfig
        elif model_name == "roberta":
            config_kwargs = {
                "hidden_size": embed_dim,
                "intermediate_size": ffwd_dim,
                "num_hidden_layers": num_layers,
                "num_attention_heads": num_heads,
                "num_labels" : num_labels,
                "problem_type": problem_type,
                "max_position_embeddings": max_seq_len
            }
            self.pretrained_model_name = "roberta-base"

        # transformers.LlamaConfig
        elif model_name == "code_llama":
            config_kwargs = {
                "hidden_size": embed_dim,
                "intermediate_size": ffwd_dim,
                "num_hidden_layers": num_layers,
                "num_attention_heads": num_heads,
                "num_labels" : num_labels,
                "problem_type": problem_type,
                "max_position_embeddings": max_seq_len
            }
            self.pretrained_model_name = "code_llama"

        else:
            raise ValueError(f"Unsupported model {model_name}")

        # The RHS of the OR (|) overwrites the LHS
        kwargs = config_kwargs | default(overwriting_config_kwargs, {})

        # Delete entries where the keys is None (i.e., were not specified)
        for k in list(kwargs.keys()):
            if kwargs[k] is None:
                del kwargs[k]

        self.model_config_kwargs = kwargs
        self.pretrained_kwargs = kwargs # These two are identical at the moment


class HFSeqClsModel(nn.Module):
    """ Simple wrapper around HF models for sequence classification """
    def __init__(self, config: HFSeqClsConfig):
        super().__init__()
        self.config = config

        if config.use_pretrained:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                config.pretrained_model_name,
                ignore_mismatched_sizes = True, # TODO: ignore_mismatched_sizes seems to fail for bert
                **config.pretrained_kwargs)
            self.model_config = self.model.config
        else:
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

    @property
    def problem_type(self):
        return self.config.problem_type

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
        attention_mask: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        """ use_inputs_ids == True:  x: (batch_size, seq_len)
            use_inputs_ids == False: x: (batch_size, seq_len, embed_dim)
        """
        # When using input ids, the result is simpler
        if use_input_ids:
            return self.model(input_ids=x, labels=labels, attention_mask=attention_mask, **kwargs)

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


