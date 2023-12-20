from typing import Optional, Union
import torch
import transformers
from transformers import AutoConfig, GPT2Config, AutoModelForSequenceClassification, \
        GPT2ForSequenceClassification, BertForSequenceClassification, \
        RobertaForSequenceClassification, LlamaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

from ..common import *

transformers.logging.set_verbosity_error()  # Be quiet!!

""" Hugging Face models for sequence classification """

HF_MODEL_NAMES = {
    "gpt2": "gpt2",
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
    "llama": "code_llama"
}

def get_hf_pretrained_name(model_name: str):
    if model_name in HF_MODEL_NAMES:
        return HF_MODEL_NAMES.get(model_name)
    else:
        raise ValueError(f"{model_name} not supported yet")


class HFSeqClsConfig:
    def __init__(
        self,
        model_name: str,
        input_dim: Optional[int] = None,
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
        self.input_dim = input_dim
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

        # transformers.LlamaConfig
        elif model_name == "llama":
            config_kwargs = {
                "hidden_size": embed_dim,
                "intermediate_size": ffwd_dim,
                "num_hidden_layers": num_layers,
                "num_attention_heads": num_heads,
                "num_labels" : num_labels,
                "problem_type": problem_type,
                "max_position_embeddings": max_seq_len
            }

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

    def to_dict(self):
        """ This is required by the HF Trainer """
        return {
            "model_name": self.model_name
        }


class HFSeqClsModel(SeqClsModel):
    """ Simple wrapper around HF models for sequence classification """
    def __init__(self, config: HFSeqClsConfig):
        super().__init__(config)

        if config.use_pretrained:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                get_hf_pretrained_name(config.model_name),
                ignore_mismatched_sizes = True, # TODO: ignore_mismatched_sizes seems to fail for bert
                **config.pretrained_kwargs)
            self.model_config = self.model.config
        else:
            self.model_config = AutoConfig.for_model(config.model_name, **config.model_config_kwargs)
            self.model = AutoModelForSequenceClassification.from_config(self.model_config)

        if config.input_dim is not None:
            self.embed_fn = nn.Linear(config.input_dim, config.embed_dim)

    @property
    def model_name(self):
        return self.config.model_name

    @property
    def input_dim(self):
        return self.config.input_dim

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
            use_inputs_ids == False: x: (batch_size, seq_len, input_dim)
        """
        # When using input ids, the result is simpler
        if use_input_ids:
            return self.model(input_ids=x, labels=labels, attention_mask=attention_mask, **kwargs)

        # Otherwise we transform the input_dim sequence into one of embed_dim
        assert x.size(-1) == self.input_dim
        x = self.embed_fn(x)

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


