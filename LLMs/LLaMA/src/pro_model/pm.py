import warnings

import torch
import torch.nn as nn

from peft.utils import _get_batch_size
from .map_layer import Mapnet
from .cross_token import ConsistencyModule
from .gate import GatingModule


class PromptTuningModelForCausalLM(nn.Module):

    def __init__(self, model, soft_prompt_length, **kwargs):
        super().__init__()

        self.model = model
        self.config = self.model.config
        self.soft_token_num = soft_prompt_length
        self.model_prepare_inputs_for_generation = self.model.base_model.model.prepare_inputs_for_generation
        self.mapnet = Mapnet(self.model.base_model.model.model.embed_tokens, self.config.hidden_size)
        self.consistnet = ConsistencyModule(self.config.hidden_size)
        self.gatenet = GatingModule(self.config.hidden_size)
        self.prefix_encoder = self.create_prefix_encoder(self.soft_token_num)
        self.prefix_tokens = torch.arange(self.soft_token_num).to(torch.int64)

    def create_prefix_encoder(self, num_prefix_tokens):
        return nn.Embedding(num_prefix_tokens, self.config.hidden_size)

    def get_soft_prompts(self, batch_size):
        prefix_encoder = self.prefix_encoder
        table_prompt_tokens = (
            self.prefix_tokens
            .unsqueeze(0)
            .expand(batch_size, -1)
        ).to(prefix_encoder.weight.device)
        soft_prompts = prefix_encoder(table_prompt_tokens)
        return soft_prompts

    def forward(
        self,
        input_ids=None,
        gate_ids=None,
        subgraphs=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        batch_size = _get_batch_size(input_ids, inputs_embeds)

        if inputs_embeds is None:
            inputs_embeds = self.model.base_model.model.model.embed_tokens(input_ids)
            gate_embeds = self.model.base_model.model.model.embed_tokens(gate_ids)

        subgraph_vector = self.mapnet(subgraphs)
        s_consistency_token = self.consistnet(subgraph_vector)

        combined_embedding = self.gatenet(gate_embeds, s_consistency_token).to(inputs_embeds.dtype)

        soft_prompts = self.get_soft_prompts(batch_size).to(inputs_embeds.dtype)

        all_prefix = torch.cat((soft_prompts, subgraph_vector, combined_embedding), dim=1)
        prefix_length = all_prefix.shape[1]

        if attention_mask is not None:
            prefix_attention_mask = torch.ones(batch_size, prefix_length).to(attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update({
            "attention_mask": attention_mask,
            "labels": labels,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        })
        if labels is not None:
            prefix_labels = torch.full((batch_size, prefix_length), -100).to(labels.device)
            kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)

        inputs_embeds = torch.cat((all_prefix, inputs_embeds), dim=1)
        return self.model(inputs_embeds=inputs_embeds, **kwargs)

    def inference(self, **kwargs):
        self.num_beams = kwargs['generation_config'].num_beams
        self.gate_ids = kwargs['gate_ids'].repeat(self.num_beams, 1)
        self.subgraphs = kwargs['subgraphs'].repeat(self.num_beams, 1, 1)

        del kwargs["gate_ids"]
        del kwargs["subgraphs"]

        outputs = self.generate(**kwargs)
        return outputs

    def generate(self, **kwargs):
        self.model.base_model.model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        self.model.base_model.generation_config = kwargs['generation_config']
        self.model.base_model.model.generation_config = kwargs['generation_config']
        try:
            outputs = self.model.generate(**kwargs)
        except Exception:
            self.model.base_model.model.prepare_inputs_for_generation = self.model_prepare_inputs_for_generation
            raise
        else:
            self.model.base_model.model.prepare_inputs_for_generation = self.model_prepare_inputs_for_generation
            return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        model_kwargs = self.model_prepare_inputs_for_generation(*args, **kwargs)

        if model_kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            model_kwargs["position_ids"] = None

        if model_kwargs["past_key_values"] is None:
            batch_size = model_kwargs["input_ids"].shape[0]
            self.device = model_kwargs["input_ids"].device
            self.batch_size = batch_size
            inputs_embeds = self.model.base_model.model.model.embed_tokens(model_kwargs["input_ids"])
            gate_embeds = self.model.base_model.model.model.embed_tokens(self.gate_ids)

            subgraph_vector = self.mapnet(self.subgraphs)
            s_consistency_token = self.consistnet(subgraph_vector)
            combined_embedding = self.gatenet(gate_embeds, s_consistency_token).to(inputs_embeds.dtype)

            soft_prompts = self.get_soft_prompts(batch_size).to(inputs_embeds.dtype)

            all_prefix = torch.cat((soft_prompts, subgraph_vector, combined_embedding), dim=1)
            self.prefix_length = all_prefix.shape[1]
            inputs_embeds = torch.cat((all_prefix, inputs_embeds), dim=1)
            model_kwargs["inputs_embeds"] = inputs_embeds
            model_kwargs["input_ids"] = None

        if model_kwargs.get("attention_mask", None) is not None:
            prefix_attention_mask = torch.ones(
                self.batch_size, self.prefix_length
            ).to(self.device)
            model_kwargs["attention_mask"] = torch.cat(
                (prefix_attention_mask, model_kwargs["attention_mask"]), dim=1
            )

        return model_kwargs
