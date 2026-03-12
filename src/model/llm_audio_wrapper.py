from abc import ABC, abstractmethod

import torch
from torch import nn

from src.model.audio import AudioAdapter, AudioEncoder, AudioFusion, AudioPostprocessing


class LLMAudioWrapperModel(nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.audio_encoder = AudioEncoder(encoder_configs=config.audio_encoder_configs)
        self.audio_adapter = AudioAdapter(adapter_configs=config.audio_adapter_configs)
        self.audio_fusion = AudioFusion(config.audio_fusion_config)
        self.audio_postprocessing = AudioPostprocessing(
            config.audio_postprocessing_config
        )
        if not config.use_explicit_audio_tokens:
            self.audio_start_emb = nn.Parameter(
                torch.randn((1, config.audio_sep_d_embed))
            )
            self.audio_end_emb = nn.Parameter(
                torch.randn((1, config.audio_sep_d_embed))
            )


class LLMAudioWrapperForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_config(self):
        return self.get_model().config

    @abstractmethod
    def get_head(self):
        pass

    def freeze_llm(self):
        for param in self.get_model().parameters():
            param.requires_grad = False
        for param in self.get_head().parameters():
            param.requires_grad = False
        # this command should freeze only llm-part
        # we unfreeze other layers
        if hasattr(self.get_model(), "audio_encoder"):
            for param in self.get_audio_encoder().parameters():
                param.requires_grad = True
        if hasattr(self.get_model(), "audio_adapter"):
            for param in self.get_audio_adapter().parameters():
                param.requires_grad = True
        if hasattr(self.get_model(), "audio_fusion"):
            for param in self.get_audio_fusion().parameters():
                param.requires_grad = True
        if hasattr(self.get_model(), "audio_postprocessing"):
            for param in self.get_audio_postprocessing().parameters():
                param.requires_grad = True
        if hasattr(self.get_model(), "audio_start_emb"):
            self.get_model().audio_start_emb.requires_grad = True
        if hasattr(self.get_model(), "audio_end_emb"):
            self.get_model().audio_end_emb.requires_grad = True

    def set_tokenizer_arguments(self, tokenizer):
        self.tokenizer_pad_token_id = tokenizer.pad_token_id
        self.tokenizer_model_max_length = tokenizer.model_max_length

    def get_audio_encoder(self):
        return self.get_model().audio_encoder

    def get_audio_adapter(self):
        return self.get_model().audio_adapter

    def get_audio_fusion(self):
        return self.get_model().audio_fusion

    def get_audio_postprocessing(self):
        return self.get_model().audio_postprocessing

    def get_audio_sep_embed(self):
        audio_start_emb = self.get_model().audio_start_emb
        audio_end_emb = self.get_model().audio_end_emb
        return audio_start_emb, audio_end_emb

    def get_audio_embeddings(self, audio, audio_attention, audio_lengths):
        """
        Get audio embeddings to put inside the LLM.

        Args:
            audio (Tensor): BxT audio.
            audio_attention (Tensor): B -- audio attention mask.
            audio_lengths (Tensor): B original audio length.
        Returns:
            audio_embeds (list[Tensor]): list of length B. Audio embeddings
                cropped based on projected audio_lengths + sep_token. Each
                element is S_i x D
        """
        audio_encoder = self.get_audio_encoder()
        audio_adapter = self.get_audio_adapter()
        audio_fusion = self.get_audio_fusion()
        audio_postprocessing = self.get_audio_postprocessing()

        audio = audio_encoder(audio, audio_attention)
        lengths = audio_encoder.calc_length(audio_lengths)

        audio = audio_adapter(audio, lengths)  # B x S x D
        lengths = audio_adapter.calc_length(lengths)

        audio = audio_fusion(audio, lengths)
        lengths = audio_fusion.calc_length(lengths)

        audio = audio_postprocessing(audio)
        lengths = audio_postprocessing.calc_length(lengths)

        # if the audio became too short, take 1 token
        lengths = lengths.clamp_min(1)
        B = lengths.shape[0]
        return [audio[i, : lengths[i]] for i in range(B)]

    def prepare_multimodal_inputs(
        self,
        inputs_system,
        inputs_start,
        inputs_main,
        outputs,
        audio,
        audio_attention,
        audio_lengths,
        padding_side="right",
        return_batch=True,
    ):
        """
        Join text and audio embeds. Audio-first.

        Args:
            inputs_system (list[Tensor]): list of system prompt ids (can be None).
            input_starts (list[Tensor]): list of start ids.
            input_mains (list[Tensor]): list of content ids.
            outputs (None | list[Tensor]): list of output ids.
            audio (Tensor): BxT -- audio input.
            audio_attention (Tensor): B -- audio attention mask.
            audio_length (Tensor): B audio input without padding.
            padding_side (str): padding side.
            return_batch (bool): if True, return batched tensor instead of a list.
        Returns:
            multimodal_embeds (Tensor): BxSxD joint embeddings.
            labels(Tensor): BxS label ids.
            position_ids (Tensor): BxS positional IDs.
            multimodal_attention_mask (Tensor): BxS -- padding mask.
        """
        batch_size = len(inputs_start)

        inputs_start_emb = [
            self.get_model().embed_tokens(inputs_start[i]) for i in range(batch_size)
        ]

        inputs_main_emb = [
            self.get_model().embed_tokens(inputs_main[i]) for i in range(batch_size)
        ]
        if outputs is not None:
            outputs_emb = [
                self.get_model().embed_tokens(outputs[i]) for i in range(batch_size)
            ]

        if audio is not None:
            audio_embeds = self.get_audio_embeddings(
                audio,
                audio_attention,
                audio_lengths,
            )
            if not self.get_config().use_explicit_audio_tokens:
                audio_start_emb, audio_end_emb = self.get_audio_sep_embed()

        # add after audio because audio conditioning is on user prompt only
        if inputs_system is not None:
            # add system prompt
            inputs_system_emb = [
                self.get_model().embed_tokens(inputs_system[i])
                for i in range(batch_size)
            ]
            inputs_start_emb = [
                torch.cat([inputs_system_emb[i], inputs_start_emb[i]], dim=0)
                for i in range(batch_size)
            ]

        multimodal_embeds = []
        multimodal_attention_mask = []
        multimodal_labels = []

        model_dtype = inputs_start_emb[0].dtype
        device = inputs_start_emb[0].device
        for i in range(batch_size):
            if audio is None:
                multimodal_embed = [inputs_start_emb[i], inputs_main_emb[i]]
                multimodal_embed_ids = [inputs_start[i], inputs_main[i]]
            else:
                if not self.get_config().use_explicit_audio_tokens:
                    multimodal_embed = [
                        inputs_start_emb[i],
                        audio_start_emb.to(model_dtype),
                        audio_embeds[i].to(model_dtype),
                        audio_end_emb.to(model_dtype),
                        inputs_main_emb[i],
                    ]
                    n_sep_ids = 2
                else:
                    multimodal_embed = [
                        inputs_start_emb[i],
                        audio_embeds[i].to(model_dtype),
                        inputs_main_emb[i],
                    ]
                    n_sep_ids = 0
                # the exact token does not matter
                # it will be ignored in labels anyway
                # attention mask will be 1 for audio, so no problem

                audio_ids = [self.tokenizer_pad_token_id] * (
                    audio_embeds[i].shape[0] + n_sep_ids
                )
                audio_ids = torch.tensor(audio_ids, device=device)
                multimodal_embed_ids = [inputs_start[i], audio_ids, inputs_main[i]]

            multimodal_embed = torch.cat(multimodal_embed, dim=0)
            multimodal_embed_ids = torch.cat(multimodal_embed_ids, dim=0)
            if outputs is not None:
                multimodal_embed_len = multimodal_embed_ids.shape[0]
                multimodal_embed = torch.cat([multimodal_embed, outputs_emb[i]], dim=0)
                multimodal_embed_ids = torch.cat(
                    [multimodal_embed_ids, outputs[i]], dim=0
                )

            multimodal_embed = multimodal_embed[: self.tokenizer_model_max_length]
            multimodal_embed_ids = multimodal_embed_ids[
                : self.tokenizer_model_max_length
            ]
            multimodal_embeds.append(multimodal_embed)

            if outputs is not None:
                multimodal_label = multimodal_embed_ids.clone()
                multimodal_label[:multimodal_embed_len] = -100  # mask input part
                multimodal_labels.append(multimodal_label)

            multimodal_attention_mask.append(
                torch.ones(
                    multimodal_embed.shape[0],
                    device=multimodal_embed.device,
                    dtype=torch.long,
                )
            )

        if return_batch:
            multimodal_embeds = torch.nn.utils.rnn.pad_sequence(
                multimodal_embeds,
                batch_first=True,
                padding_value=0,
                padding_side=padding_side,
            )
            if outputs is not None:
                multimodal_labels = torch.nn.utils.rnn.pad_sequence(
                    multimodal_labels,
                    batch_first=True,
                    padding_value=self.tokenizer_pad_token_id,
                    padding_side=padding_side,
                )
                multimodal_labels[
                    multimodal_labels == self.tokenizer_pad_token_id
                ] = -100
            else:
                multimodal_labels = None

            multimodal_attention_mask = torch.nn.utils.rnn.pad_sequence(
                multimodal_attention_mask,
                batch_first=True,
                padding_value=0,
                padding_side=padding_side,
            )
        else:
            if outputs is None:
                multimodal_labels = None

        return (
            multimodal_embeds,
            multimodal_labels,
            multimodal_attention_mask,
        )
