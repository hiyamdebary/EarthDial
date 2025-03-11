# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import warnings
from typing import Any, List, Optional, Tuple, Union

import torch.distributed as dist
import torch.utils.checkpoint
import transformers
import torch.nn.functional as F
import math
from earthdial.conversation import get_conv_template
from earthdial.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from earthdial.model.phi3.modeling_phi3 import Phi3ForCausalLM
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_internvl_chat import InternVLChatConfig
from .modeling_intern_vit import InternVisionModel

logger = logging.get_logger(__name__)

import multiprocessing

def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer',
                         'Phi3DecoderLayer', 'Qwen2DecoderLayer']
    _supports_flash_attn_2 = True

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.llm_arch_name = config.llm_config.architectures[0]

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Phi3ForCausalLM':
                self.language_model = Phi3ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        if hasattr(config, 'system_message'):
            self.system_message = config.system_message
        else:
            self.system_message = self.conv_template.system_message
        self.num_samples = 0

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)

    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        # Determine the target modules based on the architecture of the language model
        if self.llm_arch_name == 'InternLM2ForCausalLM':
            target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
        elif self.llm_arch_name == 'Phi3ForCausalLM':
            target_modules = ['mlp.down_proj', 'mlp.gate_up_proj', 'self_attn.o_proj', 'self_attn.qkv_proj']
        elif self.llm_arch_name in ['Qwen2ForCausalLM', 'LlamaForCausalLM']:
            target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                              'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
        else:
            raise NotImplemented
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone() # Text embeds [1,4096,3072]

        vit_embeds = self.extract_feature(pixel_values) # output feature [[1,256,3072]]
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C) #[4096,3072]

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id) # M
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C) # Replace all the Visual feature for the context token [256,3072]
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True

        input_embeds = input_embeds.reshape(B, N, C) #[1,4096,3072]

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None: # calculates the next-token prediction loss for a language model by aligning predictions and labels
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous() #each token’s prediction is based on the previous token, aligning predictions to target the "next" token.
            shift_labels = labels[..., 1:].contiguous() #creating an offset so that each token aligns with the one it’s meant to predict.
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size) # [4095,32030]
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels) #Computes the loss between the model’s predictions and the target labels
            if ignore_flag:
                loss = loss * 0.0

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x
    
    def bilinear_interpolate_and_concat(self,emds, final_size=(1024, 1024)):
	
        num_channels = len(emds)
        if num_channels == 1:
            # If there is only one channel
            return F.interpolate(emds[0].unsqueeze(0), size=final_size, mode='bilinear', align_corners=False).squeeze(0)
        
        # Determine the grid dimensions
        num_rows = math.ceil(math.sqrt(num_channels))
        num_cols = math.ceil(num_channels / num_rows)
        
        # Calculate width and height for each embedding
        target_width = final_size[1] // num_cols
        target_height = final_size[0] // num_rows
        
        # To ensure final concatenation yields exactly 1024x1024
        adjusted_heights = [target_height] * num_rows
        adjusted_widths = [target_width] * num_cols

        # Distribute any remaining pixels across the first few rows and columns
        height_remainder = final_size[0] - (target_height * num_rows)
        width_remainder = final_size[1] - (target_width * num_cols)
        for i in range(height_remainder):
            adjusted_heights[i] += 1
        for j in range(width_remainder):
            adjusted_widths[j] += 1

        # Interpolate each embedding and arrange in a grid
        resized_emds = []
        for i, emd in enumerate(emds):
            row = i // num_cols
            col = i % num_cols
            resized_emds.append(F.interpolate(
                emd.unsqueeze(0),
                size=(adjusted_heights[row], adjusted_widths[col]),
                mode='bilinear',
                align_corners=False
            ))

        # Concatenate the embeddings into rows, then concatenate rows into the final tensor
        grid_rows = []
        for row in range(num_rows):
            row_embeddings = resized_emds[row * num_cols:(row + 1) * num_cols]
            row_concat = torch.cat(row_embeddings, dim=-1)  # Concatenate along width
            grid_rows.append(row_concat)
        
        # Ensure each row has the same width (pad if necessary)
        max_width = max([row.shape[-1] for row in grid_rows])
        grid_rows = [F.pad(row, (0, max_width - row.shape[-1])) for row in grid_rows]

        # Concatenate rows along height to form the final tensor
        final_tensor = torch.cat(grid_rows, dim=-2)  # Concatenate rows along height
        
        final_tensor = final_tensor.squeeze(0)
        return final_tensor
    
    @torch.no_grad()
    def sequential_vit_features(self, pixel_values,mm_spatial_pool_mode):
        multiprocessing.set_start_method("spawn", force=True)
        # Determine batch size, number of channels, height, and width
        batch_size, num_channels, height, width = pixel_values.shape
        
        # List to store the outputs for each 3-channel subset across the batch
        vit_outputs = []

        # Process each batch entry individually
        for batch_idx in range(batch_size):
            single_vit_outputs = []  # Temporary list for each 3-channel group for this batch entry

            # Process in groups of 3 channels
            for i in range(0, num_channels, 3):
                # Select a subset of up to 3 channels for this batch item
                subset = pixel_values[batch_idx, i:i + 3, :, :]  # Shape: [C, H, W]

                # If fewer than 3 channels in this subset, pad by duplicating the last channel
                if subset.shape[0] < 3:
                  #  print("Subset has less than 3 bands",subset.shape)
                    padding_needed = 3 - subset.shape[0]
                    subset = torch.cat([subset] + [subset[-1:]] * padding_needed, dim=0)

                # Add batch dimension for ViT input ([1, 3, H, W])
                subset = subset.unsqueeze(0)
                # Move the input tensor to the same device and dtype as the model parameters
                subset = subset.to(device=next(self.vision_model.parameters()).device, dtype=next(self.vision_model.parameters()).dtype)

                if self.select_layer == -1:
                    # Pass to CLIP and extract features for the RGB [1,3,448,448]
                    vit_embeds = self.vision_model(
                        pixel_values=subset,
                        output_hidden_states=False,
                        return_dict=True).last_hidden_state
                else:
                    vit_embeds = self.vision_model(
                        pixel_values=subset,
                        output_hidden_states=True,
                        return_dict=True).hidden_states[self.select_layer]
                vit_embeds = vit_embeds[:, 1:, :].cpu() # [1,1024,1024] # Remove the CLS token from the model

                # Store the embedding output for this subset
                single_vit_outputs.append(vit_embeds)

            # Stack single_vit_outputs along a new dimension and apply pooling for this batch entry
            #single_vit_outputs = torch.stack(single_vit_outputs, dim=0)  # Shape: [N, 1, C, H, W]
            
            # Apply spatial pooling based on config
            if mm_spatial_pool_mode == "average":
                stack_op=torch.stack(single_vit_outputs, dim=0)
                pooled_output = stack_op.mean(dim=0)  # Average pooling across the 3-channel groups
            elif mm_spatial_pool_mode == "max":
                stack_op=torch.stack(single_vit_outputs, dim=0)
                pooled_output = stack_op.max(dim=0)  # Max pooling across the 3-channel groups
            elif mm_spatial_pool_mode == "sum":
                stack_op=torch.stack(single_vit_outputs, dim=0)
                pooled_output = stack_op.sum(dim=0)  # Sum pooling across the 3-channel groups
            elif mm_spatial_pool_mode == "bilinear":
                  pooled_output =self.bilinear_interpolate_and_concat(single_vit_outputs)
                #     resized_features = [
                #         torch.nn.functional.interpolate(feature, size=(height, width), mode='bilinear', align_corners=False)
                #         for feature in single_vit_outputs
                #     ]
                # pooled_output = torch.stack(resized_features).mean(dim=0)  # Average resized features

            vit_outputs.append(pooled_output)  # Append pooled result for this batch entry

        # Stack all batch entries and remove extra dimensions if needed
        vit_outputs  = torch.stack(vit_outputs, dim=0)  # Shape: [B, C, H, W]
        vision_embeds = vit_outputs.squeeze(0)  # s shape [2, 1024, 1024]
        if(vision_embeds.size(0)>1) :
           # print("After Vit before mean across 2 ",vision_embeds.size())
            vision_embeds=vision_embeds.squeeze(1) 
            #vision_embeds = vision_embeds.mean(dim=1)  # Average across the embedding dimension
            #print("After Vit After mean across 2 ",vision_embeds.size())

        return vision_embeds  # [B,H,W]

    def extract_feature(self, pixel_values):
        #if  pixel_values.shape[1] !=3:
        if  pixel_values.shape[2] == 1024 or pixel_values.ndim==3:
            #vit_embeds = self.sequential_vit_features(pixel_values,'average')
            vit_embeds = pixel_values #[1,1024,1024]
        else:
            if self.select_layer == -1:
                # Pass to CLIP and extract features for the RGB [1,3,448,448]
                vit_embeds = self.vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=False,
                    return_dict=True).last_hidden_state
            else:
                vit_embeds = self.vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=True,
                    return_dict=True).hidden_states[self.select_layer]
            vit_embeds = vit_embeds[:, 1:, :] # [1,1024,1024] # Remove the CLS token from the model

        h = w = int(vit_embeds.shape[1] ** 0.5) # [1,1024,1024]
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1) #([1, 32, 32, 1024])
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio) #([1, 16, 16, 4096])
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1]) #([1, 256, 4096])
        vit_embeds = self.mlp1(vit_embeds) #([1, 256, 3072])
        
        return vit_embeds
    
    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep)[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
