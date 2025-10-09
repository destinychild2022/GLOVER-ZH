from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from ..llava_arch import LlavaMetaForCausalLM, LlavaMetaModel

class LlavaQwenConfig(AutoConfig):
    model_type = "llava_qwen"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 设置千问模型的基本配置
        self.architectures = ["Qwen2ForCausalLM"]
        self.model_type = "qwen2"
        self.hidden_size = 3584
        self.intermediate_size = 18944
        self.num_hidden_layers = 28
        self.num_attention_heads = 28
        self.num_key_value_heads = 4
        self.hidden_act = "silu"
        self.max_position_embeddings = 32768
        self.initializer_range = 0.02
        self.rms_norm_eps = 1e-06
        self.use_cache = True
        self.tie_word_embeddings = False
        self.vocab_size = 152064
        self.bos_token_id = 151643
        self.eos_token_id = 151645

class LlavaQwenModel(LlavaMetaModel):
    config_class = LlavaQwenConfig
    
    def __init__(self, config):
        # 处理Qwen2.5-VL配置
        if hasattr(config, 'model_type') and 'qwen2_5_vl' in str(config.model_type):
            print(f"检测到Qwen2.5-VL配置，使用Qwen2_5_VLForConditionalGeneration加载...")
            # 对于Qwen2.5-VL，使用Qwen2_5_VLForConditionalGeneration
            from transformers import Qwen2_5_VLForConditionalGeneration
            # 从配置中获取模型路径
            model_path = getattr(config, 'mm_vision_tower', None) or getattr(config, 'model_path', None)
            if model_path and model_path != "openai/clip-vit-large-patch14":
                # 使用本地路径加载Qwen2.5-VL模型
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_path, 
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    local_files_only=True
                )
            else:
                # 如果没有指定路径或使用默认CLIP路径，使用AutoModel创建
                from transformers import AutoModel
                self.model = AutoModel.from_config(config, trust_remote_code=True)
        else:
            # 对于其他模型，使用AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        
        self.config = config

class LlavaQwenForCausalLM(LlavaMetaForCausalLM, PreTrainedModel):
    config_class = LlavaQwenConfig
    
    def __init__(self, config):
        PreTrainedModel.__init__(self, config)
        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # 处理多模态输入
        if images is not None:
            inputs_embeds = self.get_model().embed_images(images, inputs_embeds)
        
        # 调用基础模型的forward方法
        outputs = self.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # 计算损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            loss = loss_fct(shift_logits, shift_labels)
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        # 准备生成时的输入
        if past_key_values:
            input_ids = input_ids[:, -1:]
        
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            "inputs_embeds": inputs_embeds,
        } 