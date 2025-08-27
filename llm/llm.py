import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class HuggingfaceLLM:
    def __init__(self, model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        # 自动选择设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

    def __call__(self, conversation, max_new_tokens=200):
        """conversation: list of dict [{'role': 'system'|'user'|'assistant', 'content': str}, ...]"""
        inputs = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)