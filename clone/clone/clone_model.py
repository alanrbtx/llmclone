import pytorch_lightning as pl
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import torch
from collections import namedtuple


class LLMClone(pl.LightningModule):
    """
    LLMClone class create clone with specific PEFT config
    """

    def __init__(self, args):
        super().__init__()
       
        self.args = args
        self._build_clone()
        # self._build_model()
        self._build_tokenizer()


    def _build_clone(self):
        if self.args.quantized == True:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name, 
                quantization_config=bnb_config,
                device_map="auto"
            )
            
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )


    def add_adapters(self, adapter_path):
        print("TYPE: CHAT ADAPTERS")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)

        
        
    def _build_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @torch.no_grad()
    def sample(
            self,
            prompt,
            max_new_tokens=128,
            max_length=512
        ):
        
        input_ids = self.tokenizer(
            prompt,
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids

        tokens = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            
        )

        decoded_tokens = self.tokenizer.decode(
            tokens[0],
            skip_special_tokens=True
        )

        return decoded_tokens