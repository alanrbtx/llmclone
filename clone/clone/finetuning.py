from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType
    )

from datasets import load_dataset, Dataset
from .clone_model import LLMClone

def train_clone(model: LLMClone, dataset: Dataset):

    dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca")
    dataset = dataset.map(lambda example: model.tokenizer(example["prompt"], max_length=256, truncation=True), batched=True)
    dataset = dataset["train"].train_test_split(0.1, 0.9)
    model.tokenizer.pad_token_id = model.tokenizer.eos_token_id

    collator = DataCollatorForLanguageModeling(tokenizer=model.tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="llama",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        max_steps=500,
        warmup_steps=2,
        logging_steps=20,
        save_steps=2000,
        learning_rate=2e-4,
        fp16=True,
        #optim="paged_adamw_8bit",
        ddp_find_unused_parameters=False,
        push_to_hub=True
    )

    trainer = Trainer(
        model=model.model,
        args=training_args,
        data_collator=collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    trainer.train()


    model.model.save_pretrained("clone_peft")
