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

def train_clone(clone_model: LLMClone, dataset: Dataset):
    tokenizer = clone_model.tokenizer
    model = clone_model.model

    quant_model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none"
    )


    peft_model = get_peft_model(quant_model, peft_config)
    peft_model.print_trainable_parameters()


    dataset = dataset.map(lambda example: tokenizer(example["prompt"], max_length=256, truncation=True), batched=True)
    dataset = dataset.train_test_split(0.1, 0.9)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="llama",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=200,
        warmup_steps=2,
        logging_steps=10,
        save_steps=2000,
        learning_rate=2e-4,
        fp16=True,
        optim="paged_adamw_8bit",
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        data_collator=collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    trainer.train()


    peft_model.save_pretrained("clone_peft")
