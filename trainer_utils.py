from trl import SFTTrainer, SFTConfig

def create_trainer(model, tokenizer, train_dataset, eval_dataset, output_dir, learning_rate=1e-4):

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer, #deneme 
        args=SFTConfig(
            max_seq_length=2048,
            dataset_text_field="text",
            num_train_epochs=2,
            per_device_train_batch_size=8, 
            gradient_accumulation_steps=4,
            lr_scheduler_type="constant",
            learning_rate=learning_rate,
            eval_strategy="steps",
            eval_steps=500,
            warmup_ratio=0.05,
            output_dir=output_dir,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": True},
            logging_steps=1,
        )
    )
    return trainer