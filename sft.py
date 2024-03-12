from typing import List
import os

from typing import Optional, Tuple, Union
from torch.nn import CrossEntropyLoss
import torch
import argparse
from transformers import DataCollatorForLanguageModeling
from transformers import GPT2LMHeadModel, GPT2Config
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from datasets import load_dataset
import pandas as pd

class GPT2LossMaskModel(GPT2LMHeadModel):

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        loss_mask = token_type_ids[..., 1:].bool()
        labels = input_ids
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_logits = shift_logits[loss_mask]
            shift_labels = labels[..., 1:].contiguous()
            shift_labels = shift_labels[loss_mask]
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='ft_data.csv')
    parser.add_argument("--output_dir", type=str, default='model')
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()

    # Load CSV file
    df = pd.read_csv(args.data_path)

    # Create a dataset from the CSV file
    dataset = [{"source": str(source), "target": str(target)} for source, target in zip(df['source'], df['target'])]

    # Initialize the model and tokenizer
    model = GPT2LossMaskModel.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        tokenized_input = tokenizer(batch["source"], batch["target"], padding=False, truncation=True, max_length=args.max_length, return_token_type_ids=True)
        return tokenized_input

    def load_and_tokenize_dataset():
        return dataset

    all_data = load_and_tokenize_dataset()
    train_data = all_data  # For simplicity, you can use the entire dataset for training
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        evaluation_strategy="steps",
        eval_steps=5000000,
        logging_steps=20,
        gradient_accumulation_steps=1,
        num_train_epochs=150,
        weight_decay=0.1,
        warmup_steps=300,
        learning_rate=5e-5,
        save_steps=5000,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=train_data,
    )
    
    trainer.train()
