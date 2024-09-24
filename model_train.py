from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from torch.nn.utils.rnn import pad_sequence

# Clear GPU memory
torch.cuda.empty_cache()

# Load the GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = "<pad>"  
tokenizer.add_tokens([tokenizer.pad_token]) 
model = GPT2LMHeadModel.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))


dataset = load_dataset('text', data_files={'train': 'professions.txt'})


def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)


tokenized_datasets = dataset.map(tokenize_function, batched=True)


def collate_fn(batch):
    input_ids = pad_sequence([torch.tensor(item['input_ids'], dtype=torch.long) for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = (input_ids != tokenizer.pad_token_id).long() 
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': input_ids.clone()  
    }


train_dataset = tokenized_datasets['train']
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])


training_args = TrainingArguments(
    output_dir="./resultados",           
    num_train_epochs=8,                  
    per_device_train_batch_size=1,      
    per_device_eval_batch_size=1,       
    save_steps=10_000,                   
    save_total_limit=2,                  
    logging_dir='./logs',                
    logging_steps=500,                    
    evaluation_strategy="steps",         
    load_best_model_at_end=True,        
    metric_for_best_model="loss",        
    greater_is_better=False,
    max_steps=8  
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
)

trainer.train()


model.save_pretrained("./resultados")
tokenizer.save_pretrained("./resultados")
