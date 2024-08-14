# Script for fine tuning a pre-existing LLM model (Here, Llama 3 8B) on data (here Finite Element Method data). The script can handle data in the form of raw data or in the form of Questions-Answers in a csv file.
# Author: Computational Physics Group @ USC Jul 2024


#Enter the PATHS wherever relevant.
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForLanguageModeling
from transformers import get_scheduler, set_seed 
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
import pandas as pd
from peft import LoraConfig, get_peft_model, PeftModel
import time

set_seed(2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def bytes_to_giga_bytes(bytes):
  return bytes / 1024 / 1024 / 1024

print("Initial memory allocated=",bytes_to_giga_bytes(torch.cuda.max_memory_allocated()))

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    #bnb_4bit_use_double_quant=True, #Saves additional memory
)

use_refined_transcript = False
use_QA_transcript = True
do_inference = False
load_saved_model = False

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", padding_side="left", cache_dir="PATH_OF_CACHE_DIR", return_tensors="pt")

if not load_saved_model:
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", device_map="auto", quantization_config=bnb_config, cache_dir="PATH_OF_CACHE_DIR") #torch_dtype=torch.float32
    lora_config = LoraConfig( bias="none", init_lora_weights=False, use_rslora=True, lora_alpha=16, lora_dropout=0.05, peft_type="LORA", r=64, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)

if load_saved_model:
    base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", device_map="auto", quantization_config=bnb_config, cache_dir="PATH_OF_CACHE_DIR")
    peft_model_id = "ENTER_PATH_OF_SAVED_MODEL/saved_model"
    model = PeftModel.from_pretrained(base_model, peft_model_id)
    model.merge_adapter()
    print("Successfully loaded the model into memory")

model.print_trainable_parameters()
tokenizer.pad_token = tokenizer.eos_token

if use_refined_transcript:
    data_df = pd.read_csv("ENTER_PATH_OF_FILE/FEM_Garikipati_Transcripts_aggregated.CSV", dtype={'text': 'string'})
    data_df = data_df.set_axis(['id', 'text'], axis=1)
    #index_list = list(range(20, 164))
    #data_df.drop(data_df.index[index_list], inplace=True)
    
if use_QA_transcript:
    data_df = pd.read_csv("ENTER_PATH_OF_FILE/FEM_QA.csv", dtype={'question': 'string', 'answer': 'string'})
    data_df['text'] = "Question: " + data_df['question'] + " Answer: " + data_df['answer'].astype("string")
    data_df = data_df[['text']]
    #index_list = list(range(50, 8203))
    #data_df.drop(data_df.index[index_list], inplace=True)
    
    

def tokenize_function(examples):
    encoding = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=300, return_overflowing_tokens=True, stride=10)
    encoding['labels'] = encoding['input_ids'].copy()
    return encoding

dataset = Dataset.from_pandas(data_df)

if use_refined_transcript:
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["id", "text"])
    #tokenized_datasets = dataset.map(tokenize_function, batched=True)
if use_QA_transcript:
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

print("Tokenized training dataset=\n",tokenized_datasets,"[0]=",tokenized_datasets[0])

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
train_dataloader = DataLoader(tokenized_datasets, shuffle=True, batch_size=4, collate_fn=data_collator)

for batch in train_dataloader:
    break
print("\n Input batch shape:",{k: v.shape for k, v in batch.items()},"\n")

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

num_epochs = 500
num_training_steps = num_epochs * len(train_dataloader)
print("\n num_training_steps =", num_training_steps,"\n")
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=3,
    num_training_steps=num_training_steps
)

progress_bar = tqdm(range(num_training_steps))

print("Prior training memory allocated=",bytes_to_giga_bytes(torch.cuda.max_memory_allocated()))

start_time = time.time()
model.to(device)
model.train()
for epoch in range(num_epochs):
    print("Epoch:", epoch)
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "overflow_to_sample_mapping"}
        #print("Batch Shape=", {k: v.shape for k, v in batch.items()})
        outputs = model(**batch)
        #print("Output=", outputs)
        loss = outputs.loss
        print("loss=", loss)
        loss.backward()
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
    if (epoch % 100) == 0:
        output_dir=f"ENTER_PATH_TO _SAVE_MODEL/epoch{epoch}/"
        model.save_pretrained(output_dir)
        
    
end_time = time.time()

print(f"Training Completed in {end_time - start_time:.2f} seconds")
print("Post training memory allocated=",bytes_to_giga_bytes(torch.cuda.max_memory_allocated()))

# Save the model
output_dir="ENTER_PATH_TO _SAVE_MODEL"
model.save_pretrained(output_dir)

model.eval()
print("Answering your questions....")

questions = [
    "Question: What is Finite Element Method? Answer:",
    "Question: Tell me about the assembly process in Finite Element Method Answer:",
    "Question: What is continuum phyics? Answer:",
    "Question: What is deal.II? Answer:"
]
encoded_inputs = tokenizer(questions, padding=True, return_tensors="pt").to(device)
decoded_input = tokenizer.batch_decode(encoded_inputs["input_ids"], skip_special_tokens=True)
generated_ids = model.generate(**encoded_inputs, max_new_tokens=100)
text_generation = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print("\n\n Questions answered:\n\n",text_generation,"\n\n")
