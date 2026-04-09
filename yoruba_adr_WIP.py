"""
IMPORT ALL IMPORTS, DEPENDENCIES, ETC NEEDED
"""
from huggingface_hub import login
from datasets import load_dataset
import unicodedata
import pandas as pd
import torch
from transformers import (AutoTokenizer,AutoModelForSeq2SeqLM,Trainer,TrainingArguments,)
from torch.utils.data import DataLoader
import editdistance


"""
INITIAL DATASET LOAD FROM HUGGINGFACE

1. GET ACCESS TO YANKARI DATASET FROM https://huggingface.co/datasets/acflp/YANKARI
2. GENERATE A PERSONAL READ-ONLY TOKEN TO ACCESS IT
   2a. Go to your HuggingFace profile, under Settings/AccessTokens
   2b. Click '+ Create New Token'
   2c. Set Token Type to Read. Name Token. Create token. This is a 1-time display so save a copy.
   2d. Paste token in requested spot below.
"""

#INSERT YOUR HUGGINGFACE READ-ONLY TOKEN HERE INSIDE QUOTES
login("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")

#LOAD THE DATASET, EXTRACT ONLY THE TEXT ENTRIES
ds = load_dataset("acflp/YANKARI", split="train")
ds_text = ds.remove_columns([col for col in ds.column_names if col != "text"])

"""
INITIAL DATA PREPROCESSING

THIS ENDS WITH PREPPED TRAINING DATASET SPLIT
PREPPED TRAINING ENTRIES INCLUDE:
1. Input text with no diacritic markings
2. Matching target text with diacritic markings

FORMAT:
{
    "input_text": "...",
    "target_text": "..."
}
"""

#RANDOMLY SPLIT 88% TRAIN
split_1 = ds_text.train_test_split(test_size=0.12, seed=42)
train_ds = split_1['train']
temp_ds = split_1['test']
#RANDOMLY SPLIT 6% DEV, 6% TEST
split_2 = temp_ds.train_test_split(test_size=0.5, seed=42)
dev_ds = split_2['train']
test_ds = split_2['test']

#DIACRITIC STRIPPING FUNCTION
#FOR CREATING DIACRITIC-FREE INPUT ENTRIES
def strip_diacritics(text):
    return ''.join(
        #USES UNICODE TO STANDARDIZE, SUCH AS ọ TO o
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
        )

#FUNCTION TO PREP INPUT
#GIVEN ENTRY, MAKES DIACRITIC-FREE INPUT, DIACRITIC MARKED TARGET
def preprocess_hf(example):
    return {
        "input_text": strip_diacritics(example["text"]),
        "target_text": example["text"]
    }

#RUN TEXT PREPROCESSING ON ALL SPLITS
train_ds = train_ds.map(preprocess_hf, remove_columns=['text'],num_proc=6)
dev_ds = dev_ds.map(preprocess_hf, remove_columns=['text'],num_proc=6)
test_ds = test_ds.map(preprocess_hf, remove_columns=['text'],num_proc=6)

#USE TRAIN SET FOR TRAINING
dataset = train_ds 

"""
MODEL SETUP

THIS ENDS WITH BYT5 MODEL LOADED, TOKENIZATION FUNCTIONS READY
"""

#LOAD BYT5 MODEL AND TOKENIZER
#CURRENTLY USING SMALL, WE CAN TRY UPPING TO BYT5 BASE
model_name = "google/byt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#FREEZE ENCODER LAYERS
#CURRENTLY FREEZES HALF TO REDUCE TRAINING TIME, WE CAN TRY UNFREEZING MORE
num_encoder_layers = len(model.encoder.block)
for layer in model.encoder.block[:num_encoder_layers // 2]:
    for param in layer.parameters():
        param.requires_grad = False

#TOKENIZATION FUNCTION
def preprocess(example):
    #TOKENIZE INPUT
    model_inputs = tokenizer(
        example["input_text"],
        truncation=True,
        #PADS SHORT ENTRIES TO 256
        padding="max_length",
        #CAPS LONG ENTRIES TO 256
        #WE MIGHT WANT TO EXTEND THIS, OR IMPLEMENT SLIDING WINDOW, SINCE THIS LOSES INPUT INFORMATION
        max_length=1024,
    )
    #TOKENIZE TARGET, SAME SETUP
    labels = tokenizer(
        example["target_text"],
        truncation=True,
        padding="max_length",
        max_length=1024,
    )["input_ids"]

    #PADDING TOKENS
    #LIST COMPREHENSION FOR ALL LABEL IDS, WITH ALL PADDING TOKENS SET TO -100 
    #CROSS ENTROPY LOSS IGNORES VALUE -100
    labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]
    model_inputs["labels"] = labels
    return model_inputs

"""
RUN TOKENIZATION OF TRAINING SPLIT

ENDS WITH TOKENIZED TRAINING SPLIT
"""

#GET TOKENIZED DATASET
tokenized_dataset = dataset.map(preprocess)

#FORMAT TOGETHER WITH INPUTS, ATTENTION MASK, LABELS
tokenized_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)


"""
DEFINE TRAINER CLASS, SET TRAINING HYPERPARAMETERS

ENDS WITH MODEL READY TO TRAIN
"""

#TRAINER CLASS
class YorubaTrainer(Trainer):
    #INIT
    def __init__(self, byte_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.byte_weights = byte_weights

    #FUNCTION TO GET LOSS
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

#TRAINING VARIABLES
training_args = TrainingArguments(
    #SAVES MODEL PARAMETERS WHEN DONE
    output_dir="./byt5_yoruba_fixed",
    learning_rate=1e-4,
    #TRAIN BATCH SIZE, MAKE SURE SAME AS EVAL
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    #FOR PROGRESS MONITORING - PRINTS LOSS EVERY N STEPS DURING TRAINING
    logging_steps=50,
    #CURRENTLY NO INTERMITENT SAVING, WAS BREAKING
    save_strategy="no",
    report_to="none"
)

#ACTUAL TRAINER
trainer = YorubaTrainer(
    #PASSES IN ALL OF ABOVE
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

"""
RUN MODEL TRAINING
"""

#RUN THE TRAINER
trainer.train()

"""
EVALUATION

PREPARE EVAL SPLIT, TOKENIZE
"""
#Use Eval dataset split
eval_dataset = dev_ds
eval_tokenized = eval_dataset.map(preprocess)
#TOKENIZE DEV SET SAME AS TRAIN
eval_tokenized.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)

"""
SWITCH MODEL TO EVAL MODE
"""

#SET TO DEVICE, SWITCH TO EVAL MODE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

"""
DEFINE EVAL HELPER FUNCTIONS

FUNCTION TO TRANSCODE ID VALUES BACK TO ACTUAL TEXT, FOR METRICS
FUNCTION TO CALCULATE CHARACTER ACCURACY
FUNCTION TO CALCULATE CHARACTER ERROR RATE

CAN ADD FURTHER EVAL METRIC FUNCTIONS HERE
"""

#FUNCTION TO DECODE IDS BACK TO REAL TEXT
def decode(ids):
    #SKIPS SPECIAL TOKENS SO STUFF LIKE [CLS] DOESNT GET WRITTEN INTO TEXT
    return tokenizer.decode(ids, skip_special_tokens=True)

#FUNCTION FOR CHARACTER ACCURACY METRIC
def char_accuracy(preds, targets):
    total = 0
    correct = 0
    #FOR PREDICTION, TARGET
    for p, t in zip(preds, targets):
        #GET SMALLER LEN
        min_len = min(len(p), len(t))
        correct += sum(p[i] == t[i] for i in range(min_len))
        total += len(t)
    #RETURN CHARACTER ACCURACY VALUE
    return correct / total if total > 0 else 0

#FUNCTION FOR CHARACTER ERROR RATE METRIC
def cer(preds, targets):
    total_dist = 0
    total_chars = 0
    #FOR PREDICTION, TARGET
    for p, t in zip(preds, targets):
        #GET ALL EDIT DISTANCES
        total_dist += editdistance.eval(p, t)
        #GET TOTAL CHAR LENGTH
        total_chars += len(t)
    #RETURN CER VALUE
    return total_dist / total_chars if total_chars > 0 else 0

#EVAL DATALOADER
eval_loader = DataLoader(
    eval_tokenized,   # full dataset, no .select()
    batch_size=16     # keep batch size same as before
)


"""
RUN MODEL IN EVAL MODE

ENDS WITH PREDICTED AND TARGETS ENTRIES, AS REAL TEXT
"""

#INSTANSIATE PREDICTION, TARGET LISTS
all_preds = []
all_targets = []

#RUN EVAL PASS THROUGH MODEL
with torch.no_grad():
    #PER BATCH
    for batch in eval_loader:
        #INPUTS GO IN
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        #GET PREDICTIONS
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=1024
        )
        #CONVERT PREDICTIONS BACK TO REAL TEXT
        preds = [decode(g) for g in generated]
        #GET TARGETS
        targets = [
            #CONVER TARGETS BACK TO REAL TEXT
            decode([t.item() for t in label if t != -100])
            for label in batch["labels"]
        ]
        #ADD PREDICTIONS AND TARGETS (REAL TEXT) TO LISTS
        all_preds.extend(preds)
        all_targets.extend(targets)

"""
CALCULATE FINAL METRICS FOR EVAL SPLIT, PRINT
"""

#PRINT METRICS
acc = char_accuracy(all_preds, all_targets)
cer_score = cer(all_preds, all_targets)
print(f"\nCharacter Accuracy: {acc:.4f}")
print(f"CER: {cer_score:.4f}")

""""
PRINT SAMPLE SELECTION OF INPUT, TARGET, PREDICTION TRIOS

JUST FOR EYEBALLING HOW SYSTEM IS DOING
"""

#PRINT SOME ACGTUAL SAMPLE METRICS
print("\nSample Predictions:\n" + "-"*50)
for i in range(min(10, len(eval_dataset))):
    print(f"Input     : {eval_dataset[i]['input_text']}")
    print(f"Target    : {all_targets[i]}")
    print(f"Prediction: {all_preds[i]}")
    print("-"*50)

"""
SAVE MOVEL PARAMETERS TO LOCAL MAIN DIRECTORY

THIS CODE SHOULD PROBABLY BE MOVED TO ABOVE EVAL SECTION

THESE SAVED PARAMETERS CAN BE LOADED UP AGAIN LATER SO WE DONT HAVE TO RETRAIN EVERY TIME
"""
#CHANGE PATHS INSIDE QUOTES TO CHANGE SAVE DIRECTORY
model.save_pretrained("./byt5_yoruba_fixed")
tokenizer.save_pretrained("./byt5_yoruba_fixed")
