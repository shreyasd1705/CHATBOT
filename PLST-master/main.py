import tqdm
import math
from openprompt_.data_utils.text_classification_dataset import CustomProcessor
import torch
from openprompt.data_utils.utils import InputExample
import argparse
import numpy as np
from openprompt import *
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from transformers import XLMRobertaTokenizer,XLMRobertaForMaskedLM,XLMRobertaConfig,DistilBertConfig,DistilBertForMaskedLM,DistilBertTokenizer,XmodConfig,XmodForMaskedLM,AutoTokenizer,DebertaV2Tokenizer,DebertaV2Config,DebertaV2ForMaskedLM,DebertaV2ForSequenceClassification

from openprompt import PromptDataLoader

from openprompt.prompts import MixedTemplate,KnowledgeableVerbalizer

from openprompt.plms.mlm import MLMTokenizerWrapper

parser = argparse.ArgumentParser("")
parser.add_argument("--shot", type=int, default=150)
parser.add_argument("--seed", type=int, default=144)
parser.add_argument("--plm_eval_mode", action="store_true")
# <<<<<<< HEAD
parser.add_argument("--model", type=str, default='bert')
parser.add_argument("--model_name_or_path", default='bert-base-cased') # xlm-roberta-base # roberta-base # google/t5-base-lm-adapt # google/t5-xl-lm-adapt
# =======
# parser.add_argument("--model", type=str, default='roberta')
# parser.add_argument("--model_name_or_path", default='roberta-large')
# >>>>>>> 27c744522a551060ef1fda7bde07789473c33a48
parser.add_argument("--verbalizer", type=str)
parser.add_argument("--calibration", action="store_true")
parser.add_argument("--filter", default="none", type=str)
parser.add_argument("--template_id", type=int, default=0)
parser.add_argument("--dataset", type=str)
parser.add_argument("--result_file", type=str, default="../sfs_scripts/results.txt")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=20)
parser.add_argument("--kptw_lr", default=0.01, type=float)
parser.add_argument("--pred_temp", default=1.0, type=float)
parser.add_argument("--max_token_split", default=-1, type=int)
args = parser.parse_args()



import random



'''
print(args.model)
print(args.model_name_or_path)
print(args.plm_eval_mode)
print(args.max_epochs)
print(args.max_token_split)
'''
this_run_unicode = str(random.randint(0, 1e10))

from openprompt.utils.reproduciblity import set_seed

set_seed(args.seed)

from openprompt.plms import load_plm

if args.model=="xlm-roberta":
    model_config=XLMRobertaConfig.from_pretrained(args.model_name_or_path)
    plm=XLMRobertaForMaskedLM.from_pretrained(args.model_name_or_path, config=model_config)
    tokenizer=XLMRobertaTokenizer.from_pretrained(args.model_name_or_path)
    WrapperClass=MLMTokenizerWrapper

# elif args.model=="luke":
#     model_config=LukeConfig.from_pretrained(args.model_name_or_path)
#     plm=LukeForMaskedLM.from_pretrained(args.model_name_or_path, config=model_config)
#     tokenizer=mLukeTokenizer.from_pretrained(args.model_name_or_path)
#     WrapperClass=MLMTokenizerWrapper
    
elif args.model=="distilbert":
    model_config=DistilBertConfig.from_pretrained(args.model_name_or_path)
    plm=DistilBertForMaskedLM.from_pretrained(args.model_name_or_path, config=model_config)
    tokenizer=DistilBertTokenizer.from_pretrained(args.model_name_or_path)
    WrapperClass=MLMTokenizerWrapper

# elif args.model=="xmod":
#     model_config=XmodConfig.from_pretrained(args.model_name_or_path)
#     plm=XmodForMaskedLM.from_pretrained(args.model_name_or_path,config=model_config)
#     tokenizer=AutoTokenizer.from_pretrained(args.model_name_or_path)
#     WrapperClass=MLMTokenizerWrapper
    


elif args.model=="deberta":
    model_config=DebertaV2Config.from_pretrained(args.model_name_or_path)
    plm=DebertaV2ForMaskedLM.from_pretrained(args.model_name_or_path,config=model_config)
    tokenizer=DebertaV2Tokenizer.from_pretrained(args.model_name_or_path)
    WrapperClass=MLMTokenizerWrapper


# elif args.model="muril":
#     model_config=DebertaV2Config.from_pretrained(args.model_name_or_path)
#     plm=DebertaV2ForMaskedLM.from_pretrained(args.model_name_or_path,config=model_config)
#     tokenizer=DebertaV2Tokenizer.from_pretrained(args.model_name_or_path)
#     WrapperClass=MLMTokenizerWrapper


    

else:
    plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

dataset = {}

if args.dataset == "agnewstitle":
    with open('./datasets/TextClassification/' + args.dataset + '/classes.txt','r') as f:
    	labels = f.read().split('\n')[:-1]
    dataset['train'] = CustomProcessor(labels).get_examples('./datasets/TextClassification/' + args.dataset+'/',"train")
    dataset['test'] = CustomProcessor(labels).get_examples('./datasets/TextClassification/' + args.dataset+'/',"test")
    class_labels = CustomProcessor(labels).get_labels()
    scriptsbase = "TextClassification/agnewstitle"
    scriptformat = "txt"
    cutoff = 0.5
    max_seq_l = 128
    batch_s = 1

else:
    raise NotImplementedError


mytemplate = MixedTemplate(model=plm, 
			    tokenizer=tokenizer,
			    text ='This sentence: "<text_a>", is a {"mask"} question.',
			    placeholder_mapping= {'<text_a>':'text_a','<text_b>':'text_b'})



# myverbalizer = CptVerbalizer(tokenizer, classes=class_labels, candidate_frac=cutoff, pred_temp=args.pred_temp, max_token_split=args.max_token_split)\
# 				.from_file(path=f"./scripts/{scriptsbase}/cpt_verbalizer.{scriptformat}")
#print(class_labels)
#print(mytemplate.text)
myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=class_labels, candidate_frac=cutoff, pred_temp=args.pred_temp, max_token_split=args.max_token_split)\
   				.from_file(path=f"./scripts/{scriptsbase}/cpt_verbalizer.{scriptformat}")

#print(myverbalizer)


from openprompt import PromptForClassification

use_cuda = False
prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False,
                                       plm_eval_mode=args.plm_eval_mode)
if use_cuda:
    prompt_model = prompt_model.cuda()


from  sklearn.model_selection import train_test_split
dataset['train'], dataset['validation'] = train_test_split(dataset['train'], test_size=0.02, random_state=args.seed, shuffle=True)

#from openprompt.data_utils.data_sampler import FewShotSampler
#sampler = FewShotSampler(num_examples_per_label=args.shot, also_sample_dev=True, num_examples_per_label_dev=args.shot)
#dataset['train'], dataset['validation'] = sampler(dataset['train'], seed=args.seed)

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
                                    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                    decoder_max_length=3,
                                    batch_size=batch_s, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                                    truncate_method="tail")

validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
                                         tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l,
                                         decoder_max_length=3,
                                         batch_size=batch_s, shuffle=False, teacher_forcing=False,
                                         predict_eos_token=False,
                                         truncate_method="tail")

# zero-shot test
test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
                                   tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
                                   batch_size=batch_s, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                   truncate_method="tail")

print(dataset['train'])


def calculate_mcc(y_true, y_pred):
    # Count True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN)
    TP = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    TN = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 0)
    FP = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
    FN = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)

    # Calculate Matthews correlation coefficient (MCC)
    numerator = (TP * TN) - (FP * FN)
    denominator = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    
    if denominator == 0:
        mcc = 0  # Handle division by zero
    else:
        mcc = numerator / denominator
    
    return mcc



def evaluate(prompt_model, dataloader, desc):
    prompt_model.eval()
    allpreds = []
    alllabels = []
    pbar = tqdm.tqdm(dataloader, desc=desc)
    for step, inputs in enumerate(pbar):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    
    acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)
    mcc = calculate_mcc(alllabels, allpreds)
    f1 = f1_score(alllabels, allpreds, average='weighted')
    
    print(f"Accuracy: {acc}, MCC: {mcc}, F1-score: {f1}")
    
    return acc, mcc, f1





from transformers import AdamW, get_linear_schedule_with_warmup

loss_func = torch.nn.CrossEntropyLoss()


def prompt_initialize(verbalizer, prompt_model, init_dataloader):
    dataloader = init_dataloader
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Init_using_{}".format("train")):
            batch = batch.cuda()
            logits = prompt_model(batch)
        verbalizer.optimize_to_initialize()


if args.verbalizer == "cpt":
    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)
    optimizer2 = AdamW(prompt_model.verbalizer.parameters(), lr=args.kptw_lr)

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)


    scheduler2 = None

elif args.verbalizer == "manual":
    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=1e-5)

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    optimizer2 = None
    scheduler2 = None

tot_loss = 0
log_loss = 0
best_val_acc = 0

#print("Label Mapping:")
#print(class_labels)

# Label Encoding
encoded_labels = {label: i for i, label in enumerate(class_labels)}
'''
# After encoding the labels
print("Label Encoding Check:")
for i, example in enumerate(dataset['train'][:5]):  # Check the first 5 examples
    print("Raw Dataset Examples:")
    print("Text:", example)
    print()
'''
val_mcc=0
val_f1=0


for epoch in range(args.max_epochs):
    tot_loss = 0
    prompt_model.train()
    for step, inputs in tqdm.tqdm(enumerate(train_dataloader)):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']

    
        # Print out the target labels
        #print("Target Labels:", labels)

        loss = loss_func(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
        tot_loss = tot_loss + loss.item()
        optimizer1.step()
        scheduler1.step()
        optimizer1.zero_grad()
        if optimizer2 is not None:
            optimizer2.step()
            optimizer2.zero_grad()
        if scheduler2 is not None:
            scheduler2.step()
        if step % 500 == 0:
            val_acc,val_mcc,val_f1 = evaluate(prompt_model, validation_dataloader, desc="Valid")
            if val_acc >= best_val_acc:
                torch.save(prompt_model.state_dict(), f"./ckpts/{this_run_unicode}.ckpt")
                best_val_acc = val_acc
            print("Epoch {}, val_acc {:.4f}".format(epoch, val_acc), flush=True)


    val_acc,val_mcc,val_f1 = evaluate(prompt_model, validation_dataloader, desc="Valid")
    if val_acc >= best_val_acc:
        torch.save(prompt_model.state_dict(), f"./ckpts/{this_run_unicode}.ckpt")
        best_val_acc = val_acc
    print("Epoch: {}, val_acc: {}".format(epoch, val_acc), flush=True)

prompt_model.load_state_dict(torch.load(f"./ckpts/{this_run_unicode}.ckpt"))
prompt_model = prompt_model#.cuda()
test_acc,test_mcc,test_f1= evaluate(prompt_model, test_dataloader, desc="Test")



content_write = "=" * 20 + "\n"
content_write += f"dataset {args.dataset}\t"
content_write += f"temp {args.template_id}\t"
content_write += f"seed {args.seed}\t"
content_write += f"shot {args.shot}\t"
content_write += f"verb {args.verbalizer}\t"
content_write += f"cali {args.calibration}\t"
content_write += f"filt {args.filter}\t"
content_write += f"maxsplit {args.max_token_split}\t"
content_write += f"kptw_lr {args.kptw_lr}\t"
content_write += "\n"
content_write += f"model  {args.model_name_or_path}\n"
content_write += f"Acc: {test_acc:.4f}\n"
content_write += f"Mcc: {test_mcc:.4f}\n"
content_write += f"F-1: {test_f1:.4f}\n"
content_write += "\n\n"

#print(content_write)

# with open(f"{args.result_file}", "a") as fout:
#     fout.write(content_write)

import os

os.remove(f"./ckpts/{this_run_unicode}.ckpt")