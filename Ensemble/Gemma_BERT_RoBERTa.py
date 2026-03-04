import json
import numpy as np
from transformers import BertTokenizerFast, RobertaTokenizerFast, AutoTokenizer
import torch.nn as nn
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import RobertaForTokenClassification, AutoTokenizer, AutoModelForCausalLM, BertForTokenClassification
from torch.nn.utils.rnn import pad_sequence
from peft import PeftModel
import torch
import re
from datasets import Dataset
from tqdm import tqdm



with open("./datasets/HateNorm/test.json", "r") as file:
    test_json = json.load(file)

tokenizer_bert = BertTokenizerFast.from_pretrained('bert-base-uncased')
tokenizer_roberta = RobertaTokenizerFast.from_pretrained('roberta-large', add_prefix_space=True)
tokenizer_gemma = AutoTokenizer.from_pretrained("google/gemma-2-9b-it", use_fast=True)

max_length_test = 0
id2idx_test = {}
for i in range(len(test_json)):
    tokenized = tokenizer_roberta(test_json[i]['sentence_tokens'], return_offsets_mapping=True, is_split_into_words=True, padding=False)
    test_json[i]['tokenized'] = tokenized
    max_length_test = max(len(tokenized['input_ids']), max_length_test)
    id2idx_test[str(test_json[i]['Id'])] = i

max_length_test_bert = 0
for i in range(len(test_json)):
    tokenized = tokenizer_bert(test_json[i]['sentence_tokens'], return_offsets_mapping=True, is_split_into_words=True, padding=False)
    test_json[i]['tokenized_bert'] = tokenized
    max_length_test_bert = max(len(tokenized['input_ids']), max_length_test_bert)

ml_labels_test = 0
for i in range(len(test_json)):
    ml_labels_test = max(len(test_json[i]['labels']), ml_labels_test)

test_json[id2idx_test[str(56)]]['tokenized']['offset_mapping'][23] = (1,0)
test_json[id2idx_test[str(56)]]['tokenized']['offset_mapping'][27] = (1,0)
test_json[id2idx_test[str(56)]]['tokenized']['offset_mapping'][29] = (1,0)
test_json[id2idx_test[str(343)]]['tokenized']['offset_mapping'][25] = (1,0)
test_json[id2idx_test[str(343)]]['tokenized']['offset_mapping'][28] = (1,0)
test_json[id2idx_test[str(599)]]['tokenized']['offset_mapping'][13] = (1,0)
test_json[id2idx_test[str(599)]]['tokenized']['offset_mapping'][31] = (1,0)

system = """
        You are a strict hate speech detector.
        Hate speech is defined as the public expression of prejudice, hostility, or offensive remarks directed towards specific groups or individuals based on their identity characteristics, such as race, ethnicity, gender, or religious beliefs.
        The input is a list of words from the sentence. Your task is to assign each word a tag of either 1 or 0. Assign 1 to words that appear to be hate speech, and 0 to all other words.
        Format the result in JSON format, with the key “result” and the value as a list of dictionaries.
        Output should be in the following JSON format: {"result": [{"text": word1, "label": binary tag}, {"text": word2, "label": binary tag}, ...]}
"""

data_dict_test = {}
for data in test_json:
    #RoBERTa
    labels = [-100]
    tokenized = data['tokenized']
    att_labels = tokenized['attention_mask'].copy()
    i = -1
    att_labels[0] = 0
    for j in range(1, len(tokenized['offset_mapping'])):
        ofmap = tokenized['offset_mapping'][j]
        if ofmap==(0,0):
            labels.append(0)
            att_labels[j] = 0
        elif ofmap[0]==0:
            i+=1
            labels.append(data['labels'][i])
        else:
            labels.append(-100)
            att_labels[j] = 0
    pad_list = [0 for p in range(max_length_test-len(labels))]
    #pad_list_labels_roberta = [-100 for _ in range(max_length_test-len(labels))]
    
    #gemma
    target_sentence = data['sentence_tokens']
    user = f"The number of words in the input sentence is {len(target_sentence)}." + "\n" + f"input: {target_sentence}" + "\n"
    completion_list = [{"text": word, "label": tag} for word,tag in zip(data['sentence_tokens'], data['labels'])]
    completion = "{"+f'"result": {completion_list}'+"}"
    messages = [
        {"role":"user", "content":system + "\n" + user},
    ]
    prompt = tokenizer_gemma.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs_gemma = tokenizer_gemma(prompt, return_tensors="pt")

    #labels
    pad_list_labels = [0 for _ in range(ml_labels_test-len(data['labels']))]
    list_attention = [1 for _ in range(len(data['labels']))]

    #BERT for bert
    labels_bert = [-100]
    tokenized_bert = data['tokenized_bert']
    att_labels_bert = tokenized_bert['attention_mask'].copy()
    i = -1
    att_labels_bert[0] = 0
    for j in range(1, len(tokenized_bert['offset_mapping'])):
        ofmap = tokenized_bert['offset_mapping'][j]
        if ofmap==(0,0):
            labels_bert.append(-100)
            att_labels_bert[j] = 0
        elif ofmap[0]==0:
            i+=1
            labels_bert.append(data['labels'][i])
        else:
            labels_bert.append(-100)
            att_labels_bert[j] = 0
    pad_list_bert = [0 for p in range(max_length_test_bert-len(labels_bert))]

    #dict
    data_dict_test[str(data['Id'])] = {
        'ID':data['Id'], 
        'input_ids_roberta':torch.tensor(tokenized['input_ids']+pad_list), 
        'attention_mask_roberta':torch.tensor(tokenized['attention_mask']+pad_list),
        #'labels_roberta':torch.tensor(labels+pad_list_labels_roberta), 
        'offset_mapping_roberta':tokenized['offset_mapping'],
        'attention_labels_roberta':torch.tensor(att_labels+pad_list),
        'inputs_gemma': inputs_gemma,
        'prompt':prompt,
        'completion':completion,
        'sentence_tokens':data['sentence_tokens'], 
        'labels_bio':data['labels_bio'],
        'labels':torch.tensor(data['labels']+pad_list_labels),
        'attention_mask_labels':torch.tensor(list_attention+pad_list_labels),
        'usage':data['usage'],
        'input_ids_bert':torch.tensor(tokenized_bert['input_ids']+pad_list_bert), 
        'attention_mask_bert':torch.tensor(tokenized_bert['attention_mask']+pad_list_bert),
        'attention_labels_bert':torch.tensor(att_labels_bert+pad_list_bert),
    }

class EnsembleModel(nn.Module):
    def __init__(self):
        super().__init__()
        #RoBERTa
        self.roberta = RobertaForTokenClassification.from_pretrained(
            "./results/RoBERTa/"
        )
        self.roberta.eval()
        for param in self.roberta.parameters():
            param.requires_grad = False

        #Gemma
        model_name_gemma = "google/gemma-2-9b-it"
        self.gemma_tokenizer = AutoTokenizer.from_pretrained(model_name_gemma, use_fast=True)
        base_model = AutoModelForCausalLM.from_pretrained(model_name_gemma, device_map="auto")
        lora_dir = "./results/Gemma/"
        self.gemma = PeftModel.from_pretrained(base_model, lora_dir)
        self.gemma.eval()
        for param in self.gemma.parameters():
            param.requires_grad = False

        #BERT
        self.bert = BertForTokenClassification.from_pretrained(
            "./results/BERTbase/"
        )
        self.bert.eval()
        for param in self.bert.parameters():
            param.requires_grad = False

        #compute
        self.main_input_name = "input_ids_bert"

    def forward(
        self,
        input_ids_roberta=None,
        attention_mask_roberta=None,
        attention_labels_roberta=None,
        inputs_gemma=None,
        input_ids_bert=None,
        attention_mask_bert=None,
        attention_labels_bert=None,
        **kwargs,
    ):
        ### Batch Size must be 1, 
        
        #RoBERTa_prediction
        input_ids_roberta.to(self.roberta.device)
        attention_mask_roberta.to(self.roberta.device)
        with torch.no_grad():
            outputs_roberta = self.roberta(input_ids=input_ids_roberta, attention_mask=attention_mask_roberta)
        logits_roberta = outputs_roberta.logits.argmax(-1)
        pred_roberta = logits_roberta[0][attention_labels_roberta[0].bool()].tolist()

        #gemma_prediction
        with torch.no_grad():
            outputs_gemma = self.gemma.generate(
                **inputs_gemma,
                max_new_tokens=2048,
                do_sample=False,
                pad_token_id=self.gemma_tokenizer.eos_token_id,
            )
        output_text = self.gemma_tokenizer.decode(outputs_gemma[0], skip_special_tokens=True)
        generated = output_text.strip()
    
        matched_list = re.findall(r'\{\"result\": \[[^\]]*\]\}' ,generated)
        pred_gemma = []
        if len(matched_list)<=1:
            pass
        else:
            matched_text = matched_list[1].replace("}.", "},").replace("\\\'", "\'").replace("\'\"\'", "\'\\\"\'").replace("\'text\'", "\"text\"").replace("\'label\'", "\"label\"")
            matched_text = re.sub("\":\s?\'", r'": "', matched_text)
            matched_text = re.sub("\'[,.]\s?\"label", r'", "label', matched_text)
            matched_text = re.sub("},\s?\'([^{}[]]*\",\s?\"label\")", r'}, {"text": "\1', matched_text)
            matched_text = re.sub("\\\([^u\"])", r'\1', matched_text)
            pred_dict=None
            try:
                pred_dict = json.loads(matched_text)
            except json.JSONDecodeError:
                pass
            if pred_dict is not None:
                pred_result = pred_dict['result']
                if len(pred_result)!=len(pred_roberta):
                    pass
                else:
                    for pred in pred_result:
                        pred_gemma.append(pred['label'])

        #BERT_prediction
        input_ids_bert.to(self.bert.device)
        attention_mask_bert.to(self.bert.device)
        with torch.no_grad():
            outputs_bert = self.bert(input_ids=input_ids_bert, attention_mask=attention_mask_bert)
        logits_bert = outputs_bert.logits.argmax(-1)
        pred_bert = logits_bert[0][attention_labels_bert[0].bool()].tolist()
        
        
        #Prediction
        loss=None
        hidden_states = None
        if len(pred_gemma)==0: #JSON format error
            logits = torch.tensor(pred_bert).unsqueeze(0)
            hidden_states = torch.tensor([0 for _ in pred_bert]).unsqueeze(0)
        else:
            logits_list = []
            hidden_list = []
            for p_r,p_q,p_b in zip(pred_roberta, pred_gemma, pred_bert):
                if p_b==p_q:
                    logits_list.append(p_q)
                    hidden_list.append(0)
                else:
                    logits_list.append(p_r)
                    hidden_list.append(1)
            #make logits
            logits = torch.tensor(logits_list).unsqueeze(0)
            hidden_states = torch.tensor(hidden_list).unsqueeze(0)
        
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=None,
        )
    
data_list_test = list(data_dict_test.values())

model = EnsembleModel()

model.roberta.to("cuda:0")
model.gemma.to("cuda:0")
model.bert.to("cuda:0")
data_pred = []
for data in tqdm(data_list_test):
    input_ids_roberta=data['input_ids_roberta'].unsqueeze(0).to(model.roberta.device)
    attention_mask_roberta=data['attention_mask_roberta'].unsqueeze(0).to(model.roberta.device)
    attention_labels_roberta=data['attention_labels_roberta'].unsqueeze(0).to(model.roberta.device)
    input_ids_bert=data['input_ids_bert'].unsqueeze(0).to(model.bert.device)
    attention_mask_bert=data['attention_mask_bert'].unsqueeze(0).to(model.bert.device)
    attention_labels_bert=data['attention_labels_bert'].unsqueeze(0).to(model.bert.device)
    inputs_gemma=data['inputs_gemma'].to(model.gemma.device)
    labels=data['labels'].to(model.bert.device)
    attention_mask_labels=data['attention_mask_labels'].unsqueeze(0).to(model.bert.device)
    with torch.no_grad():
        output = model(
            input_ids_roberta=input_ids_roberta,
            attention_mask_roberta=attention_mask_roberta,
            attention_labels_roberta=attention_labels_roberta,
            inputs_gemma=inputs_gemma,
            input_ids_bert=input_ids_bert,
            attention_mask_bert=attention_mask_bert,
            attention_labels_bert=attention_labels_bert,
        )
    pred_list = output.logits[0].tolist()
    true_list = data['labels'][data['attention_mask_labels'].bool()].tolist()
    pred_words = []
    if output.hidden_states is not None:
        hidden = output.hidden_states[0].tolist()
    else:
        hidden = None
    for p,word in zip(pred_list, data['sentence_tokens']):
        if p==1:
            pred_words.append(word)
    acc = sum([int(a==b) for a,b in zip(true_list, pred_list)])/len(pred_list)
    data_pred.append({'ID':data['ID'], 'true':true_list, 'pred':pred_list, 'words':pred_words, 'accuracy':acc, 'usage':data['usage'], 'labels_bio':data['labels_bio'], 'selection':hidden})


for i in range(len(data_pred)):
    pred_bio = []
    pre_p = 0
    for p in data_pred[i]['pred']:
        if p==0:
            pred_bio.append('O')
        elif p==1:
            if pre_p == 0:
                pred_bio.append('B')
            else:
                pred_bio.append('I')
        pre_p = p
    data_pred[i]['pred_bio'] = pred_bio
    if len(pred_bio)!=len(data_pred[i]['labels_bio']):
        data_pred[i]['is_error'] = True
    else:
        data_pred[i]['is_error'] = False

#binary F1
cm_bi = [[0,0],[0,0]]
for data in data_pred:
    if not data['is_error']:
        for p,t in zip(data['pred'],data['true']):
            cm_bi[p][t] += 1
precision = cm_bi[1][1]/(cm_bi[1][1]+cm_bi[1][0])
recall = cm_bi[1][1]/(cm_bi[1][1]+cm_bi[0][1])
f1_bi = 2*precision*recall/(precision+recall)
print(f"binary F1: {f1_bi:.5f}")
print(f"binary precision: {precision:.5f}")
print(f"binary recall: {recall:.5f}")
#soft F1
cm_bio = [[0,0,0],[0,0,0],[0,0,0]]
bio2num = {'B':1,'I':2,'O':0}
for data in data_pred:
    if not data['is_error']:
        for p,t in zip(data['pred_bio'],data['labels_bio']):
            cm_bio[bio2num[p]][bio2num[t]] += 1
pre = (cm_bio[1][1]+cm_bio[2][2])/(sum(cm_bio[1])+sum(cm_bio[2]))
rec = (cm_bio[1][1]+cm_bio[2][2])/(cm_bio[0][1]+cm_bio[1][1]+cm_bio[2][1]+cm_bio[0][2]+cm_bio[1][2]+cm_bio[2][2])
f1_bio = 2*pre*rec/(pre+rec)
print(f"soft F1: {f1_bio:.5f}")
print(f"soft precision: {pre:.5f}")
print(f"soft recall: {rec:.5f}")
#hard F1
y_true = []
y_pred = []
for data in data_pred:
    if not data['is_error']:
        y_true.append(data['labels_bio'])
        y_pred.append(data['pred_bio'])
f1_span = f1_score(y_true, y_pred, average='macro')
pre_span = precision_score(y_true, y_pred, average='macro')
rec_span = recall_score(y_true, y_pred, average='macro')
print(f"hard F1: {f1_span:.5f}")
print(f"hard pre: {pre_span:.5f}")
print(f"hard rec: {rec_span:.5f}")