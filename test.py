from vaw import VAWInstanceLevelDataset
import torch
import json
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from models import mplug, minigpt, clip, mplug_itm
from dataset.utils import *


def prepare_text(texts: list, conv_temp):
   convs = [conv_temp.copy() for _ in range(len(texts))]
   [conv.append_message(
        conv.roles[0], '<Img><ImageHere></Img> {}'.format(text)) for conv, text in zip(convs, texts)]
   [conv.append_message(conv.roles[1], None) for conv in convs]
   texts = [conv.get_prompt() for conv in convs]
   return texts

def saveNpy(gt_numpy, model_numpy,pathGT,pathModel):
   np.save(pathGT,gt_numpy)
   np.save(pathModel,model_numpy)

def get_model(name, device):
   if name == "mplug":
      model, tokenizer, processor = mplug(device)
      return model, tokenizer, processor, True
   
   elif name == "mplug_itm":
      model, tokenizer, processor = mplug_itm(device, 'itm')
      return model, tokenizer, processor, False

   elif name == "mplug_itc":
      model, tokenizer, processor = mplug_itm(device, 'itc')
      return model, tokenizer, processor, False
   
   elif name == "minigpt":
      model, processor, chat_state = minigpt()
      return model, chat_state, processor, True

   elif name == "clip":
      model, processor, tokenizer = clip(device)
      return model, tokenizer, processor, False
   
def evaluate_vqa(dataloader, model, tokenizer_chat, name, device, n_question, attribute_types, attribute_parents):
   gt_answers = [None] * n_question
   pred_answers = [None] * n_question
   path2gt = "acc/"+name+"ar_gt.npy"
   path2model = "acc/"+name+"ar_model.npy"
   path2type = "acc/"+name+"ar_type.json"
   candidate = ["yes", "no"]
   attr_type_list = {
    "color" :[],
    "size" :[],
    "shape" :[],
    "material" :[],
    "texture":[],
    "state" : [],
    "action": [],
    "other": []
   }
   if name == "mplug":
      for batch in tqdm(dataloader): 
         images = batch['img'].to(device)
         attributes = batch['attr']
         question_ids = batch['question_id']
         t_answers = batch['ans']
         questions = tokenizer_chat(batch['question'], padding='longest', return_tensors='pt').to(device)
         answer_list = tokenizer_chat(candidate, padding="longest", return_tensors='pt').to(device)

         with torch.no_grad(): 
            topk_ids , topk_probs = model(images, questions, answer_list)

         for index, topk_id in enumerate(topk_ids):
            answer = candidate[topk_id[0]]
            parent = find_key_for_attribute(attribute_types, attributes[index])
            parent = find_key_for_attribute(attribute_parents, parent)
            question_id = question_ids[index]
            attr_type_list[parent].append(question_id)
            gt_answers[question_id] = t_answers[index]
            pred_answers[question_id] = answer

   elif name == "minigpt":
      for batch in tqdm(dataloader): 
         images = batch['img'].to(device)
         attributes = batch['attr']
         question_ids = batch['question_id']
         t_answers = batch['ans']

         texts = prepare_text(batch['question'], tokenizer_chat)
         length = len(texts)
         candidates = [candidate] * length 
         num_cand = [2] * length 
         candidates = [list(x) for x in zip(*candidates)]

         with torch.no_grad():
            answers = model.multi_select(images, texts, candidates, num_cand=num_cand)
         answers = [answer[0] for answer in answers] 
         for index, answer_id in enumerate(answers):
            answer = candidate[answer_id]
            parent = find_key_for_attribute(attribute_types, attributes[index])
            parent = find_key_for_attribute(attribute_parents, parent)
            question_id = question_ids[index]
            attr_type_list[parent].append(question_id)
            gt_answers[question_id] = t_answers[index]
            pred_answers[question_id] = answer

   gt_answers = [1 if i == 'yes' else 0 if i == "no" else -1 for i in gt_answers] 
   pred_answers = [1 if i == 'yes' else 0 if i == "no" else -1 for i in pred_answers] 
   gt_answers = np.array(gt_answers)
   pred_answers = np.array(pred_answers)

   saveNpy(gt_answers, pred_answers, pathGT=path2gt, pathModel=path2model)
   save_json(path2type, attr_type_list)

      
def evaluate_itm(dataloader, model, tokenizer, name, device, gt_numpy, attribute_index, model_numpy):
   path2gt = "map/"+"hier_"+name+"_gt.npy"
   path2model = "map/"+"hier_"+name+"_model.npy"
   if name == "mplug_itm":
      for batch in tqdm(dataloader): 
         images = batch['img'].to(device)
         attributes = batch['attr']
         instance_ids = batch['instance_id']
         t_answers = batch['ans']
         questions = tokenizer(batch['question'], padding='longest', return_tensors='pt').to(device)
         itm_scores = model(images, questions, train=False)
         for index, (instance_id, attr, t_answer) in enumerate(zip(instance_ids, attributes, t_answers)):
            gt_numpy[instance_id][attribute_index[attr]] = 1 if t_answer=='yes' else 0
            model_numpy[instance_id][attribute_index[attr]] = itm_scores[index][1].item()

   saveNpy(gt_numpy, model_numpy, pathGT=path2gt, pathModel=path2model)


def evaluate_itc(dataloader, model, tokenizer, name, device, gt_numpy, attribute_index, model_numpy):
   path2gt = "map/"+"hier_"+name+"_gt.npy"
   path2model = "map/"+"hier_"+name+"_model.npy"
   if name == "clip":
      for batch in tqdm(dataloader): 
         images = batch['img'].to(device)
         attributes = batch['attr']
         instance_ids = batch['instance_id']
         t_answers = batch['ans']
         texts = tokenizer(batch['question']).to(device)
         # print(batch['question'][0])
         with torch.no_grad():
            logits_per_image, _ = model(images, texts)
            scores = torch.sigmoid(logits_per_image/100)
            for index, (instance_id, attr, t_answer) in enumerate(zip(instance_ids, attributes, t_answers)):
               gt_numpy[instance_id][attribute_index[attr]] = 1 if t_answer=='yes' else 0
               model_numpy[instance_id][attribute_index[attr]] = scores[index][index].item()

   elif name == "mplug_itc":
      for batch in tqdm(dataloader): 
         images = batch['img'].to(device)
         attributes = batch['attr']
         instance_ids = batch['instance_id']
         t_answers = batch['ans']
         questions = tokenizer(batch['question'], padding='longest', return_tensors='pt').to(device)
         itc_scores = model(images, questions, train=False)
         itc_scores = torch.sigmoid(itc_scores)
         for index, (instance_id, attr, t_answer) in enumerate(zip(instance_ids, attributes, t_answers)):
               gt_numpy[instance_id][attribute_index[attr]] = 1 if t_answer=='yes' else 0
               model_numpy[instance_id][attribute_index[attr]] = itc_scores[index][index].item()

   saveNpy(gt_numpy, model_numpy, pathGT=path2gt, pathModel=path2model)


def main():
   batch_size = 64
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   model_task = {"mplug": "vqa",
                 "mplug_itc": "itc",
                 "mplug_itm": "itm",
                 "minigpt": "vqa",
                 "clip": "itc"}
   
   model_name = "minigpt"
   task = model_task[model_name]

   print(f"Inferencing with: {model_name}......")

   model, tokenizer_chat, processor, VQA = get_model(model_name, device) 
   vaw_dataset = VAWInstanceLevelDataset("path_to_images","path_to_annotations","test", transform=processor, VQA=VQA)
   print("-----------------------------------")
   print(vaw_dataset[0]['question'])
   print("Now we are using the standard template to conduct ITM and ITC inference, please modify VAWInstanceLevalDataset if using other prompts")
   print("-----------------------------------")
   attribute_types = vaw_dataset.attribute_types
   attribute_parents = vaw_dataset.attribute_parents
   attribute_index = vaw_dataset.attr2idx

   vaw_dataloader = DataLoader(dataset=vaw_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=vaw_dataset.mplug_collate_fn)

   n_question = len(vaw_dataset) 
   print(f" the number of questions: {n_question}")

   dataset_length = vaw_dataset.instance_id 
   gt_numpy = np.full((dataset_length,620), 2).astype(np.float32) 
   model_numpy = np.full((dataset_length,620), 2).astype(np.float32) 
   print(f" dataset_length : {dataset_length}")

   # vqa
   if task == "vqa":
      evaluate_vqa(vaw_dataloader, model, tokenizer_chat, model_name, device, n_question, attribute_types, attribute_parents)
   # itc
   elif task == "itc":
      evaluate_itc(vaw_dataloader, model, tokenizer_chat, model_name, device, gt_numpy, attribute_index, model_numpy)
   # itm
   elif task == "itm":
      evaluate_itm(vaw_dataloader, model, tokenizer_chat, model_name, device, gt_numpy, attribute_index, model_numpy)
   
   print("Saved successfully!")


if __name__ == "__main__":
   main()


