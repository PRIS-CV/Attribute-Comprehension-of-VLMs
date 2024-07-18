import logging
import torch
import copy
import os.path as op
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import datasets as datasets
from dataset import ALDataset
from dataset.utils import *


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class VAWInstanceLevelDataset(ALDataset):
    
    def __init__(self, image_path, anno_path, mode, transform=None, VQA=True, *args, **kwargs):
        
        super().__init__()
        self.image_path = image_path
        self.anno_path = anno_path
        self.transform = transform
        self.annos = [] 
        self.instance_id = 0 
        self.VQA = VQA
    
        assert mode in ['train', 'test', 'val'], 'Dataset only supports train, test, val mode.'
        self.mode = mode
        
        self.annosOld = load_json(op.join(anno_path, f'{mode}.json'))
        self.attr2idx = load_json(op.join(anno_path, 'attribute_index.json')) # attribute: id
        self.idx2attr = list(self.attr2idx.keys())   # attribute
        self.attribute_types = load_json(op.join(anno_path, 'attribute_types.json')) # parent: children
        self.attribute_parents = load_json(op.join(anno_path, 'attribute_parent_types.json')) 
        self.question = load_json(op.join(anno_path, 'question.json')) # question template
        self.prompt = load_json(op.join(anno_path, 'ITM.json')) # for ITM and ITC
        self.matrix = np.load(op.join(self.anno_path, "adjacency_matrix.npy")) # [620, 620]

        # standard prompt for clip
        self.template1 = 'The object is {attribute}.'
        self.template2 = 'The {object} is {attribute}.'
        self.template3 = 'The {type} of the object is {attribute}.'
        

        if op.exists(op.join(anno_path, 'object_index.json')):
            self.obj2idx = load_json(op.join(anno_path, 'object_index.json')) # object: id
        else:
            self.obj2idx = self.generate_object_index(anno_path)

        self.idx2obj = list(self.obj2idx.keys())    # object

        for anno in self.annosOld:
            path = op.join(self.image_path, f"{anno['image_id']}.jpg")
            pos_attr = anno['positive_attributes']
            neg_attr = anno['negative_attributes']
            object_name = anno['object_name']
            bbox = anno['instance_bbox'] # [x1, y1, w, h]
            bbox = xywh_to_xyxy(bbox) # [x1, y1, x2, y2]
            bbox = [list(map(int, bbox))]
            
            # pos_parents = self.get_questions(pos_attr, True)
            # neg_children = self.get_questions(neg_attr, False)

            # pos_attr = list(set(pos_attr + pos_parents))
            # neg_attr = list(set(neg_attr + neg_children))

            all_attributes = pos_attr + neg_attr
            other_attribute = [] 
            q_b = {} 
            for attribute in all_attributes:
                parent_key = find_key_for_attribute(self.attribute_types, attribute) 
                attr_question = self.question.get(parent_key) if self.VQA else self.prompt.get(parent_key) 
                full_question = attr_question[0].format(object=object_name, attribute=attribute) 
                
                # full_question = self.template3.format(type= parent_key, attribute=attribute)
                q_b[attribute] = full_question

            pos_attr = [x for x in pos_attr if x not in other_attribute]
            neg_attr = [x for x in neg_attr if x not in other_attribute]
                
            for pos in pos_attr: 
                attr_dict = {}
                attr_dict['attr'] = pos
                attr_dict['ans'] = "yes"
                attr_dict['img'] = path
                attr_dict['obj_name'] =  object_name
                attr_dict['instance_id'] = self.instance_id
                attr_dict['instance_bbox'] = bbox
                attr_dict['question'] = q_b[pos]
                attr_dict['question_id'] = len(self.annos)
                attr_dict['img_id'] = anno['image_id']
                self.annos.append(attr_dict)
            for neg in neg_attr: 
                attr_dict = {}
                attr_dict['attr'] = neg
                attr_dict['ans'] = "no"
                attr_dict['img'] = path
                attr_dict['obj_name'] =  object_name
                attr_dict['instance_id'] = self.instance_id
                attr_dict['instance_bbox'] = bbox
                attr_dict['question'] = q_b[neg]
                attr_dict['question_id'] = len(self.annos)
                attr_dict['img_id'] = anno['image_id']
                self.annos.append(attr_dict)

            self.instance_id += 1
              
    @staticmethod
    def generate_object_index(anno_path: str, save: bool = True) -> dict:
        object_list = []
        logger.info("Generating Object Index ..")        
        from tqdm import tqdm
        annos = load_json(op.join(anno_path, "train_part1.json"))
        annos += load_json(op.join(anno_path, "train_part2.json"))
        annos += load_json(op.join(anno_path, "val.json"))
        annos += load_json(op.join(anno_path, "test.json"))
        for anno in tqdm(annos):
            if anno['object_name'] not in object_list:
                object_list.append(anno['object_name'])
        
        obj2idx = {o: i for i, o in enumerate(object_list)}
        if save: 
            save_json(op.join(anno_path, 'object_index.json'), obj2idx)
        return obj2idx
    

    @property
    def n_objects(self) -> int:
        return len(self.obj2idx)
    
    @property
    def n_attributes(self) -> int:
        return len(self.attr2idx)
    
    @property
    def attribute_names(self) -> list:
        return list(self.attr2idx.keys())
    
    @property
    def object_names(self) -> list:
        return list(self.obj2idx.keys())

    def get_instance_ids_by_object(self, object_name: str) -> list:
        instance_ids = []
        for i, anno in enumerate(self.annos):
            if anno['object_name'] == object_name:
                instance_ids.append(i)
        return instance_ids

    def get_instance_ids_by_attribute(self, attribute_name: str) -> list:
        instance_ids = []
        for i, anno in enumerate(self.annos):
            if attribute_name in anno['positive_attributes']:
                instance_ids.append(i)
        return instance_ids

    def encode_attr(self, attr) -> int:
        return self.attr2idx[attr]
    
    def encode_obj(self, obj) -> int:
        return self.obj2idx[obj]

    def decode_attr(self, idx) -> str:
        return self.idx2attr[idx]
    
    def decode_obj(self, idx) -> str:
        return self.idx2obj[idx]

    def get_anno_by_index(self, idx=None) -> dict:
        return self.annos[idx]

    def __len__(self) -> int:
        return len(self.annos)

    def get_image_by_index(self, index, transform=None) -> Image.Image:
        anno = self.annos[index]
        path = op.join(self.image_path, f"{anno['image_id']}.jpg")
        image = Image.open(path).convert('RGB')
        bbox = anno['instance_bbox']
        bbox = xywh_to_xyxy(bbox)
        bbox = [list(map(int, bbox))]

        if transform is not None:
            image, bbox, _ = transform(image, bbox, None)
        else:
            image, bbox, _ = self.transform(image, bbox, None)

        return image    
    
    def get_questions(self, attr_list, find_parent:bool):
        parent_attributes = []
        for attr in attr_list:
            attribute_id = self.attr2idx[attr]
            parent_index = self.matrix[:, attribute_id] if find_parent else self.matrix[attribute_id, :]
            parent_index = np.where(parent_index==1)[0] 
            parent_index_new = parent_index[parent_index != attribute_id] 
            if len(parent_index_new) > 0:
                for id in parent_index_new:
                    parent = self.idx2attr[id]
                    parent_attributes.append(parent)
        return parent_attributes

    def collate_fn(self,batch):
    
        batch_questions = []
        all_attr = []
        image_tensors = []
        gt_ans = []
        obj_name = []
        question_id = []
        instance_id = []
        for i, instance in enumerate(batch):
            batch_questions.append(instance['question'])
            gt_ans.append(instance['ans'])
            all_attr.append(instance['attr'])
            image_tensors.append(instance['img'])
            obj_name.append(instance['obj_name'])
            question_id.append(instance['question_id'])
            instance_id.append(instance['instance_id'])
        merged_images = torch.cat(image_tensors, dim=0)
        batch_instance = {
        'img' : merged_images,
        'attr':all_attr,
        'question':batch_questions,
        'ans':gt_ans,
        'obj_name':obj_name,
        'question_id' : question_id,
        'instance_id':instance_id
        }
        return batch_instance    
    
    def mplug_collate_fn(self,batch):
    
        questions = []
        all_attr = []
        image_tensors = []
        gt_ans = []
        obj_name = []
        question_id = []
        instance_id = []
        for instance in batch:
            questions.append(instance['question'])
            gt_ans.append(instance['ans'])
            all_attr.append(instance['attr'])
            image_tensors.append(instance['img'])
            obj_name.append(instance['obj_name'])
            question_id.append(instance['question_id'])
            instance_id.append(instance['instance_id'])

        merged_images = torch.stack(image_tensors, dim=0) 

        batch_instance = {
        'img' : merged_images,
        'attr':all_attr,
        'question':questions,
        'ans':gt_ans,
        'obj_name':obj_name,
        'question_id' : question_id,
        'instance_id':instance_id
        }
        return batch_instance

    def __getitem__(self, index) -> dict:
        
        anno = copy.deepcopy(self.annos[index])
        object_name = anno['obj_name']
        bbox = anno['instance_bbox'] 

        object_index = self.encode_obj(object_name)
        object_index = torch.tensor(object_index, dtype=torch.long)
        img = Image.open(anno['img']).convert('RGB')
        img = img.crop(bbox[0])

        if self.transform is not None:
            img = self.transform(img)  
        anno['img'] = img

        return anno
    

class VAWRelationshipDataset(ALDataset):

    def __init__(self, image_path, anno_path, mode, transform=None):
        super().__init__()

        self.image_path = image_path
        self.anno_path = anno_path
        self.transform = transform

        self.matrix = np.load(op.join(self.anno_path, "adjacency_matrix.npy")) # [620, 620]
        self.attribute_types = load_json(op.join(self.anno_path, "attribute_types.json"))
        self.attr2ind = load_json(op.join(self.anno_path, "attribute_index.json"))
        self.id2attr = {v:k for k,v in self.attr2ind.items()}
        self.question = load_json(op.join(self.anno_path, "question.json"))

        assert mode in ['train', 'test', 'val'], 'Dataset only supports train, test, val mode.'
        
        self.data = load_json(op.join(anno_path, f'{mode}.json'))

        self.annotations = [] 
        self.pos_parents = {} # B+
        self.neg_children = {} # A-
        self.instance_id = 0
        for anno in self.data:
            path = op.join(self.image_path, f"{anno['image_id']}.jpg")
            pos_attr = anno['positive_attributes']
            neg_attr = anno['negative_attributes']
            object_name = anno['object_name']
            bbox = anno['instance_bbox'] # [x1, y1, w, h]
            bbox = xywh_to_xyxy(bbox) # [x1, y1, x2, y2]
            bbox = [list(map(int, bbox))]
            image_id = anno['image_id']

            
            self.get_questions(pos_attr, path, object_name, bbox, image_id, find_parent=True) 
            self.get_questions(neg_attr, path, object_name, bbox, image_id, find_parent=False) 

            self.instance_id += 1
        
    def get_questions(self, attr_list, img_path, obj_name, bbox, image_id, find_parent:bool):
        gt = 'yes' if find_parent else 'no'
        for attr in attr_list:
            attribute_id = self.attr2ind[attr]
            parent_index = self.matrix[:, attribute_id] if find_parent else self.matrix[attribute_id, :]
            parent_index = np.where(parent_index==1)[0] 
            parent_index_new = parent_index[parent_index != attribute_id] 
            if len(parent_index_new) == 0:
                break
            else:
                find_data = self.pos_parents if find_parent else self.neg_children
                if find_data.get(attr, None) == None:
                    find_data[attr] = {}
                # 添加该属性对应问题到列表，同时获取该属性的question id
                attr_question = self.question.get(find_key_for_attribute(self.attribute_types, attr))
                full_attr_question = attr_question[0].format(object=obj_name, attribute=attr) 
                question_id = len(self.annotations)
                self.annotations.append({
                    'attr': attr,
                    'ans': gt,
                    'img': img_path,
                    'obj_name': obj_name,
                    'instance_id': self.instance_id,
                    'instance_bbox': bbox,
                    'question': full_attr_question,
                    'question_id': question_id,
                    'img_id': image_id
                })
                find_data[attr][question_id] = []
                for id in parent_index_new:
                    parent = self.id2attr[id]
                    attr_question = self.question.get(find_key_for_attribute(self.attribute_types, parent))
                    full_question = attr_question[0].format(object=obj_name, attribute=parent) 
                    new_ques_id = len(self.annotations)
                    self.annotations.append({
                        'attr': parent,
                        'ans': gt,
                        'img': img_path,
                        'obj_name': obj_name,
                        'instance_id': self.instance_id,
                        'instance_bbox': bbox,
                        'question': full_question,
                        'question_id': new_ques_id,
                        'img_id': image_id
                    })
                    find_data[attr][question_id].append(new_ques_id)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index) -> dict:
        
        anno = copy.deepcopy(self.annotations[index])
        bbox = anno['instance_bbox'] # [[2, 248, 496, 381]] , xyxy
        img = Image.open(anno['img']).convert('RGB')
        img = img.crop(bbox[0])

        if self.transform is not None:
            img = self.transform(img)  
        anno['img'] = img

        return anno
    
    def mplug_collate_fn(self,batch):
    
        questions = []
        all_attr = []
        image_tensors = []
        gt_ans = []
        obj_name = []
        question_id = []
        instance_id = []
        for instance in batch:
            questions.append(instance['question'])
            gt_ans.append(instance['ans'])
            all_attr.append(instance['attr'])
            image_tensors.append(instance['img'])
            obj_name.append(instance['obj_name'])
            question_id.append(instance['question_id'])
            instance_id.append(instance['instance_id'])

        merged_images = torch.stack(image_tensors, dim=0) 

        batch_instance = {
        'img' : merged_images,
        'attr':all_attr,
        'question':questions,
        'ans':gt_ans,
        'obj_name':obj_name,
        'question_id' : question_id,
        'instance_id':instance_id
        }
        return batch_instance
    

class VAWImageLevelDataset(VAWInstanceLevelDataset):
    
    def __init__(self, image_path, anno_path, mode, transform=None, *args, **kwargs):
        super().__init__(image_path, anno_path, mode, transform)
        self._trans_instances_to_image()
        

    def _trans_instances_to_image(self, store=True):
        anno_path = op.join(self.anno_path, f"{self.mode}_image_level.json")
        if op.exists(anno_path):
            self.annos = load_json(anno_path)
        else:
            logger.info("Collecting Image Level Annotation From Instance Level Annotation ...")
            image_level_annos = {}
            
            for anno in tqdm(self.annos):
                
                instance = anno
                image_id = anno["image_id"]
                del instance["image_id"]
                
                try:
                    image_anno = image_level_annos[image_id]
                except:
                    image_anno = {
                        "image_id": image_id,
                        "bboxes": [], 
                        "objects": [], 
                        "polygons": [], 
                        "positive_attributes": [], 
                        "negative_attributes": []
                    }
                finally:
                    image_anno["bboxes"].append(instance["instance_bbox"])
                    image_anno["objects"].append(instance["object_name"])
                    image_anno["polygons"].append(instance["instance_polygon"])
                    image_anno["positive_attributes"].append(instance["positive_attributes"])
                    image_anno["negative_attributes"].append(instance["negative_attributes"])
                    image_level_annos[image_id] = image_anno
            
            self.annos = list(image_level_annos.values())
            if store:
                logger.info(f"Saving Image Level Annotation to {anno_path} ...")
                save_json(anno_path, self.annos)
    
    def __getitem__(self, index):
        image_anno = self.annos[index]
        objects = []
        bboxes = []
        targets = []

        path = op.join(self.image_path, f"{image_anno['image_id']}.jpg")
        image = Image.open(path).convert('RGB')

        for i, o in enumerate(image_anno["objects"]):

            object_index = self.encode_obj(o)
            objects.append(object_index)

            pos_attr = image_anno['positive_attributes'][i]
            neg_attr = image_anno['negative_attributes'][i]
            target = torch.zeros((len(self.idx2attr)), dtype=torch.float).fill_(2)
            for a in pos_attr:
                target[self.encode_attr(a)] = 1
            for a in neg_attr:
                target[self.encode_attr(a)] = 0
            targets.append(target.unsqueeze(0))
            w, h = image.size

            bbox = image_anno['bboxes'][i]
            bbox = xywh_to_xyxy(bbox)                 # [x, y, w, h] -> [x1, y1, x2, y2]
            bbox = list(map(int, bbox))
            bboxes.append(bbox)

        if self.transform is not None:
            image, bboxes, _ = self.transform(image, bboxes, None)
        
        if self.return_obj_name:
            objects = image_anno["objects"]
        else:
            objects = torch.tensor(objects, dtype=torch.long)

        targets = torch.cat(targets, dim=0)
        
        image_data = {
            'i': image,
            'o': objects,
            'b': bboxes,
            't': targets
        }
        
        return image_data

    @staticmethod
    def collate_fn(batch):
        images = torch.cat([item['i'].unsqueeze(0) for item in batch], dim=0)
        bboxes = [item['b'] for item in batch]
        targets = torch.cat([item['t'] for item in batch], dim=0)
        objects = torch.cat([item['o'] for item in batch], dim=0)
            
        return {
            'i': images,
            'o': objects,
            'b': bboxes,
            't': targets
        }
