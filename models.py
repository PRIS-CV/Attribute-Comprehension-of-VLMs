import torch
import os.path as osp
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

def visual_processor(config):
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)

        return transforms.Compose([
                transforms.Resize((config.image_res, config.image_res),
                                  interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

def mplug(device, rank=True):
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    from modelscope.models.multi_modal.mplug import CONFIG_NAME, MPlugConfig

    model_id = 'path_to_weight' # 'damo/mplug_visual-question-answering_coco_large_en'
    pipeline_vqa = pipeline(Tasks.visual_question_answering, model=model_id)
    model_dir = pipeline_vqa.model.model_dir
    model = pipeline_vqa.model.model.to(device) # model = pipeline_vqa.model.model.to(device)
    tokenizer = model.tokenizer
    

    config = MPlugConfig.from_yaml_file(
                osp.join(model_dir, CONFIG_NAME))

    def inference(image, question, answer, alpha=0, k=config.k_test):
        image = image.to(dtype=next(model.parameters()).dtype)
        image_embeds = model.visual_encoder.visual(image, skip_last_layer=True) # [128, 1297, 1024]
        image_embeds = model.dropout(
            model.visn_layer_norm(model.visn_fc(image_embeds))) # [128, 1297, 768]
        image_atts = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long).to(image.device) # [128, 1297]
        # inference
        text_output = model.text_encoder(
                question.input_ids,
                attention_mask=question.attention_mask,
                return_dict=True)   # transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions
        text_embeds = text_output.last_hidden_state # [128, 13, 768]
        fusion_output = model.fusion_encoder(
            encoder_embeds=text_embeds,
            attention_mask=question.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False)
        image_output, question_output = fusion_output  # [128, 1297, 768], [128, 13, 768]
        question_output = torch.cat([image_output, question_output], 1)
        merge_text_attention = torch.cat(
            [image_atts, question.attention_mask], 1)

        # choose from candidate list
        topk_ids, topk_probs = rank_answer(question_output, merge_text_attention, answer.input_ids, answer.attention_mask, min(k, len(answer.input_ids)))

        return topk_ids, topk_probs


    def rank_answer(question_states, question_atts, answer_ids, answer_atts, k):
        
        num_ques = question_states.size(0)
        start_ids = answer_ids[0,0].repeat(num_ques,1) # bos token  [[101]]
        
        start_output = model.text_decoder(start_ids, 
                                         encoder_hidden_states = question_states,
                                         encoder_attention_mask = question_atts,                                      
                                         return_dict = True,
                                         reduction = 'none')           
        logits = start_output.logits[:,0,:] # first token's logit
        
        # topk_probs: top-k probability 
        # topk_ids: [num_question, k]        
        answer_first_token = answer_ids[:,1]
        prob_first_token = F.softmax(logits,dim=1).index_select(dim=1, index=answer_first_token) 
        topk_probs, topk_ids = prob_first_token.topk(k,dim=1) 
        
        # answer input: [num_question*k, answer_len]                 
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids,dim=0)  
        input_atts = torch.cat(input_atts,dim=0)  

        targets_ids = input_ids.masked_fill(input_ids == model.tokenizer.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)
        
        output = model.text_decoder(input_ids, 
                                   attention_mask = input_atts, 
                                   encoder_hidden_states = question_states,
                                   encoder_attention_mask = question_atts,     
                                   labels = targets_ids,
                                   return_dict = True, 
                                   reduction = 'none')                 

        answer_loss = output.loss 
        answer_loss = answer_loss.view(input_ids.size(0),-1)
        
        # topk_prob: first token probability
        topk_probs = topk_probs.view(-1,1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss],dim=1)

        # re-calculate log probabilities for the answer sequences using chain rule
        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques,k)

        topk_probs = F.softmax(log_probs_sum, dim=-1)
        # get top-k after re-ranking
        topk_probs, rerank_id = topk_probs.topk(k,dim=1) 
        topk_ids = torch.gather(topk_ids, 1, rerank_id)    

        return topk_ids, topk_probs 

    def tile(x, dim, n_tile):
        init_dim = x.size(dim)
        repeat_idx = [1] * x.dim()
        repeat_idx[dim] = n_tile
        x = x.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        return torch.index_select(x, dim, order_index.to(x.device)) 
    
    visual_preprocessor = visual_processor(config)

    # question = tokenizer(
    #         question.lower(),
    #         padding='max_length',
    #         truncation=True,
    #         max_length=25,
    #         return_tensors='pt')
    if rank:
        return inference, tokenizer, visual_preprocessor
    else:
        return model, tokenizer, visual_preprocessor

def mplug_itm(device, mode='itm'):
    # mplug itm
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    from modelscope.models.multi_modal.mplug import CONFIG_NAME, MPlugConfig

    model = '/data/zhanghaiwen/weight/mplug_retrieval' #'damo/mplug_image-text-retrieval_flickr30k_large_en'
    pipeline_retrieval = pipeline(Tasks.image_text_retrieval, model=model)

    model_dir = pipeline_retrieval.model.model_dir
    model = pipeline_retrieval.model.model.to(device)
    tokenizer = model.tokenizer

    config = MPlugConfig.from_yaml_file(
                osp.join(model_dir, CONFIG_NAME))
    
    visual_preprocessor = visual_processor(config)

    if mode == 'itm':
        return model, tokenizer, visual_preprocessor
    elif mode == 'itc':
        def mplug_itc(image, text, train=False):
            text_output = model.text_encoder(text.input_ids, attention_mask=text.attention_mask)
            text_feat = text_output.last_hidden_state # [7, 19, 768]
            text_embed = F.normalize(model.text_proj(text_feat[:, 0, :])) # [7, 256]

            image_feat = model.visual_encoder.visual(image, skip_last_layer=True)
            image_feat = model.visn_layer_norm(model.visn_fc(image_feat)) # [7, 577, 768]
            image_embed = model.vision_proj(image_feat[:, 0, :]) # [7, 256]
            image_embed = F.normalize(image_embed, dim=-1) # [7, 256]

            sims_matrix = image_embed @ text_embed.t()

            return sims_matrix

        return mplug_itc, tokenizer, visual_preprocessor
        

def minigpt():
    import sys
    import argparse
    from transformers import StoppingCriteriaList

    sys.path.append("/home/zhanghaiwen/MiniGPT-4")
    
    from minigpt4.common.config import Config
    from minigpt4.common.registry import registry
    from minigpt4.conversation.conversation import CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

    # imports modules for registration
    # from minigpt4.datasets.builders import *
    # from minigpt4.models import *
    # from minigpt4.processors import *
    # from minigpt4.runners import *
    # from minigpt4.tasks import *

    def parse_args():
        parser = argparse.ArgumentParser(description="MiniGPT4")
        parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
        parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
        parser.add_argument(
            "--options",
            nargs="+",
            help="override some settings in the used config, the key-value pair "
            "in xxx=yyy format will be merged into config file (deprecate), "
            "change to --cfg-options instead.",
        )
        args = parser.parse_args()
        return args

    conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
                'pretrain_llama2': CONV_VISION_LLama2}

    print('Initializing Model')
    args = parse_args()
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id)) # 创建了一个MiniGPT4对象

    CONV_VISION = conv_dict[model_config.model_type]
    chat_state = CONV_VISION.copy()

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    # stop_words_ids = [[835], [2277, 29937]]
    

    return model.eval(), vis_processor, chat_state


def clip(device):
    import clip
    model, preprocess = clip.load("ViT-B/32", device=device)
    tokenize = clip.tokenize

    return model.eval(), preprocess, tokenize

def getAttMap(img, attMap, blur = True, overlap = True):
    from skimage import transform as skimage_transform
    from scipy.ndimage import filters
    from matplotlib import pyplot as plt

    attMap -= attMap.min()
    if attMap.max() > 0:
        attMap /= attMap.max()
    attMap = skimage_transform.resize(attMap, (img.shape[:2]), order = 3, mode = 'constant')
    if blur:
        attMap = filters.gaussian_filter(attMap, 0.02*max(img.shape[:2]))
        attMap -= attMap.min()
        attMap /= attMap.max()
    cmap = plt.get_cmap('jet')
    attMapV = cmap(attMap)
    attMapV = np.delete(attMapV, 3, 2)
    if overlap:
        attMap = 1*(1-attMap**0.7).reshape(attMap.shape + (1,))*img + (attMap**0.7).reshape(attMap.shape+(1,)) * attMapV
    return attMap
        
if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    ## clip---------------------------------------
    # model, vis_processor, tokenize = clip(device)
    # image = Image.open("/home/zhanghaiwen/merlion.png").convert('RGB')
    # image = vis_processor(image)
    # image = torch.stack([image, image], dim=0).to(device)
    # text = tokenize(["Sydney", "London", "Singapore"]).to(device)
    # logits_per_image, logits_per_text = model(image, text)
    # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    # print(probs)
    # print(logits_per_image)

    # # minigpt-------------------------------------
    # model, vis_processor, chat_state = minigpt()
    # image = Image.open("merlion.png").convert('RGB')
    # image = vis_processor(image).unsqueeze(0) # model.to(torch.float16)
    # question = "Is the sky blue?"
    # chat_state.append_message(chat_state.roles[0], '<Img><ImageHere></Img> {}'.format(question))
    # chat_state.append_message(chat_state.roles[1], None)
    # text = [chat_state.get_prompt()]
    # answer = model.multi_select(image.to(device), text, [['yes'], ['no']])
    # print(answer.lower())
    # answer = model.generate(image, text, max_new_tokens=1) 
    # print(answer[0].lower())

    # minigpt-itc-------------------------------------
    model, vis_processor, chat_state = minigpt()
    image = Image.open("merlion.png").convert('RGB')
    image = vis_processor(image).unsqueeze(0) # model.to(torch.float16)
    question = ['Singapore', 'London', 'Tokyo', 'Shanghai']
    sim = model.itc(image.to(device), question)
    print(sim)
