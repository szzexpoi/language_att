import warnings
warnings.filterwarnings('ignore')

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch
from glob import glob
import os
import json
import pickle
import numpy as np

# require attention mapping or not
require_att = True

# for SALICON/MIT (full description and object-only)
visual_token_start, visual_token_end = 46, 2386

patch_h, patch_w = 36, 49 # for salicon 480x640
img_size = 24

save_dir = None # Directory for saving the data
os.makedirs(save_dir, exist_ok=True)

img_dir = None # Directory for the image data (e.g., SALICON)
split = ['train', 'val']

pretrained = "lmms-lab/llama3-llava-next-8b"
model_name = "llava_llama3"


device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map,
                                                attn_implementation=None) # Add any other thing you want to pass in llava_model_args
model.eval()
model.tie_weights()

conv_template = "llava_llama_3" # Make sure you use correct chat template for different models
# conv_template = "vicuna_v1"

# full description v1
question = DEFAULT_IMAGE_TOKEN + "\nCan you describe the image in detail? The description should capture objects, their attributes, relative positions and relations. Limit to 15 words."

# # emotional attention
# question = DEFAULT_IMAGE_TOKEN + "\nCan you describe the image in detail? The description should capture the sentiments of different objects, together with their other attributes. Limit to 50 words."

conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

def token_matching(raw_token, merged_token):
    raw_counter = 0
    merged_counter = 0
    mapping = dict()
    merged_token = merged_token.split(' ')
    # take into account symbols
    processed_merged = []
    for cur_token in merged_token:
        tmp = []
        flag = False
        for symbol in [',', '.', '?', ';']:
            if symbol in cur_token:
                for cur in cur_token.split(symbol):
                    if cur == '':
                        tmp.append(symbol)
                    else:
                        tmp.append(cur)
                flag = True
                break
        if not flag:
            processed_merged.extend([cur_token])
        else:
            processed_merged.extend(tmp)
    merged_token = processed_merged

    # ignore the EOS
    while raw_counter<len(raw_token)-1:
        if raw_token[raw_counter] == merged_token[merged_counter]:
            mapping[raw_counter] = merged_counter
            raw_counter += 1
            merged_counter += 1
        else:
            flag = True
            tmp = ''
            while flag:
                tmp += raw_token[raw_counter]
                mapping[raw_counter] = merged_counter
                raw_counter += 1
                if tmp == merged_token[merged_counter]:
                    merged_counter += 1
                    flag = False
    
    return mapping

result = dict()
with torch.no_grad():
    for cur_split in split:
        if cur_split is not None:
            os.makedirs(os.path.join(save_dir, 'attention', cur_split), exist_ok=True)
            os.makedirs(os.path.join(save_dir, 'hidden_state', cur_split), exist_ok=True)
            result[cur_split] = dict()
            all_images = glob(os.path.join(img_dir, cur_split, '*.jpg'))
            # all_images = glob(os.path.join(img_dir, cur_split, '*.jpeg'))
        else:
            os.makedirs(os.path.join(save_dir, 'attention'), exist_ok=True)
            os.makedirs(os.path.join(save_dir, 'hidden_state'), exist_ok=True)            
            all_images = glob(os.path.join(img_dir, '*.jpg'))

        for check_idx, cur_img in enumerate(all_images):
            img_id = os.path.basename(cur_img)[:-4]
            # img_id = os.path.basename(cur_img)[:-5]

            image = Image.open(cur_img)
            image_tensor = process_images([image], image_processor, model.config)
            image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

            image_sizes = [image.size]

            cont = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=256,
            )

            text_outputs = tokenizer.batch_decode(cont['sequences'][0, 1:].unsqueeze(0), skip_special_tokens=True)[0]
            raw_token = [cur.replace('Ġ', '').replace('Ċ', '\n') for cur in tokenizer.convert_ids_to_tokens(cont['sequences'][0, 1:])]

            if cur_split is not None:
                result[cur_split][img_id] = dict()
                result[cur_split][img_id]['merged_token'] = text_outputs
                result[cur_split][img_id]['raw_token'] = raw_token
                # result[cur_split][img_id]['mapping'] = mapping
            else:
                result[img_id] = dict()
                result[img_id]['merged_token'] = text_outputs
                result[img_id]['raw_token'] = raw_token
                # result[cur_split][img_id]['mapping'] = mapping

            # putting things into cpu and pickle
            cur_att = []
            cur_hidden = []

            
            # ignore the first token for '\n', which is very big
            for token_idx in range(1, len(cont['attentions'])):
                # use the last attention for now
                cur_att.append(cont['attentions'][token_idx][-1].data.cpu().numpy())
                cur_hidden.append(cont['hidden_states'][token_idx][-1].data.cpu().numpy())

            if require_att:
                processed_att = {'base': [], 'patch': []}
                processed_hidden = []
                
                for i in range(len(cur_att)):
                    tmp_att = cur_att[i].squeeze()
                    base_att = tmp_att[:, visual_token_start:visual_token_start+img_size**2]
                    patch_att = tmp_att[:, visual_token_start+img_size**2:visual_token_end]
                    patch_att = patch_att.reshape([len(patch_att), patch_h, patch_w])[:, :, :-1]
                    processed_att['base'].append(base_att.reshape([len(base_att), img_size, img_size]))
                    processed_att['patch'].append(patch_att)
                    processed_hidden.append(cur_hidden[i].squeeze())

                processed_att['base'] = np.array(processed_att['base'])
                processed_att['patch'] = np.array(processed_att['patch'])
                processed_hidden = np.array(processed_hidden)

                if cur_split is not None:
                    np.save(os.path.join(save_dir, 'attention', cur_split, str(img_id)), processed_att)
                    np.save(os.path.join(save_dir, 'hidden_state', cur_split, str(img_id)), processed_hidden)
                else:
                    np.save(os.path.join(save_dir, 'attention', str(img_id)), processed_att)
                    np.save(os.path.join(save_dir, 'hidden_state', str(img_id)), processed_hidden)


                del cont
            else:
                processed_hidden = []
                for i in range(len(cur_hidden)):
                    processed_hidden.append(cur_hidden[i].squeeze())
                processed_hidden = np.array(processed_hidden)

                if cur_split is not None:
                    np.save(os.path.join(save_dir, 'hidden_state', cur_split, str(img_id)), processed_hidden)
                else:
                    np.save(os.path.join(save_dir, 'hidden_state', str(img_id)), processed_hidden)
                del cont

with open(os.path.join(save_dir, 'caption.json'), 'w') as f:
    json.dump(result, f)
