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
import json

# require attention mapping or not
require_att = True

# for SALICON/MIT (full description and object-only)
visual_token_start, visual_token_end = 46, 2386

patch_h, patch_w = 36, 49 # for salicon 480x640
img_size = 24

save_dir = None # Directory for saving the data
os.makedirs(save_dir, exist_ok=True)

img_dir = None # Directory for storing the visual/language attention map (with bbox)
split = ['language_bbox', 'visual_bbox']

pretrained = "lmms-lab/llama3-llava-next-8b"
model_name = "llava_llama3"


device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map,
                                                attn_implementation=None) # Add any other thing you want to pass in llava_model_args
model.eval()
model.tie_weights()

conv_template = "llava_llama_3" # Make sure you use correct chat template for different models

# full description v1
question = DEFAULT_IMAGE_TOKEN + "\nFocusing specifically on the red rectangle region and ignoring the rest of the image. Can you describe what you see in the rectangle in 30 words? The description should capture objects, their attributes, relative positions and relations. Start with 'In the red rectangle there...'"


conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

result = dict()
with torch.no_grad():
    for cur_split in split:
        all_images = glob(os.path.join(img_dir, cur_split, '*.jpg'))
        for cur_img in all_images:
            img_id = os.path.basename(cur_img)[:-4]

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

            if img_id not in result:
                result[img_id] = dict()
            result[img_id][cur_split] = text_outputs

with open(os.path.join(save_dir, 'language_explanation.json'), 'w') as f:
    json.dump(result, f)