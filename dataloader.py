import torch
from torchvision import transforms
from glob import glob
import os
import numpy as np
from PIL import Image
from scipy.io import loadmat
import cv2
import json

imagenet_transform = transforms.Compose([
                                # transforms.Resize((480, 640)), # original DINet
                                transforms.Resize((240, 320)), # for reweighting experiment
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                ])

class SALICON(torch.utils.data.Dataset):
    """ Dataset for reading data from salicon dataset for blind model.
    """
    def __init__(self, language_path, anno_path, img_h, img_w, split, max_len=60, percentage=None):
        self.language_path = language_path
        self.anno_path = anno_path
        self.img_h = img_h
        self.img_w = img_w
        self.split = split
        self.max_len = max_len # 100 for 50 words, 60 for 30 words
        self.percentage = percentage
        self.init_data()

    def init_data(self,):
        if self.percentage is None:
            self.img_pool = []
            valid_id = json.load(open(os.path.join(self.language_path, 'salicon_valid.json')))[self.split]
            all_data = glob(os.path.join(self.language_path, 'hidden_state', self.split, '*'))
            for cur in all_data:
                img_id = os.path.basename(cur)
                if valid_id[img_id]:
                    self.img_pool.append(img_id)
        else:
            valid_id = json.load(open(os.path.join(self.language_path, 'salicon_valid.json')))['train_'+str(self.percentage)]
            self.img_pool = list(valid_id.keys())


    def get_fixation(self, fix_data):
        #loading new salicon data
        fix_data = loadmat(fix_data, simplify_cells=True)
        h, w = fix_data['resolution']
        fix_map = np.zeros([h, w])
        count = 0
        for subj_id in range(len(fix_data['gaze'])):
            cur_subj_data = fix_data['gaze'][subj_id]['fixations']
            for fix_id in range(len(cur_subj_data)):
                if not isinstance(cur_subj_data[fix_id], np.uint16) and not isinstance(cur_subj_data[fix_id], np.uint8):
                    x, y = cur_subj_data[fix_id]
                    fix_map[int(y-1), int(x-1)] = 1
                else:
                    count += 1

        return fix_map

    def __getitem__(self, index):
        img_id = self.img_pool[index]

        # loading images
        image = Image.open(os.path.join(self.anno_path, 'images', self.split, img_id[:-4]+'.jpg')).convert('RGB')
        image = imagenet_transform(image)

        # loading pre-extracted multi-modal mapping and language features, and pad them
        # mapping
        attention_mapping = np.load(os.path.join(self.language_path,
                        'attention', self.split, img_id), allow_pickle=True).item()
        patch_attention = torch.from_numpy(attention_mapping['patch'])
        # patch_attention = patch_attention/(patch_attention.sum((2, 3), keepdim=True)+1e-15)
        patch_attention = patch_attention/(torch.amax(patch_attention, dim=(2, 3), keepdim=True)+1e-7)

        seq, head, p_h, p_w = patch_attention.shape
        valid_len = seq
        if seq<self.max_len:
            pad = torch.zeros(self.max_len-seq, head, p_h, p_w)
            patch_attention = torch.cat([patch_attention, pad], dim=0)
        base_attention = torch.from_numpy(attention_mapping['base'])
        # base_attention = base_attention/(base_attention.sum((2, 3), keepdim=True)+1e-15)
        base_attention = base_attention/(torch.amax(base_attention, dim=(2, 3), keepdim=True)+1e-7)

        _, _, b_h, b_w = base_attention.shape
        if seq<self.max_len:
            pad = torch.zeros(self.max_len-seq, head, b_h, b_w)
            base_attention = torch.cat([base_attention, pad], dim=0)

        # features
        hidden_state = np.load(os.path.join(self.language_path,
                        'hidden_state', self.split, img_id))
        hidden_state = torch.from_numpy(hidden_state)
        if seq<self.max_len:
            pad = torch.zeros(self.max_len-seq, 4096)
            hidden_state = torch.cat([hidden_state, pad], dim=0)
        
        # loading saliency maps
        saliency_map = cv2.imread(os.path.join(self.anno_path, 
                                'saliency_map', self.split, img_id[:-4]+'.png')).astype('float32')
        saliency_map = saliency_map[:,:,0]
        saliency_map = cv2.resize(saliency_map, (self.img_w, self.img_h))
        saliency_map /= saliency_map.max()
        saliency_map = torch.from_numpy(saliency_map)

        # loading the fixation maps
        fixation_map = self.get_fixation(os.path.join(self.anno_path, 'fixation', 
                                        self.split, img_id[:-4]+'.mat'))
        fixation_map = cv2.resize(fixation_map, (self.img_w, self.img_h))
        fixation_map[fixation_map>0.1] = 1
        fixation_map[fixation_map!=1]= 0

        return hidden_state.float(), base_attention.float(), patch_attention.float(), valid_len, saliency_map, fixation_map, image, img_id[:-4]
    
    def __len__(self,):
        return len(self.img_pool)
    

class MIT(torch.utils.data.Dataset):
    """ Dataset for reading data from MIT dataset for blind model. 
        Note that we only consider perspective images.
    """
    def __init__(self, language_path, anno_path, img_h, img_w, split, max_len=100):
        self.language_path = language_path
        self.anno_path = anno_path
        self.img_h = img_h
        self.img_w = img_w
        self.split = split
        self.max_len = max_len
        self.init_data()

    def init_data(self,):
        self.img_pool = []
        valid_id = json.load(open(os.path.join(self.language_path, 'mit_valid.json')))
        all_data = glob(os.path.join(self.language_path, 'hidden_state', self.split, '*'))
        for cur in all_data:
            img_id = os.path.basename(cur)
            if valid_id[img_id]:
                self.img_pool.append(img_id)

    def __getitem__(self, index):
        img_id = self.img_pool[index]

        # loading images
        image = Image.open(os.path.join(self.anno_path, 'image', 'test_reshaped', img_id[:-4]+'.jpeg')).convert('RGB')
        image = imagenet_transform(image)

        # loading pre-extracted multi-modal mapping and language features, and pad them
        # mapping
        attention_mapping = np.load(os.path.join(self.language_path,
                        'attention', self.split, img_id), allow_pickle=True).item()
        patch_attention = torch.from_numpy(attention_mapping['patch'])
        # patch_attention = patch_attention/(patch_attention.sum((2, 3), keepdim=True)+1e-15)
        patch_attention = patch_attention/(torch.amax(patch_attention, dim=(2, 3), keepdim=True)+1e-7)

        seq, head, p_h, p_w = patch_attention.shape
        valid_len = seq
        if seq<self.max_len:
            pad = torch.zeros(self.max_len-seq, head, p_h, p_w)
            patch_attention = torch.cat([patch_attention, pad], dim=0)
        base_attention = torch.from_numpy(attention_mapping['base'])
        # base_attention = base_attention/(base_attention.sum((2, 3), keepdim=True)+1e-15)
        base_attention = base_attention/(torch.amax(base_attention, dim=(2, 3), keepdim=True)+1e-7)

        _, _, b_h, b_w = base_attention.shape
        if seq<self.max_len:
            pad = torch.zeros(self.max_len-seq, head, b_h, b_w)
            base_attention = torch.cat([base_attention, pad], dim=0)

        # features
        hidden_state = np.load(os.path.join(self.language_path,
                        'hidden_state', self.split, img_id))
        hidden_state = torch.from_numpy(hidden_state)
        if seq<self.max_len:
            pad = torch.zeros(self.max_len-seq, 4096)
            hidden_state = torch.cat([hidden_state, pad], dim=0)
        
        # loading saliency maps
        saliency_map = cv2.imread(os.path.join(self.anno_path, 
                                'saliency_map', img_id[:-4]+'_fixMap.jpg')).astype('float32')
        saliency_map = saliency_map[:,:,0]
        saliency_map = cv2.resize(saliency_map, (self.img_w, self.img_h))
        saliency_map /= saliency_map.max()
        saliency_map = torch.from_numpy(saliency_map)

        # loading raw fixation maps
        fixation_map = cv2.imread(os.path.join(self.anno_path, 
                                'saliency_map', img_id[:-4]+'_fixPts.jpg')).astype('float32')
        fixation_map = fixation_map[:,:,0]/255
        fixation_map = cv2.resize(fixation_map, (self.img_w, self.img_h))
        fixation_map[fixation_map>0.5] = 1
        fixation_map[fixation_map!=1] = 0
        fixation_map = torch.from_numpy(fixation_map)       

        return hidden_state.float(), base_attention.float(), patch_attention.float(), valid_len, saliency_map, fixation_map, image, img_id[:-4]
    
    def __len__(self,):
        return len(self.img_pool)
    

class OSIE_ASD(torch.utils.data.Dataset):
    """ Dataset for reading data from OSIE-ASD dataset for blind model. 
        
    """
    def __init__(self, language_path, anno_path, img_h, img_w, split, max_len=100, asd_label='asd', 
                subj_id=None, temporal_step=1, temporal_id=None):
        self.language_path = language_path
        self.anno_path = anno_path
        self.img_h = img_h
        self.img_w = img_w
        self.split = split
        self.max_len = max_len
        self.asd_label = asd_label
        self.subj_id = subj_id
        self.temporal_step = temporal_step
        self.temporal_id = temporal_id
        self.init_data()

    def init_data(self,):
        self.img_pool = []
        self.all_img_pool = dict() # use for asd personalized attention as sometimes a subject miss a few images
        if self.split != 'all':
            valid_id = json.load(open(os.path.join(self.language_path, 'osie_valid.json')))[self.split]
        else:
            valid_id = json.load(open(os.path.join(self.language_path, 'osie_valid.json')))['train']
            valid_id.update(json.load(open(os.path.join(self.language_path, 'osie_valid.json')))['val'])
        all_data = glob(os.path.join(self.language_path, 'hidden_state', '*'))
        for cur in all_data:
            img_id = os.path.basename(cur)
            if img_id in valid_id:
                self.img_pool.append(img_id)
                self.all_img_pool[img_id[:-4]] = True

    def set_subj(self, subj_id):
        self.subj_id = subj_id
        if self.temporal_id is None:
            valid_data = glob(os.path.join(self.anno_path, 'personalized_attention',
                                        self.asd_label, 'saliency_map', self.subj_id, '*.jpg'))
        else:
            valid_data = glob(os.path.join(self.anno_path, 'personalized_attention_temporal', 
                                           self.asd_label, 'period_'+str(self.temporal_id), 
                                        'saliency_map', self.subj_id, '*.jpg'))            
        valid_data = [os.path.basename(cur)[:-4] for cur in valid_data]
        self.img_pool = [cur+'.npy' for cur in valid_data if cur in self.all_img_pool]

    def set_asd_label(self, asd_label):
        self.asd_label = asd_label

    def __getitem__(self, index):
        img_id = self.img_pool[index]

        image = Image.open(os.path.join(self.anno_path, 'images', img_id[:-4]+'.jpg')).convert('RGB')
        image = imagenet_transform(image)

        # loading pre-extracted multi-modal mapping and language features, and pad them
        # mapping
        attention_mapping = np.load(os.path.join(self.language_path,
                        'attention', img_id), allow_pickle=True).item()
        patch_attention = torch.from_numpy(attention_mapping['patch'])
        # patch_attention = patch_attention/(patch_attention.sum((2, 3), keepdim=True)+1e-15)
        patch_attention = patch_attention/(torch.amax(patch_attention, dim=(2, 3), keepdim=True)+1e-7)

        seq, head, p_h, p_w = patch_attention.shape
        valid_len = seq
        if seq<self.max_len:
            pad = torch.zeros(self.max_len-seq, head, p_h, p_w)
            patch_attention = torch.cat([patch_attention, pad], dim=0)
        base_attention = torch.from_numpy(attention_mapping['base'])
        # base_attention = base_attention/(base_attention.sum((2, 3), keepdim=True)+1e-15)
        base_attention = base_attention/(torch.amax(base_attention, dim=(2, 3), keepdim=True)+1e-7)

        _, _, b_h, b_w = base_attention.shape
        if seq<self.max_len:
            pad = torch.zeros(self.max_len-seq, head, b_h, b_w)
            base_attention = torch.cat([base_attention, pad], dim=0)

        # features
        hidden_state = np.load(os.path.join(self.language_path,
                        'hidden_state', img_id))
        hidden_state = torch.from_numpy(hidden_state)
        if seq<self.max_len:
            pad = torch.zeros(self.max_len-seq, 4096)
            hidden_state = torch.cat([hidden_state, pad], dim=0)
        
        # loading saliency maps
        if self.temporal_step == 1:
            if self.subj_id is None and self.temporal_id is None:
                saliency_map = cv2.imread(os.path.join(self.anno_path, 
                                        'saliency_map', self.asd_label, img_id[:-4]+'.jpg')).astype('float32')
            elif self.subj_id is not None and self.temporal_id is not None:
                saliency_map = cv2.imread(os.path.join(self.anno_path, 
                                        'personalized_attention_temporal', self.asd_label, 'period_'+str(self.temporal_id), 
                                        'saliency_map', self.subj_id, img_id[:-4]+'.jpg')).astype('float32')            
            elif self.temporal_id is not None:
                saliency_map = cv2.imread(os.path.join(self.anno_path, 
                                        'temporal_map', self.asd_label, 'period_'+str(self.temporal_id), 
                                        'saliency_map', img_id[:-4]+'.jpg')).astype('float32')            
            else:
                saliency_map = cv2.imread(os.path.join(self.anno_path, 'personalized_attention',
                                        self.asd_label, 'saliency_map', self.subj_id, img_id[:-4]+'.jpg')).astype('float32')

            saliency_map = saliency_map[:,:,0]
            saliency_map = cv2.resize(saliency_map, (self.img_w, self.img_h))
            saliency_map /= saliency_map.max()
            saliency_map = torch.from_numpy(saliency_map)

            # loading raw fixation maps
            if self.subj_id is None and self.temporal_id is None:
                fixation_map = cv2.imread(os.path.join(self.anno_path, 
                                        'fixation_map', self.asd_label, img_id[:-4]+'.jpg')).astype('float32')
            elif self.subj_id is not None and self.temporal_id is not None:
                fixation_map = cv2.imread(os.path.join(self.anno_path, 
                                        'personalized_attention_temporal', self.asd_label, 'period_'+str(self.temporal_id), 
                                        'fixation_map', self.subj_id, img_id[:-4]+'.jpg')).astype('float32') 
            elif self.temporal_id is not None:
                fixation_map = cv2.imread(os.path.join(self.anno_path, 
                        'temporal_map', self.asd_label, 'period_'+str(self.temporal_id), 
                        'fixation_map', img_id[:-4]+'.jpg')).astype('float32')                
            else:
                fixation_map = cv2.imread(os.path.join(self.anno_path, 'personalized_attention',
                                        self.asd_label, 'fixation_map', self.subj_id, img_id[:-4]+'.jpg')).astype('float32')

            fixation_map = fixation_map[:,:,0]/255
            fixation_map = cv2.resize(fixation_map, (self.img_w, self.img_h))
            fixation_map[fixation_map>0.5] = 1
            fixation_map[fixation_map!=1] = 0
            fixation_map = torch.from_numpy(fixation_map)   

        else:
            saliency_map = []
            for t in range(1, self.temporal_step+1):
                cur_map = cv2.imread(os.path.join(self.anno_path, 
                                        'temporal_map', self.asd_label, 'period_'+str(t), 
                                        'saliency_map', img_id[:-4]+'.jpg')).astype('float32')
                cur_map = cur_map[:,:,0]
                cur_map = cv2.resize(cur_map, (self.img_w, self.img_h))
                cur_map /= cur_map.max()
                cur_map = torch.from_numpy(cur_map)
                saliency_map.append(cur_map.unsqueeze(0))
            saliency_map = torch.cat(saliency_map, dim=0)

            fixation_map = []
            for t in range(1, self.temporal_step+1):
                cur_fix = cv2.imread(os.path.join(self.anno_path, 
                        'temporal_map', self.asd_label, 'period_'+str(t), 
                        'fixation_map', img_id[:-4]+'.jpg')).astype('float32')

                cur_fix = cur_fix[:,:,0]/255
                cur_fix = cv2.resize(cur_fix, (self.img_w, self.img_h))
                cur_fix[cur_fix>0.5] = 1
                cur_fix[cur_fix!=1] = 0
                cur_fix = torch.from_numpy(cur_fix)
                fixation_map.append(cur_fix.unsqueeze(0))  
            fixation_map = torch.cat(fixation_map, dim=0)     

        return hidden_state.float(), base_attention.float(), patch_attention.float(), valid_len, saliency_map, fixation_map, image, img_id[:-4]
    
    def __len__(self,):
        return len(self.img_pool)


class Emotion(torch.utils.data.Dataset):
    """ Dataset for reading data from Emotion saliency dataset for blind model.    
    """
    def __init__(self, language_path, anno_path, img_h, img_w, split, max_len=100):
        self.language_path = language_path
        self.anno_path = anno_path
        self.img_h = img_h
        self.img_w = img_w
        self.split = split
        self.max_len = max_len
        self.init_data()

    def init_data(self,):
        self.img_pool = []
        if self.split != 'all':
            valid_id = json.load(open(os.path.join(self.language_path, 'emotion.json')))[self.split]
        else:
            valid_id_file = json.load(open(os.path.join(self.language_path, 'emotion.json')))
            valid_id = valid_id_file['train']
            valid_id.update(valid_id_file['val'])

        all_data = glob(os.path.join(self.language_path, 'hidden_state', '*'))
        for cur in all_data:
            img_id = os.path.basename(cur)
            if img_id in valid_id:
                self.img_pool.append(img_id)

    def __getitem__(self, index):
        img_id = self.img_pool[index]

        # loading images
        image = Image.open(os.path.join(self.anno_path, 'image', img_id[:-4]+'.jpg')).convert('RGB')
        image = imagenet_transform(image)

        # loading pre-extracted multi-modal mapping and language features, and pad them
        # mapping
        attention_mapping = np.load(os.path.join(self.language_path,
                        'attention', img_id), allow_pickle=True).item()
        patch_attention = torch.from_numpy(attention_mapping['patch'])
        # patch_attention = patch_attention/(patch_attention.sum((2, 3), keepdim=True)+1e-15)
        patch_attention = patch_attention/(torch.amax(patch_attention, dim=(2, 3), keepdim=True)+1e-7)

        seq, head, p_h, p_w = patch_attention.shape
        valid_len = seq
        if seq<self.max_len:
            pad = torch.zeros(self.max_len-seq, head, p_h, p_w)
            patch_attention = torch.cat([patch_attention, pad], dim=0)
        base_attention = torch.from_numpy(attention_mapping['base'])
        # base_attention = base_attention/(base_attention.sum((2, 3), keepdim=True)+1e-15)
        base_attention = base_attention/(torch.amax(base_attention, dim=(2, 3), keepdim=True)+1e-7)

        _, _, b_h, b_w = base_attention.shape
        if seq<self.max_len:
            pad = torch.zeros(self.max_len-seq, head, b_h, b_w)
            base_attention = torch.cat([base_attention, pad], dim=0)

        # features
        hidden_state = np.load(os.path.join(self.language_path,
                        'hidden_state', img_id))
        hidden_state = torch.from_numpy(hidden_state)
        if seq<self.max_len:
            pad = torch.zeros(self.max_len-seq, 4096)
            hidden_state = torch.cat([hidden_state, pad], dim=0)
        
        # loading saliency maps
        saliency_map = cv2.imread(os.path.join(self.anno_path, 
                                'saliency_map', 'Continous_map', img_id[:-4]+'.jpg')).astype('float32')
        saliency_map = saliency_map[:,:,0]
        saliency_map = cv2.resize(saliency_map, (self.img_w, self.img_h))
        saliency_map /= saliency_map.max()
        saliency_map = torch.from_numpy(saliency_map)

        # loading raw fixation maps
        fixation_map = cv2.imread(os.path.join(self.anno_path, 
                                'saliency_map', 'Binary_map', img_id[:-4]+'.jpg')).astype('float32')
        fixation_map = fixation_map[:,:,0]/255
        fixation_map = cv2.resize(fixation_map, (self.img_w, self.img_h))
        fixation_map[fixation_map>0.5] = 1
        fixation_map[fixation_map!=1] = 0
        fixation_map = torch.from_numpy(fixation_map)       

        return hidden_state.float(), base_attention.float(), patch_attention.float(), valid_len, saliency_map, fixation_map, image, img_id[:-4]
    
    def __len__(self,):
        return len(self.img_pool)
    

class SALICON_Image(torch.utils.data.Dataset):
    """ Dataset for reading data from salicon dataset for standard visual model.
    """
    def __init__(self, anno_path, language_path, img_h, img_w, split, percentage=None):
        self.anno_path = anno_path
        self.img_h = img_h
        self.img_w = img_w
        self.split = split
        self.percentage = percentage
        self.language_path = language_path
        self.init_data()

    def init_data(self,):
        self.img_pool = [os.path.basename(cur) for cur in 
                         glob(os.path.join(self.anno_path, 'images', self.split, '*.jpg'))]
        
        if self.percentage is None:
            valid_id = json.load(open(os.path.join(self.language_path, 'salicon_valid.json')))[self.split]
            self.img_pool = [cur for cur in self.img_pool if valid_id[cur[:-4]+'.npy']]
        else:
            valid_id = json.load(open(os.path.join(self.language_path, 'salicon_valid.json')))['train_'+str(self.percentage)]
            self.img_pool = [cur for cur in self.img_pool if cur[:-4]+'.npy' in valid_id]


    def get_fixation(self, fix_data):
        #loading new salicon data
        fix_data = loadmat(fix_data, simplify_cells=True)
        h, w = fix_data['resolution']
        fix_map = np.zeros([h, w])
        count = 0
        for subj_id in range(len(fix_data['gaze'])):
            cur_subj_data = fix_data['gaze'][subj_id]['fixations']
            for fix_id in range(len(cur_subj_data)):
                if not isinstance(cur_subj_data[fix_id], np.uint16) and not isinstance(cur_subj_data[fix_id], np.uint8):
                    x, y = cur_subj_data[fix_id]
                    fix_map[int(y-1), int(x-1)] = 1
                else:
                    count += 1

        return fix_map

    def __getitem__(self, index):
        img_id = self.img_pool[index]

        # loading images
        image = Image.open(os.path.join(self.anno_path, 'images', self.split, img_id)).convert('RGB')
        image = imagenet_transform(image)
        
        # loading saliency maps
        saliency_map = cv2.imread(os.path.join(self.anno_path, 
                                'saliency_map', self.split, img_id[:-4]+'.png')).astype('float32')
        saliency_map = saliency_map[:,:,0]
        saliency_map = cv2.resize(saliency_map, (self.img_w, self.img_h))
        saliency_map /= saliency_map.max()
        saliency_map = torch.from_numpy(saliency_map)

        # loading the fixation maps
        fixation_map = self.get_fixation(os.path.join(self.anno_path, 'fixation', 
                                        self.split, img_id[:-4]+'.mat'))
        fixation_map = cv2.resize(fixation_map, (self.img_w, self.img_h))
        fixation_map[fixation_map>0.1] = 1
        fixation_map[fixation_map!=1]= 0

        return image, saliency_map, fixation_map, img_id[:-4]
    
    def __len__(self,):
        return len(self.img_pool)
    

class MIT_Image(torch.utils.data.Dataset):
    """ Dataset for reading data from MIT dataset for visual model. 
        Note that we only consider perspective images.
    """
    def __init__(self, anno_path, img_h, img_w):
        self.anno_path = anno_path
        self.img_h = img_h
        self.img_w = img_w
        self.init_data()

    def init_data(self,):
        self.img_pool = [os.path.basename(cur) for cur in 
                         glob(os.path.join(self.anno_path, 'image', 'test_reshaped', '*.jpeg'))]

    def __getitem__(self, index):
        img_id = self.img_pool[index]

        # loading images
        image = Image.open(os.path.join(self.anno_path, 'image', 'test_reshaped', img_id)).convert('RGB')
        image = imagenet_transform(image)

        # loading saliency maps
        saliency_map = cv2.imread(os.path.join(self.anno_path, 
                                'saliency_map', img_id[:-5]+'_fixMap.jpg')).astype('float32')
        saliency_map = saliency_map[:,:,0]
        saliency_map = cv2.resize(saliency_map, (self.img_w, self.img_h))
        saliency_map /= saliency_map.max()
        saliency_map = torch.from_numpy(saliency_map)

        # loading raw fixation maps
        fixation_map = cv2.imread(os.path.join(self.anno_path, 
                                'saliency_map', img_id[:-5]+'_fixPts.jpg')).astype('float32')
        fixation_map = fixation_map[:,:,0]/255
        fixation_map = cv2.resize(fixation_map, (self.img_w, self.img_h))
        fixation_map[fixation_map>0.1] = 1
        fixation_map[fixation_map!=1] = 0
        fixation_map = torch.from_numpy(fixation_map)       

        return image, saliency_map, fixation_map, img_id[:-5]
    
    def __len__(self,):
        return len(self.img_pool)
    