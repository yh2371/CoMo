import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import utils.paramUtil as paramUtil
from torch.utils.data._utils.collate import default_collate

def collate_fn(batch):
    batch.sort(key=lambda x: x[2], reverse=True)
    return default_collate(batch)

'''For use of training text-2-motion generative model'''
class Text2MotionDataset(data.Dataset):
    def __init__(self, dataset_name, unit_length = 4, codebook_size = 1024):
        
        self.max_length = 64
        self.pointer = 0
        self.dataset_name = dataset_name
        self.codebook_size = codebook_size

        self.unit_length = unit_length # downsample across frames

        self.mot_end_idx = codebook_size
        self.mot_pad_idx = codebook_size + 1

        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.code_dir = pjoin(self.data_root, 'codes') 
            self.keyword_dir = pjoin(self.data_root, 'keyword_embeddings') 
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 26 if unit_length == 8 else 51
            dim_pose = 263
            kinematic_chain = paramUtil.t2m_kinematic_chain
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.code_dir = pjoin(self.data_root, 'codes') 
            self.keyword_dir = pjoin(self.data_root, 'keyword_embeddings') 
            self.joints_num = 21
            radius = 240 * 8
            fps = 12.5
            dim_pose = 251
            self.max_motion_length = 26 if unit_length == 8 else 51
            kinematic_chain = paramUtil.kit_kinematic_chain
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'

        split_file = pjoin(self.data_root, 'train.txt')
        self.mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        self.std = np.load(pjoin(self.meta_dir, 'std.npy'))

        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        data_dict = {}
        count = 0
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                code_indices = np.load(pjoin(self.code_dir, name + '.npy'))
                keywords = np.load(pjoin(self.keyword_dir, name + '.npy'))

                if (len(motion)) < 40 or (len(motion) >= 200):
                    continue

                m_token_list = code_indices.reshape(1,-1,self.codebook_size) #N x T x code_num

                m_length = m_token_list.shape[1]
                m_length = (m_length // self.unit_length) * self.unit_length
                idx = random.randint(0, m_token_list.shape[1] - m_length)
                m_token_list = m_token_list[:,idx:idx+m_length,:][:,::self.unit_length,:]
                motion = motion[idx:idx+m_length,:]  
             
                # Read text
                count += 1
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    text_data = []
                    flag = False
                    lines = f.readlines()

                    for line in lines:
                        try:
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            t_tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = t_tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                n_motion = motion[int(f_tag*fps) : int(to_tag*fps)]
                                if (len(n_motion)) < 40 or (len(n_motion) >= 200):
                                    continue

                                m_token_list_new = [tokens[int(f_tag*fps/unit_length) : int(to_tag*fps/unit_length)] for tokens in m_token_list if int(f_tag*fps/unit_length) < int(to_tag*fps/unit_length)]
                            
                                if len(m_token_list_new) == 0:
                                    continue
                                new_name = '%s_%f_%f'%(name, f_tag, to_tag)

                                data_dict[new_name] = {'m_token_list': m_token_list_new,
                                                       'text':[text_dict],
                                                       'keywords': keywords,
                                                       'motion': n_motion}
                                new_name_list.append(new_name)
                        except:
                            pass

                if flag:
                    data_dict[name] = {'m_token_list': m_token_list,
                                       'text':text_data,
                                       'keywords': keywords,
                                       'motion': motion}
                    new_name_list.append(name)
            except:
                pass
        self.data_dict = data_dict
        self.name_list = new_name_list
        print(count)

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        m_token_list, text_list, keywords, motion = data['m_token_list'], data['text'], data['keywords'], data['motion']

        m_tokens = random.choice(m_token_list)

        text_data = random.choice(text_list)
        caption= text_data['caption']

        # Randomly select keywords
        keyword_indices = np.arange(0,33,3) + np.random.randint(0,3,size = (11,))
        keyword_data = torch.from_numpy(keywords[keyword_indices])

        proba = np.random.rand(1)[0]
        mask = torch.bernoulli(proba * torch.ones((11,)))
        mask = mask.round().to(dtype=torch.int64)
        keyword_data[mask] = 0

        coin = np.random.choice([False, False, True])

        if coin:
            # drop one token at the head or tail
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
                motion = motion[:-4]
            else:
                m_tokens = m_tokens[1:]
                motion = motion[4:]
        motion = (motion - self.mean) / self.std
        m_length = len(motion)

        max_motion_length = 200
        if m_length < max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)

        m_tokens_len = m_tokens.shape[0] #t x code_num     
        m_tokens_new = m_tokens
         

        if m_tokens_len+1 < self.max_motion_length:
            end_token = np.zeros((1, self.codebook_size+2)) 
            end_token[:,self.mot_end_idx] = 1

            pad_tokens = np.zeros((self.max_motion_length-1-m_tokens_len, self.codebook_size+2))
            pad_tokens[:,self.mot_pad_idx] = 1
            
            m_tokens_new = np.concatenate([m_tokens_new, np.zeros((m_tokens_new.shape[0],2))], axis = 1) #motion_length x code_num+2
            m_tokens_new = np.concatenate([m_tokens_new, end_token, pad_tokens], axis = 0) #max_motion_length x code_num+2

        else: 
            end_token = np.zeros((1, self.codebook_size+2)) 
            end_token[:,self.mot_end_idx] = 1 #consider end token
            m_tokens_new = np.concatenate([m_tokens_new, np.zeros((m_tokens_new.shape[0],2))], axis = 1) #motion_length x code_num+2
            m_tokens_new = np.concatenate([m_tokens_new, end_token], axis = 0) #max_motion_length x code_num+2

        return caption, m_tokens_new, m_tokens_len, keyword_data, motion, m_length 
        
def DATALoader(dataset_name,
                batch_size, codebook_size, unit_length=4,
                num_workers = 8) : 

    train_loader = torch.utils.data.DataLoader(Text2MotionDataset(dataset_name, codebook_size = codebook_size, unit_length=unit_length),
                                              batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn,
                                              drop_last = True)
    

    return train_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

