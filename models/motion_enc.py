import torch.nn as nn
import torch
from models.encdec import Encoder
from utils.codebook import *

# Optional encoder module for obtaining pose codes
class MotionEnc(nn.Module):
    def __init__(self,
                 args,
                 nb_code=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        self.nb_joints = 21 if args.dataname == 'kit' else 22

        self.encoder = Encoder(251 if args.dataname == 'kit' else 263, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

    def forward(self, x):
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        logits = self.postprocess(x_encoder)
        
        return logits 

    def k_hot(self, logits, if_categorial = False):
        # Ensure mutual exclusivity within groups
        for cat in vq_to_range:
            if cat < 70:
                end, start = vq_to_range[cat]
                if if_categorial:
                    dist = Categorical(torch.nn.functional.softmax(logits[:,start:end+1],dim = -1))
                    idx = dist.sample()
                else:
                    idx = torch.argmax(logits[:,:,start:end+1], dim = -1)
                logits[:,:,start:end+1] = 0
                logits[torch.arange(logits.shape[0]),torch.arange(logits.shape[1]), start+idx] = 1
        return logits

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x