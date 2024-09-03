import torch.nn as nn
import torch
from models.encdec import Decoder

class MotionDec(nn.Module):
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

        self.decoder = Decoder(251 if args.dataname == 'kit' else 263, code_dim, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        self.codebook = nn.Embedding(self.num_code, self.code_dim)
        #Initialized random/uniformly
        self.codebook.weight.data.uniform_(-1.0 / self.num_code, 1.0 / self.code_dim)

    def forward(self, code_indices = None):
        N, T, _ = code_indices.shape
        codes_flattened = code_indices.contiguous().view(-1, self.num_code)

        #Retrieve latent representation as linear combination of codes
        z = torch.matmul(codes_flattened, self.codebook.weight).view((-1, self.code_dim))
        z = z.view(N, T, -1).permute(0, 2, 1).contiguous()

        ## decoder
        x_decoder = self.decoder(z)
        x_out = self.postprocess(x_decoder)
        
        return x_out
        
    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0,2,1).float()
        return x


    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0,2,1)
        return x