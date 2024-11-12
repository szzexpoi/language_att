import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block
import numpy as np
from DINet import DINet
from transalnet import TranSalNet

class Learned_Prior(nn.Module):
    def __init__(self, N=16, row=15, col=20): #N is the number of center-bias maps
        super(Learned_Prior,self).__init__()
        self.mu = nn.Parameter(torch.randn(2, N), requires_grad=True)
        self.sigma = nn.Parameter(torch.randn(2, N), requires_grad=True)
        self.x_grid = self.init_grid(N, row, col, axis=0)
        self.y_grid = self.init_grid(N, row, col, axis=1)
        self.row = row
        self.col = col
        self.N = N

    def init_grid(self,N,row,col,axis): #initializing grid for computing spatial gaussian
        if axis == 0 :
            e = float(row)/col
            e1 = (1-e)/2
            e2 = e1 + e
            grid = np.linspace(e1, e2, row).reshape(row, 1)
            grid = np.repeat(grid, col, axis=1)
        else:
            grid = np.linspace(0, 1, col).reshape(1, col)
            grid = np.repeat(grid, row, axis=0)

        grid = np.repeat(grid.reshape(1, row, col), N, axis=0).astype('float32')
        grid = nn.Parameter(torch.from_numpy(grid), requires_grad=False)

        return grid

    def forward(self, batch_size):
        x_grid = self.x_grid.view(self.N, -1)
        y_grid = self.y_grid.view(self.N, -1)
        mu_x = self.mu[0].unsqueeze(1).expand_as(x_grid)
        mu_y = self.mu[1].unsqueeze(1).expand_as(y_grid)
        sigma_x = self.sigma[0].unsqueeze(1).expand_as(x_grid)
        sigma_y = self.sigma[1].unsqueeze(1).expand_as(y_grid)

        #clipping mu and sigma
        mu_y = torch.clamp(mu_y, min=0.25, max=0.75)
        mu_x = torch.clamp(mu_x, min=0.35, max=0.65)
        sigma_y = torch.clamp(sigma_y, min=0.1, max=0.9)
        sigma_x = torch.clamp(sigma_x, min=0.2, max=0.8)

        #gaussian equation
        x_grid = torch.pow(x_grid - mu_x,2)
        y_grid = torch.pow(y_grid - mu_y,2)
        x_grid = torch.div(x_grid,2*torch.pow(sigma_x, 2))
        y_grid = torch.div(y_grid,2*torch.pow(sigma_y, 2))
        grid = torch.exp(-(x_grid+y_grid))
        grid = torch.div(grid,2*np.pi*torch.mul(sigma_x, sigma_y))
        max_grid = grid.max(1, keepdim=True)[0].expand_as(grid).contiguous()
        grid = torch.div(grid, max_grid)
        grid = grid.view(self.N, self.row, self.col).unsqueeze(0).repeat(batch_size, 1, 1, 1)

        return grid


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Blind_Transformer(nn.Module):
    """ A "Blind" Transformer model for
        saliency prediction. The principal idea is to 
        compute the weights of tokens inside the image captions,
        and leverage pre-extracted multi-modal mapping for deriving
        the visual saliency maps.
    """
    def __init__(self, input_dim=4096, input_head=32, num_head=4, depth=4, img_h=36, img_w=48, max_len=70):
        super(Blind_Transformer, self).__init__()
        self.input_head = input_head
        # initialize (and freeze) pos_embed by sin-cos embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, input_dim), requires_grad=False)  # fixed sin-cos embedding 
        pos_embed = get_1d_sincos_pos_embed_from_grid(input_dim, np.arange(max_len))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # computing the attention weights for language tokens
        self.language_decoder = nn.ModuleList([Block(input_dim, num_head, 4., qkv_bias=True, 
                                           qk_scale=None, norm_layer=nn.LayerNorm)
                                           for _ in range(depth)])
        self.language_attn = nn.Linear(input_dim, 1)

        
        # learnable piror maps
        self.learned_prior = Learned_Prior(16, img_h, img_w)

        # use a collection of convolutional layers to project the multi-head attention to saliency maps
        self.sal_decoder = nn.Sequential(nn.Conv2d(input_head+16, 32, kernel_size=5, padding=2, stride=1),
                                      nn.ReLU(),
                                      nn.Upsample(scale_factor=2),
                                      nn.Conv2d(32, 16, kernel_size=5, padding=2, stride=1),
                                      nn.ReLU(),
                                      nn.Upsample(scale_factor=2),
                                      nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1),
                                      nn.ReLU(),
                                      nn.Upsample(scale_factor=2),
                                      nn.Conv2d(16, 8, kernel_size=3, padding=1, stride=1),
                                      nn.ReLU(),
                                      nn.Conv2d(8, 4, kernel_size=3, padding=1, stride=1),
                                      nn.ReLU(),
                                      nn.Conv2d(4, 1, kernel_size=3, padding=1, stride=1),
                                      )
        

    def forward(self, language, multi_modal_mapping, valid_len, get_lang_att=False):
        x = language + self.pos_embed
        for blk in self.language_decoder:
            x = blk(x, is_causal=False)
            x = F.dropout(x, 0.3)        
        language_att = self.language_attn(x).squeeze(-1)
        
        # filter out the padded token
        b, seq = language_att.shape
        binary_mask = torch.ones(b, seq).bool().cuda()
        for i in range(b):
            binary_mask[i, :valid_len[i]] = 0
        language_att = language_att.masked_fill(binary_mask, -1e4)
        language_att = F.softmax(language_att, dim=-1)
        raw_language_att = language_att.clone()

        # weight the multi-modal mapping based on the language attention
        language_att = language_att.view(b, seq, 1, 1, 1).expand(b, seq, self.input_head, 1, 1)
        visual_attention = (language_att*multi_modal_mapping).sum(1)
        
        # derive the saliency maps
        prior_map = self.learned_prior(b)
        visual_attention = torch.cat([visual_attention, prior_map], dim=1)
        saliency_map = self.sal_decoder(visual_attention)
        saliency_map = F.interpolate(saliency_map, (240, 320), mode='bilinear').squeeze(1)
        b, h, w = saliency_map.shape 
        saliency_map = F.softmax(saliency_map.view(b, h*w), dim=-1).view(b, h, w)

        if get_lang_att:
            return saliency_map, raw_language_att
        else:
            return saliency_map
    

class Blind_Transformer_MS(nn.Module):
    """ A multi-scale "Blind" Transformer model for
        saliency prediction. The principal idea is to 
        compute the weights of tokens inside the image captions,
        and leverage pre-extracted multi-modal mapping for deriving
        the visual saliency maps.
    """
    def __init__(self, input_dim=4096, input_head=32, num_head=4, depth=4, img_h=36, img_w=48, max_len=70, temporal_step=1):
        super(Blind_Transformer_MS, self).__init__()
        self.input_head = input_head
        # initialize (and freeze) pos_embed by sin-cos embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, input_dim), requires_grad=False)  # fixed sin-cos embedding 
        pos_embed = get_1d_sincos_pos_embed_from_grid(input_dim, np.arange(max_len))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # computing the attention weights for language tokens
        self.language_decoder = nn.ModuleList([Block(input_dim, num_head, 4., qkv_bias=True, 
                                           qk_scale=None, norm_layer=nn.LayerNorm)
                                           for _ in range(depth)])
        self.temporal_step = temporal_step
        self.input_dim = input_dim
        self.language_attn = nn.Linear(input_dim, 1)

        # learnable piror maps for patch attention
        self.learned_prior = Learned_Prior(16, img_h, img_w)

        # learnable piror maps for base attention
        self.learned_prior_base = Learned_Prior(16, 24, 24)

        # use a collection of convolutional layers to project the multi-head attention to saliency maps
        self.sal_decoder = nn.Sequential(nn.Conv2d(input_head+16, 32, kernel_size=5, padding=2, stride=1),
                                      nn.ReLU(),
                                      nn.Upsample(scale_factor=2),
                                      nn.Conv2d(32, 16, kernel_size=5, padding=2, stride=1),
                                      nn.ReLU(),
                                      nn.Upsample(scale_factor=2),
                                      nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1),
                                      nn.ReLU(),
                                      nn.Upsample(scale_factor=2),
                                      )
        
        self.sal_header = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1),
                                        nn.ReLU(),
                                        nn.Conv2d(16, 8, kernel_size=3, padding=1, stride=1),
                                        nn.ReLU(),
                                        nn.Conv2d(8, 1, kernel_size=3, padding=1, stride=1),
                                      )

    def set_temporal_layer(self, ):
        """ reset the temporal layer after loading the pretrained weights.
        """
        ori_weight = self.language_attn.weight.data
        ori_bias =  self.language_attn.bias.data
        new_weight = ori_weight.repeat(self.temporal_step, 1)
        new_bias = ori_bias.repeat(self.temporal_step, )
        self.language_attn = nn.Linear(self.input_dim, self.temporal_step)
        self.language_attn.weight.data = new_weight
        self.language_attn.bias.data = new_bias

    def forward(self, language, patch_mapping, base_mapping, valid_len, get_lang_att=False):
        x = language + self.pos_embed
        for blk in self.language_decoder:
            x = blk(x, is_causal=False)
            x = F.dropout(x, 0.3)

        if self.temporal_step == 1:
            language_att = self.language_attn(x).squeeze(-1)
            # filter out the padded token
            b, seq = language_att.shape
            binary_mask = torch.ones(b, seq).bool().cuda()
            for i in range(b):
                binary_mask[i, :valid_len[i]] = 0
            language_att = language_att.masked_fill(binary_mask, -1e4)
            language_att = F.softmax(language_att, dim=-1)
            raw_language_att = language_att

            # weight the patch and base attention mapping based on the language attention
            language_att = language_att.view(b, seq, 1, 1, 1).expand(b, seq, self.input_head, 1, 1)
            patch_attention = (language_att*patch_mapping).sum(1)
            base_attention = (language_att*base_mapping).sum(1)
        else:
            language_att = self.language_attn(x).permute(0, 2, 1)
            b, t, seq = language_att.shape
            binary_mask = torch.ones(b, t, seq).bool().cuda()
            for i in range(b):
                binary_mask[i, :, :valid_len[i]] = 0
            language_att = language_att.masked_fill(binary_mask, -1e4)
            language_att = F.softmax(language_att, dim=-1)
            raw_language_att = language_att
            language_att = language_att.view(b, t, seq, 1, 1, 1).expand(b, t, seq, self.input_head, 1, 1)
            patch_attention = (language_att*patch_mapping.unsqueeze(1)).sum(2)
            base_attention = (language_att*base_mapping.unsqueeze(1)).sum(2)
            patch_attention = patch_attention.view(b*t, self.input_head, 
                                patch_attention.shape[-2], patch_attention.shape[-1]) 
            base_attention = base_attention.view(b*t, 
                        self.input_head, base_attention.shape[-2], base_attention.shape[-1])
            b = b*t                
        
        # derive the saliency maps
        prior_map_patch = self.learned_prior(b)
        patch_attention = torch.cat([patch_attention, prior_map_patch], dim=1)
        patch_attention = self.sal_decoder(patch_attention)
        patch_attention = F.interpolate(patch_attention, (240, 320), mode='bilinear')

        prior_map_base = self.learned_prior_base(b)
        base_attention = torch.cat([base_attention, prior_map_base], dim=1)
        base_attention = self.sal_decoder(base_attention) # use shared decoder
        base_attention = F.interpolate(base_attention, (240, 320), mode='bilinear')

        visual_attention = torch.cat([base_attention, patch_attention], dim=1)
        saliency_map = self.sal_header(visual_attention).squeeze(1)
        b, h, w = saliency_map.shape 
        saliency_map = F.softmax(saliency_map.view(b, h*w), dim=-1).view(b, h, w)

        if self.temporal_step!=1:
            saliency_map = saliency_map.view(b//self.temporal_step, self.temporal_step, h, w)

        if get_lang_att:
            return saliency_map, raw_language_att
        else:
            return saliency_map
    
    def language_only(self, language, valid_len):
        x = language + self.pos_embed
        for blk in self.language_decoder:
            x = blk(x, is_causal=False)
            x = F.dropout(x, 0.3)        
        language_att = self.language_attn(x).squeeze(-1)        

        # filter out the padded token
        b, seq = language_att.shape
        binary_mask = torch.ones(b, seq).bool().cuda()
        for i in range(b):
            binary_mask[i, :valid_len[i]] = 0
        language_att = language_att.masked_fill(binary_mask, -1e4)
        language_att = F.softmax(language_att, dim=-1)
        return language_att
    
class Blind_Transformer_MS_NOATT(nn.Module):
    """ A multi-scale "Blind" Transformer model for
        saliency prediction. The principal idea is to 
        compute the weights of tokens inside the image captions,
        and leverage pre-extracted multi-modal mapping for deriving
        the visual saliency maps.
    """
    def __init__(self, input_dim=4096, input_head=32, num_head=4, depth=4, img_h=36, img_w=48, max_len=70):
        super(Blind_Transformer_MS_NOATT, self).__init__()
        self.input_head = input_head
        self.input_dim = input_dim

        # learnable piror maps for patch attention
        self.learned_prior = Learned_Prior(16, img_h, img_w)

        # learnable piror maps for base attention
        self.learned_prior_base = Learned_Prior(16, 24, 24)

        # use a collection of convolutional layers to project the multi-head attention to saliency maps
        self.sal_decoder = nn.Sequential(nn.Conv2d(input_head+16, 32, kernel_size=5, padding=2, stride=1),
                                      nn.ReLU(),
                                      nn.Upsample(scale_factor=2),
                                      nn.Conv2d(32, 16, kernel_size=5, padding=2, stride=1),
                                      nn.ReLU(),
                                      nn.Upsample(scale_factor=2),
                                      nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1),
                                      nn.ReLU(),
                                      nn.Upsample(scale_factor=2),
                                      )
        
        self.sal_header = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1),
                                        nn.ReLU(),
                                        nn.Conv2d(16, 8, kernel_size=3, padding=1, stride=1),
                                        nn.ReLU(),
                                        nn.Conv2d(8, 1, kernel_size=3, padding=1, stride=1),
                                      )

    def forward(self, language, patch_mapping, base_mapping, valid_len):
        b, seq, d = language.shape
        language_att = torch.ones(b, seq).cuda()
        # filter out the padded token
        binary_mask = torch.ones(b, seq).bool().cuda()
        for i in range(b):
            binary_mask[i, :valid_len[i]] = 0
        language_att = language_att.masked_fill(binary_mask, -1e4)
        language_att = F.softmax(language_att, dim=-1)

        # weight the patch and base attention mapping based on the language attention
        language_att = language_att.view(b, seq, 1, 1, 1).expand(b, seq, self.input_head, 1, 1)
        patch_attention = (language_att*patch_mapping).sum(1)
        base_attention = (language_att*base_mapping).sum(1)
        
        # derive the saliency maps
        prior_map_patch = self.learned_prior(b)
        patch_attention = torch.cat([patch_attention, prior_map_patch], dim=1)
        patch_attention = self.sal_decoder(patch_attention)
        patch_attention = F.interpolate(patch_attention, (240, 320), mode='bilinear')

        prior_map_base = self.learned_prior_base(b)
        base_attention = torch.cat([base_attention, prior_map_base], dim=1)
        base_attention = self.sal_decoder(base_attention) # use shared decoder
        base_attention = F.interpolate(base_attention, (240, 320), mode='bilinear')

        visual_attention = torch.cat([base_attention, patch_attention], dim=1)
        saliency_map = self.sal_header(visual_attention).squeeze(1)
        b, h, w = saliency_map.shape 
        saliency_map = F.softmax(saliency_map.view(b, h*w), dim=-1).view(b, h, w)

        return saliency_map
    

class Blind_Transformer_MS_Reweight(nn.Module):
    """ A multi-scale "Blind" Transformer model for
        saliency prediction. It does have a visual module,
        but it is used to reweight the attention map predicted
        by the blind model.
    """
    def __init__(self, input_dim=4096, input_head=32, num_head=4, depth=4, img_h=36, img_w=48, max_len=70,
                 joint_training=False, reweight_module='dinet'):
        super(Blind_Transformer_MS_Reweight, self).__init__()
        self.input_head = input_head
        self.joint_training = joint_training

        if self.joint_training:
            hidden_dim = 512
            self.lang_proj = nn.Linear(input_dim, hidden_dim)
        else:
            hidden_dim = input_dim

        # initialize (and freeze) pos_embed by sin-cos embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, hidden_dim), requires_grad=False)  # fixed sin-cos embedding 
        pos_embed = get_1d_sincos_pos_embed_from_grid(hidden_dim, np.arange(max_len))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # computing the attention weights for language tokens
        self.language_decoder = nn.ModuleList([Block(hidden_dim, num_head, 4., qkv_bias=True, 
                                           qk_scale=None, norm_layer=nn.LayerNorm)
                                           for _ in range(depth)])
        self.language_attn = nn.Linear(hidden_dim, 1)

        
        # learnable piror maps for patch attention
        self.learned_prior = Learned_Prior(16, img_h, img_w)

        # learnable piror maps for base attention
        self.learned_prior_base = Learned_Prior(16, 24, 24)

        # use a collection of convolutional layers to project the multi-head attention to saliency maps
        self.sal_decoder = nn.Sequential(nn.Conv2d(input_head+16, 32, kernel_size=5, padding=2, stride=1),
                                      nn.ReLU(),
                                      nn.Upsample(scale_factor=2),
                                      nn.Conv2d(32, 16, kernel_size=5, padding=2, stride=1),
                                      nn.ReLU(),
                                      nn.Upsample(scale_factor=2),
                                      nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1),
                                      nn.ReLU(),
                                      nn.Upsample(scale_factor=2),
                                      )
        
        self.sal_header = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1),
                                        nn.ReLU(),
                                        nn.Conv2d(16, 8, kernel_size=3, padding=1, stride=1),
                                        nn.ReLU(),
                                        nn.Conv2d(8, 1, kernel_size=3, padding=1, stride=1),
                                      )
        
        for module in [self.language_decoder, self.language_attn, self.learned_prior, self.learned_prior_base,
                       self.sal_decoder, self.sal_header]:
            for para in module.parameters():
                para.requires_grad = False if not joint_training else True

        if reweight_module == 'dinet':
            self.reweight_module = DINet()
        else:
            self.reweight_module = TranSalNet()

    def forward(self, language, patch_mapping, base_mapping, valid_len, img=None, get_lang_att=False):
        if self.joint_training:
            language = self.lang_proj(language)
        x = language + self.pos_embed
        for blk in self.language_decoder:
            x = blk(x, is_causal=False)
            x = F.dropout(x, 0.3)        
        language_att = self.language_attn(x).squeeze(-1)
        
        # filter out the padded token
        b, seq = language_att.shape
        binary_mask = torch.ones(b, seq).bool().cuda()
        for i in range(b):
            binary_mask[i, :valid_len[i]] = 0
        language_att = language_att.masked_fill(binary_mask, -1e4)
        language_att = F.softmax(language_att, dim=-1)
        raw_language_att = language_att.clone()

        # weight the patch and base attention mapping based on the language attention
        language_att = language_att.view(b, seq, 1, 1, 1).expand(b, seq, self.input_head, 1, 1)
        patch_attention = (language_att*patch_mapping).sum(1)
        base_attention = (language_att*base_mapping).sum(1)
        
        # derive the saliency maps
        prior_map_patch = self.learned_prior(b)
        patch_attention = torch.cat([patch_attention, prior_map_patch], dim=1)
        patch_attention = self.sal_decoder(patch_attention)
        patch_attention = F.interpolate(patch_attention, (240, 320), mode='bilinear')

        prior_map_base = self.learned_prior_base(b)
        base_attention = torch.cat([base_attention, prior_map_base], dim=1)
        base_attention = self.sal_decoder(base_attention) # use shared decoder
        base_attention = F.interpolate(base_attention, (240, 320), mode='bilinear')

        visual_attention = torch.cat([base_attention, patch_attention], dim=1)
        ori_saliency_map = self.sal_header(visual_attention)
        b, _, h, w = ori_saliency_map.shape 
        ori_saliency_map = F.softmax(ori_saliency_map.view(b, h*w), dim=-1).view(b, h, w)

        # reweighting
        weight_map = self.reweight_module(img).squeeze(1)
        saliency_map = ori_saliency_map*weight_map # standard
        # saliency_map = (1+ori_saliency_map)*weight_map # with residual 

        # saliency_map = F.softmax(saliency_map.view(b, h*w), dim=-1).view(b, h, w)

        if get_lang_att:
            ori_saliency_map = ori_saliency_map.squeeze(1)
            # ori_saliency_map = F.softmax(ori_saliency_map.view(b, h*w), dim=-1).view(b, h, w)
            return saliency_map, raw_language_att, ori_saliency_map, weight_map.squeeze(1)
        else:
            return saliency_map
            # return weight_map


class Blind_Transformer_MS_Reweight_Lang(nn.Module):
    """ A multi-scale "Blind" Transformer model for
        saliency prediction. Instead of using a visual module for
        reweighting, it takes advantage of a "blind" module that takes
        into account the attention weights learned before.
    """
    def __init__(self, input_dim=4096, input_head=32, num_head=4, depth=4, img_h=36, img_w=48, max_len=70):
        super(Blind_Transformer_MS_Reweight_Lang, self).__init__()
        self.input_head = input_head
        # initialize (and freeze) pos_embed by sin-cos embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, input_dim), requires_grad=False)  # fixed sin-cos embedding 
        pos_embed = get_1d_sincos_pos_embed_from_grid(input_dim, np.arange(max_len))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # computing the attention weights for language tokens
        self.language_decoder = nn.ModuleList([Block(input_dim, num_head, 4., qkv_bias=True, 
                                           qk_scale=None, norm_layer=nn.LayerNorm)
                                           for _ in range(depth)])
        self.language_attn = nn.Linear(input_dim, 1)

        
        # learnable piror maps for patch attention
        self.learned_prior = Learned_Prior(16, img_h, img_w)

        # learnable piror maps for base attention
        self.learned_prior_base = Learned_Prior(16, 24, 24)

        # use a collection of convolutional layers to project the multi-head attention to saliency maps
        self.sal_decoder = nn.Sequential(nn.Conv2d(input_head+16, 32, kernel_size=5, padding=2, stride=1),
                                      nn.ReLU(),
                                      nn.Upsample(scale_factor=2),
                                      nn.Conv2d(32, 16, kernel_size=5, padding=2, stride=1),
                                      nn.ReLU(),
                                      nn.Upsample(scale_factor=2),
                                      nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1),
                                      nn.ReLU(),
                                      nn.Upsample(scale_factor=2),
                                      )
        
        self.sal_header = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1),
                                        nn.ReLU(),
                                        nn.Conv2d(16, 8, kernel_size=3, padding=1, stride=1),
                                        nn.ReLU(),
                                        nn.Conv2d(8, 1, kernel_size=3, padding=1, stride=1),
                                      )
        
        for module in [self.language_decoder, self.language_attn, self.learned_prior, self.learned_prior_base,
                       self.sal_decoder, self.sal_header]:
            for para in module.parameters():
                para.requires_grad = False

        # reweighting
        hidden_dim = 512
        self.reweight_pos_embed = nn.Parameter(torch.zeros(1, max_len, hidden_dim), requires_grad=False)  # fixed sin-cos embedding 
        reweight_pos_embed = get_1d_sincos_pos_embed_from_grid(hidden_dim, np.arange(max_len))
        self.reweight_pos_embed.data.copy_(torch.from_numpy(reweight_pos_embed).float().unsqueeze(0))
        self.reweight_proj = nn.Linear(input_dim, hidden_dim)
        self.reweight_language_decoder = nn.ModuleList([Block(hidden_dim, num_head, 4., qkv_bias=True, 
                                           qk_scale=None, norm_layer=nn.LayerNorm)
                                           for _ in range(depth)])

        self.reweight_attn = nn.Linear(hidden_dim, 1)
        self.reweight_sal_decoder =  nn.Sequential(nn.Conv2d(input_head+16, 32, kernel_size=5, padding=2, stride=1),
                                      nn.ReLU(),
                                      nn.Upsample(scale_factor=2),
                                      nn.Conv2d(32, 16, kernel_size=5, padding=2, stride=1),
                                      nn.ReLU(),
                                      nn.Upsample(scale_factor=2),
                                      nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1),
                                      nn.ReLU(),
                                      nn.Upsample(scale_factor=2),
                                      )
        self.reweight_sal_header = nn.Sequential(nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1),
                                        nn.ReLU(),
                                        nn.Conv2d(16, 8, kernel_size=3, padding=1, stride=1),
                                        nn.ReLU(),
                                        nn.Conv2d(8, 1, kernel_size=3, padding=1, stride=1),
                                      )
        # learnable piror maps for patch attention
        self.reweight_learned_prior = Learned_Prior(16, img_h, img_w)

        # learnable piror maps for base attention
        self.reweight_learned_prior_base = Learned_Prior(16, 24, 24)

    def forward(self, language, patch_mapping, base_mapping, valid_len, img=None, get_lang_att=False):
        x = language + self.pos_embed
        for blk in self.language_decoder:
            x = blk(x, is_causal=False)
            x = F.dropout(x, 0.3)        
        language_att = self.language_attn(x).squeeze(-1)
        
        # filter out the padded token
        b, seq = language_att.shape
        binary_mask = torch.ones(b, seq).bool().cuda()
        for i in range(b):
            binary_mask[i, :valid_len[i]] = 0
        language_att = language_att.masked_fill(binary_mask, -1e4)
        language_att = F.softmax(language_att, dim=-1)
        raw_language_att = language_att.clone()

        # weight the patch and base attention mapping based on the language attention
        language_att = language_att.view(b, seq, 1, 1, 1).expand(b, seq, self.input_head, 1, 1)
        patch_attention = (language_att*patch_mapping).sum(1)
        base_attention = (language_att*base_mapping).sum(1)
        
        # derive the saliency maps
        prior_map_patch = self.learned_prior(b)
        patch_attention = torch.cat([patch_attention, prior_map_patch], dim=1)
        patch_attention = self.sal_decoder(patch_attention)
        patch_attention = F.interpolate(patch_attention, (240, 320), mode='bilinear')

        prior_map_base = self.learned_prior_base(b)
        base_attention = torch.cat([base_attention, prior_map_base], dim=1)
        base_attention = self.sal_decoder(base_attention) # use shared decoder
        base_attention = F.interpolate(base_attention, (240, 320), mode='bilinear')

        visual_attention = torch.cat([base_attention, patch_attention], dim=1)
        ori_saliency_map = self.sal_header(visual_attention)
        b, _, h, w = ori_saliency_map.shape 
        ori_saliency_map = F.softmax(ori_saliency_map.view(b, h*w), dim=-1).view(b, h, w)

        # reweighting with the same language features
        reweight_lang = self.reweight_proj(language)
        reweight_x = reweight_lang + self.reweight_pos_embed
        for blk in self.reweight_language_decoder:
            reweight_x = blk(reweight_x, is_causal=False)
            reweight_x = F.dropout(reweight_x, 0.3)        
        reweight_attn = self.reweight_attn(reweight_x).squeeze(-1)
        reweight_attn = reweight_attn.masked_fill(binary_mask, -1e4)
        reweight_attn = F.softmax(reweight_attn, dim=-1)        

        reweight_attn = reweight_attn.view(b, seq, 1, 1, 1).expand(b, seq, self.input_head, 1, 1)
        reweight_patch_attention = (reweight_attn*patch_mapping).sum(1)
        reweight_base_attention = (reweight_attn*base_mapping).sum(1)
        
        # derive the saliency maps
        reweight_prior_map_patch = self.reweight_learned_prior(b)
        reweight_patch_attention = torch.cat([reweight_patch_attention, reweight_prior_map_patch], dim=1)
        reweight_patch_attention = self.reweight_sal_decoder(reweight_patch_attention)
        reweight_patch_attention = F.interpolate(reweight_patch_attention, (240, 320), mode='bilinear')

        reweight_prior_map_base = self.reweight_learned_prior_base(b)
        reweight_base_attention = torch.cat([reweight_base_attention, reweight_prior_map_base], dim=1)
        reweight_base_attention = self.reweight_sal_decoder(reweight_base_attention) # use shared decoder
        reweight_base_attention = F.interpolate(reweight_base_attention, (240, 320), mode='bilinear')

        reweight_attention = torch.cat([reweight_base_attention, reweight_patch_attention], dim=1)
        reweight_attention = self.reweight_sal_header(reweight_attention)
        b, _, h, w = reweight_attention.shape 
        reweight_attention = F.softmax(reweight_attention.view(b, h*w), dim=-1).view(b, h, w)

        saliency_map = ori_saliency_map*reweight_attention
        # saliency_map = F.softmax(saliency_map.view(b, h*w), dim=-1).view(b, h, w)

        if get_lang_att:
            ori_saliency_map = ori_saliency_map.squeeze(1)
            # ori_saliency_map = F.softmax(ori_saliency_map.view(b, h*w), dim=-1).view(b, h, w)
            return saliency_map, raw_language_att, ori_saliency_map, reweight_attention.squeeze(1)
        else:
            return saliency_map
            # return weight_map


class Blind_Transformer_MRI(nn.Module):
    """ A "Blind" Transformer model for predicting the fMRI responses
        based on language features (and learned attention)
    """
    def __init__(self, input_dim=4096, num_head=4, depth=4, max_len=100, num_vertex=1000):
        super(Blind_Transformer_MRI, self).__init__()
        # initialize (and freeze) pos_embed by sin-cos embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len+1, input_dim), requires_grad=False)  # fixed sin-cos embedding 
        pos_embed = get_1d_sincos_pos_embed_from_grid(input_dim, np.arange(max_len+1))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim), requires_grad=True)

        # computing the attention weights for language tokens
        self.mri_decoder = Block(input_dim, num_head, 4., qkv_bias=True, 
                                           qk_scale=None, norm_layer=nn.LayerNorm)
        
        self.mri_head = nn.Linear(input_dim, num_vertex)

    def forward(self, lang_feat, lang_att=None, valid_len=None, get_attention=False):
        lang_feat = torch.cat([self.cls_token.expand(len(lang_feat), 1, lang_feat.shape[-1]), lang_feat], dim=1)
        x = lang_feat + self.pos_embed

        if get_attention:
            x, attn = self.mri_decoder(x, is_causal=False, return_attention=True, valid_len=valid_len)
            return attn[:, :, 0, 1:]

        x = self.mri_decoder(x, is_causal=False, valid_len=valid_len)
        cls_token = x[:, 0]
        pred = self.mri_head(cls_token)        
        return pred

class Blind_Transformer_MRI_v2(nn.Module):
    """ A "Blind" Transformer model for predicting the fMRI responses
        based on language features (and learned attention)
    """
    def __init__(self, input_dim=4096, num_head=4, depth=4, max_len=100, num_vertex=1000):
        super(Blind_Transformer_MRI_v2, self).__init__()
        # initialize (and freeze) pos_embed by sin-cos embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, input_dim), requires_grad=False)  # fixed sin-cos embedding 
        pos_embed = get_1d_sincos_pos_embed_from_grid(input_dim, np.arange(max_len))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # computing the attention weights for language tokens
        self.language_decoder = nn.ModuleList([Block(input_dim, num_head, 4., qkv_bias=True, 
                                           qk_scale=None, norm_layer=nn.LayerNorm)
                                           for _ in range(depth)])
        self.language_attn = nn.Linear(input_dim, 1)
        self.mri_head = nn.Linear(input_dim, num_vertex)

    def forward(self, lang_feat, lang_att=None, valid_len=None, get_attention=False):
        x = lang_feat + self.pos_embed

        for blk in self.language_decoder:
            x = blk(x, is_causal=False, valid_len=valid_len)
            x = F.dropout(x, 0.3)        
        language_att = self.language_attn(x).squeeze(-1)
        # filter out the padded token
        b, seq = language_att.shape
        binary_mask = torch.ones(b, seq).bool().cuda()
        for i in range(b):
            binary_mask[i, :valid_len[i]] = 0
        language_att = language_att.masked_fill(binary_mask, -1e4)
        language_att = F.softmax(language_att, dim=-1)

        lang_feat = (language_att.unsqueeze(-1)*lang_feat).sum(1)
        pred = self.mri_head(lang_feat)        
        if not get_attention:
            return pred
        else:
            return language_att


class Blind_Transformer_ASD(nn.Module):
    """ A "Blind" Transformer model for predicting the autism labels based on
        the weighted language caption.
    """
    def __init__(self, input_dim=4096, num_head=4, depth=4, max_len=100):
        super(Blind_Transformer_ASD, self).__init__()
        hidden_dim = 512
        # initialize (and freeze) pos_embed by sin-cos embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, input_dim), requires_grad=False)  # fixed sin-cos embedding 
        pos_embed = get_1d_sincos_pos_embed_from_grid(input_dim, np.arange(max_len))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # use a single Transformer block for dimensionality reduction
        self.language_projector = Block(input_dim, num_head, 4., qkv_bias=True, 
                                           qk_scale=None, norm_layer=nn.LayerNorm)   
        self.projector_layer = nn.Linear(input_dim, hidden_dim)     

        # decoding the weighted language tokens
        self.language_decoder = nn.ModuleList([Block(hidden_dim, num_head, 4., qkv_bias=True, 
                                           qk_scale=None, norm_layer=nn.LayerNorm)
                                           for _ in range(depth)])
        
        # use a single Transformer block with CLS token for predicting asd label
        self.asd_decoder = Block(hidden_dim, num_head, 4., qkv_bias=True, 
                                           qk_scale=None, norm_layer=nn.LayerNorm)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim), requires_grad=True)
        self.asd_head = nn.Linear(hidden_dim, 1)


    def forward(self, lang_feat, lang_att):
        x = lang_feat*lang_att.unsqueeze(-1) + self.pos_embed
        x = self.language_projector(x, is_causal=False)
        x = self.projector_layer(F.dropout(x, 0.3))

        for blk in self.language_decoder:
            x = blk(x, is_causal=False)
            x = F.dropout(x, 0.3)

        # concatenate with CLS for extracting ASD features
        x = torch.cat([self.cls_token.expand(len(x), 1, x.shape[-1]), x], dim=1)
        x = self.asd_decoder(x)
        pred = torch.sigmoid(self.asd_head(x[:, 0]))
        
        return pred
    

class Blind_Transformer_ASD_slim(nn.Module):
    """ A "Blind" Transformer model for predicting the autism labels based on
        the weighted language caption.
    """
    def __init__(self, input_dim=4096, num_head=4, depth=4, max_len=100):
        super(Blind_Transformer_ASD_slim, self).__init__()
        # initialize (and freeze) pos_embed by sin-cos embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len+1, input_dim), requires_grad=False)  # fixed sin-cos embedding 
        pos_embed = get_1d_sincos_pos_embed_from_grid(input_dim, np.arange(max_len+1))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # use a single Transformer block with CLS token for predicting asd label
        self.asd_decoder = Block(input_dim, num_head, 4., qkv_bias=True, 
                                           qk_scale=None, norm_layer=nn.LayerNorm)
        self.cls_token = nn.Parameter(torch.rand(1, 1, input_dim), requires_grad=True)
        self.asd_head = nn.Linear(input_dim, 1)


    def forward(self, lang_feat, lang_att):
        x = lang_feat*lang_att.unsqueeze(-1)
        # concatenate with CLS for extracting ASD features
        x = torch.cat([self.cls_token.expand(len(x), 1, x.shape[-1]), x], dim=1)
        x = x + self.pos_embed
        x = self.asd_decoder(x)
        pred = torch.sigmoid(self.asd_head(F.dropout(x[:, 0], 0.3)))
        
        return pred
    

class Blind_Transformer_ASD_temporal(nn.Module):
    """ A "Blind" Transformer model for predicting the autism labels based on
        the weighted language caption.
    """
    def __init__(self, input_dim=4096, hidden_dim=256, num_head=4, depth=4, max_len=100):
        super(Blind_Transformer_ASD_temporal, self).__init__()
        self.lang_projector = nn.Linear(input_dim, hidden_dim)
        self.temporal_projector = nn.Linear(4, 128)

        # initialize (and freeze) pos_embed by sin-cos embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len+1, hidden_dim+128), requires_grad=False)  # fixed sin-cos embedding 
        pos_embed = get_1d_sincos_pos_embed_from_grid(hidden_dim+128, np.arange(max_len+1))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # use a single Transformer block with CLS token for predicting asd label
        self.asd_decoder = nn.ModuleList([Block(hidden_dim+128, num_head, 4., qkv_bias=True, 
                                           qk_scale=None, norm_layer=nn.LayerNorm)
                                           for _ in range(depth)])
        
        self.cls_token = nn.Parameter(torch.rand(1, 1, hidden_dim+128), requires_grad=True)
        self.asd_head = nn.Linear(hidden_dim+128, 1)


    def forward(self, lang_feat, lang_att, valid_len):
        lang_feat = self.lang_projector(lang_feat)
        temporal_feat = self.temporal_projector(lang_att)
        x = torch.cat([lang_feat, temporal_feat], dim=-1)
        # concatenate with CLS for extracting ASD features
        x = torch.cat([self.cls_token.expand(len(x), 1, x.shape[-1]), x], dim=1)
        x = x + self.pos_embed
        for blk in self.asd_decoder:
            x = blk(x, valid_len=valid_len)
            x = F.dropout(x, 0.3)
        pred = torch.sigmoid(self.asd_head(x[:, 0]))
        
        return pred