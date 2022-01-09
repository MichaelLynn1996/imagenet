import math
import timm
import torch
import torch.nn as nn
from timm.models import PatchEmbed, PosConv
from timm.models.registry import register_model
from timm.models.layers import to_2tuple, trunc_normal_


class PCViT(nn.Module):
    def __init__(self, base_vit='vit_base_patch32_384', pretrained=False, **kwargs):
        super(PCViT, self).__init__()

        # initial base vision transformer model
        self.v = timm.create_model(base_vit, pretrained=pretrained, embed_layer=PatchEmbed, **kwargs)

        if self.v.num_tokens == 2:
            del self.v.dist_token
        del self.v.cls_token

        self.v.pos_embed = PosConv(self.v.num_features, self.v.num_features)
        fan_out = self.v.pos_embed.proj.kernel_size[0] * self.v.pos_embed.proj.kernel_size[1] * self.v.pos_embed.proj.out_channels
        fan_out //= self.v.pos_embed.proj.groups
        self.v.pos_embed.proj.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if self.v.pos_embed.proj.bias is not None:
            self.v.pos_embed.proj.bias.data.zero_()

    def forward(self, x):
        B = x.shape[0]
        x, size = self.v.patch_embed(x)
        x = self.v.pos_drop(x)
        x = self.v.blocks(x)
        x = self.v.pos_embed(x)
        x = self.v.norm(x)
        # if self.dist_token is None:
        #     return self.pre_logits(x[:, 0])
        # else:
        #     return x[:, 0], x[:, 1]
        x = x.mean(dim=1)  # GAP here
        return self.v.head(x)

    def forward_features(self, x):
        B = x.shape[0]
        for i, (embed, drop, blocks, pos_blk) in enumerate(
                zip(self.patch_embeds, self.pos_drops, self.blocks, self.pos_block)):
            x, size = embed(x)
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                if j == 0:
                    x = pos_blk(x, size)  # PEG here
            if i < len(self.depths) - 1:
                x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
        x = self.norm(x)
        return x.mean(dim=1)  # GAP here


@register_model
def pcivt(base_vit='vit_base_patch32_384', pretrained=False, **kwargs):
    model = PCViT(base_vit, pretrained, **kwargs)
    return model
