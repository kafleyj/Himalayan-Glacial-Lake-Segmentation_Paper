import torch
from torch import nn
import torch.nn.functional as F

def build_model(name):
    if name in ('unet','deeplabv3plus'):
        import segmentation_models_pytorch as smp
        if name=='unet': return smp.Unet(encoder_name='resnet34',encoder_weights=None,in_channels=3,classes=1)
        return smp.DeepLabV3Plus(encoder_name='resnet34',encoder_weights=None,in_channels=3,classes=1)
    if name=='segformer_b0':
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        cfg=SegformerConfig(num_labels=2, depths=[2,2,2,2],hidden_sizes=[32,64,160,256],num_attention_heads=[1,2,5,8],sr_ratios=[8,4,2,1],patch_sizes=[7,3,3,3],strides=[4,2,2,2],mlp_ratios=[4,4,4,4],decoder_hidden_size=256)
        return SegformerBinary(SegformerForSemanticSegmentation(cfg))
    raise ValueError('Unknown architecture: '+name)

class SegformerBinary(nn.Module):
    def __init__(self, base): super().__init__(); self.base=base
    def forward(self,x):
        logits=self.base(pixel_values=x).logits
        logits=F.interpolate(logits,size=x.shape[-2:],mode='bilinear',align_corners=False)
        return logits[:,1:2]-logits[:,0:1]
