import os
import clip
import torch
from torch import nn

class RSID(nn.Module):
    def __init__(self, img_encoder='ViT-L/14', num_class=2, text_option=1):
        super().__init__()
        self.clip_model, self.preprocess = clip.load(img_encoder, device='cpu', jit=False)

    def forward(self, image_input):
        image_feats = self.clip_model.encode_image(image_input)
        return image_feats


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

