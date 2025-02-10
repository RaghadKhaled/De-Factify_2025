import os
import clip
import torch
from torch import nn

class RSID(nn.Module):
    def __init__(self, img_encoder='ViT-L/14', num_class=2, text_option=1):
        super().__init__()
        self.clip_model, self.preprocess = clip.load(img_encoder, device='cpu', jit=False)

        if num_class == 2:

            if text_option == 1:
                self.text_input = clip.tokenize(['Real', 'Synthetic'])
                self.test_txt_input = torch.cat([clip.tokenize(f"Real"), clip.tokenize(f"Synthetic")]).cuda()

            elif text_option == 2:

                self.text_input = torch.cat([clip.tokenize(f"A real photo."), clip.tokenize(f"A synthetic photo.")]).cuda()

                self.test_txt_input = torch.cat([clip.tokenize(f"A real photo."), clip.tokenize(f"A synthetic photo.")]).cuda()

        elif num_class == 6:

            if text_option == 1:
                self.text_input = clip.tokenize(['Real', 'Synthetic SD21', 'Synthetic SDXL', 'Synthetic SD3', 'Synthetic DALLE3', 'Synthetic Midjourney'])
                self.test_txt_input = torch.cat([clip.tokenize(f"Real"), clip.tokenize(f"Synthetic SD21"), clip.tokenize(f"Synthetic SDXL"), clip.tokenize(f"Synthetic SD3"), clip.tokenize(f"Synthetic DALLE3"), clip.tokenize(f"Synthetic Midjourney")]).cuda()


            elif text_option == 2:

                self.text_input = torch.cat([
                    clip.tokenize(f"A real photo."), 
                    clip.tokenize(f"A synthetic SD21 photo."), 
                    clip.tokenize(f"A synthetic SDXL photo."), 
                    clip.tokenize(f"A synthetic SD3 photo."), 
                    clip.tokenize(f"A synthetic DALLE3 photo."), 
                    clip.tokenize(f"A synthetic Midjourney photo.")]).cuda()


                self.test_txt_input = torch.cat([
                    clip.tokenize(f"A real photo."), 
                    clip.tokenize(f"A synthetic SD21 photo."), 
                    clip.tokenize(f"A synthetic SDXL photo."), 
                    clip.tokenize(f"A synthetic SD3 photo."), 
                    clip.tokenize(f"A synthetic DALLE3 photo."), 
                    clip.tokenize(f"A synthetic Midjourney photo.")]).cuda()
                    

    def forward(self, image_input, text_input=" ", num_class=2, isTrain=True, isTest=False, text_option=1):

        if isTrain:

            if text_option == 2:

                text_input = torch.squeeze(text_input)

                logits_per_image, _ = self.clip_model(image_input, text_input)
                return logits_per_image, None
            else:
                logits_per_image, _ = self.clip_model(image_input, self.text_input.to(image_input.device))
                return None, logits_per_image
        elif isTest:

            if num_class == 2:
                if text_option == 1:
                    # option 1
                    self.test_txt_input = torch.cat([
                        clip.tokenize(f"Real"), 
                        clip.tokenize(f"Synthetic")]).cuda()

                elif text_option == 2:
                    self.test_txt_input = torch.cat([
                        clip.tokenize(f"A real photo."), 
                        clip.tokenize(f"A synthetic photo.")]).cuda()

            elif num_class == 6:

                if text_option == 1:
                    # option 1
                    self.test_txt_input = torch.cat([
                        clip.tokenize(f"Real"), 
                        clip.tokenize(f"Synthetic SD21"), 
                        clip.tokenize(f"Synthetic SDXL"), 
                        clip.tokenize(f"Synthetic SD3"), 
                        clip.tokenize(f"Synthetic DALLE3"), 
                        clip.tokenize(f"Synthetic Midjourney")]).cuda()
                    
                elif text_option == 2:
                    self.test_txt_input = torch.cat([
                        clip.tokenize(f"A real photo."), 
                        clip.tokenize(f"A synthetic SD21 photo."), 
                        clip.tokenize(f"A synthetic SDXL photo."), 
                        clip.tokenize(f"A synthetic SD3 photo."), 
                        clip.tokenize(f"A synthetic DALLE3 photo."), 
                        clip.tokenize(f"A synthetic Midjourney photo.")]).cuda()

            image_feats = self.clip_model.encode_image(image_input)
            image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)


            text_feats = self.clip_model.encode_text(self.test_txt_input)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

            return text_feats, image_feats



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

