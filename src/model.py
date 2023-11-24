import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers



class CLIPModelModified(nn.Module):
    def __init__(self, clip_model):
        super(CLIPModelModified, self).__init__()
        self.clip_model = clip_model


    def forward(self, inputs):

        outputs = self.clip_model(**inputs)
        outputs = {'text': outputs.text_embeds, 'image': outputs.image_embeds}
        return outputs
    




class MultiModalSiameseNetwork(nn.Module):
    def __init__(self, multimodal_network):
        super(MultiModalSiameseNetwork, self).__init__()
        self.multimodal_network = multimodal_network


    def forward_once(self, inputs, normalize_outputs=True):
        
        outputs = self.multimodal_network(inputs)
        fused_outputs = torch.cat((outputs['text'], outputs['image']), dim=1)

        if normalize_outputs:
            fused_outputs = F.normalize(fused_outputs, dim=1)

        return fused_outputs



    def forward(self, inputs1, inputs2, normalize_outputs=True):

        outputs1 = self.forward_once(inputs1, normalize_outputs=normalize_outputs)
        outputs2 = self.forward_once(inputs2, normalize_outputs=normalize_outputs)

        return outputs1, outputs2




class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):

        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
