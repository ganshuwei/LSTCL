import torch
import torch.nn as nn
from torch.nn import LayerNorm as norm_layer
class ConcatFusionModel(nn.Module):
    def __init__(self, num_classes = 7, embed_dim_list = [768,768]):
        nn.Module.__init__(self)
        self.fusion_feature_dim = sum(embed_dim_list)
        self.nr_actions = num_classes

        self.fusion = nn.Sequential(
            nn.Linear(self.fusion_feature_dim, self.fusion_feature_dim, bias=False),
            nn.BatchNorm1d(self.fusion_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.fusion_feature_dim, self.fusion_feature_dim, bias=False),
            nn.BatchNorm1d(self.fusion_feature_dim),
            nn.ReLU(inplace=True)
        )

        self.head = (
            nn.Linear(self.fusion_feature_dim,num_classes)
        )



    def fine_tune(self, restore_path, parameters_to_train=['classifier']):
        weights = torch.load(restore_path)['state_dict']
        new_weights = {}
        for k, v in weights.items():
            if not 'fc' in k and not 'classifier.4' in k:
                new_weights[k.replace('module.', '')] = v
        self.load_state_dict(new_weights, strict=False)
        print('Num of weights in restore dict {}'.format(len(new_weights.keys())))

        frozen_weights = 0
        for name, param in self.named_parameters():
            if not 'classifier.4' in name:
                param.requires_grad = False
                frozen_weights += 1
            else:
                print('Training : {}'.format(name))
        print('Number of frozen weights {}'.format(frozen_weights))
        assert frozen_weights != 0, 'You are trying to fine tune, but no weights are frozen!!! ' \
                                    'Check the naming convention of the parameters'
    
    def get_num_layers(self):
        return sum(1 for _ in self.modules() if isinstance(_, (nn.Linear, nn.BatchNorm1d, nn.ReLU)))



    def forward(self, feature_Surg, feature_App):
        concat_feature = torch.cat([feature_Surg, feature_App], -1)
        fusion_feature = self.fusion(concat_feature)
        class_output = self.head(fusion_feature)
        return class_output

