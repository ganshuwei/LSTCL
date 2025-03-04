import torch
import torch.nn as nn
class CombinedModel(nn.Module):
    def __init__(self, model, rgbmodel, fusionmodel):
        super(CombinedModel, self).__init__()
        self.model = model
        self.rgbmodel = rgbmodel
        self.fusionmodel = fusionmodel
        #self.fusionmodel = fusionmodel

    def forward(self, samples, current_frame):
        # Step 1: 主模型提取特征
        feature_model,output_class_model = self.model(samples)
        feature_rgb,output_class_rgb = self.rgbmodel(current_frame)
        output_class_fusion = self.fusionmodel(feature_model,feature_rgb)


        # Step 2: RGB 模型提取特征
        #output_class_rgb, feature_rgb = self.rgbmodel(current_frame)

        # Step 3: 融合模型融合特征
        #output_class_fusion = self.fusionmodel(feature_model, feature_rgb)

        return output_class_model, output_class_rgb,output_class_fusion
    

    def get_num_layers(self):
        # 假设每个子模型都有 get_num_layers 方法
        model_layers = self.model.get_num_layers() if hasattr(self.model, "get_num_layers") else 0
        rgb_layers = self.rgbmodel.get_num_layers() if hasattr(self.rgbmodel, "get_num_layers") else 0
        #fusion_layers = self.fusionmodel.get_num_layers() if hasattr(self.fusionmodel, "get_num_layers") else 0
        return model_layers + rgb_layers

    @torch.jit.ignore
    def no_weight_decay(self):
        return self.model.no_weight_decay()

