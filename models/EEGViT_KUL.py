import torch
import transformers
from transformers import ViTModel
import torch
from torch import nn
import transformers

class EEGViT_KUL(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=256,
            kernel_size=(1, 9),
            stride=(1, 9),
            padding=(0,2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)
        config = transformers.ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            num_channels=256,
            image_size=(64,14),
            patch_size=(4,1)
        )
        model = ViTModel(config)
        model.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(4, 1), stride=(4, 1), padding=(0,0), groups=256)
        #model.pooler.activation = torch.nn.Sequential(torch.nn.Dropout(p=0.1),torch.nn.Linear(768, 64, bias=True),torch.nn.ReLU(),torch,nn.Linear(64, 2, bias=True))
        model.pooler.activation = torch.nn.Sequential(torch.nn.Dropout(p=0.1),
                                                       torch.nn.Linear(768,2,bias=True))
        self.ViT = model
        
    def forward(self,x):
        x=self.conv1(x)
        x=self.batchnorm1(x)
        x=self.ViT(x).pooler_output
        
        return x
    

class EEGViT_KUL_pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=256,
            kernel_size=(1, 9),
            stride=(1, 9),
            padding=(0,2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)
        model_name = "/media/nchen/CYB/EEGViT/vit-base-patch16-224"
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 256})
        config.update({'image_size': (64,14)})
        config.update({'patch_size': (4,1)})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(4, 1), stride=(4, 1), padding=(0,0), groups=256)
        model.classifier=torch.nn.Sequential(torch.nn.Linear(768,1000,bias=True),
                                     torch.nn.Dropout(p=0.1),
                                     torch.nn.Linear(1000,2,bias=True))
        self.ViT = model
        
    def forward(self,x):
        x=self.conv1(x)
        x=self.batchnorm1(x)
        x=self.ViT.forward(x).logits
        
        return x
    

class EEGViT_KUL_CL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=256,
            kernel_size=(1, 9),
            stride=(1, 9),
            padding=(0,2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)
        config = transformers.ViTConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            num_channels=256,
            image_size=(64,14),
            patch_size=(4,1)
        )
        model = ViTModel(config)
        model.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(4, 1), stride=(4, 1), padding=(0,0), groups=256)
        #model.pooler.activation = torch.nn.Sequential(torch.nn.Dropout(p=0.1),torch.nn.Linear(768, 64, bias=True),torch.nn.ReLU(),torch,nn.Linear(64, 2, bias=True))
        model.pooler.activation = torch.nn.Sequential(torch.nn.Dropout(p=0.1),
                                                       torch.nn.Linear(768,128,bias=True))
        self.ViT = model
        self.eeg_dim = 64
        self.feat_dim = args.feature_dim
        
    def forward(self,total):
        x = total[:, :, :self.eeg_dim, :]
        att_stimulus = total[:, :, self.eeg_dim:self.eeg_dim+self.feat_dim, :]
        unatt_stimulus = total[:, :, self.eeg_dim+self.feat_dim:self.eeg_dim+self.feat_dim*2, :]
        x=self.conv1(x)
        x=self.batchnorm1(x)
        x=self.ViT(x).pooler_output
        feature = torch.cat((x.unsqueeze(dim=1), att_stimulus.squeeze(dim=1), unatt_stimulus.squeeze(dim=1)), dim=1)
        
        return feature  #(batch, dim_total, time)
    

class EEGViT_KUL_pretrained_CL(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=256,
            kernel_size=(1, 9),
            stride=(1, 9),
            padding=(0,2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)
        model_name = "/media/nchen/CYB/EEGViT/vit-base-patch16-224"
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 256})
        config.update({'image_size': (64,14)})
        config.update({'patch_size': (4,1)})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(4, 1), stride=(4, 1), padding=(0,0), groups=256)
        # model.classifier=torch.nn.Sequential(torch.nn.Linear(768,1000,bias=True),
        #                              torch.nn.Dropout(p=0.1),
        #                              torch.nn.Linear(1000,128,bias=True))
        model.classifier=torch.nn.Sequential(torch.nn.Linear(768,128,bias=True))

        self.ViT = model
        self.eeg_dim = 64
        self.feat_dim = args.feature_dim
        
    def forward(self,total):
        x = total[:, :, :self.eeg_dim, :]
        att_stimulus = total[:, :, self.eeg_dim:self.eeg_dim+self.feat_dim, :]
        unatt_stimulus = total[:, :, self.eeg_dim+self.feat_dim:self.eeg_dim+self.feat_dim*2, :]
        
        x=self.conv1(x)
        x=self.batchnorm1(x)
        x=self.ViT.forward(x).logits

        feature = torch.cat((x.unsqueeze(dim=1), att_stimulus.squeeze(dim=1), unatt_stimulus.squeeze(dim=1)), dim=1)
        
        return feature

class EEGViT_KUL_pretrained_Restruct(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=256,
            kernel_size=(1, 9),
            stride=(1, 9),
            padding=(0,2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)
        model_name = "/media/nchen/CYB/EEGViT/vit-base-patch16-224"
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 256})
        config.update({'image_size': (64,14)})
        config.update({'patch_size': (4,1)})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(4, 1), stride=(4, 1), padding=(0,0), groups=256)
        # model.classifier=torch.nn.Sequential(torch.nn.Linear(768,1000,bias=True),
        #                              torch.nn.Dropout(p=0.1),
        #                              torch.nn.Linear(1000,128,bias=True))
        model.classifier=torch.nn.Sequential(torch.nn.Linear(768, 128, bias=True))


        self.ViT = model
        self.eeg_dim = 64
        self.feat_dim = args.feature_dim
        
    def forward(self,total):
        x = total[:, :, :self.eeg_dim, :]
        att_stimulus = total[:, :, self.eeg_dim:self.eeg_dim+self.feat_dim, :]
        unatt_stimulus = total[:, :, self.eeg_dim+self.feat_dim:self.eeg_dim+self.feat_dim*2, :]
        
        x=self.conv1(x)
        x=self.batchnorm1(x)
        x=self.ViT.forward(x).logits

        feature = torch.cat((x.unsqueeze(dim=1), att_stimulus.squeeze(dim=1), unatt_stimulus.squeeze(dim=1)), dim=1)
        
        return feature
    
class EEGViT_KUL_pretrained_wav2vec(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=256,
            kernel_size=(1, 9),
            stride=(1, 9),
            padding=(0, 2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)
        model_name = "/media/nchen/CYB/EEGViT/vit-base-patch16-224"
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 256})
        config.update({'image_size': (64, 14)})
        config.update({'patch_size': (4, 1)})
        config.update({'output_hidden_states': True})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config,
                                                                       ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(4, 1), stride=(4, 1),
                                                                           padding=(0, 0), groups=256)
        # model.classifier=torch.nn.Sequential(torch.nn.Linear(768,1000,bias=True),
        #                              torch.nn.Dropout(p=0.1),
        #                              torch.nn.Linear(1000,128,bias=True))
        model.classifier = torch.nn.Sequential(torch.nn.Linear(768, 128, bias=True))
        self.avgpooling = torch.nn.AvgPool1d(6)

        self.ViT = model
        self.eeg_dim = 64
        self.feat_dim = args.feature_dim

    def forward(self, x):

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.ViT.forward(x)
        output = x.logits
        feat = list(x.hidden_states)
        feature = torch.tensor([]).to(output.device)
        for i in range(len(feat)):
            feature = torch.concatenate([feature, feat[i][:, 0, :].unsqueeze(1)], dim=1)

        feature = self.avgpooling(feature)

        return feature, output

class EEGViT_KUL_pretrained_wav2vec_frequency_dropout(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=256,
            kernel_size=(1, 9),
            stride=(1, 9),
            padding=(0, 2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)
        # model_name =  "/media/nchen/CYB/EEGViT/vit-base-patch16-224"
        model_name = args.PT_model
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 256})
        config.update({'image_size': (64, 14)})
        config.update({'patch_size': (4, 1)})
        config.update({'output_hidden_states': True})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config,
                                                                       ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(4, 1), stride=(4, 1),
                                                                           padding=(0, 0), groups=256)
        # model.classifier=torch.nn.Sequential(torch.nn.Linear(768,1000,bias=True),
        #                              torch.nn.Dropout(p=0.1),
        #                              torch.nn.Linear(1000,128,bias=True))
        model.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.2, inplace=False),
                                                torch.nn.Linear(768, 128, bias=True))
        self.avgpooling = torch.nn.AvgPool1d(6)
        self.dropout = torch.nn.Dropout(p=0.2, inplace=False)

        self.ViT = model
        self.eeg_dim = 64
        self.feat_dim = args.feature_dim

    def forward(self, x):
        b, _, band, f, t = x.shape
        x = torch.reshape(x, (-1, 1, 64, 128))

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.ViT.forward(x)
        output = x.logits
        # b, d=x.hidden_states
        feat = list(x.hidden_states)
        feature = torch.tensor([]).to(output.device)
        for i in range(len(feat)):
            feature = torch.concatenate([feature, self.dropout(feat[i][:, 0, :].unsqueeze(1))], dim=1)

        feature = self.avgpooling(feature)
        feature = torch.reshape(feature, (b, -1, band, t))
        output = torch.reshape(output, (b, band, t))
        return feature, output
    
class EEGViT_DTU_pretrained_wav2vec_frequency(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=256,
            kernel_size=(1, 9),
            stride=(1, 9),
            padding=(0, 2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)
        # model_name = "D:/EEGViT/vit-base-patch16-224"
        model_name = args.PT_model
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 256})
        config.update({'image_size': (64, 14)})
        config.update({'patch_size': (4, 1)})
        config.update({'output_hidden_states': True})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config,
                                                                       ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(4, 1), stride=(4, 1),
                                                                           padding=(0, 0), groups=256)
        # model.classifier=torch.nn.Sequential(torch.nn.Linear(768,1000,bias=True),
        #                              torch.nn.Dropout(p=0.1),
        #                              torch.nn.Linear(1000,128,bias=True))
        model.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.8, inplace=False),
                                               torch.nn.Linear(768, 128, bias=True))
        self.avgpooling = torch.nn.AvgPool1d(6)
        self.pooling = torch.nn.AdaptiveAvgPool2d((224,1))
        self.Conv1 = torch.nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1)
        self.Conv2 = torch.nn.Conv1d(in_channels=14, out_channels=128, kernel_size=1)
        self.transconv = torch.nn.ConvTranspose1d(in_channels=14, out_channels=128, kernel_size=1, stride=1, padding=0, output_padding=0, bias=True)

        self.ViT = model
        self.eeg_dim = 64
        self.feat_dim = args.feature_dim

    def forward(self, x):
        b, _, band, f, t = x.shape
        x = torch.reshape(x, (-1, 1, 64, 128))

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.ViT.forward(x)
        output = x.logits
        feat = list(x.hidden_states)
        feature = torch.tensor([]).to(output.device)
        for i in range(len(feat)):
            fea = self.pooling(feat[i][:, 1:, :])
            fea = fea.reshape((b, 16, 14))
            fea = self.Conv2(self.Conv1(fea).permute([0, 2, 1])).permute([0, 2, 1])
            # fea = torch.nn.functional.interpolate(fea, scale_factor=9)
            # fea = self.transconv(fea.permute([0, 2, 1]))
            feature = torch.concatenate([feature, fea], dim=1)

        #feature = self.avgpooling(feature)
        feature = torch.reshape(feature, (b, -1, band, t))
        output = torch.reshape(output, (b, band, t))
        return feature, output
    
class EEGViT_KUL_pretrained_wav2vec_projection(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=256,
            kernel_size=(1, 9),
            stride=(1, 9),
            padding=(0, 2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)
        model_name = "/media/nchen/CYB/EEGViT/vit-base-patch16-224"
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 256})
        config.update({'image_size': (64, 14)})
        config.update({'patch_size': (4, 1)})
        config.update({'output_hidden_states': True})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config,
                                                                       ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(4, 1), stride=(4, 1),
                                                                           padding=(0, 0), groups=256)
        # model.classifier=torch.nn.Sequential(torch.nn.Linear(768,1000,bias=True),
        #                              torch.nn.Dropout(p=0.1),
        #                              torch.nn.Linear(1000,128,bias=True))
        model.classifier = torch.nn.Sequential(torch.nn.Linear(768, 128, bias=True))
        self.avgpooling = torch.nn.AvgPool1d(6)
        self.project = nn.Linear(1, 64)

        self.ViT = model
        self.eeg_dim = 64
        self.feat_dim = args.feature_dim

    def forward(self, x):

        b, _, f, t = x.shape
        x = torch.reshape(x, (-1, 1, 64, 128))

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.ViT.forward(x)
        output = (x.logits).unsqueeze(-1)
        output = self.project(output)
        feat = list(x.hidden_states)
        feature = torch.tensor([]).to(output.device)
        for i in range(len(feat)):
            feature = torch.concatenate([feature, feat[i][:, 0, :].unsqueeze(1)], dim=1)

        feature = self.avgpooling(feature)


        feature = torch.reshape(feature, (b, -1, t))
        output = torch.reshape(output, (b, f, t))
        return feature, output
    
class EEGViT_KUL_pretrained_wav2vec_projection_all(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=256,
            kernel_size=(1, 9),
            stride=(1, 9),
            padding=(0, 2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)
        model_name = "/media/nchen/CYB/EEGViT/vit-base-patch16-224"
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 256})
        config.update({'image_size': (64, 14)})
        config.update({'patch_size': (4, 1)})
        config.update({'output_hidden_states': True})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config,
                                                                       ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(4, 1), stride=(4, 1),
                                                                           padding=(0, 0), groups=256)
        # model.classifier=torch.nn.Sequential(torch.nn.Linear(768,1000,bias=True),
        #                              torch.nn.Dropout(p=0.1),
        #                              torch.nn.Linear(1000,128,bias=True))
        model.classifier = torch.nn.Sequential(torch.nn.Linear(768, 128, bias=True))
        self.avgpooling = torch.nn.AvgPool1d(6)
        self.project = nn.Linear(1, 64)

        self.ViT = model
        self.eeg_dim = 64
        self.feat_dim = args.feature_dim

    def forward(self, x):

        b, _, f, t = x.shape
        x = torch.reshape(x, (-1, 1, 64, 128))

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.ViT.forward(x)
        output = (x.logits).unsqueeze(-1)
        output = self.project(output)
        feat = list(x.hidden_states)
        feature = torch.tensor([]).to(output.device)
        for i in range(len(feat)):
            feature = torch.concatenate([feature, feat[i][:, 0, :].unsqueeze(1)], dim=1)

        feature = self.avgpooling(feature)

        feature = self.project(feature.unsqueeze(-1))
        feature = torch.reshape(feature, (b, -1, f, t))
        output = torch.reshape(output, (b, f, t))
        return feature, output
    
class EEGViT_KUL_pretrained_wav2vec_frequency(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=256,
            kernel_size=(1, 9),
            stride=(1, 9),
            padding=(0, 2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)
        # model_name =  "/media/nchen/CYB/EEGViT/vit-base-patch16-224"
        model_name = args.PT_model
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 256})
        config.update({'image_size': (64, 14)})
        config.update({'patch_size': (4, 1)})
        config.update({'output_hidden_states': True})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config,
                                                                       ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(4, 1), stride=(4, 1),
                                                                           padding=(0, 0), groups=256)
        # model.classifier=torch.nn.Sequential(torch.nn.Linear(768,1000,bias=True),
        #                              torch.nn.Dropout(p=0.1),
        #                              torch.nn.Linear(1000,128,bias=True))
        model.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.4, inplace=True),
                                               torch.nn.Linear(768, 128, bias=True))
        self.avgpooling = torch.nn.AvgPool1d(6)
        self.dropout = torch.nn.Dropout(p=0.4)

        self.ViT = model
        self.eeg_dim = 64
        self.feat_dim = args.feature_dim

    def forward(self, x):
        b, _, band, f, t = x.shape
        x = torch.reshape(x, (-1, 1, 64, 128))

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.ViT.forward(x)
        output = x.logits
        feat = list(x.hidden_states)
        feature = torch.tensor([]).to(output.device)
        for i in range(len(feat)):
            feature = torch.concatenate([feature, self.dropout(feat[i][:, 0, :]).unsqueeze(1)], dim=1)

        feature = self.avgpooling(feature)
        feature = torch.reshape(feature, (b, -1, band, t))
        output = torch.reshape(output, (b, band, t))
        return feature, output

class EEGViT_KUL_pretrained_wav2vec_frequency_sum(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=256,
            kernel_size=(1, 9),
            stride=(1, 9),
            padding=(0, 2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)
        # model_name =  "/media/nchen/CYB/EEGViT/vit-base-patch16-224"
        model_name = args.PT_model
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 256})
        config.update({'image_size': (64, 14)})
        config.update({'patch_size': (4, 1)})
        config.update({'output_hidden_states': True})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config,
                                                                       ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(4, 1), stride=(4, 1),
                                                                           padding=(0, 0), groups=256)
        # model.classifier=torch.nn.Sequential(torch.nn.Linear(768,1000,bias=True),
        #                              torch.nn.Dropout(p=0.1),
        #                              torch.nn.Linear(1000,128,bias=True))
        model.classifier = torch.nn.Sequential( torch.nn.Linear(768, 384, bias=True),
                                                torch.nn.GELU(),
                                                torch.nn.Dropout(p=0.1),
                                                torch.nn.LayerNorm(384),
                                                torch.nn.Linear(384, 2, bias=True))
        self.avgpooling = torch.nn.AvgPool1d(6)
        self.dropout = torch.nn.Dropout(p=0.4)

        self.ViT = model
        self.eeg_dim = 64
        self.feat_dim = args.feature_dim

    def forward(self, x):
        b, _, band, f, t = x.shape
        x = torch.reshape(x, (-1, 1, 64, 128))

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.ViT.forward(x)
        output = x.logits
        #print(output.shape)
        feat = list(x.hidden_states)
        feature = torch.tensor([]).to(output.device)
        for i in range(len(feat)):
            feature = torch.concatenate([feature, self.dropout(feat[i][:, 0, :]).unsqueeze(1)], dim=1)

        feature = self.avgpooling(feature)
        feature = torch.reshape(feature, (b, -1, band, t))
        output = torch.reshape(output, (b, band, 2))
        return feature, output
    
class EEGViT_KUL_pretrained_wav2vec_frequency_sum_768(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=256,
            kernel_size=(1, 9),
            stride=(1, 9),
            padding=(0, 2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)
        # model_name =  "/media/nchen/CYB/EEGViT/vit-base-patch16-224"
        model_name = args.PT_model
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 256})
        config.update({'image_size': (64, 14)})
        config.update({'patch_size': (4, 1)})
        config.update({'output_hidden_states': True})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config,
                                                                       ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(4, 1), stride=(4, 1),
                                                                           padding=(0, 0), groups=256)
        # model.classifier=torch.nn.Sequential(torch.nn.Linear(768,1000,bias=True),
        #                              torch.nn.Dropout(p=0.1),
        #                              torch.nn.Linear(1000,128,bias=True))
        model.classifier = torch.nn.Sequential( torch.nn.Linear(768, 384, bias=True),
                                                torch.nn.GELU(),
                                                torch.nn.Dropout(p=0.1),
                                                torch.nn.LayerNorm(384),
                                                torch.nn.Linear(384, 2, bias=True))
        self.avgpooling = torch.nn.AvgPool1d(6)
        self.dropout = torch.nn.Dropout(p=0.4)

        self.ViT = model
        self.eeg_dim = 64
        self.feat_dim = args.feature_dim

    def forward(self, x):
        b, _, band, f, t = x.shape
        x = torch.reshape(x, (-1, 1, 64, 128))

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.ViT.forward(x)
        output = x.logits
        #print(output.shape)
        feat = list(x.hidden_states)
        feature = torch.tensor([]).to(output.device)
        for i in range(len(feat)):
            feature = torch.concatenate([feature, self.dropout(feat[i][:, 0, :]).unsqueeze(1)], dim=1)

        #feature = self.avgpooling(feature)
        feature = torch.reshape(feature, (b, -1, band, 768))
        output = torch.reshape(output, (b, band, 2))
        return feature, output
    
class EEGViT_KUL_pretrained_wav2vec_frequency_sum_768_cs(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=256,
            kernel_size=(1, 9),
            stride=(1, 9),
            padding=(0, 2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)
        # model_name =  "/media/nchen/CYB/EEGViT/vit-base-patch16-224"
        model_name = args.PT_model
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 256})
        config.update({'image_size': (64, 14)})
        config.update({'patch_size': (4, 1)})
        config.update({'output_hidden_states': True})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config,
                                                                       ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(4, 1), stride=(4, 1),
                                                                           padding=(0, 0), groups=256)
        # model.classifier=torch.nn.Sequential(torch.nn.Linear(768,1000,bias=True),
        #                              torch.nn.Dropout(p=0.1),
        #                              torch.nn.Linear(1000,128,bias=True))
        model.classifier = torch.nn.Sequential( torch.nn.Linear(768, 384, bias=True),
                                                torch.nn.GELU(),
                                                torch.nn.Dropout(p=0.1),
                                                torch.nn.LayerNorm(384),
                                                torch.nn.Linear(384, 2, bias=True))
        self.avgpooling = torch.nn.AvgPool1d(6)
        self.dropout = torch.nn.Dropout(p=0.4)

        self.ViT = model
        self.eeg_dim = 64
        self.feat_dim = args.feature_dim
        self.feature_layer_index = args.feature_layer_index

    def forward(self, x):
        b, _, band, f, t = x.shape
        x = torch.reshape(x, (-1, 1, 64, 128))

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.ViT.forward(x)
        output = x.logits
        #print(output.shape)
        feat = list(x.hidden_states)
        feature = torch.tensor([]).to(output.device)
        for i in range(len(self.feature_layer_index)):
            feature = torch.concatenate([feature, self.dropout(feat[int(self.feature_layer_index[i]/2)][:, 0, :]).unsqueeze(1)], dim=1)

        #feature = self.avgpooling(feature)
        feature = torch.reshape(feature, (b, -1, band, 768))
        output = torch.reshape(output, (b, band, 2))
        return feature, output

class EEGViT_KUL_pretrained_wav2vec_frequency_sum_768_2s(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=256,
            kernel_size=(1, int(9*2)),
            stride=(1, int(9*2)),
            padding=(0, 2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)
        # model_name =  "/media/nchen/CYB/EEGViT/vit-base-patch16-224"
        model_name = args.PT_model
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 256})
        config.update({'image_size': (64, 14)})
        config.update({'patch_size': (4, 1)})
        config.update({'output_hidden_states': True})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config,
                                                                       ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(4, 1), stride=(4, 1),
                                                                           padding=(0, 0), groups=256)
        # model.classifier=torch.nn.Sequential(torch.nn.Linear(768,1000,bias=True),
        #                              torch.nn.Dropout(p=0.1),
        #                              torch.nn.Linear(1000,128,bias=True))
        model.classifier = torch.nn.Sequential( torch.nn.Linear(768, 384, bias=True),
                                                torch.nn.GELU(),
                                                torch.nn.Dropout(p=0.1),
                                                torch.nn.LayerNorm(384),
                                                torch.nn.Linear(384, 2, bias=True))
        self.avgpooling = torch.nn.AvgPool1d(6)
        self.dropout = torch.nn.Dropout(p=0.4)

        self.ViT = model
        self.eeg_dim = 64
        self.feat_dim = args.feature_dim

    def forward(self, x):
        b, _, band, f, t = x.shape
        x = torch.reshape(x, (-1, 1, 64, int(128*2)))

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.ViT.forward(x)
        output = x.logits
        #print(output.shape)
        feat = list(x.hidden_states)
        feature = torch.tensor([]).to(output.device)
        for i in range(len(feat)):
            feature = torch.concatenate([feature, self.dropout(feat[i][:, 0, :]).unsqueeze(1)], dim=1)

        #feature = self.avgpooling(feature)
        feature = torch.reshape(feature, (b, -1, band, 768))
        output = torch.reshape(output, (b, band, 2))
        return feature, output
    
class EEGViT_KUL_pretrained_wav2vec_frequency_sum_768_2s_new(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=256,
            kernel_size=(1, 9),
            stride=(1, 9),
            padding=(0, 2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)
        # model_name =  "/media/nchen/CYB/EEGViT/vit-base-patch16-224"
        model_name = args.PT_model
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 256})
        config.update({'image_size': (64, 14)})
        config.update({'patch_size': (4, 1)})
        config.update({'output_hidden_states': True})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config,
                                                                       ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(4, 1), stride=(4, 1),
                                                                           padding=(0, 0), groups=256)
        # model.classifier=torch.nn.Sequential(torch.nn.Linear(768,1000,bias=True),
        #                              torch.nn.Dropout(p=0.1),
        #                              torch.nn.Linear(1000,128,bias=True))
        model.classifier = torch.nn.Sequential( torch.nn.Linear(768, 384, bias=True),
                                                torch.nn.GELU(),
                                                torch.nn.Dropout(p=0.1),
                                                torch.nn.LayerNorm(384),
                                                torch.nn.Linear(384, 2, bias=True))
        self.avgpooling = torch.nn.AvgPool1d(6)
        self.dropout = torch.nn.Dropout(p=0.4)

        self.ViT = model
        self.eeg_dim = 64
        self.feat_dim = args.feature_dim

    def forward(self, x):
        b, _, band, f, t = x.shape
        x = torch.reshape(x, (-1, 1, 64, 128))

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.ViT.forward(x)
        output = x.logits
        #print(output.shape)
        feat = list(x.hidden_states)
        feature = torch.tensor([]).to(output.device)
        for i in range(len(feat)):
            feature = torch.concatenate([feature, self.dropout(feat[i][:, 0, :]).unsqueeze(1)], dim=1)

        #feature = self.avgpooling(feature)
        feature = torch.reshape(feature, (b, -1, band, int(768*2)))
        output = torch.reshape(output, (b, band, -1, 2))
        output = nn.functional.softmax(output.mean(dim=2), dim=-1)
        return feature, output
    
class EEGViT_KUL_pretrained_wav2vec_frequency_sum_768_05s_new(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=256,
            kernel_size=(1, 5),
            stride=(1, 5),
            padding=(0, 5),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)
        # model_name =  "/media/nchen/CYB/EEGViT/vit-base-patch16-224"
        model_name = args.PT_model
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 256})
        config.update({'image_size': (64, 14)})
        config.update({'patch_size': (4, 1)})
        config.update({'output_hidden_states': True})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config,
                                                                       ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(4, 1), stride=(4, 1),
                                                                           padding=(0, 0), groups=256)
        # model.classifier=torch.nn.Sequential(torch.nn.Linear(768,1000,bias=True),
        #                              torch.nn.Dropout(p=0.1),
        #                              torch.nn.Linear(1000,128,bias=True))
        model.classifier = torch.nn.Sequential( torch.nn.Linear(768, 384, bias=True),
                                                torch.nn.GELU(),
                                                torch.nn.Dropout(p=0.1),
                                                torch.nn.LayerNorm(384),
                                                torch.nn.Linear(384, 2, bias=True))
        self.avgpooling = torch.nn.AvgPool1d(6)
        self.dropout = torch.nn.Dropout(p=0.4)

        self.ViT = model
        self.eeg_dim = 64
        self.feat_dim = args.feature_dim

    def forward(self, x):
        b, _, band, f, t = x.shape
        x = torch.reshape(x, (-1, 1, 64, 64))

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.ViT.forward(x)
        output = x.logits
        #print(output.shape)
        feat = list(x.hidden_states)
        feature = torch.tensor([]).to(output.device)
        for i in range(len(feat)):
            feature = torch.concatenate([feature, self.dropout(feat[i][:, 0, :]).unsqueeze(1)], dim=1)

        #feature = self.avgpooling(feature)
        feature = torch.reshape(feature, (b, -1, band, 768))
        output = torch.reshape(output, (b, band, 2))
        # output = output.mean(dim=2)
        return feature, output

class EEGViT_KUL_pretrained_wav2vec_frequency_sum_768_15s_new(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=256,
            kernel_size=(1, 13),
            stride=(1, 13),
            padding=(0, 0),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)
        # model_name =  "/media/nchen/CYB/EEGViT/vit-base-patch16-224"
        model_name = args.PT_model
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 256})
        config.update({'image_size': (64, 14)})
        config.update({'patch_size': (4, 1)})
        config.update({'output_hidden_states': True})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config,
                                                                       ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(4, 1), stride=(4, 1),
                                                                           padding=(0, 0), groups=256)
        # model.classifier=torch.nn.Sequential(torch.nn.Linear(768,1000,bias=True),
        #                              torch.nn.Dropout(p=0.1),
        #                              torch.nn.Linear(1000,128,bias=True))
        model.classifier = torch.nn.Sequential( torch.nn.Linear(768, 384, bias=True),
                                                torch.nn.GELU(),
                                                torch.nn.Dropout(p=0.1),
                                                torch.nn.LayerNorm(384),
                                                torch.nn.Linear(384, 2, bias=True))
        self.avgpooling = torch.nn.AvgPool1d(6)
        self.dropout = torch.nn.Dropout(p=0.4)

        self.ViT = model
        self.eeg_dim = 64
        self.feat_dim = args.feature_dim

    def forward(self, x):
        b, _, band, f, t = x.shape
        x = torch.reshape(x, (-1, 1, 64, 192))

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.ViT.forward(x)
        output = x.logits
        #print(output.shape)
        feat = list(x.hidden_states)
        feature = torch.tensor([]).to(output.device)
        for i in range(len(feat)):
            feature = torch.concatenate([feature, self.dropout(feat[i][:, 0, :]).unsqueeze(1)], dim=1)

        #feature = self.avgpooling(feature)
        feature = torch.reshape(feature, (b, -1, band, 768))
        output = torch.reshape(output, (b, band, 2))
        # output = output.mean(dim=2)
        return feature, output

class EEGViT_KUL_pretrained_wav2vec_frequency_sum_768_3s(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=256,
            kernel_size=(1, int(9*3)),
            stride=(1, int(9*3)),
            padding=(0, 2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)
        # model_name =  "/media/nchen/CYB/EEGViT/vit-base-patch16-224"
        model_name = args.PT_model
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 256})
        config.update({'image_size': (64, 14)})
        config.update({'patch_size': (4, 1)})
        config.update({'output_hidden_states': True})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config,
                                                                       ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(4, 1), stride=(4, 1),
                                                                           padding=(0, 0), groups=256)
        # model.classifier=torch.nn.Sequential(torch.nn.Linear(768,1000,bias=True),
        #                              torch.nn.Dropout(p=0.1),
        #                              torch.nn.Linear(1000,128,bias=True))
        model.classifier = torch.nn.Sequential( torch.nn.Linear(768, 384, bias=True),
                                                torch.nn.GELU(),
                                                torch.nn.Dropout(p=0.1),
                                                torch.nn.LayerNorm(384),
                                                torch.nn.Linear(384, 2, bias=True))
        self.avgpooling = torch.nn.AvgPool1d(6)
        self.dropout = torch.nn.Dropout(p=0.4)

        self.ViT = model
        self.eeg_dim = 64
        self.feat_dim = args.feature_dim

    def forward(self, x):
        b, _, band, f, t = x.shape
        x = torch.reshape(x, (-1, 1, 64, int(128*3)))

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.ViT.forward(x)
        output = x.logits
        #print(output.shape)
        feat = list(x.hidden_states)
        feature = torch.tensor([]).to(output.device)
        for i in range(len(feat)):
            feature = torch.concatenate([feature, self.dropout(feat[i][:, 0, :]).unsqueeze(1)], dim=1)

        #feature = self.avgpooling(feature)
        feature = torch.reshape(feature, (b, -1, band, 768))
        output = torch.reshape(output, (b, band, 2))
        return feature, output

# from peft import LoraConfig, get_peft_model
class EEGViT_KUL_pretrained_wav2vec_frequency_sum_LORA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=256,
            kernel_size=(1, 9),
            stride=(1, 9),
            padding=(0, 2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)
        model_name = "/media/c1/CYB/EEGViT/vit-base-patch16-224"
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 256})
        config.update({'image_size': (64, 14)})
        config.update({'patch_size': (4, 1)})
        config.update({'output_hidden_states': True})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config,
                                                                       ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(4, 1), stride=(4, 1),
                                                                           padding=(0, 0), groups=256)

        # config = transformers.ViTConfig(
        #     hidden_size=768,
        #     num_hidden_layers=12,
        #     num_attention_heads=12,
        #     intermediate_size=3072,
        #     hidden_dropout_prob=0.1,
        #     attention_probs_dropout_prob=0.1,
        #     initializer_range=0.02,
        #     num_channels=256,
        #     image_size=(64, 14),
        #     patch_size=(4, 1)
        # )
        # model = ViTModel(config)
        # model.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(4, 1), stride=(4, 1),
        #                                                                    padding=(0, 0), groups=256)

        # model.classifier=torch.nn.Sequential(torch.nn.Linear(768,1000,bias=True),
        #                              torch.nn.Dropout(p=0.1),
        #                              torch.nn.Linear(1000,128,bias=True))
        # model.classifier = torch.nn.Sequential( torch.nn.Dropout(p=0.1),
        #                                         torch.nn.Linear(768, 128, bias=True))
                                                #torch.nn.AvgPool1d(6),)
        model.classifier = torch.nn.Sequential( torch.nn.Linear(768, 384, bias=True),
                                                torch.nn.Dropout(p=0.1, inplace=True),
                                               torch.nn.Linear(384, 2, bias=True))
        self.avgpooling = torch.nn.AvgPool1d(6)
        # self.avgpooling = torch.nn.AdaptiveAvgPool1d(128)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.batchnorm = torch.nn.BatchNorm1d(768)
        self.layernorm = torch.nn.LayerNorm(768)

        self.ViT = model
        self.eeg_dim = 64
        self.feat_dim = args.feature_dim

        config = LoraConfig(
            r=8,  # LoRA 的秩
            lora_alpha=16,  # LoRA 的缩放因子
            target_modules=["query", "value"],  # 目标模块
            lora_dropout=0.1,  # Dropout 概率
            bias="none",  # 是否更新偏置
            modules_to_save=["classifier"],  # 指定分类器需要被微调
        )
        self.lora_model = get_peft_model(model, config)

    def forward(self, x):
        b, _, band, f, t = x.shape
        x = torch.reshape(x, (-1, 1, 64, 128))

        x = self.conv1(x)
        x = self.batchnorm1(x)
        #x = self.ViT.forward(x)
        x = self.lora_model(x)
        output = x.logits
        feat = list(x.hidden_states)
        feature = torch.tensor([]).to(output.device)
        # output = x.pooler_output[-1]
        # feat = list(x.pooler_output)
        # feature = torch.tensor([]).to(output.device)
        for i in range(len(feat)):
            # t =  self.batchnorm(self.dropout(feat[i][:, 0, :])).unsqueeze(1)
            feature = torch.concatenate([feature, self.dropout(feat[i][:, 0, :]).unsqueeze(1)], dim=1)

        feature = self.avgpooling(feature)
        feature = torch.reshape(feature, (b, -1, band, t))
        output = torch.reshape(output, (b, band, 2))
        return feature, output
    

class EEGViT_KUL_pretrained_wavlm_frequency(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=256,
            kernel_size=(1, 9),
            stride=(1, 9),
            padding=(0, 2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)
        model_name = "/media/c1/CYB/EEGViT/vit-base-patch16-224"
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 256})
        config.update({'image_size': (64, 14)})
        config.update({'patch_size': (4, 1)})
        config.update({'output_hidden_states': True})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config,
                                                                       ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(4, 1), stride=(4, 1),
                                                                           padding=(0, 0), groups=256)

        # config = transformers.ViTConfig(
        #     hidden_size=768,
        #     num_hidden_layers=12,
        #     num_attention_heads=12,
        #     intermediate_size=3072,
        #     hidden_dropout_prob=0.1,
        #     attention_probs_dropout_prob=0.1,
        #     initializer_range=0.02,
        #     num_channels=256,
        #     image_size=(64, 14),
        #     patch_size=(4, 1)
        # )
        # model = ViTModel(config)
        # model.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(4, 1), stride=(4, 1),
        #                                                                    padding=(0, 0), groups=256)

        # model.classifier=torch.nn.Sequential(torch.nn.Linear(768,1000,bias=True),
        #                              torch.nn.Dropout(p=0.1),
        #                              torch.nn.Linear(1000,128,bias=True))
        model.classifier = torch.nn.Sequential( torch.nn.Dropout(p=0.1),
                                                torch.nn.Linear(768, 128, bias=True))
                                                #torch.nn.AvgPool1d(6),)
        self.avgpooling = torch.nn.AvgPool1d(6)
        # self.avgpooling = torch.nn.AdaptiveAvgPool1d(128)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.batchnorm = torch.nn.BatchNorm1d(768)
        self.layernorm = torch.nn.LayerNorm(768)

        self.ViT = model
        self.eeg_dim = 64
        self.feat_dim = args.feature_dim

    def forward(self, x):
        b, _, band, f, t = x.shape
        x = torch.reshape(x, (-1, 1, 64, 128))

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.ViT.forward(x)
        output = x.logits
        feat = list(x.hidden_states)
        feature = torch.tensor([]).to(output.device)
        # output = x.pooler_output[-1]
        # feat = list(x.pooler_output)
        # feature = torch.tensor([]).to(output.device)
        for i in range(len(feat)):
            # t =  self.batchnorm(self.dropout(feat[i][:, 0, :])).unsqueeze(1)
            feature = torch.concatenate([feature, self.dropout(feat[i][:, 0, :]).unsqueeze(1)], dim=1)

        feature = self.avgpooling(feature)
        feature = torch.reshape(feature, (b, -1, band, t))
        output = torch.reshape(output, (b, band, t))
        return feature, output
    
from peft import LoraConfig, get_peft_model
class EEGViT_KUL_pretrained_wav2vec_frequency_LORA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=256,
            kernel_size=(1, 9),
            stride=(1, 9),
            padding=(0, 2),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)
        model_name = "/media/c1/CYB/EEGViT/vit-base-patch16-224"
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 256})
        config.update({'image_size': (64, 14)})
        config.update({'patch_size': (4, 1)})
        config.update({'output_hidden_states': True})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config,
                                                                       ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(4, 1), stride=(4, 1),
                                                                           padding=(0, 0), groups=256)

        # config = transformers.ViTConfig(
        #     hidden_size=768,
        #     num_hidden_layers=12,
        #     num_attention_heads=12,
        #     intermediate_size=3072,
        #     hidden_dropout_prob=0.1,
        #     attention_probs_dropout_prob=0.1,
        #     initializer_range=0.02,
        #     num_channels=256,
        #     image_size=(64, 14),
        #     patch_size=(4, 1)
        # )
        # model = ViTModel(config)
        # model.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(4, 1), stride=(4, 1),
        #                                                                    padding=(0, 0), groups=256)

        # model.classifier=torch.nn.Sequential(torch.nn.Linear(768,1000,bias=True),
        #                              torch.nn.Dropout(p=0.1),
        #                              torch.nn.Linear(1000,128,bias=True))
        model.classifier = torch.nn.Sequential( torch.nn.Dropout(p=0.1),
                                                torch.nn.Linear(768, 128, bias=True))
                                                #torch.nn.AvgPool1d(6),)
        self.avgpooling = torch.nn.AvgPool1d(6)
        # self.avgpooling = torch.nn.AdaptiveAvgPool1d(128)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.batchnorm = torch.nn.BatchNorm1d(768)
        self.layernorm = torch.nn.LayerNorm(768)

        self.ViT = model
        self.eeg_dim = 64
        self.feat_dim = args.feature_dim

        config = LoraConfig(
            r=8,  # LoRA 的秩
            lora_alpha=16,  # LoRA 的缩放因子
            target_modules=["query", "value"],  # 目标模块
            lora_dropout=0.1,  # Dropout 概率
            bias="none",  # 是否更新偏置
            modules_to_save=["classifier"],  # 指定分类器需要被微调
        )
        self.lora_model = get_peft_model(model, config)

    def forward(self, x):
        b, _, band, f, t = x.shape
        x = torch.reshape(x, (-1, 1, 64, 128))

        x = self.conv1(x)
        x = self.batchnorm1(x)
        #x = self.ViT.forward(x)
        x = self.lora_model(x)
        output = x.logits
        feat = list(x.hidden_states)
        feature = torch.tensor([]).to(output.device)
        # output = x.pooler_output[-1]
        # feat = list(x.pooler_output)
        # feature = torch.tensor([]).to(output.device)
        for i in range(len(feat)):
            # t =  self.batchnorm(self.dropout(feat[i][:, 0, :])).unsqueeze(1)
            feature = torch.concatenate([feature, self.dropout(feat[i][:, 0, :]).unsqueeze(1)], dim=1)

        feature = self.avgpooling(feature)
        feature = torch.reshape(feature, (b, -1, band, t))
        output = torch.reshape(output, (b, band, t))
        return feature, output

    
class EEGViT_KUL_pretrained_wav2vec_frequency_channel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=256,
            kernel_size=(5, 2),
            stride=(5, 2),
            padding=(5, 0),
            bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(256, False)
        # model_name =  "/media/nchen/CYB/EEGViT/vit-base-patch16-224"
        model_name = args.PT_model
        config = transformers.ViTConfig.from_pretrained(model_name)
        config.update({'num_channels': 256})
        config.update({'image_size': (14, 64)})
        config.update({'patch_size': (1, 4)})
        config.update({'output_hidden_states': True})

        model = transformers.ViTForImageClassification.from_pretrained(model_name, config=config,
                                                                       ignore_mismatched_sizes=True)
        model.vit.embeddings.patch_embeddings.projection = torch.nn.Conv2d(256, 768, kernel_size=(1, 4), stride=(1, 4),
                                                                           padding=(0, 0), groups=256)
        # model.classifier=torch.nn.Sequential(torch.nn.Linear(768,1000,bias=True),
        #                              torch.nn.Dropout(p=0.1),
        #                              torch.nn.Linear(1000,128,bias=True))
        model.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.4, inplace=True),
                                               torch.nn.Linear(768, 128, bias=True))
        self.avgpooling = torch.nn.AdaptiveAvgPool1d(768)
        self.dropout = torch.nn.Dropout(p=0.4)

        self.ViT = model
        self.eeg_dim = 64
        self.feat_dim = args.feature_dim

    def forward(self, x):
        b, _, band, f, t = x.shape
        x = torch.reshape(x, (-1, 1, 64, 128))

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.ViT.forward(x)
        output = x.logits
        feat = list(x.hidden_states)
        feature = torch.tensor([]).to(output.device)
        for i in range(len(feat)):
            feature = torch.concatenate([feature, self.dropout(feat[i][:, 0, :]).unsqueeze(1)], dim=1)

        feature = self.avgpooling(feature)
        feature = torch.reshape(feature, (b, -1, band, t))
        output = torch.reshape(output, (b, band, t))
        return feature, output