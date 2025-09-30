import numpy as np

import torch
import torch.nn as nn
import random
from .util import initialize_weights
from .util import NystromAttention
from .util import BilinearFusion
from .util import SNN_Block
from .util import MultiheadAttention


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,  # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,  # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class Transformer_P(nn.Module):
    def __init__(self, feature_dim=512):
        super(Transformer_P, self).__init__()
        # Encoder
        self.pos_layer = PPEG(dim=feature_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        # Decoder

    def forward(self, features):
        # ---->pad
        H = features.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([features, features[:, :add_length, :]], dim=1)  # [B, N, 512]
        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)
        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]
        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]
        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]
        # ---->cls_token
        h = self.norm(h)
        return h[:, 0], h[:, 1:]


class Transformer_G(nn.Module):
    def __init__(self, feature_dim=512):
        super(Transformer_G, self).__init__()
        # Encoder
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        # Decoder

    def forward(self, features):
        # ---->pad
        cls_tokens = self.cls_token.expand(features.shape[0], -1, -1).cuda()
        h = torch.cat((cls_tokens, features), dim=1)
        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]
        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]
        # ---->cls_token
        h = self.norm(h)
        return h[:, 0], h[:, 1:]

class Private_encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(Private_encoder, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        return x

class Share_encoder(nn.Module):
    def __init__(self, input_dim, ouput_dim):
        super(Share_encoder, self).__init__()
        self.fc_m = nn.Linear(input_dim, ouput_dim)
        self.fc_p = nn.Linear(input_dim, ouput_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,f_m,f_p): #(1,256)
        f_m = self.fc_m(f_m) #(1,128)
        f_p = self.fc_p(f_p)
        m_p = torch.concat((f_m,f_p),dim=1) #(1,256)
        p_m = torch.concat((f_p,f_m),dim=1)

        attn = torch.mm(m_p.transpose(1,0),p_m) #(256,256)
        attn_score = self.softmax(attn)
        attn_m_p = attn_score[:,:128]
        attn_p_m = attn_score[:128,:]
        p_m = torch.mm(p_m,attn_m_p) #(1,256)x(256,128)->(1,128)
        m_p = torch.mm(attn_p_m,m_p.transpose(1,0)).transpose(1,0) #(128,256)x(256,1)->(128,1)->(1,128)
        return (p_m+m_p)/2 #(1,128)

    
class Gate(nn.Module):
    def __init__(self, input_dim=128,output_dim=1):
        super(Gate, self).__init__()
        self.fc = nn.Linear(input_dim,output_dim)
        self.softmax = nn.Softmax(dim=0)
    def forward(self,x): #(1,512)
        score = self.softmax(self.fc(x)) #(1,4)
        return score

class expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(expert, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        return x

class DeReF(nn.Module):
    def __init__(self, omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4, fusion="concat", model_size="small"):
        super(DeReF, self).__init__()
        self.seg_length=[2,4,8,16,32,64]
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.fusion = fusion
        ###
        self.size_dict = {
            "pathomics": {"small": [1024, 256, 256], "large": [1024, 512, 256]},
            "genomics": {"small": [1024, 256], "large": [1024, 1024, 1024, 256]},
        }
        # Pathomics Embedding Network
        hidden = self.size_dict["pathomics"][model_size]
        fc = []
        for idx in range(len(hidden) - 1):
            fc.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(0.25))
        self.pathomics_fc = nn.Sequential(*fc)
        # Genomic Embedding Network
        hidden = self.size_dict["genomics"][model_size]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.genomics_fc = nn.ModuleList(sig_networks)

        # Encoder
        self.pathomics_encoder = Transformer_P(feature_dim=hidden[-1])
        self.genomics_encoder = Transformer_G(feature_dim=hidden[-1])

        # Private Encoder
        self.m_private = Private_encoder(input_dim=256, hidden_dim=256, output_dim=128, dropout_rate=0.1)
        self.p_private = Private_encoder(input_dim=256, hidden_dim=256, output_dim=128, dropout_rate=0.1)
        # Share Encoder
        self.com_encoder = Share_encoder(input_dim=256, ouput_dim=128)
        self.red_encoder = Share_encoder(input_dim=256, ouput_dim=128)
        # Expert
        self.exp1 = expert(input_dim=512, hidden_dim=64, output_dim=128, dropout_rate=0.1)
        self.exp2 = expert(input_dim=512, hidden_dim=64, output_dim=128, dropout_rate=0.1)
        self.exp3 = expert(input_dim=512, hidden_dim=64, output_dim=128, dropout_rate=0.1)
        self.exp4 = expert(input_dim=512, hidden_dim=64, output_dim=128, dropout_rate=0.1)
        self.gate = Gate(input_dim=128, output_dim=1)
        # Classification Layer
        if self.fusion == "concat":
            self.mm = nn.Sequential(
                *[nn.Linear(hidden[-1] * 2, hidden[-1]), nn.ReLU(), nn.Linear(hidden[-1], hidden[-1]), nn.ReLU()]
            )
        elif self.fusion == "bilinear":
            self.mm = BilinearFusion(dim1=hidden[-1], dim2=hidden[-1], scale_dim1=8, scale_dim2=8, mmhid=hidden[-1])
        else:
            raise NotImplementedError("Fusion [{}] is not implemented".format(self.fusion))

        self.classifier = nn.Linear(hidden[-1]*2, self.n_classes)

        self.apply(initialize_weights)

    def forward(self, **kwargs):
        # meta genomics and pathomics features
        x_path = kwargs["x_path"]
        x_omic = [kwargs["x_omic%d" % i] for i in range(1, 7)]

        # Enbedding
        # genomics embedding
        genomics_features = [self.genomics_fc[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)]
        genomics_features = torch.stack(genomics_features).unsqueeze(0)  # [1, 6, 256]
        # pathomics embedding
        pathomics_features = self.pathomics_fc(x_path).unsqueeze(0)

        # encoder
        # pathomics encoder
        cls_token_pathomics_encoder, patch_token_pathomics_encoder = self.pathomics_encoder(
            pathomics_features)  # cls token + patch tokens #[1,1,256]
        # genomics encoder
        cls_token_genomics_encoder, patch_token_genomics_encoder = self.genomics_encoder(
            genomics_features)  # cls token + patch tokens

        feature_g = self.m_private(cls_token_genomics_encoder)  # (1,128)
        feature_p = self.p_private(cls_token_pathomics_encoder)  # (1,128)
        feature_comm = self.com_encoder(cls_token_genomics_encoder, cls_token_pathomics_encoder)  # (1,128)
        feature_red = self.red_encoder(cls_token_genomics_encoder, cls_token_pathomics_encoder)  # (1,128)

        cat_total = torch.concat([feature_g, feature_p, feature_comm, feature_red], dim=0)  # (4,128)
        seg_l = random.choice(self.seg_length)
        cat_new = cat_total.reshape(4, seg_l, 128 // seg_l).transpose(1, 0).reshape(1, -1)

        out1 = self.exp1(cat_new)  # (1,128)
        out2 = self.exp2(cat_new)
        out3 = self.exp3(cat_new)
        out4 = self.exp4(cat_new)

        out_total = torch.concat([out1, out2, out3, out4], dim=0)  # (4,128)

        score = self.gate(cat_total)

        out_total = out_total * score

        # predict
        logits = self.classifier(out_total.view(1,-1))  # [1, n_classes]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return  [feature_g, feature_p, feature_comm, feature_red],  score,  hazards, S, logits 
