import torch
import torch.nn as nn
from layers import *
import torch.nn.functional as F
import math
import numpy as np



class Fusion(nn.Module):
    def __init__(self, fusion_dim, nbit):
        super(Fusion, self).__init__()
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU()
        )
        self.hash = nn.Sequential(
            nn.Linear(fusion_dim, nbit),
            nn.BatchNorm1d(nbit),
            nn.Tanh()
        )

    def forward(self, x, y):
        fused_feat = self.fusion(torch.cat([x, y], dim=-1))
        hash_code = self.hash(fused_feat)
        return hash_code


class BitImportanceNetwork(nn.Module):
    """轻量级比特重要性评估网络"""
    def __init__(self, nbit):
        super(BitImportanceNetwork, self).__init__()
        self.nbit = nbit
        # 使用两层全连接网络估计每个比特的重要性
        self.importance_net = nn.Sequential(
            nn.Linear(nbit, nbit//2),
            nn.ReLU(),
            nn.Linear(nbit//2, nbit)
        )
        
    def forward(self, h):
        # h shape: [batch_size, nbit]
        # 输出重要性权重矩阵 w [batch_size, nbit]
        w = self.importance_net(h)
        return w


class PCH(nn.Module):
    def __init__(self, args):
        super(PCH, self).__init__()
        self.image_dim = args.image_dim
        self.text_dim = args.text_dim

        self.img_hidden_dim = args.img_hidden_dim
        self.txt_hidden_dim = args.txt_hidden_dim
        self.common_dim = args.img_hidden_dim[-1]
        self.nbit = int(args.nbit)
        self.classes = args.classes
        self.batch_size = args.batch_size
        assert self.img_hidden_dim[-1] == self.txt_hidden_dim[-1]

        self.fusionnn = Fusion(fusion_dim=self.common_dim, nbit=self.nbit)

        self.imageMLP = MLP(hidden_dim=self.img_hidden_dim, act=nn.Tanh(),dropout=args.mlpdrop)

        self.textMLP = MLP(hidden_dim=self.txt_hidden_dim, act=nn.Tanh(),dropout=args.mlpdrop)
        self.imglabelatt = AttentionLayer(self.common_dim, self.classes, self.common_dim)
        self.txtlabelatt = AttentionLayer(self.common_dim, self.classes, self.common_dim)

        self.Fcfusion = nn.Linear(2 * self.common_dim, self.common_dim)


        self.fusion_layer = nn.Sequential(
            nn.Linear(self.nbit, self.common_dim),
            nn.ReLU()
        )
        self.hash_output = nn.Sequential(
            nn.Linear(self.common_dim, self.nbit),
            nn.Tanh())
        self.hashfc = nn.Sequential(
            nn.Linear(self.nbit, 2 * self.nbit),
            nn.Sigmoid(),
            nn.Linear(2 * self.nbit, self.nbit))
        
        self.weight = nn.Parameter(torch.randn(self.nbit))  # 可学习的权重
        nn.init.normal_(self.weight, 0.25, 1/self.nbit)
        self.centroids = nn.Parameter(torch.randn(self.classes, self.nbit)).to(dtype=torch.float32)
        self.classify = nn.Linear(self.nbit, self.classes)
        
        # 添加比特重要性评估网络
        self.bit_importance_net = BitImportanceNetwork(self.nbit)
        # 存储全局比特重要性排序
        self.global_bit_ranking = None

    def forward(self, image, text, label, target_length=None):
        self.batch_size = len(image)
        imageH = self.imageMLP(image)
        textH = self.textMLP(text)
        imagefine = self.imglabelatt(imageH, label)
        textfine = self.txtlabelatt(textH, label)

        img_feature = self.Fcfusion(torch.cat((imageH, imagefine), 1))
        text_feature = self.Fcfusion(torch.cat((textH, textfine), 1))

        fused_fine = self.fusionnn(img_feature, text_feature)

        cfeat_concat = self.fusion_layer(fused_fine)    
        code = self.hash_output(cfeat_concat)
        
        if target_length is not None and self.global_bit_ranking is not None:
            # 使用全局比特重要性排序进行自适应截断
            selected_indices = self.global_bit_ranking[:target_length]
            selected_code = code[:, selected_indices]
            # 对于检索任务，只需要返回截断后的哈希码，不需要分类预测
            return selected_code, None
        else:
            # 返回完整长度的哈希码及分类预测
            return code, self.classify(code)
    
    def compute_global_bit_importance(self, dataloader):
        """
        计算全局比特重要性排序
        """
        self.eval()
        total_importance_weights = None
        count = 0
        
        with torch.no_grad():
            for idx, img_feat, txt_feat, label in dataloader:
                img_feat = img_feat.cuda()
                txt_feat = txt_feat.cuda()
                label = label.cuda()
                
                # 获取哈希码（不进行截断）
                H, _ = self(img_feat, txt_feat, label)
                
                # 计算重要性权重
                W = self.bit_importance_net(H)  # [batch_size, nbit]
                
                # 累积重要性权重
                if total_importance_weights is None:
                    total_importance_weights = W.sum(dim=0)  # [nbit]
                else:
                    total_importance_weights += W.sum(dim=0)
                
                count += img_feat.size(0)
        
        # 计算全局重要性得分
        global_scores = total_importance_weights / count
        
        # 计算全局比特重要性排序
        _, sorted_indices = torch.sort(global_scores, descending=True)
        self.global_bit_ranking = sorted_indices.cpu().numpy()
        
        return self.global_bit_ranking


class AttentionLayer(nn.Module):
    def __init__(self, data_dim, label_dim, hidden_dim, n_heads=4):
        super(AttentionLayer, self).__init__()

        assert hidden_dim % n_heads == 0

        self.data_dim = data_dim
        self.label_dim = label_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.fc_q = nn.Linear(label_dim, hidden_dim)
        self.fc_k = nn.Linear(data_dim, hidden_dim)
        self.fc_v = nn.Linear(data_dim, hidden_dim)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).cuda()
        self.dense = nn.Linear(hidden_dim, data_dim)    
        self.bn = nn.BatchNorm1d(data_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, data_tensor, label_tensor):
        batch_size = data_tensor.shape[0]

        Q = self.fc_q(label_tensor).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3).cuda()
        K = self.fc_k(data_tensor).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3).cuda()
        V = self.fc_v(data_tensor).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3).cuda()

        att_map = torch.softmax((torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale), dim=-1)
        output = torch.matmul(att_map, V).view(batch_size, -1)

        output = self.dense(output)
        output = self.bn(output)
        output = self.relu(output)

        return output
