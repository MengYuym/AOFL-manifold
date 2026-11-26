import sys
import csv
import json
from datetime import datetime


sys.path.append('Loss')
sys.path.append('Model')
sys.path.append('Dataset')

from IEMOCAPDataset import IEMOCAPDataset
from MELDDataset import MELDDataset
from AOFL_Model import AOFL_model

import torch.nn as nn
from torch.utils.data import DataLoader
from optparse import OptionParser
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score
import warnings
import os
warnings.filterwarnings('ignore')
from torch.utils.data.sampler import SubsetRandomSampler
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch

from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
# import umap
import time

from tqdm import tqdm
# import seaborn as sns



class AOFL():

    def __init__(self, dataset, batch_size, num_epochs, learning_rate, weight_decay,
                 num_layers, model_dim, num_heads, hidden_dim, dropout_rate, dropout_rec
                 , CE_loss_param, multi_attn_flag, device):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.dropout_rec = dropout_rec
        self.CE_loss_param = CE_loss_param
        self.multi_attn_flag = multi_attn_flag
        self.device = device

        self.best_valid_f1 = float("-inf")
        self.best_test_f1 = 0.0
        self.best_epoch = 1
        self.best_test_report = None

        self.get_dataloader()
        self.get_model()
        self.get_loss()
        self.get_optimizer()
        # 用于保存特征
        self.shared_text_features = []
        self.specific_text_features = []
        self.shared_audio_features = []
        self.specific_audio_features = []
        self.shared_visual_features = []
        self.specific_visual_features = []
        self.labels = []
        self.fc_out = []
        self.writer = SummaryWriter(log_dir=f"runs/{self.dataset}_angular_loss_logs")
        self.total_text_cosloss, self.total_audio_cosloss, self.total_visual_cosloss = 0.0, 0.0, 0.0
        self.bilinear = nn.Bilinear(model_dim//2,model_dim//2,model_dim//2,bias=False)

    def count_parameters(self):
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")


    def get_train_valid_sampler(self, train_dataset, valid=0.1):
        size = len(train_dataset)
        idx = list(range(size))
        split = int(valid * size)
        np.random.shuffle(idx)
        self.train_indices = idx[split:]
        self.valid_indices = idx[:split]
        return SubsetRandomSampler(self.train_indices), SubsetRandomSampler(self.valid_indices)



    def get_dataloader(self, valid=0.05):
        if self.dataset == 'IEMOCAP':
            train_dataset = IEMOCAPDataset(train=True)
            test_dataset = IEMOCAPDataset(train=False)
        elif self.dataset == 'MELD':
            train_dataset = MELDDataset(train=True)
            test_dataset = MELDDataset(train=False)

        if self.dataset == 'MELD':
            manifest_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         'Data', 'MELD', 'official_split_ids.json')
            with open(manifest_path, encoding='utf-8') as handle:
                split_ids = json.load(handle)
            valid_ids = set(split_ids['validation_ids'])
            test_ids = set(split_ids['test_ids'])
            available = set(train_dataset.keys)
            if (available != set(split_ids['train_ids']) | valid_ids
                    or set(split_ids['train_ids']) & valid_ids
                    or set(test_dataset.keys) != test_ids or available & test_ids):
                raise ValueError('MELD split manifest does not match the loaded dataset')
            self.train_indices = [i for i, key in enumerate(train_dataset.keys) if key not in valid_ids]
            self.valid_indices = [i for i, key in enumerate(train_dataset.keys) if key in valid_ids]
            train_sampler = SubsetRandomSampler(self.train_indices)
            valid_sampler = SubsetRandomSampler(self.valid_indices)
            self.split_strategy = 'MELD official dev recovered by full-dialogue text and label matching'
            print(f'MELD split: {len(self.train_indices)} train dialogues, '
                  f'{len(self.valid_indices)} validation dialogues, {len(test_dataset)} test dialogues')
        else:
            train_sampler, valid_sampler = self.get_train_valid_sampler(train_dataset, valid)
            self.split_strategy = f'random dialogue holdout, fraction={valid}' 
        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size,
                                           sampler=train_sampler, collate_fn=train_dataset.collate_fn, num_workers=0)
        self.valid_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size,
                                           sampler=valid_sampler, collate_fn=train_dataset.collate_fn, num_workers=0)
        self.test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.batch_size,
                                          collate_fn=test_dataset.collate_fn, shuffle=False, num_workers=0)

    def get_class_counts(self):
        class_counts = torch.zeros(self.num_classes).to(self.device)

        for _, data in enumerate(self.train_dataloader):
            _, _, _, _, _, padded_labels = [d.to(self.device) for d in data]
            padded_labels = padded_labels.reshape(-1)
            labels = padded_labels[padded_labels != -1]
            class_counts += torch.bincount(labels, minlength=self.num_classes)

        return class_counts

    def get_model(self):
        if self.dataset == 'IEMOCAP':
            self.num_classes = 6
            self.n_speakers = 2
        elif self.dataset == 'MELD':
            self.num_classes = 7
            self.n_speakers = 9

        roberta_dim = 768
        D_m_audio = 512
        D_m_visual = 1000
        listener_state = False
        D_e = self.model_dim
        D_p = self.model_dim
        D_g = self.model_dim
        D_h = self.model_dim
        D_a = self.model_dim
        context_attention = 'simple'
        hidden_dim = self.hidden_dim
        dropout_rate = self.dropout_rate
        num_layers = self.num_layers
        num_heads = self.num_heads

        self.model = AOFL_model(self.dataset, self.multi_attn_flag, roberta_dim, hidden_dim, dropout_rate, num_layers,
                              self.model_dim, num_heads, D_m_audio, D_m_visual, D_g, D_p, D_e,
                              D_h, self.num_classes, self.n_speakers,
                              listener_state, context_attention, D_a, self.dropout_rec, self.device).to(self.device)

    def get_loss(self):
        self.CE_loss = nn.CrossEntropyLoss()

    def save_tsne(self, epoch):
        """
        保存 t-SNE 图，每个 epoch 都保存，绘制所有共享和特定特征
        """
        print(f"Saving t-SNE plot for epoch {epoch}")

        # 拼接特征
        shared_text_features = np.concatenate(self.shared_text_features, axis=0)
        specific_text_features = np.concatenate(self.specific_text_features, axis=0)
        shared_audio_features = np.concatenate(self.shared_audio_features, axis=0)
        specific_audio_features = np.concatenate(self.specific_audio_features, axis=0)
        shared_visual_features = np.concatenate(self.shared_visual_features, axis=0)
        specific_visual_features = np.concatenate(self.specific_visual_features, axis=0)
        labels = np.concatenate(self.labels, axis=0)  # 合并标签

        # 限制样本数量
        max_samples = 1000
        shared_text_features = shared_text_features[:max_samples]
        specific_text_features = specific_text_features[:max_samples]
        shared_audio_features = shared_audio_features[:max_samples]
        specific_audio_features = specific_audio_features[:max_samples]
        shared_visual_features = shared_visual_features[:max_samples]
        specific_visual_features = specific_visual_features[:max_samples]

        # 拼接所有特征
        combined_features = np.concatenate([
            shared_text_features,
            specific_text_features,
            shared_audio_features,
            specific_audio_features,
            shared_visual_features,
            specific_visual_features,
        ], axis=0)

        # 为每种特征创建标签
        combined_labels = (
                ["Shared Text"] * len(shared_text_features) +
                ["Specific Text"] * len(specific_text_features) +
                ["Shared Audio"] * len(shared_audio_features) +
                ["Specific Audio"] * len(specific_audio_features) +
                ["Shared Visual"] * len(shared_visual_features) +
                ["Specific Visual"] * len(specific_visual_features)
        )

        # 检查特征和标签长度是否一致
        assert len(combined_features) == len(combined_labels), "Feature and label lengths do not match!"

        # 使用 t-SNE 降维
        tsne = TSNE(n_components=2,  perplexity=30,  random_state=42)
        reduced_features = tsne.fit_transform(combined_features)

        # 绘制 t-SNE 图
        plt.figure(figsize=(10, 8))

        # 定义颜色和标记
        markers = {
            "Shared Text": "o",
            "Specific Text": "o",
            "Shared Audio": "o",
            "Specific Audio": "o",
            "Shared Visual": "o",
            "Specific Visual": "o",
        }
        colors = {
            "Shared Text": "steelblue",
            "Specific Text": "lightskyblue",
            "Shared Audio": "firebrick",
            "Specific Audio": 'pink',
            "Shared Visual": "darkorange",
            "Specific Visual": "gold",
        }

        # 根据标签绘制散点图
        for label in set(combined_labels):
            idx = np.array([i for i, l in enumerate(combined_labels) if l == label])
            plt.scatter(
                reduced_features[idx, 0],
                reduced_features[idx, 1],
                c=colors[label],
                marker=markers[label],
                label=label,
                alpha=0.6,
            )

        # 添加图例和标题
        plt.legend()
        plt.title('t-SNE of Shared and Specific Features')

        # 保存图像
        output_dir = f"./pic/AOFL_non_Contr_{self.dataset}/"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{self.dataset}_all_features_epoch_{epoch}.png")
        plt.savefig(output_path)
        plt.close()
    def plot_vector_features(self, epoch):
        """
        绘制共享特征和特定特征的三维向量图，确保六个向量的起点一致
        """
        print(f"Saving 3D vector visualization for epoch {epoch}")

        # 将所有特征拼接成一个列表
        shared_text_features = np.concatenate(self.shared_text_features, axis=0)
        specific_text_features = np.concatenate(self.specific_text_features, axis=0)
        shared_audio_features = np.concatenate(self.shared_audio_features, axis=0)
        specific_audio_features = np.concatenate(self.specific_audio_features, axis=0)
        shared_visual_features = np.concatenate(self.shared_visual_features, axis=0)
        specific_visual_features = np.concatenate(self.specific_visual_features, axis=0)

        # 选择特定的特征向量（例如选取第一个样本）
        # feature_vectors = [
        #     shared_text_features[5], specific_text_features[5],
        #     shared_audio_features[5], specific_audio_features[5],
        #     shared_visual_features[5], specific_visual_features[5]
        # ]
        feature_vectors = [
            F.normalize(torch.tensor(shared_text_features[5]), dim=-1).numpy(),
            F.normalize(torch.tensor(specific_text_features[5]), dim=-1).numpy(),
            F.normalize(torch.tensor(shared_audio_features[5]), dim=-1).numpy(),
            F.normalize(torch.tensor(specific_audio_features[5]), dim=-1).numpy(),
            F.normalize(torch.tensor(shared_visual_features[5]), dim=-1).numpy(),
            F.normalize(torch.tensor(specific_visual_features[5]), dim=-1).numpy()
        ]
        labels = [
            "Shared Text", "Specific Text",
            "Shared Audio", "Specific Audio",
            "Shared Visual", "Specific Visual"
        ]

        # 设置颜色
        colors = [
            'firebrick', 'lightpink',  # Shared Text, Specific Text
            'cornflowerblue', 'lightskyblue',  # Shared Audio, Specific Audio
            'slateblue', 'plum'  # Shared Visual, Specific Visual
        ]

        # 步骤1：归一化 + 缩放
        scale_factor = 2.0
        feature_vectors_normalized = [
            (F.normalize(torch.tensor(vec, dtype=torch.float32), p=2, dim=0) * scale_factor).numpy()
            for vec in feature_vectors
        ]

        # 步骤2：自动设置坐标轴范围
        all_vectors = np.stack(feature_vectors_normalized, axis=0)
        max_range = np.max(np.abs(all_vectors)) * 1.2
        # 创建一个3D图形
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制每个特征向量
        for i, vector in enumerate(feature_vectors):
            ax.quiver(0, 0, 0, vector[0], vector[1], vector[2], color=colors[i], label=labels[i],length=10, linewidth=3, arrow_length_ratio=0.1)

        # 设置图形的显示范围
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

        # 添加图例
        ax.legend()

        # 标题
        ax.set_title(f'3D Feature Vectors Visualization at Epoch {epoch}')

        # 保存图像
        output_dir = f"./3d_pic/AOFL/vector_visualization_3d_{self.dataset}/"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{self.dataset}_epoch_{epoch}.png")
        plt.savefig(output_path)
        plt.close()

    def save_tsne_test(self, epoch):
        """
        保存 t-SNE 图，基于测试集绘制分类效果
        """
        print(f"Saving t-SNE plot for test dataset at epoch {epoch}")

        # 初始化存储特征和标签的列表
        test_features = []
        test_labels = []

        # 遍历测试集，提取特征和标签
        for _, data in enumerate(self.test_dataloader):
            padded_texts, padded_audios, padded_visuals, padded_speaker_masks, padded_utterance_masks, padded_labels = [
                d.to(self.device) for d in data]
            padded_labels = padded_labels.reshape(-1)
            labels = padded_labels[padded_labels != -1]

            # 获取模型输出的特征
            fc_outputs, _, _, _, \
                _, _, _,\
                _, _, _,  = self.model(
                padded_texts, padded_audios, padded_visuals, padded_speaker_masks, padded_utterance_masks, padded_labels
            )

            # 将特征和标签添加到列表中
            test_features.append(fc_outputs.cpu().detach().numpy())
            test_labels.append(labels.cpu().detach().numpy())

        # 合并所有特征和标签
        test_features = np.concatenate(test_features, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)

        # 使用 t-SNE 降维
        tsne = TSNE(n_components=2, perplexity=30, n_iter=10000,random_state=42)
        reduced_features = tsne.fit_transform(test_features)

        # 绘制 t-SNE 图
        plt.figure(figsize=(10, 8))
        class_names = range(self.num_classes)  # 假设类别从 0 到 num_classes-1
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'pink']  # 根据类别定义颜色

        for class_label in class_names:
            idx = test_labels == class_label
            plt.scatter(
                reduced_features[idx, 0], reduced_features[idx, 1],
                label=f'Class {class_label}', alpha=0.6
            )

        plt.legend()
        plt.title(f't-SNE Visualization of Test Dataset at Epoch {epoch}')

        # 保存图像
        output_dir = f"./pic/AOFL_class_tsne_without_{self.dataset}/"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{self.dataset}_test_epoch_{epoch}.png")
        plt.savefig(output_path)
        plt.close()


    def get_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.95, patience=10, threshold=1e-6)


    def angular_loss(self, shared, private, target_cos):
        # 计算共享特征和私有特征之间的角度
        cosine_similarity = F.cosine_similarity(shared, private, dim=-1)
        angular_loss = torch.mean((cosine_similarity - target_cos) ** 2)
        return angular_loss


    #
    def efficient_contrastive_loss(self,shared_text, shared_audio, shared_visual, labels,
                                                     temperature=0.1):
        """
        高效标签感知跨模态对比损失（只使用不同标签样本作为负样本）
        """
        batch_size = shared_text.shape[0]
        device = shared_text.device

        # Step 1: 拼接模态 [3, B, D]
        all_modalities = torch.stack([shared_text, shared_audio, shared_visual], dim=0)  # [3, B, D]

        # Step 2: 为每个样本随机选择 anchor 和 positive 模态索引（保证不同模态）
        modal_indices = torch.randint(0, 3, (batch_size,), device=device)
        pos_indices = (modal_indices + torch.randint(1, 3, (batch_size,), device=device)) % 3

        anchor_features = all_modalities[modal_indices, torch.arange(batch_size)]  # [B, D]
        positive_features = all_modalities[pos_indices, torch.arange(batch_size)]  # [B, D]

        # Step 3: L2 归一化
        anchor_features = F.normalize(anchor_features, p=2, dim=-1)
        positive_features = F.normalize(positive_features, p=2, dim=-1)

        # Step 4: 相似度矩阵 logits[i, j] = anchor_i · positive_j
        logits = torch.matmul(anchor_features, positive_features.T) / temperature  # [B, B]
        # positive_score = F.sigmoid(self.bilinear(anchor_features,positive_features))
        # print("positive_score",positive_score.shape)


        # Step 5: 只保留标签不同的样本作为负样本，构造 mask
        label_matrix = labels.unsqueeze(1).expand(batch_size, batch_size)  # [B, B]
        mask = label_matrix != label_matrix.T  # True 表示标签不同（保留）

        # Step 6: 强制把对角线保留为 True（正样本）
        mask.fill_diagonal_(True)

        # Step 7: 用 mask 过滤 logits：把不该参与的负样本设为一个极小值
        logits_masked = logits.masked_fill(~mask, float('-inf'))

        # Step 8: 构造目标（正样本在对角线）
        target = torch.arange(batch_size, device=device)

        # Step 9: CrossEntropyLoss（忽略那些被设为 -inf 的负样本）
        loss = F.cross_entropy(logits_masked, target)

        return loss

    def contrastive_loss_label(self, shared_text, shared_audio, shared_visual, labels, temperature=0.07):
        """
        标签感知 InfoNCE 对比损失：
        每个样本随机选一个模态为 anchor，另一个模态为 positive，
        然后只使用标签不同的样本作为负样本（正样本仍在对角线）。
        """
        batch_size = shared_text.shape[0]
        device = shared_text.device

        # Step 1: 拼接模态 [B, 3, D]
        all_modalities = torch.stack([shared_text, shared_audio, shared_visual], dim=1)  # [B, 3, D]

        # Step 2: 随机选 anchor 和 positive（不同模态）
        modal_indices = torch.randint(0, 3, (batch_size,), device=device)
        pos_indices = (modal_indices + torch.randint(1, 3, (batch_size,), device=device)) % 3
        anchor_features = all_modalities[torch.arange(batch_size), modal_indices]  # [B, D]
        positive_features = all_modalities[torch.arange(batch_size), pos_indices]  # [B, D]

        # Step 3: L2 归一化
        anchor_features = F.normalize(anchor_features, p=2, dim=-1)
        positive_features = F.normalize(positive_features, p=2, dim=-1)

        # Step 4: 相似度矩阵 logits[i, j] = anchor_i · positive_j
        sim_matrix = torch.matmul(anchor_features, positive_features.T) / temperature  # [B, B]

        # Step 5: 计算标签不同的掩码（True 表示标签不同）
        label_matrix = labels.unsqueeze(1) != labels.unsqueeze(0)  # [B, B]
        label_matrix = label_matrix.to(device)

        # Step 6: 对角线必须保留（正样本）
        label_matrix[torch.arange(batch_size), torch.arange(batch_size)] = True

        # Step 7: 计算 exp 相似度（用掩码筛掉标签相同的负样本）
        exp_sim = torch.exp(sim_matrix)  # [B, B]
        masked_exp_sim = exp_sim * label_matrix  # 只保留标签不同的负样本
        all_score = masked_exp_sim.sum(dim=1)  # [B]
        pos_score = torch.diag(exp_sim)  # [B]

        # Step 8: 计算最终 InfoNCE 损失
        loss = (-torch.log(pos_score / all_score)).mean()

        return loss



    def multi_contrastive_loss_label(self, shared_anchor, shared_x1, shared_x2, specific_anchor,  labels, temperature=0.07):
        """
        标签感知 InfoNCE 对比损失：
        每个样本随机选一个模态为 anchor，另一个模态为 positive，
        然后只使用标签不同的样本作为负样本（正样本仍在对角线）。
        """
        batch_size = shared_anchor.shape[0]
        device = shared_anchor.device

        # Step 1: 拼接模态 [B, 3, D]
        other_modalities = torch.stack([shared_x1, shared_x2], dim=1)  # [B, 3, D]
        # Step 2: 随机选 anchor 和 positive（不同模态）
        pos_indices = torch.ones_like(torch.randint(0, 2, (batch_size,), device=device))
        positive_features = other_modalities[torch.arange(batch_size), pos_indices]  # [B, D]

        # Step 3: L2 归一化
        anchor_features = F.normalize(shared_anchor, p=2, dim=-1)
        positive_features = F.normalize(positive_features, p=2, dim=-1)
        negative_features = F.normalize(specific_anchor, p=2, dim=-1)

        # Step 4: 相似度矩阵 logits[i, j] = anchor_i · positive_j
        sim_matrix = torch.matmul(anchor_features, positive_features.T) / temperature  # [B, B]
        neg_matrix = torch.matmul(anchor_features, negative_features.T) / temperature
        all_matrix = neg_matrix.clone()

        # Step 5: 计算标签不同的掩码（True 表示标签不同）
        label_matrix = labels.unsqueeze(1) != labels.unsqueeze(0)  # [B, B]
        label_matrix = label_matrix.to(device)

        # Step 6: 对角线必须保留（正样本）
        label_matrix[torch.arange(batch_size), torch.arange(batch_size)] = True
        all_matrix[torch.arange(batch_size), torch.arange(batch_size)] = sim_matrix[
            torch.arange(batch_size), torch.arange(batch_size)].clone()
        # Step 7: 计算 exp 相似度（用掩码筛掉标签相同的负样本）
        exp_sim = torch.exp(all_matrix)  # [B, B]
        masked_exp_sim = exp_sim * label_matrix  # 只保留标签不同的负样本
        all_score = masked_exp_sim.sum(dim=1)  # [B]
        pos_score = torch.diag(exp_sim)  # [B]

        # Step 8: 计算最终 InfoNCE 损失
        loss = (-torch.log(pos_score / all_score)).mean()

        return loss


    def contrastive_loss_label_4_9(self, shared_text, shared_audio, shared_visual,
                               specific_text, specific_audio, specific_visual,
                               labels, temperature=0.07):
        """
        标签感知 InfoNCE 对比损失：
        每个样本随机选一个模态为 anchor，另一个模态为 positive，
        然后只使用标签不同的样本作为负样本（正样本仍在对角线）。
        """
        batch_size = shared_text.shape[0]
        device = shared_text.device

        # Step 1: 拼接模态 [B, 3, D]
        all_modalities = torch.stack([shared_text, shared_audio, shared_visual], dim=1)  # [B, 3, D]
        all_specific_modalities = torch.stack([specific_text, specific_audio, specific_visual], dim=1)  # [B, 3, D]

        # Step 2: 随机选 anchor 和 positive（不同模态）
        modal_indices = torch.randint(0, 3, (batch_size,), device=device)
        pos_indices = (modal_indices + torch.randint(1, 3, (batch_size,), device=device)) % 3
        anchor_features = all_modalities[torch.arange(batch_size), modal_indices]  # [B, D]
        positive_features = all_modalities[torch.arange(batch_size), pos_indices]  # [B, D]

        # 这里可调参 0.1-0.9
        positive_features_spec = all_specific_modalities[torch.arange(batch_size), modal_indices]  # [B, D]
        rand_index = torch.randint(positive_features.size(0), (int(positive_features.size(0) * 0.2),)).cuda()
        positive_features[rand_index] = positive_features_spec[rand_index].clone()

        # Step 3: L2 归一化
        anchor_features = F.normalize(anchor_features, p=2, dim=-1)
        positive_features = F.normalize(positive_features, p=2, dim=-1)

        # Step 4: 相似度矩阵 logits[i, j] = anchor_i · positive_j
        sim_matrix = torch.matmul(anchor_features, positive_features.T) / temperature  # [B, B]

        # Step 5: 计算标签不同的掩码（True 表示标签不同）
        label_matrix = labels.unsqueeze(1) != labels.unsqueeze(0)  # [B, B]
        label_matrix = label_matrix.to(device)

        # Step 6: 对角线必须保留（正样本）
        label_matrix[torch.arange(batch_size), torch.arange(batch_size)] = True

        # Step 7: 计算 exp 相似度（用掩码筛掉标签相同的负样本）
        exp_sim = torch.exp(sim_matrix)  # [B, B]
        masked_exp_sim = exp_sim * label_matrix  # 只保留标签不同的负样本
        all_score = masked_exp_sim.sum(dim=1)  # [B]
        pos_score = torch.diag(exp_sim)  # [B]
        # Step 8: 计算最终 InfoNCE 损失
        loss = (-torch.log(pos_score / all_score)).mean()

        return loss

    def train_or_eval_model_per_epoch(self, dataloader, epoch, train):
        if train == 'Train':
            self.model.train()
        else:
            self.model.eval()
        Contrastive_loss_final_list = []
        total_loss_list, CE_loss_list, AAO_loss_list, Cos_loss_list, AAC_loss_list, Contrastive_loss_list = [], [], [], [], [], []
        all_labels, all_preds = [], []
        total_loss = 0.0
        inference_times = []
        start = time.time()
        for _, data in tqdm(enumerate(dataloader), total=len(dataloader)):

            if train == 'Train':
                self.optimizer.zero_grad()

            padded_texts, padded_audios, padded_visuals, padded_speaker_masks, padded_utterance_masks, padded_labels = [
                d.to(self.device) for d in data]
            padded_labels = padded_labels.reshape(-1)
            labels = padded_labels[padded_labels != -1]
            # print('11111',labels.shape)
            (mlp_outputs, share_text, share_audio, share_visual,
             specific_text, specific_audio, specific_visual,
             AAO_loss, AAC_loss, cos_loss) = self.model(padded_texts, padded_audios, padded_visuals,
                                                        padded_speaker_masks, padded_utterance_masks, padded_labels)

            #
            # # 保存共享特征和特定特征
            if train != 'Train':
                self.shared_text_features.append(share_text.cpu().detach().numpy())  # 共享特征
                self.specific_text_features.append(specific_text.cpu().detach().numpy())   # 特定特征
                self.shared_audio_features.append(share_audio.cpu().detach().numpy())  # 共享特征
                self.specific_audio_features.append(specific_audio.cpu().detach().numpy())   # 特定特征
                self.shared_visual_features.append(share_visual.cpu().detach().numpy())  # 共享特征
                self.specific_visual_features.append(specific_visual.cpu().detach().numpy())   # 特定特征
                self.labels.append(labels.cpu().detach().numpy())                # 标签




            contrastive_loss = self.contrastive_loss_label(share_text, share_visual, share_audio, labels, temperature=1)



            CE_loss = self.CE_loss(mlp_outputs, labels)

            if epoch < 15:
                loss = CE_loss * self.CE_loss_param
            else:
                loss = 10* CE_loss * self.CE_loss_param + 0.09 * AAO_loss + 1 * contrastive_loss   # best


            total_loss += loss.item()
            total_loss_list.append(loss.item())
            CE_loss_list.append(CE_loss.item())
            AAO_loss_list.append(AAO_loss.item())
            Cos_loss_list.append(cos_loss.item())
            AAC_loss_list.append(AAC_loss.item())
            Contrastive_loss_list.append(contrastive_loss.item())

            if train == 'Train':
                loss.backward()
                self.optimizer.step()

            preds = torch.argmax(mlp_outputs, dim=-1)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())


        if train == 'Train':
            print('{0}-th total_loss:{1:.4f}, CE_loss:{2:.4f}, AAO_loss:{3:.4f}, Contrastive_loss:{4:.4f},'
                  ' Cos_loss:{5:.4f}, AAC_loss:{6:.4f}'.format(
                epoch, (sum(total_loss_list) / len(total_loss_list)), (sum(CE_loss_list) / len(CE_loss_list)),
                (sum(AAO_loss_list) / len(AAO_loss_list)), (sum(Contrastive_loss_list) / len(Contrastive_loss_list)),
                (sum(Cos_loss_list) / len(Cos_loss_list)), (sum(AAC_loss_list) / len(AAC_loss_list))))

        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        avg_f1 = round(f1_score(all_labels, all_preds, average='weighted') * 100, 4)
        avg_acc = round(accuracy_score(all_labels, all_preds) * 100, 4)
        report = classification_report(all_labels, all_preds, digits=4)



        end = time.time()
        inference_times.append(end - start)

        print(f"Avg inference time per batch: {np.mean(inference_times):.4f} seconds")

        if train == 'Test':
            print(report)
        return (round(total_loss, 4), avg_f1, avg_acc, report)

    def train_or_eval_linear_model(self):
        # Each run has its own directory; existing results are never overwritten.
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = os.path.abspath(getattr(self, "requested_output_dir", None) or os.path.join(f"save_model_{self.dataset}", run_id))
        os.makedirs(output_dir, exist_ok=False)
        print(f"Run results directory: {output_dir}")
        self.writer.close()
        self.writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))
        config_keys = ("dataset", "batch_size", "num_epochs", "learning_rate",
                       "weight_decay", "num_layers", "model_dim", "num_heads",
                       "hidden_dim", "dropout_rate", "dropout_rec",
                       "CE_loss_param", "multi_attn_flag")
        config = {key: getattr(self, key) for key in config_keys}
        config.update(csr_mode=getattr(self, "csr_mode", "vector"),
                      subspace_margin=getattr(self, "subspace_margin", 0.1),
                      refinement=getattr(self, "refinement", "baseline"),
                      subspace_rank=getattr(self, "subspace_rank", 4),
                      device=str(self.device), torch_initial_seed=torch.initial_seed(),
                      selection_metric="validation_weighted_f1_percent",
                      tie_policy="keep_earliest_epoch", loss_reduction="sum_of_batch_losses",
                      train_indices=self.train_indices, valid_indices=self.valid_indices,
                      split_strategy=self.split_strategy,
                      train_dialogue_ids=[self.train_dataloader.dataset.keys[i] for i in self.train_indices],
                      valid_dialogue_ids=[self.valid_dataloader.dataset.keys[i] for i in self.valid_indices],
                      test_dialogue_ids=list(self.test_dataloader.dataset.keys),
                      loss_schedule={"epochs_1_to_14": "CE_loss_param * CE",
                                     "epochs_15_onward": "10 * CE_loss_param * CE + 0.09 * AAO + CEN"})
        def save_json(name, value):
            path = os.path.join(output_dir, name)
            with open(path + ".tmp", "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2)
            os.replace(path + ".tmp", path)

        save_json("config.json", config)
        self.best_valid_f1 = float("-inf")
        fields = ["epoch", "loss_stage", "learning_rate", "next_learning_rate",
                  "train_loss", "train_f1", "train_acc", "valid_loss", "valid_f1",
                  "valid_acc", "test_loss", "test_f1", "test_acc", "epoch_seconds"]
        total_start = time.time()
        try:
            with open(os.path.join(output_dir, "metrics.csv"), "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                handle.flush()
                for e in range(1, self.num_epochs + 1):
                    epoch_start = time.time()
                    lr = self.optimizer.param_groups[0]["lr"]
                    # Feature caches are only needed for the current evaluation.
                    for name in ("shared_text_features", "specific_text_features",
                                 "shared_audio_features", "specific_audio_features",
                                 "shared_visual_features", "specific_visual_features", "labels", "fc_out"):
                        setattr(self, name, [])
                    train_loss, train_f1, train_acc, train_report = self.train_or_eval_model_per_epoch(self.train_dataloader, e, train="Train")
                    with torch.no_grad():
                        valid_loss, valid_f1, valid_acc, valid_report = self.train_or_eval_model_per_epoch(self.valid_dataloader, e, train="Valid")
                        test_loss, test_f1, test_acc, test_report = self.train_or_eval_model_per_epoch(self.test_dataloader, e, train="Test")
                    self.scheduler.step(valid_loss)
                    row = dict(epoch=e, loss_stage="CE_only" if e < 15 else "combined",
                               learning_rate=lr, next_learning_rate=self.optimizer.param_groups[0]["lr"],
                               train_loss=train_loss, train_f1=train_f1, train_acc=train_acc,
                               valid_loss=valid_loss, valid_f1=valid_f1, valid_acc=valid_acc,
                               test_loss=test_loss, test_f1=test_f1, test_acc=test_acc,
                               epoch_seconds=round(time.time() - epoch_start, 4))
                    writer.writerow(row)
                    handle.flush()
                    report_dir = os.path.join(output_dir, "reports")
                    os.makedirs(report_dir, exist_ok=True)
                    with open(os.path.join(report_dir, f"epoch_{e:03d}.txt"), "w", encoding="utf-8") as report_file:
                        report_file.write(f"Epoch {e}\n\nTrain\n{train_report}\nValid\n{valid_report}\nTest\n{test_report}")
                    for split in ("train", "valid", "test"):
                        for metric in ("loss", "f1", "acc"):
                            self.writer.add_scalar(f"{split}/{metric}", row[f"{split}_{metric}"], e)
                    self.writer.add_scalar("train/learning_rate", lr, e)
                    self.writer.flush()
                    if valid_f1 > self.best_valid_f1:
                        checkpoint = dict(epoch=e, model_state_dict=self.model.state_dict(),
                                          optimizer_state_dict=self.optimizer.state_dict(),
                                          scheduler_state_dict=self.scheduler.state_dict(),
                                          metrics=row, config=config,
                                          selection_metric="validation_weighted_f1_percent")
                        checkpoint_path = os.path.join(output_dir, "best_model.pth")
                        torch.save(checkpoint, checkpoint_path + ".tmp")
                        os.replace(checkpoint_path + ".tmp", checkpoint_path)
                        self.best_valid_f1 = valid_f1
                        self.best_epoch = e
                        self.best_test_f1 = test_f1
                        self.best_test_report = test_report
                        save_json("best_metrics.json", row)
                        with open(os.path.join(output_dir, "best_report.txt"), "w", encoding="utf-8") as report_file:
                            report_file.write(f"Selected by validation F1; epoch {e}\n\nValid\n{valid_report}\nTest\n{test_report}")
                    print(f"Best validation F1: {self.best_valid_f1} at epoch {self.best_epoch}; corresponding test F1: {self.best_test_f1}")
            # Keep the original full-model format for the final epoch only.
            torch.save(self.model, os.path.join(output_dir, "final_model.pth"))
            save_json("summary.json", dict(best_epoch=self.best_epoch,
                      best_valid_f1=self.best_valid_f1, test_f1_at_best_valid=self.best_test_f1,
                      final_epoch_metrics=row,
                      total_training_minutes=round((time.time() - total_start) / 60, 4)))
            print(self.best_test_report)
            print(f"Results and models saved: {output_dir}")
            print(f"Total training time: {(time.time() - total_start) / 60:.2f} minutes")
        finally:
            self.writer.close()


def get_args():
    parser = OptionParser()
    parser.add_option('--csr_mode', default='vector', type='choice', choices=['vector', 'subspace'])
    parser.add_option('--subspace_margin', default=0.1, type='float')
    parser.add_option('--refinement', default='baseline', type='choice', choices=['baseline', 'qr', 'free', 'directional'])
    parser.add_option('--subspace_rank', default=4, type='int')
    parser.add_option('--seed', default=2025, type='int', help='random seed')
    parser.add_option('--output_dir', default=None, help='new directory for this run')
    parser.add_option('--dataset', dest='dataset', default='MELD', type='str', help='MELD or IEMOCAP')
    parser.add_option('--batch_size', dest='batch_size', default=64, type='int', help='64 for IEMOCAP and 100 for MELD')
    parser.add_option('--num_epochs', dest='num_epochs', default=100, type='int', help='number of epochs')
    #iemocp
    parser.add_option('--learning_rate', dest='learning_rate', default=0.0001, type='float', help='learning rate')
    #meld
    parser.add_option('--weight_decay', dest='weight_decay', default=0.0001, type='float',
                      help='weight decay parameter')
    parser.add_option('--num_layers', dest='num_layers', default=4, type='int', help='number of layers in MultiAttn')
    parser.add_option('--model_dim', dest='model_dim', default=256, type='int', help='model dimension in MultiAttn')
    parser.add_option('--num_heads', dest='num_heads', default=4, type='int', help='number of heads in MultiAttn')
    parser.add_option('--hidden_dim', dest='hidden_dim', default=1024, type='int', help='hidden dimension in MultiAttn')
    parser.add_option('--dropout_rate', dest='dropout_rate', default=0.3, type='float', help='dropout rate')
    parser.add_option('--dropout_rec', dest='dropout_rec', default=0, type='float', help='dropout rec')
    parser.add_option('--CE_loss_param', dest='CE_loss_param', default=1, type='float',
                      help='coefficient of Cross Entropy loss')

    parser.add_option('--multi_attn_flag', dest='multi_attn_flag', default=True, help='Multimodal fusion')

    (options, _) = parser.parse_args()

    return options


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    args = get_args()
    dataset = args.dataset
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    num_layers = args.num_layers
    model_dim = args.model_dim
    num_heads = args.num_heads
    hidden_dim = args.hidden_dim
    dropout_rate = args.dropout_rate
    dropout_rec = args.dropout_rec

    CE_loss_param = args.CE_loss_param


    multi_attn_flag = args.multi_attn_flag
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed = args.seed

    set_seed(seed)

    AOFL_train = AOFL(dataset, batch_size, num_epochs, learning_rate,
                                   weight_decay, num_layers, model_dim, num_heads, hidden_dim,
                                   dropout_rate, dropout_rec,
                                   CE_loss_param,  multi_attn_flag,
                                   device)

    AOFL_train.refinement = args.refinement
    AOFL_train.subspace_rank = args.subspace_rank
    if args.refinement != 'baseline' or args.csr_mode != 'vector':
        raise ValueError('GAPD/subspace experiments archived; use baseline and vector CSR.')
    AOFL_train.requested_output_dir = args.output_dir
    AOFL_train.count_parameters()
    AOFL_train.train_or_eval_linear_model()


