import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
'''
Bidirectional cross-attention layers.
'''


class BidirectionalCrossAttention(nn.Module):

    def __init__(self, model_dim, Q_dim, K_dim, V_dim):
        super().__init__()

        self.query_matrix = nn.Linear(model_dim, Q_dim)
        self.key_matrix = nn.Linear(model_dim, K_dim)
        self.value_matrix = nn.Linear(model_dim, V_dim)

    def bidirectional_scaled_dot_product_attention(self, Q, K, V):
        score = torch.bmm(Q, K.transpose(-1, -2))
        scaled_score = score / (K.shape[-1] ** 0.5)
        attention = torch.bmm(F.softmax(scaled_score, dim=-1), V)

        return attention

    def forward(self, query, key, value):
        Q = self.query_matrix(query)
        K = self.key_matrix(key)
        V = self.value_matrix(value)
        attention = self.bidirectional_scaled_dot_product_attention(Q, K, V)
        # print('Q',Q.shape)
        # print('attention',attention.shape)
        return attention


'''
Multi-head bidirectional cross-attention layers.
'''


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, model_dim, Q_dim, K_dim, V_dim):
        super().__init__()

        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList(
            [BidirectionalCrossAttention(model_dim, Q_dim, K_dim, V_dim) for _ in range(self.num_heads)]
        )
        self.projection_matrix = nn.Linear(num_heads * V_dim, model_dim)

    def forward(self, query, key, value):
        heads = [self.attention_heads[i](query, key, value) for i in range(self.num_heads)]
        # print('heads', torch.cat(heads, dim=-1).shape)
        multihead_attention = self.projection_matrix(torch.cat(heads, dim=-1))

        return multihead_attention


'''
A feed-forward network, which operates as a key-value memory.
'''


class Feedforward(nn.Module):

    def __init__(self, model_dim, hidden_dim, dropout_rate):
        super().__init__()

        self.linear_W1 = nn.Linear(model_dim, hidden_dim)
        self.linear_W2 = nn.Linear(hidden_dim, model_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(self.linear_W2(self.relu(self.linear_W1(x))))


'''
Residual connection to smooth the learning process.
'''


class AddNorm(nn.Module):

    def __init__(self, model_dim, dropout_rate):
        super().__init__()

        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, sublayer):
        output = self.layer_norm(x + self.dropout(sublayer(x)))

        return output

class SharedEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, output_dim)
        # self.encoder = nn.Linear(input_dim, output_dim,bias=False)


    def forward(self, x):
        return self.encoder(x)
class MLPEncoder(nn.Module):
    def __init__(self, input_dim,  output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim), #128 64
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)  #64 64
        )

    def forward(self, x):
        return self.encoder(x)




class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer(x)

#---

class PrivateEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, output_dim)
        # self.encoder = nn.Linear(input_dim, output_dim,bias=False)

    def forward(self, x):
        return self.encoder(x)


class MLPPrivateEncoder(nn.Module):
    def __init__(self, input_dim,  output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class TransformerPrivateEncoder(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.transformer(x)


# #our
class AttnLayer(nn.Module):

    def __init__(self, num_heads, model_dim, hidden_dim, dropout_rate):
        super().__init__()

        Q_dim = K_dim = V_dim = model_dim // num_heads
        self.attn_1 = MultiHeadAttention(num_heads, model_dim, Q_dim, K_dim, V_dim)
        self.add_norm_1 = AddNorm(model_dim, dropout_rate)
        self.ff = Feedforward(model_dim, hidden_dim, dropout_rate)
        self.add_norm_2 = AddNorm(model_dim, dropout_rate)

    def forward(self, query_modality, modality_A):
        attn_output_1 = self.add_norm_1(query_modality,
                                        lambda query_modality: self.attn_1(query_modality, modality_A, modality_A))
        ff_output = self.add_norm_2(attn_output_1, self.ff)

        return ff_output


class MAttn(nn.Module):

    def __init__(self, num_layers, model_dim, num_heads, hidden_dim, dropout_rate):
        super().__init__()

        self.mattn_layers = nn.ModuleList([
            AttnLayer(num_heads, model_dim, hidden_dim, dropout_rate) for _ in range(num_layers)])

    def forward(self, query_modality, modality_A):
        for mattn_layer in self.mattn_layers:
            query_modality = mattn_layer(query_modality, modality_A)

        return query_modality

class CosinePredictor(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        # 定义线性层
        self.target_cos_predictor = nn.Linear(2 * model_dim, 1)

    def forward(self, shared, specific):
        # 拼接共享特征和特定特征
        input_features = torch.cat([shared, specific], dim=-1)
        # 通过线性层预测 cos(θ)
        predicted_cos_theta = torch.tanh(self.target_cos_predictor(input_features))
        return predicted_cos_theta



class Multi_Disentangle_Model(nn.Module):

    def __init__(self, num_layers, model_dim, num_heads, hidden_dim, dropout_rate):
        super().__init__()


        self.shared_encoder = MLPEncoder(model_dim,model_dim)


        self.private_encoders = nn.ModuleDict({
            'text': MLPPrivateEncoder(model_dim, model_dim),
            'audio': MLPPrivateEncoder(model_dim,model_dim),
            'visual': MLPPrivateEncoder(model_dim, model_dim)
        })


        # Attention layers
        self.multiattn_text = MAttn(num_layers, model_dim, num_heads, hidden_dim, dropout_rate)
        self.multiattn_audio = MAttn(num_layers, model_dim, num_heads, hidden_dim, dropout_rate)
        self.multiattn_visual = MAttn(num_layers, model_dim, num_heads, hidden_dim, dropout_rate)

        # Linear layers to predict target cosθ for each modality
        self.target_cos_predictors = nn.ModuleDict({
            'text': nn.Linear(model_dim*2, 1),
            'audio': nn.Linear(model_dim*2, 1),
            'visual': nn.Linear(model_dim*2, 1)
        })
        self.cosine_predictors = nn.ModuleDict({
            'text': CosinePredictor(model_dim),
            'audio': CosinePredictor(model_dim),
            'visual': CosinePredictor(model_dim),
        })

        self.fusion_layer = nn.Linear(model_dim * 3, model_dim)



    def contrastive_loss_label_aware(self, shared_text, shared_audio, shared_visual, labels, temperature=1.0):
        """
        标签感知的跨模态对比损失。
        正样本：同一标签 & 不同模态；
        负样本：不同标签 & 任意模态。
        """
        batch_size = shared_text.shape[0]
        all_modalities = ['text', 'audio', 'visual']
        modality_features = {
            'text': shared_text,
            'audio': shared_audio,
            'visual': shared_visual
        }

        total_loss = 0.0
        num_pairs = 0

        for i in range(batch_size):
            anchor_label = labels[i].item()

            for anchor_modal, pos_modal in [('text', 'audio'), ('text', 'visual'), ('audio', 'visual')]:
                if anchor_modal == pos_modal:
                    continue

                anchor = modality_features[anchor_modal][i]
                positive = modality_features[pos_modal][i]

                # 构建负样本（不同标签）
                negative_samples = []
                for j in range(batch_size):
                    if labels[j].item() != anchor_label:
                        for neg_modal in all_modalities:
                            negative = modality_features[neg_modal][j]
                            negative_samples.append(negative.unsqueeze(0))

                if len(negative_samples) == 0:
                    continue  # 如果标签都一样，跳过当前样本

                negatives = torch.cat(negative_samples, dim=0)  # [N, dim]

                sim_pos = F.cosine_similarity(anchor.unsqueeze(0), positive.unsqueeze(0)) / temperature
                sim_negs = F.cosine_similarity(anchor.unsqueeze(0), negatives) / temperature

                logits = torch.cat([sim_pos, sim_negs], dim=0)
                labels_ce = torch.zeros(logits.shape[0], dtype=torch.long).to(anchor.device)

                loss = F.cross_entropy(logits.unsqueeze(0), labels_ce.unsqueeze(0))
                total_loss += loss
                num_pairs += 1

        return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0, requires_grad=True).to(shared_text.device)






    def forward(self, text_features, audio_features, visual_features):
        # print(text_features.shape)

        shared_text = self.shared_encoder(text_features)
        exclusive_text = self.private_encoders['text'](text_features)

        shared_audio = self.shared_encoder(audio_features)
        exclusive_audio = self.private_encoders['audio'](audio_features)

        shared_visual = self.shared_encoder(visual_features)
        exclusive_visual = self.private_encoders['visual'](visual_features)

        shared_text_tsne, shared_audio_tsne, shared_visual_tsne = shared_text.clone(), shared_audio.clone(), shared_visual.clone()
        orth_text_tsne, orth_audio_tsne, orth_visual_tsne = exclusive_text.clone(), exclusive_audio.clone(), exclusive_visual.clone()

        # ============================
        cos_theta_text_actual = F.cosine_similarity(shared_text, exclusive_text, dim=-1)
        cos_theta_audio_actual = F.cosine_similarity(shared_audio, exclusive_audio, dim=-1)
        cos_theta_visual_actual = F.cosine_similarity(shared_visual, exclusive_visual, dim=-1)

        orth_text = exclusive_text - cos_theta_text_actual.unsqueeze(-1) * exclusive_text
        orth_audio = exclusive_audio - cos_theta_audio_actual.unsqueeze(-1) * exclusive_audio
        orth_visual = exclusive_visual - cos_theta_visual_actual.unsqueeze(-1) * exclusive_visual
        # Optional experiment replaces OPR only; baseline remains the default.
        if hasattr(self, 'subspace_refiners'):
            orth_text = self.subspace_refiners['text'](shared_text, exclusive_text)
            orth_audio = self.subspace_refiners['audio'](shared_audio, exclusive_audio)
            orth_visual = self.subspace_refiners['visual'](shared_visual, exclusive_visual)
        # ============================

        fused_text = self.multiattn_text(shared_text, shared_text)
        fused_audio = self.multiattn_audio(shared_audio, shared_audio)
        fused_visual = self.multiattn_visual(shared_visual, shared_visual)



        final_text = torch.cat((fused_text, orth_text), dim=-1)
        final_audio = torch.cat((fused_audio, orth_audio), dim=-1)
        final_visual = torch.cat((fused_visual, orth_visual), dim=-1)


        return (final_text, final_audio, final_visual,
                shared_text_tsne, shared_audio_tsne, shared_visual_tsne,
                orth_text_tsne, orth_audio_tsne, orth_visual_tsne)


