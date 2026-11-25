from DialogueRNN import BiModel
from Multi_Disentangle import Multi_Disentangle_Model
from MLP import MLP
import torch
import torch.nn as nn
import torch.nn.functional as F




class AOFL_model(nn.Module):

    def __init__(self, dataset, multi_attn_flag, roberta_dim, hidden_dim, dropout, num_layers,
                 model_dim, num_heads, D_m_audio, D_m_visual, D_g, D_p, D_e, D_h,
                 n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec, device):
        super().__init__()
        self.dataset = dataset
        self.multi_attn_flag = multi_attn_flag

        self.text_fc = nn.Linear(roberta_dim, model_dim)
        self.text_dialoguernn = BiModel(model_dim, D_g, D_p, D_e, D_h, dataset,
                                        n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec,
                                        dropout, device)

        self.audio_fc = nn.Linear(D_m_audio, model_dim)
        self.audio_dialoguernn = BiModel(model_dim, D_g, D_p, D_e, D_h, dataset,
                                         n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec,
                                         dropout, device)
        self.visual_fc = nn.Linear(D_m_visual, model_dim)
        self.visual_dialoguernn = BiModel(model_dim, D_g, D_p, D_e, D_h, dataset,
                                          n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec,
                                          dropout, device)

        self.multiattn = Multi_Disentangle_Model(num_layers, model_dim, num_heads, hidden_dim, dropout)
        self.fc = nn.Linear(model_dim * 6, model_dim)
        if self.dataset == 'MELD':
            self.mlp = MLP(model_dim, model_dim * 2, n_classes, dropout)
        elif self.dataset == 'IEMOCAP':
            self.mlp = MLP(model_dim, model_dim, n_classes, dropout)

        self.target_cos_predictors = nn.ModuleDict({
            'text': nn.Linear(model_dim*2, 1),
            'audio': nn.Linear(model_dim*2, 1),
            'visual': nn.Linear(model_dim*2, 1)
        })
        self.modality_dropout_prob = 0.05

        self.angle_param_dict = nn.ParameterDict({
            'text': nn.Parameter(torch.tensor(0.5)),  # 初始值在 sigmoid 范围内
            'audio': nn.Parameter(torch.tensor(0.5)),
            'visual': nn.Parameter(torch.tensor(0.5)),
        })

    def apply_modality_dropout(self, text_feat, audio_feat, visual_feat):
        if self.training:  # 只在训练阶段应用
            if torch.rand(1).item() < self.modality_dropout_prob:
                text_feat = torch.zeros_like(text_feat)
            if torch.rand(1).item() < self.modality_dropout_prob:
                audio_feat = torch.zeros_like(audio_feat)
            if torch.rand(1).item() < self.modality_dropout_prob:
                visual_feat = torch.zeros_like(visual_feat)
        return text_feat, audio_feat, visual_feat

    def angle_constraint_loss(self, shared, exclusive, modality):
        cos_theta = F.cosine_similarity(shared, exclusive, dim=-1)  # 实际的 cos(θ)
        # expected_cos = torch.sigmoid(self.angle_param_dict[modality]) * 2 - 1  # 映射到 [-
        expected_cos = torch.sigmoid(self.angle_param_dict[modality])    # 映射到 [-
        # expected_cos = torch.tanh(self.angle_param_dict[modality])   # 映射到 [-1, 1]
        return torch.mean(torch.abs(cos_theta - expected_cos))  # 使用 L1 损失


    def cosine_loss(self, shared, exclusive):
        cos_theta = F.cosine_similarity(shared, exclusive, dim=-1)
        return torch.mean(torch.abs(cos_theta))

    def cosine_rank_loss(self, shared, exclusive, other_shared):
        cos_theta_small = F.cosine_similarity(shared, exclusive, dim=-1)

        pos_indices = torch.ones_like(torch.randint(0, 2, (shared.shape[0],), device=shared.device))
        sample = other_shared[torch.arange(shared.shape[0]), pos_indices]  # [B, D]
        cos_theta_Large = F.cosine_similarity(shared, sample, dim=-1)

        return F.relu(cos_theta_small - cos_theta_Large).mean()

    def RMSE_loss(self, share_feature, specific_feature, modal_name):
        cos_theta_text_actual = F.cosine_similarity(share_feature, specific_feature, dim=-1)
        input_text_features = torch.cat([share_feature, specific_feature], dim=-1)
        cos_theta_text_predicted = torch.tanh(self.target_cos_predictors[modal_name](input_text_features).squeeze(-1))
        # 计算均方误差（MSE）
        mse = torch.mean((cos_theta_text_predicted - cos_theta_text_actual) ** 2)
        # 计算均方根误差（RMSE）
        rmse = torch.sqrt(mse)

        return rmse

    def forward(self, texts, audios, visuals, speaker_masks, utterance_masks, padded_labels):

        text_features = self.text_fc(texts)

        if self.dataset == 'IEMOCAP':
            # print('DDD',text_features.shape)
            text_features = self.text_dialoguernn(text_features, speaker_masks, utterance_masks)
            print('')
            #

        audio_features = self.audio_fc(audios)
        audio_features = self.audio_dialoguernn(audio_features, speaker_masks, utterance_masks)

        visual_features = self.visual_fc(visuals)
        visual_features = self.visual_dialoguernn(visual_features, speaker_masks, utterance_masks)
        text_features = text_features.transpose(0, 1)
        audio_features = audio_features.transpose(0, 1)
        visual_features = visual_features.transpose(0, 1)

        if self.multi_attn_flag == True:
            (final_text, final_audio, final_visual,
             share_text, share_audio, share_visual,
             specific_text, specific_audio, specific_visual) = self.multiattn(text_features, audio_features, visual_features)


        final_text = final_text.reshape(-1, final_text.shape[-1])
        final_text = final_text[padded_labels != -1]
        final_audio = final_audio.reshape(-1, final_audio.shape[-1])
        final_audio = final_audio[padded_labels != -1]
        final_visual = final_visual.reshape(-1, final_visual.shape[-1])
        final_visual = final_visual[padded_labels != -1]


        share_text = share_text.reshape(-1, share_text.shape[-1])
        share_text = share_text[padded_labels != -1]
        share_audio = share_audio.reshape(-1, share_audio.shape[-1])
        share_audio = share_audio[padded_labels != -1]
        share_visual = share_visual.reshape(-1, share_visual.shape[-1])
        share_visual = share_visual[padded_labels != -1]

        specific_text = specific_text.reshape(-1, specific_text.shape[-1])
        specific_text = specific_text[padded_labels != -1]
        specific_audio = specific_audio.reshape(-1, specific_audio.shape[-1])
        specific_audio = specific_audio[padded_labels != -1]
        specific_visual = specific_visual.reshape(-1, specific_visual.shape[-1])
        specific_visual = specific_visual[padded_labels != -1]


        final_text, final_audio, final_visual = self.apply_modality_dropout(
            final_text, final_audio,final_visual
        )


        # 示例：在你特征提取后加
        fused_features = torch.cat((final_text, final_audio, final_visual), dim=-1)
        fc_outputs = self.fc(fused_features)

        mlp_outputs = self.mlp(fc_outputs)
        # print('mlp_outputs',mlp_outputs.shape)
        #

        #cosine_rank_loss############################################################################################
        cos_loss_text = self.cosine_rank_loss(share_text, specific_text,
                                              torch.cat((share_audio.unsqueeze(1), share_visual.unsqueeze(1)), 1))
        cos_loss_audio = self.cosine_rank_loss(share_audio, specific_audio,
                                               torch.cat((share_text.unsqueeze(1), share_visual.unsqueeze(1)), 1))
        cos_loss_visual = self.cosine_rank_loss(share_visual, specific_visual,
                                                torch.cat((share_text.unsqueeze(1), share_audio.unsqueeze(1)), 1))
        cos_loss = (cos_loss_text + cos_loss_audio + cos_loss_visual)


        if hasattr(self, 'subspace_ranking'):
            cos_loss = self.subspace_ranking(
                [share_text, share_audio, share_visual],
                [specific_text, specific_audio, specific_visual])

        #AAC_loss###################################################################################################
        mse_loss_text = self.RMSE_loss(share_text, specific_text, 'text')
        mse_loss_audio = self.RMSE_loss(share_audio, specific_audio, 'audio')
        mse_loss_visual = self.RMSE_loss(share_visual, specific_visual, 'visual')
        AAC_loss = (mse_loss_text + mse_loss_audio + mse_loss_visual)
        aao_loss =  0.5*AAC_loss +  0.005* cos_loss

        return (mlp_outputs, share_text, share_audio, share_visual, specific_text, specific_audio,
                specific_visual, aao_loss, AAC_loss, cos_loss)

