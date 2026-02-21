# =============================================================================
# inference_engine/model_definition.py — 与训练模型完全匹配的推理模型
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# =============================================================================
# 1️⃣ CNN Encoder
# =============================================================================
class CNNEncoder(nn.Module):
    """CNN特征提取器"""
    def __init__(self, pretrained=False):
        super().__init__()
        resnet = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        feature_map = self.features(x)
        global_features = self.pool(feature_map).view(x.size(0), -1)
        return feature_map, global_features


# =============================================================================
# 2️⃣ Disease-Aware Attention
# =============================================================================
class DiseaseAwareAttention(nn.Module):
    """疾病感知注意力机制"""
    def __init__(self, feature_dim=512, hidden_dim=512, disease_dim=14):
        super().__init__()
        self.attn = nn.Linear(feature_dim + hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
        self.disease_proj = nn.Sequential(
            nn.Linear(disease_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )

    def forward(self, features, hidden, disease_features=None):
        hidden = hidden.unsqueeze(1).repeat(1, features.size(1), 1)
        if disease_features is not None:
            disease_context = self.disease_proj(disease_features)
            hidden = hidden + disease_context.unsqueeze(1)
        energy = torch.tanh(self.attn(torch.cat((features, hidden), dim=2)))
        attention = self.v(energy).squeeze(2)
        alpha = torch.softmax(attention, dim=1)
        context = (features * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha


# =============================================================================
# 3️⃣ Disease-Aware LSTM Decoder
# =============================================================================
class DiseaseAwareLSTMDecoder(nn.Module):
    """疾病感知的报告生成解码器"""
    def __init__(self, vocab_size, embed_dim=512, hidden_size=512, disease_dim=14):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = DiseaseAwareAttention(512, hidden_size, disease_dim)
        self.disease_encoder = nn.Sequential(
            nn.Linear(disease_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256)
        )
        self.lstm = nn.LSTM(embed_dim + 512 + 256, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions, disease_features=None):
        B, C, H, W = features.size()
        features = features.view(B, C, -1).permute(0, 2, 1)
        embeddings = self.embedding(captions)
        
        if disease_features is not None:
            disease_context = self.disease_encoder(disease_features)
        else:
            disease_context = torch.zeros(B, 256).to(features.device)

        h = torch.zeros(1, B, self.hidden_size).to(features.device)
        c = torch.zeros(1, B, self.hidden_size).to(features.device)
        outputs = []

        for t in range(embeddings.size(1)):
            context, _ = self.attention(features, h[-1], disease_features)
            lstm_input = torch.cat((embeddings[:, t, :], context, disease_context), dim=1).unsqueeze(1)
            out, (h, c) = self.lstm(lstm_input, (h, c))
            outputs.append(self.fc(out.squeeze(1)))

        return torch.stack(outputs, dim=1)

    def generate(self, features, disease_features, sos_id, eos_id, max_len=50,
                 temperature=1.0, top_k=None, top_p=None):
        """
        生成序列（支持temperature、top_k和top_p采样）
        """
        B, C, H, W = features.size()
        features = features.view(B, C, -1).permute(0, 2, 1)
        
        if disease_features is not None:
            disease_context = self.disease_encoder(disease_features)
        else:
            disease_context = torch.zeros(B, 256).to(features.device)

        h = torch.zeros(1, B, self.hidden_size).to(features.device)
        c = torch.zeros(1, B, self.hidden_size).to(features.device)
        inputs = torch.tensor([sos_id] * B).to(features.device).unsqueeze(1)
        outputs = []

        for _ in range(max_len):
            embedding = self.embedding(inputs[:, -1])
            context, _ = self.attention(features, h[-1], disease_features)
            lstm_input = torch.cat((embedding, context, disease_context), dim=1).unsqueeze(1)
            out, (h, c) = self.lstm(lstm_input, (h, c))
            logits = self.fc(out.squeeze(1))
            
            # 应用temperature
            logits = logits / temperature
            
            # 应用top_k采样
            if top_k is not None and top_k > 0:
                top_k_actual = min(top_k, logits.size(-1))
                top_k_values, top_k_indices = torch.topk(logits, k=top_k_actual, dim=-1)
                logits_filtered = torch.full_like(logits, float('-inf'))
                logits_filtered.scatter_(1, top_k_indices, top_k_values)
                logits = logits_filtered
            
            # 应用top_p采样
            if top_p is not None and top_p > 0.0 and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    -1, sorted_indices, sorted_indices_to_remove
                )
                logits = logits.masked_fill(indices_to_remove, float('-inf'))
            
            # 采样
            probs = F.softmax(logits, dim=-1)
            predicted = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            outputs.append(predicted)
            inputs = torch.cat([inputs, predicted.unsqueeze(1)], dim=1)
            
            if (predicted == eos_id).all():
                break

        return torch.stack(outputs, dim=1)


# =============================================================================
# 4️⃣ IU 报告生成模型（推理版本）
# =============================================================================
class IUReportGenerator(nn.Module):
    """疾病感知的报告生成器（推理版本）"""
    def __init__(self, vocab_size, use_disease_features=True):
        super().__init__()
        self.use_disease_features = use_disease_features
        self.vocab_size = vocab_size
        self.encoder = CNNEncoder(pretrained=False)
        self.decoder = DiseaseAwareLSTMDecoder(vocab_size)
        self.disease_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 14)
        )

    def forward(self, images, captions, disease_labels=None):
        feature_map, global_features = self.encoder(images)
        if disease_labels is None and self.use_disease_features:
            disease_logits = self.disease_classifier(global_features)
            disease_features = torch.sigmoid(disease_logits)
        else:
            disease_features = disease_labels
        return self.decoder(feature_map, captions, disease_features)

    def generate_report(self, images, sos_id, eos_id, max_len=50,
                       temperature=1.0, top_k=None, top_p=None, use_sampling=True):
        """
        生成报告（支持多样性参数）
        """
        self.eval()
        with torch.no_grad():
            feature_map, global_features = self.encoder(images)
            
            if self.use_disease_features:
                disease_logits = self.disease_classifier(global_features)
                disease_features = torch.sigmoid(disease_logits)
            else:
                disease_features = None
            
            if use_sampling:
                # 使用采样模式
                generated_ids = self.decoder.generate(
                    feature_map, 
                    disease_features, 
                    sos_id, 
                    eos_id, 
                    max_len,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
            else:
                # 贪婪解码
                generated_ids = self.decoder.generate(
                    feature_map, 
                    disease_features, 
                    sos_id, 
                    eos_id, 
                    max_len,
                    temperature=1.0,
                    top_k=None,
                    top_p=None
                )
            
            return generated_ids


# =============================================================================
# 疾病名称（用于调试）
# =============================================================================
DISEASE_NAMES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax"
]
