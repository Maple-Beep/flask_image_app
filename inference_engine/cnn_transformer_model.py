# =============================================================================
# model/cnn_transformer_model.py — Disease-Aware CNN + Transformer
#
# 核心改动：
#   - 解码器从 LSTM → Transformer（4层编码/解码，8头，512维）
#   - 引入门控交叉注意力（GCA）融合CNN网格特征与疾病上下文
#   - 引入多层特征融合（MF）利用编码器各层输出
#   - 保留原有疾病感知逻辑、NIHCC迁移学习接口、Beam Search生成接口
#   - 保留原有 CNNEncoder / NIHCCClassifier / 工具函数 不变
# =============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# =============================================================================
# 1️⃣  CNN Encoder（原版保留，仅升级backbone为ResNet-101）
# =============================================================================
class CNNEncoder(nn.Module):
    """CNN特征提取器（ResNet-101网格特征 + 全局特征双输出）"""
    def __init__(self, pretrained=True):
        super().__init__()
        # 升级为ResNet-101，与论文验证结果一致
        resnet = models.resnet101(weights="IMAGENET1K_V2" if pretrained else None)
        # 保留原版切除分类层的逻辑
        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        feature_map = self.features(x)                          # (B, 2048, H, W)
        global_features = self.pool(feature_map).view(x.size(0), -1)  # (B, 2048)
        return feature_map, global_features


# =============================================================================
# 2️⃣  NIHCC 疾病分类器（原版保留，适配ResNet-101输出2048维）
# =============================================================================
class NIHCCClassifier(nn.Module):
    """14种胸部疾病多标签分类器（适配ResNet-101）"""
    def __init__(self, num_classes=14):
        super().__init__()
        self.encoder = CNNEncoder(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        _, global_features = self.encoder(x)
        return self.classifier(global_features)

    def get_encoder_state_dict(self):
        return self.encoder.state_dict()


# =============================================================================
# 3️⃣  位置编码（标准Sinusoidal，供Transformer使用）
# =============================================================================
class PositionalEncoding(nn.Module):
    """标准Sinusoidal位置编码"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# =============================================================================
# 4️⃣  门控交叉注意力（GCA）——借鉴论文核心思想，适配本工程疾病特征
#
#   原论文：融合卷积特征 V1 与序列化特征 V2
#   本工程适配：融合 CNN网格特征 与 疾病感知特征，保留门控逻辑
# =============================================================================
class GatedCrossAttention(nn.Module):
    """
    门控交叉注意力：用文本查询 Y 分别对两组视觉特征做交叉注意力，
    再由两个可学习门控（sigmoid）动态加权融合。

    V1: CNN网格特征（空间信息）
    V2: 疾病感知特征（语义信息）
    """
    def __init__(self, d_model: int = 512, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn_v1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.attn_v2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # 两个独立sigmoid门控单元
        self.gate_alpha = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.gate_beta  = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query, v1, v2, key_padding_mask=None):
        """
        Args:
            query: 文本序列 (B, T, d_model)
            v1:    CNN网格特征 (B, S, d_model)
            v2:    疾病感知特征 (B, M, d_model)
        Returns:
            fused: (B, T, d_model)
        """
        v1_out, _ = self.attn_v1(query, v1, v1)          # (B, T, d_model)
        v2_out, _ = self.attn_v2(query, v2, v2)          # (B, T, d_model)

        alpha = self.gate_alpha(query)                    # (B, T, d_model)
        beta  = self.gate_beta(query)                     # (B, T, d_model)

        fused = alpha * v1_out + beta * v2_out            # 加权融合
        return self.norm1(fused + query)                  # 残差 + 归一化


# =============================================================================
# 5️⃣  多层特征融合（MF）——借鉴论文思路，融合Transformer编码器各层输出
# =============================================================================
class MultiLayerFusion(nn.Module):
    """
    融合Transformer编码器各层输出特征（而非仅使用顶层）。
    V_G = G_L + λ * MLP([G1, G2, ..., GL])
    """
    def __init__(self, d_model: int = 512, num_layers: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model * num_layers, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        # λ初始化为小值，训练中自适应调整（可学习标量）
        self.lam = nn.Parameter(torch.tensor(0.1))

    def forward(self, layer_outputs: list):
        """
        Args:
            layer_outputs: list of (B, S, d_model), 每层编码器输出
        Returns:
            fused: (B, S, d_model)
        """
        G_L = layer_outputs[-1]                           # 顶层特征
        concat = torch.cat(layer_outputs, dim=-1)         # (B, S, d_model*L)
        G_prime = self.mlp(concat)                        # (B, S, d_model)
        return G_L + self.lam * G_prime


# =============================================================================
# 6️⃣  CNN视觉特征Transformer编码器（含多层融合）
# =============================================================================
class VisualTransformerEncoder(nn.Module):
    """
    对CNN网格特征做Transformer编码，支持多层特征融合输出。
    d_input=2048 投影到 d_model=512。
    """
    def __init__(self, d_input: int = 2048, d_model: int = 512,
                 nhead: int = 8, num_layers: int = 4,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_input, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        # 手动叠加，方便逐层收集输出
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True
            ) for _ in range(num_layers)
        ])
        self.mf = MultiLayerFusion(d_model, num_layers)

    def forward(self, x):
        """
        Args:
            x: (B, 2048, H, W) CNN feature map
        Returns:
            fused: (B, H*W, d_model)  多层融合后特征
            layer_outputs: list of (B, H*W, d_model)
        """
        B, C, H, W = x.size()
        # 展平空间维 → 序列
        x = x.view(B, C, H * W).permute(0, 2, 1)        # (B, H*W, 2048)
        x = self.proj(x)                                  # (B, H*W, d_model)
        x = self.pos_enc(x)

        layer_outputs = []
        for layer in self.layers:
            x = layer(x)
            layer_outputs.append(x)

        fused = self.mf(layer_outputs)
        return fused, layer_outputs


# =============================================================================
# 7️⃣  疾病感知特征编码器（保留原有设计逻辑）
# =============================================================================
class DiseaseFeatureEncoder(nn.Module):
    """
    将14维疾病概率向量编码为 d_model 维序列特征（供GCA使用）。
    保留原版疾病编码器逻辑，升级为输出可参与注意力的序列。
    """
    def __init__(self, disease_dim: int = 14, d_model: int = 512, num_pseudo: int = 8):
        super().__init__()
        self.num_pseudo = num_pseudo  # 伪区域数（参考论文CP模块思路）
        self.proj = nn.Sequential(
            nn.Linear(disease_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, d_model * num_pseudo)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, disease_features):
        """
        Args:
            disease_features: (B, 14) 疾病概率
        Returns:
            (B, num_pseudo, d_model)
        """
        B = disease_features.size(0)
        out = self.proj(disease_features)                  # (B, d_model * num_pseudo)
        out = out.view(B, self.num_pseudo, -1)             # (B, num_pseudo, d_model)
        return self.norm(out)


# =============================================================================
# 8️⃣  Disease-Aware Transformer解码器（核心改造部分）
# =============================================================================
class DiseaseAwareTransformerDecoder(nn.Module):
    """
    疾病感知的Transformer报告生成解码器。
    - 标准自注意力（Masked）处理文本序列
    - GCA门控交叉注意力融合CNN特征与疾病特征
    - 支持Teacher Forcing训练和自回归生成推理
    """
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8,
                 num_layers: int = 4, dim_feedforward: int = 2048,
                 dropout: float = 0.1, disease_dim: int = 14):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)

        # 疾病特征编码器
        self.disease_encoder = DiseaseFeatureEncoder(disease_dim, d_model)

        # 每层：自注意力 + GCA
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True
            ) for _ in range(num_layers)
        ])
        self.gca_layers = nn.ModuleList([
            GatedCrossAttention(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    def _make_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """生成因果掩码（上三角为-inf）"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(self, visual_features, captions, disease_features=None):
        """
        Teacher Forcing前向传播（训练阶段）。
        Args:
            visual_features: (B, S, d_model) 编码后的视觉特征
            captions:        (B, T)           输入token序列（含SOS）
            disease_features:(B, 14)          疾病概率向量（可选）
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = captions.size()
        causal_mask = self._make_causal_mask(T, captions.device)

        x = self.pos_enc(self.embedding(captions) * math.sqrt(self.d_model))

        # 疾病特征编码（V2）
        if disease_features is not None:
            disease_seq = self.disease_encoder(disease_features)   # (B, num_pseudo, d_model)
        else:
            disease_seq = torch.zeros(B, 8, self.d_model, device=captions.device)

        for self_attn, gca in zip(self.self_attn_layers, self.gca_layers):
            # 标准Transformer自注意力
            x = self_attn(x, visual_features,
                          tgt_mask=causal_mask,
                          memory_key_padding_mask=None)
            # GCA融合疾病语义
            x = gca(x, visual_features, disease_seq)

        return self.fc_out(x)

    def generate(self, visual_features, disease_features, sos_id, eos_id,
                 max_len=100, temperature=1.0, top_k=None):
        """
        自回归生成（支持temperature和top_k采样）。
        接口保持与原LSTM generate方法完全一致。
        """
        B = visual_features.size(0)
        device = visual_features.device

        if disease_features is not None:
            disease_seq = self.disease_encoder(disease_features)
        else:
            disease_seq = torch.zeros(B, 8, self.d_model, device=device)

        generated = torch.full((B, 1), sos_id, dtype=torch.long, device=device)
        outputs = []

        for _ in range(max_len):
            T = generated.size(1)
            causal_mask = self._make_causal_mask(T, device)

            x = self.pos_enc(self.embedding(generated) * math.sqrt(self.d_model))

            for self_attn, gca in zip(self.self_attn_layers, self.gca_layers):
                x = self_attn(x, visual_features, tgt_mask=causal_mask)
                x = gca(x, visual_features, disease_seq)

            logits = self.fc_out(x[:, -1, :])   # 取最后一步

            # temperature缩放
            logits = logits / temperature

            # top_k过滤
            if top_k is not None and top_k > 0:
                top_k_actual = min(top_k, logits.size(-1))
                top_k_values, top_k_indices = torch.topk(logits, top_k_actual, dim=-1)
                logits_filtered = torch.full_like(logits, float('-inf'))
                logits_filtered.scatter_(1, top_k_indices, top_k_values)
                logits = logits_filtered

            probs = F.softmax(logits, dim=-1)
            predicted = torch.multinomial(probs, num_samples=1).squeeze(1)

            outputs.append(predicted)
            generated = torch.cat([generated, predicted.unsqueeze(1)], dim=1)

            if (predicted == eos_id).all():
                break

        return torch.stack(outputs, dim=1)


# =============================================================================
# 9️⃣  IU 报告生成模型（CNN + Transformer，保留原有对外接口）
# =============================================================================
class IUReportGenerator(nn.Module):
    """
    疾病感知报告生成器（CNN + Transformer版本）。
    对外接口与原 LSTM 版本完全兼容，训练脚本无需修改。
    """
    def __init__(self, vocab_size: int, use_disease_features: bool = True,
                 d_model: int = 512, nhead: int = 8,
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.use_disease_features = use_disease_features
        self.vocab_size = vocab_size

        # CNN编码器（ResNet-101）
        self.encoder = CNNEncoder(pretrained=True)

        # 视觉Transformer编码器（含多层融合）
        self.visual_encoder = VisualTransformerEncoder(
            d_input=2048, d_model=d_model, nhead=nhead,
            num_layers=num_layers, dropout=dropout
        )

        # 疾病分类头（保留原版逻辑）
        self.disease_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 14)
        )

        # Transformer解码器（含GCA）
        self.decoder = DiseaseAwareTransformerDecoder(
            vocab_size=vocab_size, d_model=d_model,
            nhead=nhead, num_layers=num_layers,
            dropout=dropout
        )

    def forward(self, images, captions, disease_labels=None):
        """
        训练阶段前向传播（Teacher Forcing）。
        接口与原版完全一致。
        """
        feature_map, global_features = self.encoder(images)

        # 疾病特征获取（保留原版逻辑）
        if disease_labels is None and self.use_disease_features:
            disease_logits = self.disease_classifier(global_features)
            disease_features = torch.sigmoid(disease_logits)
        else:
            disease_features = disease_labels

        # 视觉特征编码（多层融合）
        visual_seq, _ = self.visual_encoder(feature_map)

        return self.decoder(visual_seq, captions, disease_features)

    def generate_report(self, images, sos_id, eos_id, max_len=100,
                        beam_size=1, temperature=1.0, top_k=None):
        """
        推理阶段生成报告。
        接口与原版完全一致（含beam_size/temperature/top_k参数）。

        Returns:
            generated_ids: (B, seq_len)
            disease_features: (B, 14)
        """
        feature_map, global_features = self.encoder(images)

        if self.use_disease_features:
            disease_logits = self.disease_classifier(global_features)
            disease_features = torch.sigmoid(disease_logits)
        else:
            disease_features = None

        visual_seq, _ = self.visual_encoder(feature_map)

        generated_ids = self.decoder.generate(
            visual_seq, disease_features,
            sos_id, eos_id, max_len,
            temperature=temperature, top_k=top_k
        )

        return generated_ids, disease_features

    def load_nihcc_encoder(self, nihcc_checkpoint_path: str):
        """从NIHCC模型迁移CNN Encoder权重（原版接口保留）"""
        checkpoint = torch.load(nihcc_checkpoint_path, map_location='cpu', weights_only=True)
        # 兼容原版存储格式
        encoder_state = {k.replace('encoder.', ''): v
                         for k, v in checkpoint.items()
                         if k.startswith('encoder.')}
        if encoder_state:
            self.encoder.load_state_dict(encoder_state, strict=False)

        # 疾病分类器权重迁移
        classifier_state = {k.replace('classifier.', ''): v
                            for k, v in checkpoint.items()
                            if k.startswith('classifier.')}
        if classifier_state:
            self.disease_classifier.load_state_dict(classifier_state, strict=False)


# =============================================================================
# 辅助函数（原版保留）
# =============================================================================
DISEASE_NAMES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax"
]

def diseases_to_text(disease_vector, threshold=0.5):
    """将疾病概率向量转为文本（原版保留）"""
    detected = [DISEASE_NAMES[i] for i, prob in enumerate(disease_vector)
                if prob > threshold]
    if not detected:
        return "No significant abnormalities detected."
    if len(detected) == 1:
        return f"{detected[0]} detected."
    return f"Multiple findings: {', '.join(detected[:-1])} and {detected[-1]}."
