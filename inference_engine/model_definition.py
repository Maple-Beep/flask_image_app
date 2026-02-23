# inference_engine/model_definition.py — 推理侧模型（与训练工程结构完全一致）
# model/cnn_transformer_model.py — Disease-Aware CNN + Transformer (v3)
#
# v3 修复清单：
# 1. 移除 VisualTransformerEncoder 中的死代码（未使用的 encoder_layer 对象）
# 2. generate_report 包裹 torch.no_grad()，推理不积累梯度
# 3. load_nihcc_encoder 完全重写：兼容 train_nihcc 的 model.state_dict() 保存格式
# 4. generate_report 统一返回 (ids, disease_features) tuple，全链路一致
# 5. IUReportGenerator.__init__ 中 pretrained 参数正确按阶段传递
# =============================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# =============================================================================
# 1️⃣  CNN Encoder
# =============================================================================
class CNNEncoder(nn.Module):
    """ResNet-101 特征提取器（双输出：网格特征图 + 全局池化特征）"""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet101(weights=None)  # 推理时权重由checkpoint提供
        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor):
        feature_map = self.features(x)                              # (B, 2048, H, W)
        global_features = self.pool(feature_map).view(x.size(0), -1)  # (B, 2048)
        return feature_map, global_features


# =============================================================================
# 2️⃣  NIHCC 分类器（第一阶段预训练）
# =============================================================================
class NIHCCClassifier(nn.Module):
    """14 种胸部疾病多标签分类器"""
    def __init__(self, num_classes: int = 14):
        super().__init__()
        self.encoder = CNNEncoder(pretrained=False)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, global_features = self.encoder(x)
        return self.classifier(global_features)

    def get_encoder_state_dict(self) -> dict:
        return self.encoder.state_dict()


# =============================================================================
# 3️⃣  位置编码
# =============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# =============================================================================
# 4️⃣  门控交叉注意力（GCA）
# =============================================================================
class GatedCrossAttention(nn.Module):
    """
    双路门控交叉注意力。
    V1: CNN 网格特征（空间信息）
    V2: 疾病感知特征（语义信息）
    fused = sigmoid(W_a·query)·CrossAttn(query,V1) + sigmoid(W_b·query)·CrossAttn(query,V2)
    """
    def __init__(self, d_model: int = 512, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn_v1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.attn_v2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.gate_alpha = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.gate_beta  = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query: torch.Tensor, v1: torch.Tensor,
                v2: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        v1_out, _ = self.attn_v1(query, v1, v1)
        v2_out, _ = self.attn_v2(query, v2, v2)
        alpha = self.gate_alpha(query)
        beta  = self.gate_beta(query)
        fused = alpha * v1_out + beta * v2_out
        return self.norm(fused + query)               # 残差 + LayerNorm


# =============================================================================
# 5️⃣  多层特征融合（MF）
# =============================================================================
class MultiLayerFusion(nn.Module):
    """V_G = G_L + λ · MLP([G1‥GL])，λ 可学习"""
    def __init__(self, d_model: int = 512, num_layers: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model * num_layers, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.lam = nn.Parameter(torch.tensor(0.1))

    def forward(self, layer_outputs: list) -> torch.Tensor:
        G_L = layer_outputs[-1]
        concat = torch.cat(layer_outputs, dim=-1)     # (B, S, d_model*L)
        G_prime = self.mlp(concat)
        return G_L + self.lam * G_prime


# =============================================================================
# 6️⃣  视觉 Transformer 编码器（含多层融合）
# =============================================================================
class VisualTransformerEncoder(nn.Module):
    """CNN 网格特征 → Transformer 编码 → 多层融合输出"""
    def __init__(self, d_input: int = 2048, d_model: int = 512,
                 nhead: int = 8, num_layers: int = 4,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_input, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        # FIX: 移除死代码（之前多创建了一个未使用的 encoder_layer 对象）
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True,
                norm_first=True               # Pre-LN：训练更稳定
            ) for _ in range(num_layers)
        ])
        self.mf = MultiLayerFusion(d_model, num_layers)

    def forward(self, x: torch.Tensor):
        """
        Args:  x: (B, 2048, H, W)
        Returns: fused (B, H*W, d_model), layer_outputs list
        """
        B, C, H, W = x.size()
        x = x.view(B, C, H * W).permute(0, 2, 1)    # (B, H*W, 2048)
        x = self.proj(x)                              # (B, H*W, d_model)
        x = self.pos_enc(x)
        layer_outputs = []
        for layer in self.layers:
            x = layer(x)
            layer_outputs.append(x)
        fused = self.mf(layer_outputs)
        return fused, layer_outputs


# =============================================================================
# 7️⃣  疾病特征编码器
# =============================================================================
class DiseaseFeatureEncoder(nn.Module):
    """14 维疾病概率 → num_pseudo 个伪区域特征序列（供 GCA 使用）"""
    def __init__(self, disease_dim: int = 14, d_model: int = 512, num_pseudo: int = 8):
        super().__init__()
        self.num_pseudo = num_pseudo
        self.proj = nn.Sequential(
            nn.Linear(disease_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, d_model * num_pseudo)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, disease_features: torch.Tensor) -> torch.Tensor:
        B = disease_features.size(0)
        out = self.proj(disease_features).view(B, self.num_pseudo, -1)
        return self.norm(out)


# =============================================================================
# 8️⃣  Disease-Aware Transformer 解码器
# =============================================================================
class DiseaseAwareTransformerDecoder(nn.Module):
    """每层：Masked Self-Attn + GCA 门控交叉注意力"""
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8,
                 num_layers: int = 4, dim_feedforward: int = 2048,
                 dropout: float = 0.1, disease_dim: int = 14):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        self.disease_encoder = DiseaseFeatureEncoder(disease_dim, d_model)
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout, batch_first=True,
                norm_first=True               # Pre-LN
            ) for _ in range(num_layers)
        ])
        self.gca_layers = nn.ModuleList([
            GatedCrossAttention(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)

    @staticmethod
    def _make_causal_mask(sz: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def _encode_disease(self, disease_features, B: int, device: torch.device):
        if disease_features is not None:
            return self.disease_encoder(disease_features)
        return torch.zeros(B, 8, self.d_model, device=device)

    def forward(self, visual_features: torch.Tensor, captions: torch.Tensor,
                disease_features=None) -> torch.Tensor:
        B, T = captions.size()
        causal_mask = self._make_causal_mask(T, captions.device)
        x = self.pos_enc(self.embedding(captions) * math.sqrt(self.d_model))
        disease_seq = self._encode_disease(disease_features, B, captions.device)
        for self_attn, gca in zip(self.self_attn_layers, self.gca_layers):
            x = self_attn(x, visual_features, tgt_mask=causal_mask)
            x = gca(x, visual_features, disease_seq)
        return self.fc_out(x)

    def generate(self, visual_features: torch.Tensor, disease_features,
                 sos_id: int, eos_id: int, max_len: int = 100,
                 temperature: float = 1.0, top_k: int = None,
                 top_p: float = None) -> torch.Tensor:
        """
        自回归生成，支持 temperature / top_k / top_p。
        修复：用 finished 掩码追踪已结束样本，确保 EOS 后不再生成，
        输出统一 padding 到同一长度，避免不同样本长度导致的 stack 失败。
        """
        B = visual_features.size(0)
        device = visual_features.device
        disease_seq = self._encode_disease(disease_features, B, device)
        generated = torch.full((B, 1), sos_id, dtype=torch.long, device=device)
        # 追踪哪些样本已生成 EOS
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        output_ids = torch.full((B, max_len), eos_id, dtype=torch.long, device=device)

        for step in range(max_len):
            T = generated.size(1)
            causal_mask = self._make_causal_mask(T, device)
            x = self.pos_enc(self.embedding(generated) * math.sqrt(self.d_model))
            for self_attn, gca in zip(self.self_attn_layers, self.gca_layers):
                x = self_attn(x, visual_features, tgt_mask=causal_mask)
                x = gca(x, visual_features, disease_seq)
            logits = self.fc_out(x[:, -1, :])               # (B, vocab)

            logits = logits / max(temperature, 1e-8)

            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                vals, idx = torch.topk(logits, k, dim=-1)
                logits_f = torch.full_like(logits, float('-inf'))
                logits = logits_f.scatter_(1, idx, vals)

            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                logits = logits.masked_fill(
                    remove.scatter(-1, sorted_idx, remove), float('-inf')
                )

            probs = F.softmax(logits, dim=-1)
            predicted = torch.multinomial(probs, num_samples=1).squeeze(1)
            # 已结束的样本强制输出 EOS
            predicted = predicted.masked_fill(finished, eos_id)
            output_ids[:, step] = predicted
            generated = torch.cat([generated, predicted.unsqueeze(1)], dim=1)
            finished = finished | (predicted == eos_id)
            if finished.all():
                break

        return output_ids


# =============================================================================
# 9️⃣  IU 报告生成模型（对外完整接口）
# =============================================================================
class IUReportGenerator(nn.Module):
    """
    Disease-Aware 报告生成器：ResNet-101 + Transformer Decoder with GCA。
    v3 修复：
      - generate_report 包裹 no_grad
      - load_nihcc_encoder 正确解析 train_nihcc 保存的 state_dict 格式
      - generate_report 统一返回 (generated_ids, disease_features) tuple
    """
    def __init__(self, vocab_size: int, use_disease_features: bool = True,
                 d_model: int = 512, nhead: int = 8,
                 num_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.use_disease_features = use_disease_features
        self.vocab_size = vocab_size

        self.encoder = CNNEncoder(pretrained=False)
        self.visual_encoder = VisualTransformerEncoder(
            d_input=2048, d_model=d_model, nhead=nhead,
            num_layers=num_layers, dropout=dropout
        )
        self.disease_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 14)
        )
        self.decoder = DiseaseAwareTransformerDecoder(
            vocab_size=vocab_size, d_model=d_model,
            nhead=nhead, num_layers=num_layers, dropout=dropout
        )

    def forward(self, images: torch.Tensor, captions: torch.Tensor,
                disease_labels=None) -> torch.Tensor:
        """Teacher Forcing 前向传播（训练阶段）"""
        feature_map, global_features = self.encoder(images)
        if disease_labels is None and self.use_disease_features:
            disease_features = torch.sigmoid(self.disease_classifier(global_features))
        else:
            disease_features = disease_labels
        visual_seq, _ = self.visual_encoder(feature_map)
        return self.decoder(visual_seq, captions, disease_features)

    @torch.no_grad()
    def generate_report(self, images: torch.Tensor,
                        sos_id: int, eos_id: int, max_len: int = 100,
                        beam_size: int = 1, temperature: float = 0.8,
                        top_k: int = 30, top_p: float = 0.9):
        """
        推理接口（v3：包裹 no_grad，统一返回 tuple）。
        Returns:
            generated_ids: (B, seq_len)  token ID 序列
            disease_probs: (B, 14)       疾病概率（可选输出）
        """
        self.eval()
        feature_map, global_features = self.encoder(images)
        if self.use_disease_features:
            disease_probs = torch.sigmoid(self.disease_classifier(global_features))
        else:
            disease_probs = None
        visual_seq, _ = self.visual_encoder(feature_map)
        generated_ids = self.decoder.generate(
            visual_seq, disease_probs,
            sos_id, eos_id, max_len,
            temperature=temperature, top_k=top_k, top_p=top_p
        )
        return generated_ids, disease_probs

    def load_nihcc_encoder(self, nihcc_checkpoint_path: str) -> None:
        """
        从 NIHCC 检查点迁移 CNN Encoder 权重。
        v3 修复：兼容 train_nihcc 用 model.state_dict() 直接保存的格式。
        train_nihcc 保存的 key 格式：
          encoder.features.0.weight  → CNNEncoder.features.0.weight
          classifier.0.weight        → disease_classifier 中对应权重
        """
        if not torch.cuda.is_available():
            map_loc = 'cpu'
        else:
            map_loc = None  # 默认 GPU

        raw = torch.load(nihcc_checkpoint_path, map_location=map_loc, weights_only=True)

        # train_nihcc 保存两种格式：
        #   格式A: torch.save(model.state_dict(), path)        → 直接是 state_dict
        #   格式B: torch.save({'model_state_dict': ...}, path) → 带 key 的 dict
        if isinstance(raw, dict) and 'model_state_dict' in raw:
            state_dict = raw['model_state_dict']
        else:
            state_dict = raw   # 格式A

        # 提取 encoder 部分（key 前缀是 "encoder."）
        encoder_sd = {
            k[len('encoder.'):]: v
            for k, v in state_dict.items()
            if k.startswith('encoder.')
        }
        if encoder_sd:
            missing, unexpected = self.encoder.load_state_dict(encoder_sd, strict=False)
            print(f"  CNN Encoder 迁移成功: {len(encoder_sd)} 个参数")
            if missing:
                print(f"  缺失 key（{len(missing)}个）: {missing[:3]}")
        else:
            print("  ⚠ 未找到 encoder.* 前缀的权重，跳过 CNN Encoder 迁移")

        # 提取 disease_classifier 部分（key 前缀是 "classifier."）
        cls_sd = {
            k[len('classifier.'):]: v
            for k, v in state_dict.items()
            if k.startswith('classifier.')
        }
        if cls_sd:
            self.disease_classifier.load_state_dict(cls_sd, strict=False)
            print(f"  disease_classifier 迁移成功: {len(cls_sd)} 个参数")


# =============================================================================
# 辅助：疾病名称 & 概率 → 文本
# =============================================================================
DISEASE_NAMES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax"
]


def diseases_to_text(disease_probs: torch.Tensor, threshold: float = 0.5) -> str:
    """疾病概率向量 → 描述文本"""
    detected = [DISEASE_NAMES[i] for i, p in enumerate(disease_probs) if p > threshold]
    if not detected:
        return "No significant abnormalities detected."
    if len(detected) == 1:
        return f"{detected[0]} detected."
    return f"Multiple findings: {', '.join(detected[:-1])} and {detected[-1]}."
