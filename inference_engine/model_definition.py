# model_definition.py
import torch
import torch.nn as nn
import torchvision.models as models
from config import Config  # ✅ 新增：导入配置


class CNNEncoder(nn.Module):
    """
    CNN 编码器（基于 ResNet18）
    输出: (batch, CNN_OUT_FEATURES, 7, 7) —— 保留空间维度！
    """

    def __init__(self, out_features: int = Config.CNN_OUT_FEATURES, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)

        # 移除 avgpool 和 fc 层
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            # 注意：不再包含 resnet.avgpool！
        )

        # 添加 1x1 卷积投影到目标维度
        self.proj = nn.Conv2d(512, out_features, kernel_size=1)

        # 初始化投影层
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, 224, 224)
        Returns:
            (batch, out_features, 7, 7)
        """
        x = self.features(x)  # (batch, 512, 7, 7)
        x = self.proj(x)  # (batch, out_features, 7, 7)
        return x


class LSTMDecoder(nn.Module):
    """
    LSTM 解码器
    接收 2D 特征图，内部做全局平均池化用于初始化
    """

    def __init__(
            self,
            vocab_size: int = Config.VOCAB_SIZE,
            embed_dim: int = 512,
            hidden_size: int = Config.LSTM_HIDDEN_SIZE,
            num_layers: int = Config.LSTM_NUM_LAYERS,
            dropout: float = Config.LSTM_DROPOUT,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.output_proj = nn.Linear(hidden_size, vocab_size)

        # 用于从 CNN 特征初始化 LSTM 隐藏状态
        self.init_h = nn.Linear(Config.CNN_OUT_FEATURES, hidden_size)
        self.init_c = nn.Linear(Config.CNN_OUT_FEATURES, hidden_size)

        # 初始化
        nn.init.xavier_uniform_(self.init_h.weight)
        nn.init.zeros_(self.init_h.bias)
        nn.init.xavier_uniform_(self.init_c.weight)
        nn.init.zeros_(self.init_c.bias)

    def forward(self, encoder_features: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_features: (batch, C, H, W) ← 2D 特征图
            target_ids: (batch, seq_len)
        Returns:
            (batch, seq_len, vocab_size)
        """
        batch_size = encoder_features.shape[0]

        # 全局平均池化用于初始化 LSTM（保持与旧模型行为一致）
        pooled = encoder_features.mean(dim=[2, 3])  # (batch, C)

        h0 = self.init_h(pooled).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = self.init_c(pooled).unsqueeze(0).repeat(self.num_layers, 1, 1)

        embedded = self.embedding(target_ids)  # (batch, seq_len, embed_dim)
        lstm_out, _ = self.lstm(embedded, (h0, c0))
        outputs = self.output_proj(lstm_out)  # (batch, seq_len, vocab_size)
        return outputs

    @torch.no_grad()
    def generate(self, encoder_features: torch.Tensor, sos_id: int, eos_id: int, max_len: int = Config.MAX_REPORT_LEN):
        """
        自回归生成报告（用于推理）
        """
        batch_size = encoder_features.shape[0]
        device = encoder_features.device

        # 初始化
        pooled = encoder_features.mean(dim=[2, 3])
        h = self.init_h(pooled).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c = self.init_c(pooled).unsqueeze(0).repeat(self.num_layers, 1, 1)

        input_ids = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)
        generated = [sos_id]

        for _ in range(max_len - 1):
            embedded = self.embedding(input_ids[:, -1:])
            lstm_out, (h, c) = self.lstm(embedded, (h, c))
            logits = self.output_proj(lstm_out.squeeze(1))
            next_token = torch.argmax(logits, dim=-1)
            generated.append(next_token.item())

            if next_token.item() == eos_id:
                break

            input_ids = torch.cat([input_ids, next_token.unsqueeze(1)], dim=1)

        return generated


class MedicalReportModel(nn.Module):
    """
    完整的医疗报告生成模型
    """

    def __init__(self, vocab_size: int = Config.VOCAB_SIZE):
        super().__init__()
        self.encoder = CNNEncoder(out_features=Config.CNN_OUT_FEATURES, pretrained=True)
        self.decoder = LSTMDecoder(vocab_size=vocab_size)

    def forward(self, images: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)  # (batch, C, 7, 7)
        outputs = self.decoder(features, target_ids)
        return outputs

    @torch.no_grad()
    def generate_report(self, images: torch.Tensor, sos_id: int, eos_id: int, max_len: int = Config.MAX_REPORT_LEN):
        features = self.encoder(images)
        return self.decoder.generate(features, sos_id, eos_id, max_len)