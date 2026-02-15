# =============================================================================
# inference_engine/model_definition.py — CNN + LSTM 模型定义 (完全解耦版)
#
# 此版本不依赖任何外部配置文件，所有参数均通过构造函数传入。
# 这使得模型定义干净、独立，易于测试和复用。
# =============================================================================

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List


class CNNEncoder(nn.Module):
    """
    基于 ResNet-18 的 CNN 特征提取器。
    """

    def __init__(self, out_features: int = 256, pretrained: bool = False):
        super().__init__()
        resnet = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )
        self.projection = nn.Linear(512, out_features)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.projection(x)
        return x


class LSTMDecoder(nn.Module):
    """
    LSTM 序列解码器，用于生成报告文本。
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        cnn_out_features: int = 256,  # <-- 新增：接收CNN输出维度
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.embedding.weight.data[0] = 0.0

        # 使用传入的 cnn_out_features 初始化LSTM
        self.init_h = nn.Linear(cnn_out_features, hidden_size)
        self.init_c = nn.Linear(cnn_out_features, hidden_size)
        nn.init.xavier_uniform_(self.init_h.weight)
        nn.init.zeros_(self.init_h.bias)
        nn.init.xavier_uniform_(self.init_c.weight)
        nn.init.zeros_(self.init_c.bias)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.output_proj = nn.Linear(hidden_size, vocab_size)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    @torch.no_grad()
    def generate(self, encoder_features: torch.Tensor, sos_id: int, eos_id: int, max_len: int = 100) -> torch.Tensor:
        device = encoder_features.device
        h = self.init_h(encoder_features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c = self.init_c(encoder_features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        input_ids = torch.full((encoder_features.size(0), 1), sos_id, dtype=torch.long, device=device)

        generated_tokens: List[torch.Tensor] = []
        finished = torch.zeros(encoder_features.size(0), dtype=torch.bool, device=device)

        for _ in range(max_len):
            embedded = self.embedding(input_ids)
            lstm_out, (h, c) = self.lstm(embedded, (h, c))
            logits = self.output_proj(lstm_out)
            next_token = logits.argmax(dim=-1)
            generated_tokens.append(next_token)
            finished = finished | (next_token.squeeze(-1) == eos_id)
            input_ids = next_token
            if finished.all():
                break

        generated_ids = torch.cat(generated_tokens, dim=1)
        if generated_ids.size(1) < max_len:
            pad = torch.zeros(encoder_features.size(0), max_len - generated_ids.size(1), dtype=torch.long, device=device)
            generated_ids = torch.cat([generated_ids, pad], dim=1)

        return generated_ids[:, :max_len]


class IUReportGenerator(nn.Module):
    """
    IU Xray 报告生成模型，专为推理设计。
    """

    def __init__(
        self,
        vocab_size: int,
        cnn_out_features: int = 256,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        lstm_dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = CNNEncoder(out_features=cnn_out_features, pretrained=False)
        self.decoder = LSTMDecoder(
            vocab_size=vocab_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout,
            cnn_out_features=cnn_out_features,  # <-- 传递给Decoder
        )

    @torch.no_grad()
    def generate_report(self, images: torch.Tensor, sos_id: int, eos_id: int, max_len: int = 100) -> torch.Tensor:
        features = self.encoder(images)
        return self.decoder.generate(features, sos_id, eos_id, max_len)