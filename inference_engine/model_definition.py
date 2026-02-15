# =============================================================================
# inference_engine/model_definition.py — CNN + LSTM 模型定义 (增强版)
#
# 修复说明：
# 1. 添加温度采样（Temperature Sampling）
# 2. 添加Top-K和Top-P（Nucleus）采样
# 3. 增加生成多样性，避免总是输出相同的报告
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        cnn_out_features: int = 256,
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
    def generate(
        self, 
        encoder_features: torch.Tensor, 
        sos_id: int, 
        eos_id: int, 
        max_len: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        use_sampling: bool = True
    ) -> torch.Tensor:
        """
        生成报告序列
        
        Args:
            encoder_features: 编码器输出的特征 [batch_size, feature_dim]
            sos_id: 开始标记ID
            eos_id: 结束标记ID
            max_len: 最大生成长度
            temperature: 温度参数，越高越随机（推荐0.7-1.0）
            top_k: Top-K采样参数（0表示不使用）
            top_p: Top-P (Nucleus) 采样参数（0表示不使用）
            use_sampling: 是否使用采样（False则使用贪婪解码）
        """
        device = encoder_features.device
        batch_size = encoder_features.size(0)
        
        # 初始化隐藏状态
        h = self.init_h(encoder_features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c = self.init_c(encoder_features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        # 初始输入（SOS token）
        input_ids = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)

        generated_tokens: List[torch.Tensor] = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(max_len):
            # 嵌入当前输入
            embedded = self.embedding(input_ids)
            
            # LSTM前向传播
            lstm_out, (h, c) = self.lstm(embedded, (h, c))
            
            # 投影到词汇表大小
            logits = self.output_proj(lstm_out)  # [batch_size, 1, vocab_size]
            logits = logits.squeeze(1)  # [batch_size, vocab_size]
            
            # 应用温度
            if use_sampling and temperature != 1.0:
                logits = logits / temperature
            
            # 生成下一个token
            if use_sampling:
                # 使用采样策略
                next_token = self._sample_next_token(logits, top_k=top_k, top_p=top_p)
            else:
                # 贪婪解码
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            generated_tokens.append(next_token)
            
            # 检查是否生成了EOS token
            finished = finished | (next_token.squeeze(-1) == eos_id)
            
            # 更新输入为当前生成的token
            input_ids = next_token.unsqueeze(1) if next_token.dim() == 1 else next_token
            
            # 如果所有序列都结束了，提前停止
            if finished.all():
                break

        # 拼接生成的tokens
        generated_ids = torch.cat(generated_tokens, dim=1)
        
        # 填充到max_len（如果需要）
        if generated_ids.size(1) < max_len:
            pad = torch.zeros(
                batch_size, 
                max_len - generated_ids.size(1), 
                dtype=torch.long, 
                device=device
            )
            generated_ids = torch.cat([generated_ids, pad], dim=1)

        return generated_ids[:, :max_len]
    
    def _sample_next_token(
        self, 
        logits: torch.Tensor, 
        top_k: int = 0, 
        top_p: float = 0.0
    ) -> torch.Tensor:
        """
        从logits中采样下一个token
        
        Args:
            logits: 未归一化的对数概率 [batch_size, vocab_size]
            top_k: Top-K采样，保留概率最高的k个token
            top_p: Top-P (Nucleus) 采样，保留累积概率达到p的最小token集合
        """
        # Top-K采样
        if top_k > 0:
            # 获取top-k的值和索引
            top_k_vals, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            # 将其他位置设为负无穷
            logits_filtered = torch.full_like(logits, float('-inf'))
            logits_filtered.scatter_(-1, top_k_indices, top_k_vals)
            logits = logits_filtered
        
        # Top-P (Nucleus) 采样
        if top_p > 0.0 and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 移除累积概率超过top_p的token
            sorted_indices_to_remove = cumulative_probs > top_p
            # 保留第一个超过阈值的token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # 将要移除的token设为负无穷
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        # 转换为概率并采样
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token


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
            cnn_out_features=cnn_out_features,
        )

    @torch.no_grad()
    def generate_report(
        self, 
        images: torch.Tensor, 
        sos_id: int, 
        eos_id: int, 
        max_len: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        use_sampling: bool = True
    ) -> torch.Tensor:
        """
        生成医疗报告
        
        推荐参数设置：
        - temperature=0.7~0.9: 适度的随机性
        - top_k=30~50: 限制候选词数量
        - top_p=0.85~0.95: Nucleus采样
        - use_sampling=True: 使用采样而非贪婪解码
        """
        features = self.encoder(images)
        return self.decoder.generate(
            features, 
            sos_id, 
            eos_id, 
            max_len,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_sampling=use_sampling
        )
