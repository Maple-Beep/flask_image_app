# ========================================================================
# flask_image_app/models/cnn_lstm_model.py
# 模型架构定义
# ========================================================================

import torch
import torch.nn as nn
import torchvision.models as models

class CNNEncoder(nn.Module):
    """CNN 编码器，基于 ResNet-18"""
    def __init__(self, embed_size=256):
        super(CNNEncoder, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-1]  # 移除最后的全连接层
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)  # [batch, 512, 1, 1]
        features = features.view(features.size(0), -1)  # [batch, 512]
        features = self.embed(features)  # [batch, embed_size]
        return self.dropout(features)

class LSTMDecoder(nn.Module):
    """LSTM 解码器"""
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(LSTMDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions)  # [batch, seq_len, embed_size]
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)  # [batch, seq_len+1, embed_size]
        hiddens, _ = self.lstm(embeddings)  # [batch, seq_len+1, hidden_size]
        outputs = self.linear(hiddens)  # [batch, seq_len+1, vocab_size]
        return outputs

class IUReportGenerator(nn.Module):
    """完整的报告生成模型"""
    def __init__(self, vocab_size, embed_size=256, hidden_size=256, num_layers=1):
        super(IUReportGenerator, self).__init__()
        self.encoder = CNNEncoder(embed_size)
        self.decoder = LSTMDecoder(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def generate_report(self, image, sos_id, eos_id, max_len=100, beam_size=5, temperature=0.8, top_k=20):
        """
        生成单张图像的报告（简化版，使用 Top-K 采样）
        """
        self.eval()
        with torch.no_grad():
            feature = self.encoder(image.unsqueeze(0))  # [1, embed_size]
            sampled_ids = [sos_id]
            inputs = torch.tensor([sos_id], device=image.device).unsqueeze(0)  # [1, 1]

            for _ in range(max_len):
                outputs = self.decoder(feature, inputs)
                predictions = outputs[:, -1, :] / temperature

                # Top-K 采样
                if top_k > 0:
                    top_k_vals, top_k_idxs = torch.topk(predictions, top_k, dim=-1)
                    probabilities = torch.softmax(top_k_vals, dim=-1)
                    next_token_idx = torch.multinomial(probabilities, 1).item()
                    predicted_id = top_k_idxs[0, next_token_idx].item()
                else:
                    predicted_id = predictions.argmax(1).item()

                sampled_ids.append(predicted_id)
                if predicted_id == eos_id:
                    break
                inputs = torch.cat([inputs, torch.tensor([[predicted_id]], device=image.device)], dim=1)

        return torch.tensor(sampled_ids, device=image.device)