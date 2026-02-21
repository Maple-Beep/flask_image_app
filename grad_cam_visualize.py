# ========================================================================
# grad_cam_visualize.py - Grad-CAM 热力图可视化（修复版）
# 适配模型: 疾病感知的CNN + LSTM报告生成模型
# ========================================================================
import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from inference_engine.engine import MedicalReportEngine


class GradCAM:
    """Grad-CAM 实现（修复版）"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None

        # 注册 hook（修复签名）
        self.target_layer.register_forward_hook(self.save_feature_maps)
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_feature_maps(self, module, input, output):
        """保存特征图 - 正确的hook签名"""
        self.feature_maps = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        """保存梯度 - 正确的hook签名"""
        self.gradients = grad_output[0].detach()

    def __call__(self, image_tensor, max_len=10):
        """
        生成 Grad-CAM 热力图
        :param image_tensor: 预处理后的图像 [1, C, H, W]
        :param max_len: 最大生成长度（控制计算量）
        :return: 热力图 numpy array [H, W]
        """
        # 保存原始模式
        was_training = self.model.training
        self.model.train()
        image_tensor.requires_grad_(True)

        try:
            with torch.enable_grad():
                # 前向传播：获取特征图和全局特征
                feature_map, global_features = self.model.encoder(image_tensor)  # [B, 512, 7, 7], [B, 512]

                # 获取疾病特征
                disease_logits = self.model.disease_classifier(global_features)
                disease_features = torch.sigmoid(disease_logits)  # [B, 14]

                # 初始化LSTM状态
                B = image_tensor.size(0)
                h = torch.zeros(1, B, 512, device=image_tensor.device)
                c = torch.zeros(1, B, 512, device=image_tensor.device)

                # 准备特征用于注意力
                features_for_attn = feature_map.view(B, 512, -1).permute(0, 2, 1)  # [B, 49, 512]

                # 编码疾病特征
                disease_context = self.model.decoder.disease_encoder(disease_features)  # [B, 256]

                # 第一个token: SOS
                sos_id = 1  # 假设SOS token ID是1
                input_ids = torch.tensor([[sos_id]], dtype=torch.long, device=image_tensor.device)

                # 执行一步解码
                embedding = self.model.decoder.embedding(input_ids[:, -1])  # [B, 512]
                
                # 计算注意力
                context, _ = self.model.decoder.attention(
                    features_for_attn, h[-1], disease_features
                )  # [B, 512]
                
                # LSTM输入
                lstm_input = torch.cat(
                    (embedding, context, disease_context), dim=1
                ).unsqueeze(1)  # [B, 1, 1280]
                
                out, (h, c) = self.model.decoder.lstm(lstm_input, (h, c))
                logits = self.model.decoder.fc(out.squeeze(1))  # [B, vocab_size]

                # 对所有logit求和作为loss
                score = logits[0].sum()

                # 反向传播
                self.model.zero_grad()
                score.backward(retain_graph=False)

            # === 计算 Grad-CAM ===
            if self.gradients is None or self.feature_maps is None:
                raise RuntimeError("未能捕获梯度或特征图")

            gradients = self.gradients  # [1, 512, 7, 7]
            feature_maps = self.feature_maps  # [1, 512, 7, 7]

            # 全局平均池化梯度 → 权重 [512]
            weights = torch.mean(gradients, dim=[0, 2, 3])  # [512]

            # 加权求和特征图
            cam = torch.zeros(feature_maps.shape[2:], device=feature_maps.device)  # [7, 7]
            for i, w in enumerate(weights):
                cam += w * feature_maps[0, i, :, :]

            cam = F.relu(cam)
            cam = cam.cpu().numpy()
            cam = cv2.resize(cam, (image_tensor.shape[3], image_tensor.shape[2]))  # [224, 224]
            cam = cam - np.min(cam)
            cam = cam / (np.max(cam) + 1e-8)

            return cam

        finally:
            # 恢复模型原始模式
            if was_training:
                self.model.train()
            else:
                self.model.eval()


def preprocess_image(image_path, img_size=(224, 224), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """图像预处理"""
    from torchvision import transforms
    image = Image.open(image_path).convert('RGB')
    resized_image = image.resize(img_size, Image.Resampling.LANCZOS)
    original_for_overlay = np.array(resized_image)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    tensor = transform(resized_image).unsqueeze(0)
    return tensor, original_for_overlay


def overlay_heatmap(original_img, cam, alpha=0.6):
    """将热力图叠加到原图"""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * original_img).astype(np.uint8)
    return overlay


def main():
    if len(sys.argv) < 2:
        print("使用方法: python grad_cam_visualize.py <图片路径>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"错误: 图片不存在 {image_path}")
        sys.exit(1)

    print(f"🖼️  加载图片: {image_path}")

    # === 初始化引擎 ===
    config = Config()
    engine_config = {
        'MODEL_PATH': config.MODEL_PATH,
        'VOCAB_PATH': config.VOCAB_PATH,
        'IMG_SIZE': config.IMG_SIZE,
        'IMG_MEAN': config.IMG_MEAN,
        'IMG_STD': config.IMG_STD,
        'VOCAB_SIZE': config.VOCAB_SIZE,
        'CNN_OUT_FEATURES': config.CNN_OUT_FEATURES,
        'LSTM_HIDDEN_SIZE': config.LSTM_HIDDEN_SIZE,
        'LSTM_NUM_LAYERS': config.LSTM_NUM_LAYERS,
        'LSTM_DROPOUT': config.LSTM_DROPOUT,
        'MAX_REPORT_LEN': config.MAX_REPORT_LEN,
        'PAD_TOKEN_ID': config.PAD_TOKEN_ID,
        'SOS_TOKEN_ID': config.SOS_TOKEN_ID,
        'EOS_TOKEN_ID': config.EOS_TOKEN_ID,
    }

    engine = MedicalReportEngine(config_dict=engine_config, debug=True)
    model = engine.model

    if model is None:
        print("❌ 模型加载失败")
        sys.exit(1)

    # === 获取 ResNet 的 layer4 ===
    target_layer = model.encoder.features[7]  # resnet.layer4
    print(f"🎯 Hook 目标层: model.encoder.features[7] (ResNet layer4)")

    # === 预处理图像 ===
    input_tensor, original_img = preprocess_image(
        image_path,
        img_size=config.IMG_SIZE,
        mean=config.IMG_MEAN,
        std=config.IMG_STD
    )
    input_tensor = input_tensor.to(engine.device)

    # === 生成 Grad-CAM ===
    print("🔥 生成Grad-CAM热力图...")
    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam(input_tensor, max_len=10)

    # === 叠加热力图 ===
    overlay = overlay_heatmap(original_img, cam)

    # === 保存结果 ===
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"gradcam_{base_name}.png"
    
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title("Original Image", fontsize=14)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cam, cmap='jet')
    plt.title("Grad-CAM Heatmap", fontsize=14)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay", fontsize=14)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Grad-CAM 结果已保存至: {output_path}")

    # === 生成 AI 报告 ===
    print("\n📝 生成AI报告...")
    report = engine.generate(image_path)
    print(f"\nAI报告:\n{report}")
    
    # === 显示疾病检测 ===
    print("\n🔬 疾病检测分析...")
    with torch.no_grad():
        feature_map, global_features = model.encoder(input_tensor)
        disease_logits = model.disease_classifier(global_features)
        disease_probs = torch.sigmoid(disease_logits)[0].cpu().numpy()
    
    from inference_engine.model_definition import DISEASE_NAMES
    
    print("\n检测到的异常（概率>0.3）：")
    detected = False
    for i, (name, prob) in enumerate(zip(DISEASE_NAMES, disease_probs)):
        if prob > 0.3:
            print(f"  - {name}: {prob:.2%}")
            detected = True
    
    if not detected:
        print("  未检测到明显异常")
    
    print(f"\n🎉 完成！结果已保存到 {output_path}")


if __name__ == '__main__':
    main()
