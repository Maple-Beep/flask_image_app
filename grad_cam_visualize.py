# ========================================================================
# grad_cam_visualize.py - 为你的医疗报告生成模型生成 Grad-CAM 热力图
# 适配模型: ResNet18 + LSTM (来自 model_definition.py)
# 修正：支持 CUDA + LSTM 反向传播，自动管理 model.train()/eval() 模式
# ========================================================================
import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 添加项目路径以便导入本地模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from inference_engine.engine import MedicalReportEngine


class GradCAM:
    """Grad-CAM 实现，适配 ResNet-based 医疗图像描述模型"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None

        # 注册前向和反向 hook
        self.target_layer.register_forward_hook(self.save_feature_maps)
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_feature_maps(self, module, input, output):
        self.feature_maps = output.detach()

    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, image_tensor, sos_id, eos_id, max_len=20):
        """
        生成 Grad-CAM 热力图
        :param image_tensor: 预处理后的图像 [1, C, H, W]
        :param sos_id: 开始符 ID
        :param eos_id: 结束符 ID
        :param max_len: 最大生成长度（控制计算量）
        :return: 热力图 numpy array [H, W]
        """
        # 保存原始模式，并强制进入 train 模式以支持 LSTM backward
        was_training = self.model.training
        self.model.train()
        image_tensor.requires_grad_(True)

        try:
            with torch.enable_grad():
                # 前向：获取 CNN 特征
                cnn_features = self.model.encoder(image_tensor)  # [B, 256]

                # 初始化 LSTM 隐藏状态
                h = self.model.decoder.init_h(cnn_features).unsqueeze(0).repeat(
                    self.model.decoder.num_layers, 1, 1
                )
                c = self.model.decoder.init_c(cnn_features).unsqueeze(0).repeat(
                    self.model.decoder.num_layers, 1, 1
                )

                # 第一个输入 token: SOS
                input_ids = torch.full(
                    (1, 1), sos_id, dtype=torch.long, device=image_tensor.device
                )

                # 执行一步解码（我们只关心第一个预测词的梯度）
                embedded = self.model.decoder.embedding(input_ids)
                lstm_out, (h, c) = self.model.decoder.lstm(embedded, (h, c))
                logits = self.model.decoder.output_proj(lstm_out).squeeze(1)  # [1, vocab_size]

                # 对第一个预测词的所有 logit 求和（得到 scalar 用于 backward）
                score = logits[0].sum()

                # 清零梯度并反向传播
                self.model.zero_grad()
                score.backward(retain_graph=False)

            # === 计算 Grad-CAM ===
            if self.gradients is None or self.feature_maps is None:
                raise RuntimeError("未能捕获梯度或特征图，请检查 hook 是否注册成功")

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
    """与 engine.py 一致的预处理流程，并返回同样尺寸的原始图像用于可视化"""
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

    # === 初始化引擎（仅用于加载模型）===
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

    # === 获取 ResNet 的 layer4 模块（索引 7）===
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
    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam(
        input_tensor,
        sos_id=config.SOS_TOKEN_ID,
        eos_id=config.EOS_TOKEN_ID,
        max_len=10
    )

    # === 叠加热力图 ===
    overlay = overlay_heatmap(original_img, cam)

    # === 保存结果 ===
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"gradcam_{base_name}.png"
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(cam, cmap='jet')
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Grad-CAM 结果已保存至: {output_path}")

    # === 同时生成 AI 报告用于对比 ===
    report = engine.generate(image_path)
    print(f"\n📝 AI 生成报告:\n{report}")


if __name__ == '__main__':
    main()