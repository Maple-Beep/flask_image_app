# =============================================================================
# grad_cam_visualize.py — Grad-CAM 热力图（最终修复版）
#
# 本次修复：
# 1. GradCAM 对 BN 和 Dropout 全部 eval（热力图稳定，不受 batch 影响）
# 2. 梯度反传目标改为 GAP 后激活值的 L2 范数（比 max 更鲁棒）
# 3. 移除 generate 调用中的 use_sampling 参数（engine.generate 已无此参数）
# 4. 异常处理更完善，避免 hook 残留
# =============================================================================

import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from inference_engine.engine import MedicalReportEngine
from inference_engine.model_definition import DISEASE_NAMES


# =============================================================================
# Grad-CAM（修复 BN 问题）
# =============================================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self._feature_maps = None
        self._gradients = None
        self._hooks = [
            target_layer.register_forward_hook(self._save_features),
            target_layer.register_full_backward_hook(self._save_grads),
        ]

    def _save_features(self, module, inp, out):
        self._feature_maps = out          # 保留计算图（不 detach，反传需要）

    def _save_grads(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __call__(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        生成归一化 Grad-CAM 热力图 [0,1]。
        修复：BN 和 Dropout 全部切换到 eval 模式，保证热力图稳定。
        """
        self.model.zero_grad()

        # ── 模式设置 ──
        # 整体 eval（BN 用训练统计量，Dropout 关闭）
        self.model.eval()
        # 仅对 CNN encoder 启用梯度
        for param in self.model.encoder.parameters():
            param.requires_grad_(True)

        try:
            with torch.enable_grad():
                feature_map, global_feat = self.model.encoder(image_tensor)
                # 反传目标：GAP 后特征的 L2 范数（比单个 max 更稳健）
                target = global_feat.norm()
                target.backward()

            if self._gradients is None or self._feature_maps is None:
                print("⚠  Grad-CAM 未获取到梯度，返回零热力图")
                return np.zeros((224, 224), dtype=np.float32)

            gradients  = self._gradients                      # (1, C, H, W)
            feat_maps  = self._feature_maps.detach()          # (1, C, H, W)

            # 全局平均池化梯度 → 通道权重
            weights = gradients.mean(dim=[2, 3], keepdim=True)   # (1, C, 1, 1)
            cam = (weights * feat_maps).sum(dim=1, keepdim=True)  # (1, 1, H, W)
            cam = F.relu(cam).squeeze().cpu()                     # (H, W)

            # 归一化 [0,1]
            cam_min, cam_max = cam.min(), cam.max()
            if (cam_max - cam_min) > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                print("⚠  Grad-CAM 激活范围极小，热力图可能无意义")
                cam = torch.zeros_like(cam)

            return cam.numpy()

        finally:
            # 恢复模型到推理 eval 状态
            self.model.eval()


# =============================================================================
# 图像处理工具
# =============================================================================
def preprocess_image(image_path: str, img_size=(224, 224),
                     mean=None, std=None):
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    image = Image.open(image_path).convert('RGB')
    original = np.array(image.resize(img_size))
    return tf(image).unsqueeze(0), original


def overlay_heatmap(original: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    cam_up  = cv2.resize(cam, (original.shape[1], original.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_up), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return np.clip(alpha * heatmap + (1 - alpha) * original, 0, 255).astype(np.uint8)


# =============================================================================
# 主程序
# =============================================================================
def main():
    if len(sys.argv) < 2:
        print("使用方法: python grad_cam_visualize.py <图片路径>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"错误: 图片不存在 {image_path}")
        sys.exit(1)

    print(f"🖼️  图片: {image_path}")

    config = Config()
    engine_cfg = {
        'MODEL_PATH':    config.MODEL_PATH,
        'VOCAB_PATH':    config.VOCAB_PATH,
        'IMG_SIZE':      config.IMG_SIZE,
        'IMG_MEAN':      config.IMG_MEAN,
        'IMG_STD':       config.IMG_STD,
        'VOCAB_SIZE':    config.VOCAB_SIZE,
        'D_MODEL':       config.D_MODEL,
        'NHEAD':         config.NHEAD,
        'NUM_LAYERS':    config.NUM_LAYERS,
        'DROPOUT':       config.DROPOUT,
        'MAX_REPORT_LEN': config.MAX_REPORT_LEN,
        'PAD_TOKEN_ID':  config.PAD_TOKEN_ID,
        'SOS_TOKEN_ID':  config.SOS_TOKEN_ID,
        'EOS_TOKEN_ID':  config.EOS_TOKEN_ID,
    }

    engine = MedicalReportEngine(config_dict=engine_cfg, debug=True)
    model  = engine.model
    if model is None:
        print("❌ 模型加载失败")
        sys.exit(1)

    # ResNet-101 layer4（第 7 个子模块，输出 2048 通道）
    target_layer = model.encoder.features[7]
    print(f"🎯 Hook 目标: encoder.features[7] (ResNet-101 layer4, 2048ch)")

    tensor, original = preprocess_image(
        image_path, img_size=config.IMG_SIZE,
        mean=config.IMG_MEAN, std=config.IMG_STD
    )
    tensor = tensor.to(engine.device)

    print("🔥 生成 Grad-CAM...")
    grad_cam = GradCAM(model, target_layer)
    try:
        cam = grad_cam(tensor)
    finally:
        grad_cam.remove_hooks()

    overlay = overlay_heatmap(original, cam)

    base  = os.path.splitext(os.path.basename(image_path))[0]
    out_p = f"gradcam_{base}.png"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original);  axes[0].set_title("Original");  axes[0].axis('off')
    im = axes[1].imshow(cam, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM"); axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[2].imshow(overlay);  axes[2].set_title("Overlay");  axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(out_p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 热力图已保存: {out_p}")

    # 生成报告（修复：无 use_sampling 参数）
    print("\n📝 生成 AI 报告...")
    report = engine.generate(image_path, temperature=0.8, top_k=30, top_p=0.9)
    print(f"\nAI 报告:\n{report}")

    # 疾病检测
    print("\n🔬 疾病检测...")
    probs = engine.get_disease_probs(image_path)
    if probs is not None:
        detected = [(n, p) for n, p in zip(DISEASE_NAMES, probs) if p > 0.3]
        if detected:
            for name, prob in detected:
                print(f"  - {name}: {prob:.2%}")
        else:
            print("  未检测到明显异常")
        top3 = sorted(zip(DISEASE_NAMES, probs), key=lambda x: x[1], reverse=True)[:3]
        print("\n  Top-3:")
        for name, prob in top3:
            print(f"    {name}: {prob:.2%}")

    print(f"\n🎉 完成！结果已保存: {out_p}")


if __name__ == '__main__':
    main()
