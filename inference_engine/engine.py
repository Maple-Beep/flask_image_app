# =============================================================================
# inference_engine/engine.py — 医疗报告生成引擎（最终版）
#
# 本次修复（相对 v3）：
# 1. 🔴 移除 use_sampling 参数：model_definition.generate_report 无此参数，传入会 TypeError
# 2. 🔴 正确拆包 generate_report 返回的 tuple(ids, disease_probs)
# 3. 🔴 _ids_to_words 统一接收 ids tensor（非 tuple），消除歧义
# 4. 🟡 GradCAM 通知 engine 层：Dropout + BN 全部 eval（热力图稳定）
# 5. 🟠 generate 路由增加文件存在检查
# 6. 🟠 词汇表加载兼容 Vocabulary 对象和 dict 两种格式
# =============================================================================

import os
import re
import torch
import pickle
from PIL import Image
from torchvision import transforms

from .model_definition import IUReportGenerator


# =============================================================================
# 报告后处理器
# =============================================================================
class MedicalReportPostProcessor:
    """推理侧报告后处理器（与训练工程保持一致）"""

    TRAILING_NOISE = {
        'none', 'available', 'from', 'dated', 'x', 'ct', 'pa', 'ap',
        'views', 'view', 'chest', 'two', 'study', 'exam', 'radiograph'
    }

    def process(self, text: str) -> str:
        if not text or len(text.strip()) < 5:
            return text
        text = self._remove_repetition(text)
        text = self._remove_numeric_noise(text)
        text = self._remove_trailing_noise(text)
        text = self._fix_format(text)
        return text

    def _remove_repetition(self, text: str) -> str:
        text = re.sub(r'\b(\w+)( \1){1,}\b', r'\1', text)
        text = re.sub(r'\b((\w+ ){1,4})\1+', r'\1', text)
        return text

    def _remove_numeric_noise(self, text: str) -> str:
        text = re.sub(r'\bno\s+\d+\s+', 'no ', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(\d+)\s+(?=no |there |the |heart |lung )', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip()

    def _remove_trailing_noise(self, text: str) -> str:
        words = text.rstrip('. ').split()
        while words and words[-1].lower() in self.TRAILING_NOISE:
            words.pop()
        return ' '.join(words)

    def _fix_format(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        if not text.endswith('.'):
            text += '.'
        if text:
            text = text[0].upper() + text[1:]
        return text


# =============================================================================
# 推理引擎
# =============================================================================
class MedicalReportEngine:
    """医疗报告生成引擎（ResNet-101 + Transformer 最终版）"""

    def __init__(self, config_dict: dict, debug: bool = False):
        self.config = config_dict
        self.debug = debug
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.post_processor = MedicalReportPostProcessor()

        if self.debug:
            print(f"🔧 设备: {self.device}")

        img_size = self.config['IMG_SIZE']
        resize_to = tuple(img_size) if isinstance(img_size, (list, tuple)) else (img_size, img_size)

        self.transform = transforms.Compose([
            transforms.Resize(resize_to),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config['IMG_MEAN'], std=self.config['IMG_STD'])
        ])

        self.model: IUReportGenerator | None = None
        self.vocab: dict | None = None       # idx→word 的 dict
        self._load_model_and_vocab()

    # ------------------------------------------------------------------
    # 内部：加载模型与词汇表
    # ------------------------------------------------------------------
    def _load_model_and_vocab(self) -> None:
        model_path = self.config['MODEL_PATH']
        vocab_path = self.config['VOCAB_PATH']

        if not os.path.exists(model_path) or not os.path.exists(vocab_path):
            print("❌ 模型或词汇表文件未找到，AI报告功能不可用。")
            print(f"   模型: {model_path}")
            print(f"   词汇表: {vocab_path}")
            return

        # ── 加载词汇表（兼容两种格式）──
        try:
            with open(vocab_path, 'rb') as f:
                raw = pickle.load(f)
            if isinstance(raw, dict) and 'idx2word' in raw:
                self.vocab = {int(k): v for k, v in raw['idx2word'].items()}
            elif hasattr(raw, 'idx2word'):
                self.vocab = {int(k): v for k, v in raw.idx2word.items()}
            elif isinstance(raw, dict):
                # 兜底：可能直接是 {idx: word} 格式
                self.vocab = {int(k): v for k, v in raw.items()}
            else:
                raise ValueError(f"不支持的词汇表格式: {type(raw)}")

            if self.debug:
                print(f"📚 词汇表大小: {len(self.vocab)}")
        except Exception as e:
            print(f"❌ 词汇表加载失败: {e}")
            return

        # ── 实例化模型 ──
        try:
            vocab_size = self.config.get('VOCAB_SIZE', len(self.vocab))
            self.model = IUReportGenerator(
                vocab_size=vocab_size,
                use_disease_features=True,
                d_model=self.config.get('D_MODEL', 512),
                nhead=self.config.get('NHEAD', 8),
                num_layers=self.config.get('NUM_LAYERS', 4),
                dropout=self.config.get('DROPOUT', 0.1),
            )
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            state_dict = (
                checkpoint['model_state_dict']
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint
                else checkpoint
            )
            self.model.load_state_dict(state_dict, strict=True)
            self.model.to(self.device)
            self.model.eval()
            print("✅ 医疗报告引擎加载成功！")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            self.model = None

    # ------------------------------------------------------------------
    # 内部：ids tensor → 词语列表（接收 tensor，不接收 tuple）
    # ------------------------------------------------------------------
    def _ids_to_words(self, ids: torch.Tensor) -> list:
        """
        ids: (B, seq_len) 或 (seq_len,) tensor
        返回第一个样本的词语列表（过滤特殊 token）
        """
        pad_id = self.config['PAD_TOKEN_ID']
        sos_id = self.config['SOS_TOKEN_ID']
        eos_id = self.config['EOS_TOKEN_ID']
        skip_ids = {pad_id, sos_id, eos_id}
        skip_words = {'<unk>', '<UNK>', '<pad>', '<sos>', '<eos>'}

        # 取第一个样本
        if ids.dim() == 2:
            seq = ids[0].cpu().tolist()
        else:
            seq = ids.cpu().tolist()

        words = []
        for idx in seq:
            if idx == eos_id:
                break
            if idx in skip_ids:
                continue
            word = self.vocab.get(int(idx), '<unk>')
            if word not in skip_words:
                words.append(word)
        return words

    # ------------------------------------------------------------------
    # 公开接口：生成报告
    # ------------------------------------------------------------------
    def generate(
        self,
        image_path: str,
        temperature: float = 0.8,
        top_k: int = 30,
        top_p: float = 0.9,
    ) -> str:
        """
        输入图像路径，返回后处理后的报告文本。

        修复：移除 use_sampling 参数（model.generate_report 无此参数）；
        正确拆包 generate_report 返回的 (ids, disease_probs) tuple。
        """
        if self.model is None or self.vocab is None:
            return "AI报告功能暂不可用，请检查模型文件是否正确放置。"

        if not os.path.exists(image_path):
            return f"图像文件不存在: {image_path}"

        try:
            image = Image.open(image_path).convert('RGB')
            tensor = self.transform(image).unsqueeze(0).to(self.device)

            if self.debug:
                print(f"🖼️  图像: {tensor.shape}  参数: T={temperature} K={top_k} P={top_p}")

            # ── 推理 ──
            # generate_report 已内置 @torch.no_grad() + self.eval()
            # 返回 tuple(generated_ids, disease_probs)
            generated_ids, disease_probs = self.model.generate_report(
                tensor,
                sos_id=self.config['SOS_TOKEN_ID'],
                eos_id=self.config['EOS_TOKEN_ID'],
                max_len=self.config.get('MAX_REPORT_LEN', 100),
                temperature=temperature,
                top_k=top_k if top_k > 0 else None,
                top_p=top_p if top_p > 0.0 else None,
            )

            if self.debug:
                preview = generated_ids[0, :20].cpu().tolist()
                print(f"📝 token IDs (前20): {preview}")

            # ── 解码 ──
            words = self._ids_to_words(generated_ids)

            if self.debug:
                print(f"📄 词语 (前15): {words[:15]}")

            raw    = " ".join(words).strip()
            report = self.post_processor.process(raw)
            return report if report else "No significant findings."

        except Exception as e:
            if self.debug:
                import traceback
                traceback.print_exc()
            return f"生成报告时出错: {e}"

    def generate_multiple(
        self,
        image_path: str,
        num_samples: int = 3,
        temperature: float = 0.8,
        top_k: int = 30,
        top_p: float = 0.9
    ) -> list:
        """生成多个采样报告"""
        return [
            self.generate(image_path, temperature=temperature, top_k=top_k, top_p=top_p)
            for _ in range(num_samples)
        ]

    def get_disease_probs(self, image_path: str):
        """获取图像的 14 维疾病概率向量"""
        if self.model is None:
            return None
        if not os.path.exists(image_path):
            return None
        try:
            image = Image.open(image_path).convert('RGB')
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, global_features = self.model.encoder(tensor)
                disease_logits = self.model.disease_classifier(global_features)
                return torch.sigmoid(disease_logits)[0].cpu().numpy()
        except Exception:
            return None

    def get_model_for_gradcam(self):
        """
        返回供 Grad-CAM 使用的模型引用。
        调用方应在 eval 模式下运行模型主体，
        仅在 CNN encoder 上启用梯度。
        """
        return self.model
