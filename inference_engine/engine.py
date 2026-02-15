# inference_engine/engine.py
# ========================================================================
# flask_image_app/inference_engine/engine.py
# 医疗报告生成引擎 - 完全解耦版 (已修复词汇表加载和Resize问题)
# ========================================================================

import os
import torch
import pickle
from PIL import Image
from torchvision import transforms

# ✅ 现在只导入同级目录下的模型定义，无任何外部依赖
from .model_definition import IUReportGenerator


class MedicalReportEngine:
    """医疗报告生成引擎"""

    def __init__(self, config_dict):
        """
        初始化引擎。
        :param config_dict: 一个包含所有必要配置项的字典。
        """
        self.config = config_dict
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- ✅ 智能处理 IMG_SIZE 配置 ---
        img_size = self.config['IMG_SIZE']
        if isinstance(img_size, int):
            resize_args = (img_size, img_size)
        elif isinstance(img_size, (list, tuple)) and len(img_size) == 2:
            resize_args = tuple(img_size)
        else:
            raise ValueError(f"IMG_SIZE must be an int or a tuple/list of length 2. Got: {img_size}")

        self.transform = transforms.Compose([
            transforms.Resize(resize_args),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config['IMG_MEAN'], std=self.config['IMG_STD'])
        ])
        self.model = None
        self.vocab = None
        self._load_model_and_vocab()

    def _load_model_and_vocab(self):
        """加载模型权重和词汇表"""
        model_path = self.config['MODEL_PATH']
        vocab_path = self.config['VOCAB_PATH']

        if not os.path.exists(model_path) or not os.path.exists(vocab_path):
            self.model = None
            self.vocab = None
            print("❌ 警告: 模型或词汇表文件未找到，AI报告功能将不可用。")
            return

        # 加载词汇表 (它是一个字典，包含 'idx2word', 'word2idx' 等键)
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)

        # 使用传入的配置字典来实例化模型
        self.model = IUReportGenerator(
            vocab_size=self.config['VOCAB_SIZE'],
            cnn_out_features=self.config['CNN_OUT_FEATURES'],
            lstm_hidden_size=self.config['LSTM_HIDDEN_SIZE'],
            lstm_num_layers=self.config['LSTM_NUM_LAYERS'],
            lstm_dropout=self.config['LSTM_DROPOUT'],
        )

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device).eval()
        print("✅ 医疗报告引擎加载成功！")

    def generate(self, image_path: str) -> str:
        """输入图像路径，返回生成的报告文本"""
        if self.model is None or self.vocab is None:
            return "AI报告功能暂不可用。"

        try:
            image = Image.open(image_path).convert('RGB')
            tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate_report(
                    tensor,
                    sos_id=self.config['SOS_TOKEN_ID'],
                    eos_id=self.config['EOS_TOKEN_ID'],
                    max_len=self.config['MAX_REPORT_LEN']
                )

            # --- 修正：正确地从字典中访问 idx2word ---
            words = []
            for idx in output_ids[0].cpu().numpy():
                if idx == self.config['EOS_TOKEN_ID']:
                    break
                if idx not in [self.config['PAD_TOKEN_ID'], self.config['SOS_TOKEN_ID']]:
                    # self.vocab 是一个 dict, 'idx2word' 是它的 key
                    word = self.vocab['idx2word'].get(int(idx), '<unk>')
                    if word != '<UNK>' and word != '<unk>':
                        words.append(word)

            report = " ".join(words).strip()
            if report and not report.endswith('.'):
                report += '.'
            return report.capitalize()

        except Exception as e:
            return f"生成报告时出错: {str(e)}"