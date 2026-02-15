# inference_engine/engine.py
# ========================================================================
# flask_image_app/inference_engine/engine.py
# 医疗报告生成引擎 - 增强版（添加多样性采样和调试功能）
# ========================================================================

import os
import torch
import pickle
from PIL import Image
from torchvision import transforms

# ✅ 导入增强版的模型定义
from .model_definition import IUReportGenerator


class MedicalReportEngine:
    """医疗报告生成引擎（增强版）"""

    def __init__(self, config_dict, debug=False):
        """
        初始化引擎。
        :param config_dict: 一个包含所有必要配置项的字典。
        :param debug: 是否启用调试模式
        """
        self.config = config_dict
        self.debug = debug
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.debug:
            print(f"🔧 使用设备: {self.device}")

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

        # 加载词汇表
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        
        if self.debug:
            print(f"📚 词汇表大小: {len(self.vocab.get('idx2word', {}))}")

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

    def generate(
        self, 
        image_path: str,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        use_sampling: bool = True
    ) -> str:
        """
        输入图像路径，返回生成的报告文本
        
        参数说明：
        - temperature: 温度参数（0.7-1.0推荐），越高越随机
        - top_k: Top-K采样（30-50推荐），限制候选词数量
        - top_p: Top-P采样（0.85-0.95推荐），Nucleus采样
        - use_sampling: 是否使用采样（True推荐，False则使用贪婪解码）
        """
        if self.model is None or self.vocab is None:
            return "AI报告功能暂不可用。"

        try:
            # 加载并预处理图像
            image = Image.open(image_path).convert('RGB')
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            if self.debug:
                print(f"🖼️  图像形状: {tensor.shape}")
                print(f"🎲 生成参数: temperature={temperature}, top_k={top_k}, top_p={top_p}, sampling={use_sampling}")

            # 生成报告ID序列
            with torch.no_grad():
                output_ids = self.model.generate_report(
                    tensor,
                    sos_id=self.config['SOS_TOKEN_ID'],
                    eos_id=self.config['EOS_TOKEN_ID'],
                    max_len=self.config['MAX_REPORT_LEN'],
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    use_sampling=use_sampling
                )
            
            if self.debug:
                print(f"📝 生成的token IDs (前20个): {output_ids[0][:20].cpu().tolist()}")

            # 将ID转换为文本
            words = []
            token_ids = output_ids[0].cpu().numpy()
            
            for idx in token_ids:
                if idx == self.config['EOS_TOKEN_ID']:
                    break
                if idx not in [self.config['PAD_TOKEN_ID'], self.config['SOS_TOKEN_ID']]:
                    word = self.vocab['idx2word'].get(int(idx), '<unk>')
                    if word not in ['<UNK>', '<unk>', '<pad>', '<PAD>']:
                        words.append(word)
            
            if self.debug:
                print(f"📄 生成的词语: {words[:15]}...")

            # 拼接并格式化报告
            report = " ".join(words).strip()
            
            # 基本的后处理
            if report:
                # 确保句子以句号结尾
                if not report.endswith('.'):
                    report += '.'
                # 首字母大写
                report = report[0].upper() + report[1:] if len(report) > 1 else report.upper()
            else:
                report = "No significant findings."
            
            if self.debug:
                print(f"✅ 最终报告: {report}")
            
            return report

        except Exception as e:
            error_msg = f"生成报告时出错: {str(e)}"
            if self.debug:
                import traceback
                print(f"❌ {error_msg}")
                traceback.print_exc()
            return error_msg
    
    def generate_multiple(
        self, 
        image_path: str, 
        num_samples: int = 3,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> list:
        """
        为同一张图片生成多个不同的报告
        
        这对于诊断多样性问题很有用
        """
        reports = []
        for i in range(num_samples):
            if self.debug:
                print(f"\n--- 生成报告 #{i+1} ---")
            report = self.generate(
                image_path,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                use_sampling=True
            )
            reports.append(report)
        return reports
