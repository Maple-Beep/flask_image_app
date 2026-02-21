# =============================================================================
# inference_engine/engine.py — 医疗报告生成引擎（匹配训练模型）
# =============================================================================

import os
import torch
import pickle
from PIL import Image
from torchvision import transforms

from .model_definition import IUReportGenerator


class MedicalReportEngine:
    """医疗报告生成引擎（匹配训练模型版本）"""

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

        # 图像预处理
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
            print(f"   模型路径: {model_path}")
            print(f"   词汇表路径: {vocab_path}")
            return

        # 加载词汇表
        try:
            with open(vocab_path, 'rb') as f:
                self.vocab = pickle.load(f)
            
            if self.debug:
                print(f"📚 词汇表加载成功")
                if isinstance(self.vocab, dict):
                    print(f"   词汇表大小: {len(self.vocab.get('idx2word', {}))}")
        except Exception as e:
            print(f"❌ 词汇表加载失败: {str(e)}")
            self.vocab = None
            return

        # 实例化模型
        try:
            self.model = IUReportGenerator(
                vocab_size=self.config['VOCAB_SIZE'],
                use_disease_features=True
            )
            
            if self.debug:
                print(f"📦 模型结构创建成功")
            
            # 加载权重
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            if self.debug:
                print(f"📂 检查点加载成功")
                if isinstance(checkpoint, dict):
                    print(f"   检查点键: {list(checkpoint.keys())}")
            
            # 提取state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # 加载state_dict
            self.model.load_state_dict(state_dict, strict=True)
            self.model.to(self.device)
            self.model.eval()
            
            print("✅ 医疗报告引擎加载成功！")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            import traceback
            if self.debug:
                traceback.print_exc()
            self.model = None
            return

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
            return "AI报告功能暂不可用。请检查模型文件是否正确放置。"

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
                    top_k=top_k if top_k > 0 else None,
                    top_p=top_p if top_p > 0.0 else None,
                    use_sampling=use_sampling
                )
            
            if self.debug:
                print(f"📝 生成的token IDs (前20个): {output_ids[0][:20].cpu().tolist()}")

            # 将ID转换为文本
            words = []
            token_ids = output_ids[0].cpu().numpy()
            
            # 获取idx2word映射
            if isinstance(self.vocab, dict):
                idx2word = self.vocab.get('idx2word', self.vocab)
            else:
                idx2word = self.vocab
            
            for idx in token_ids:
                if idx == self.config['EOS_TOKEN_ID']:
                    break
                if idx not in [self.config['PAD_TOKEN_ID'], self.config['SOS_TOKEN_ID']]:
                    word = idx2word.get(int(idx), '<unk>')
                    if word not in ['<UNK>', '<unk>', '<pad>', '<PAD>', '<SOS>', '<sos>', '<EOS>', '<eos>']:
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
