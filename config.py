# ========================================================================
# flask_image_app/config.py
# Flask 应用配置文件（Transformer版本，对应 ResNet-101 + Transformer 架构）
# ========================================================================

import os


class Config:
    # --- Flask 基础配置 ---
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'a_very_secret_key_that_you_should_change_in_production'
    SQLALCHEMY_DATABASE_URI = (
        os.environ.get('DATABASE_URL') or
        'sqlite:///' + os.path.join(os.path.abspath(os.path.dirname(__file__)), 'instance', 'database.db')
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = 'static/uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

    # --- 模型和词汇表路径 ---
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, 'iu_best.pth')
    VOCAB_PATH = os.path.join(BASE_DIR, 'vocabulary.pkl')

    # --- Transformer 模型架构参数（必须与训练时一致）---
    VOCAB_SIZE = 3047           # 词汇表大小（由训练工程决定，使用前请确认）
    D_MODEL = 512               # Transformer 隐藏维度
    NHEAD = 8                   # 注意力头数
    NUM_LAYERS = 4              # 编码器/解码器层数
    DROPOUT = 0.1               # Transformer dropout
    MAX_REPORT_LEN = 100        # 报告最大 token 数

    # --- 图像处理 ---
    IMG_SIZE = (224, 224)
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]

    # --- Token ID（必须与训练时一致）---
    PAD_TOKEN_ID = 0
    SOS_TOKEN_ID = 1
    EOS_TOKEN_ID = 2
