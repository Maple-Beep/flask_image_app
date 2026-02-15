# ========================================================================
# flask_image_app/config.py
# Flask 应用配置文件
# ========================================================================

import os

class Config:
    # --- Flask 基础配置 ---
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'a_very_secret_key_that_you_should_change_in_production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(os.path.abspath(os.path.dirname(__file__)), 'instance', 'database.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = 'static/uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 限制上传文件大小为 16MB

    # --- 模型和词汇表路径 ---
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, 'iu_best.pth')
    VOCAB_PATH = os.path.join(BASE_DIR, 'vocabulary.pkl')

    # --- 模型架构超参数 (必须与训练时一致) ---
    CNN_OUT_FEATURES = 256          # CNN 输出特征维度
    LSTM_HIDDEN_SIZE = 256          # LSTM 隐藏层维度
    LSTM_NUM_LAYERS = 2             # LSTM 层数
    LSTM_DROPOUT = 0.3              # LSTM Dropout 率
    VOCAB_SIZE = 3047                # 词汇表大小
    MAX_REPORT_LEN = 100            # 报告最大长度

    # --- 图像处理 ---
    IMG_SIZE = (224, 224)
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]

    # --- 其他 ---
    PAD_TOKEN_ID = 0
    SOS_TOKEN_ID = 1
    EOS_TOKEN_ID = 2