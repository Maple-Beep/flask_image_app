# 📋 胸部X光报告生成系统 - Flask Web应用

<div align="center">
<br>
<img src="https://img.shields.io/badge/Python-3.8+-blue.svg">
<img src="https://img.shields.io/badge/PyTorch-2.0+-orange.svg">
<img src="https://img.shields.io/badge/Flask-2.0+-green.svg">
<img src="https://img.shields.io/badge/CUDA-11.7+-brightgreen.svg">
<br><br>
</div>

## 📌 项目概述

> 基于深度学习**CNN + LSTM**架构 + **Flask Web框架**的胸部X光影像报告生成系统

### ✨ 核心特性

| 模块 | 功能 | 技术栈 |
|:-----|:-----|:--------|
| 🧠 **AI引擎** | X光图像→医学报告 | CNN编码器 + LSTM解码器 |
| 🔐 **用户系统** | 双角色认证管理 | Flask-Login + SQLAlchemy |
| 📤 **文件管理** | 图像上传/存储 | Werkzeug + Pillow |
| 📊 **可视化** | 个人/管理面板 | Jinja2 + CSS |

---

## ⚠️ 关键文件注意事项 · **必读**

### 🔴 **文件1：`iu_best.pth`** - 训练好的模型权重

#### 📍 **存放位置**
```bash
flask_image_app/
├── ⚠️ iu_best.pth                    # ✅ 必须放在项目根目录
├── app.py
├── config.py
└── ...
🔧 获取方式
bash
# 方案A：从训练结果复制（推荐）
cp /path/to/training/output/checkpoints/iu_best.pth ./

# 方案B：重新训练（耗时2-3小时）
python train_nihcc.py && python train_iu.py
cp output/checkpoints/iu_best.pth ./
✅ 版本验证
python
# 检查模型是否可加载
import torch
checkpoint = torch.load('iu_best.pth', map_location='cpu')
print("📦 模型包含的键:", list(checkpoint.keys()))
print("📏 CNN输出维度:", checkpoint['model_state_dict']['encoder.projection.weight'].shape[1])
⚠️ 常见问题
问题	现象	解决方案
文件缺失	AI报告功能不可用	检查文件是否存在
维度不匹配	RuntimeError: size mismatch	检查config.py中的超参数
版本错误	KeyError: 'model_state_dict'	检查模型文件是否完整
🔵 文件2：vocabulary.pkl - 序列化词汇表
📍 存放位置
bash
flask_image_app/
├── ⚠️ vocabulary.pkl                 # ✅ 必须放在项目根目录
├── app.py
├── config.py
└── ...
🔧 格式要求
python
# ✅ 正确格式 - 必须为字典
vocab = {
    'word2idx': {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, 'lung': 3, ...},
    'idx2word': {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: 'lung', ...}
}

# 🔍 验证脚本
import pickle
with open('vocabulary.pkl', 'rb') as f:
    vocab = pickle.load(f)
    print("📋 词汇表类型:", type(vocab))
    print("🔑 包含的键:", list(vocab.keys()))
    print("📊 词汇表大小:", len(vocab['idx2word']))
⚠️ 常见问题
问题	现象	解决方案
类型错误	'Vocabulary' object is not subscriptable	确保是字典格式
键错误	KeyError: 'idx2word'	检查字典键名
乱码输出	报告全是<unk>	检查VOCAB_SIZE配置
🔗 文件关联性检查
python
# 📝 完整检查脚本 - check_files.py
import torch
import pickle
import os

def check_files():
    """检查两个文件是否匹配"""
    print("\n" + "="*50)
    print("🔍 文件完整性检查")
    print("="*50)
    
    # 1. 检查文件存在性
    files = ['iu_best.pth', 'vocabulary.pkl']
    for f in files:
        status = "✅" if os.path.exists(f) else "❌"
        print(f"{status} {f}")
    
    if not all(os.path.exists(f) for f in files):
        print("❌ 文件缺失，请先复制模型文件")
        return False
    
    # 2. 检查模型与词汇表匹配
    checkpoint = torch.load('iu_best.pth', map_location='cpu')
    model_vocab_size = checkpoint['model_state_dict']['decoder.embedding.weight'].shape[0]
    
    with open('vocabulary.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab['idx2word'])
    
    print(f"\n📏 模型期望词汇表大小: {model_vocab_size}")
    print(f"📏 实际词汇表大小: {vocab_size}")
    
    if model_vocab_size == vocab_size:
        print("✅ 模型与词汇表匹配 ✓")
        return True
    else:
        print("❌ 模型与词汇表不匹配！请使用同一训练生成的配对文件")
        return False

if __name__ == '__main__':
    check_files()
📁 完整目录结构
text
flask_image_app/
│
├── ⚠️ iu_best.pth                    # [必需] 训练好的模型权重
├── ⚠️ vocabulary.pkl                  # [必需] 序列化词汇表
│
├── 📄 app.py                          # 主应用（路由/认证/模型集成）
├── 📄 config.py                       # 全局配置
├── 📄 requirements.txt                 # 依赖清单
├── 📄 README.md                        # 本文档
│
├── 📂 inference_engine/                # 推理引擎模块
│   ├── __init__.py
│   ├── engine.py                        # MedicalReportEngine
│   ├── model_definition.py              # CNN + LSTM模型定义
│   └── vocabulary.py                    # 词汇表工具类
│
├── 📂 instance/                         # 数据库
│   └── database.db                       # SQLite数据库
│
├── 📂 static/                           # 静态资源
│   ├── 📂 css/
│   │   └── style.css                     # 全局样式
│   └── 📂 uploads/                        # 用户上传图像
│
└── 📂 templates/                         # 页面模板
    ├── base.html                          # 基础模板
    ├── index.html                         # 首页
    ├── login.html                         # 登录页
    ├── register.html                      # 注册页
    ├── admin_login.html                    # 管理员登录
    ├── forgot_password.html                # 密码重置
    ├── upload_image.html                   # 上传图片
    ├── user_profile.html                   # 个人中心
    ├── admin_profile.html                  # 管理员面板
    ├── manage_users.html                   # 用户管理
    └── manage_images.html                  # 图片管理
⚙️ 配置说明
📄 config.py 关键参数
python
import os

class Config:
    # ---------- Flask 基础配置 ----------
    SECRET_KEY = 'your-secret-key-here'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(
        os.path.abspath(os.path.dirname(__file__)), 
        'instance', 'database.db'
    )
    UPLOAD_FOLDER = 'static/uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    
    # 🔴 [重要] 模型路径 - 必须与文件实际位置一致
    MODEL_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'iu_best.pth')
    VOCAB_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'vocabulary.pkl')
    
    # 🔴 [重要] 模型超参数 - 必须与训练时完全一致
    CNN_OUT_FEATURES = 256      # CNN输出维度
    LSTM_HIDDEN_SIZE = 256      # LSTM隐藏层维度
    LSTM_NUM_LAYERS = 2         # LSTM层数
    LSTM_DROPOUT = 0.3          # Dropout率
    VOCAB_SIZE = 3047           # ⚠️ 必须与词汇表大小一致
    MAX_REPORT_LEN = 100        # 报告最大长度
    
    # ---------- 图像预处理 ----------
    IMG_SIZE = (224, 224)       # ResNet-18标准输入
    IMG_MEAN = [0.485, 0.456, 0.406]  # ImageNet均值
    IMG_STD = [0.229, 0.224, 0.225]   # ImageNet标准差
    
    # ---------- 特殊Token ----------
    PAD_TOKEN_ID = 0  # <PAD>
    SOS_TOKEN_ID = 1  # <SOS>
    EOS_TOKEN_ID = 2  # <EOS>
🚀 快速开始
1️⃣ 环境准备
bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate      # Linux/Mac
# 或
venv\Scripts\activate         # Windows

# 安装依赖
pip install -r requirements.txt
2️⃣ 模型文件准备
bash
# 复制模型文件到项目根目录
cp /path/to/your/model/iu_best.pth ./
cp /path/to/your/model/vocabulary.pkl ./

# 运行检查脚本
python check_files.py
3️⃣ 启动应用
bash
python app.py
预期输出：

text
初始管理员账户已创建: 用户名 'admin', 密码 'admin123'
✅ 医疗报告引擎加载成功！  # 如果文件正确
 * Running on http://127.0.0.1:5000
📖 使用指南
👤 普通用户流程
步骤	操作	路径
1	注册账户	/register
2	登录系统	/login
3	上传图片	/upload_image
4	生成报告	点击"生成AI报告"按钮
5	查看报告	显示在图片下方
👑 管理员功能
功能	路径	说明
登录	/admin_login	默认账户: admin/admin123
面板	/admin_profile	查看系统统计
用户管理	/manage_users	查看/删除用户
图片管理	/manage_images	查看/删除所有图片
🔐 密码找回流程
text
1. 点击"忘记密码?"
2. 输入注册邮箱
3. 查看终端获取验证码
4. 输入验证码
5. 设置新密码
🧠 模型架构
text
输入图像 (3×224×224)
       ↓
┌─────────────────────┐
│    CNN编码器        │  ← ResNet-18特征提取
│   (ResNet-18)       │    输出: 512维特征
└────────┬────────────┘
         ↓
┌─────────────────────┐
│   特征投影层        │  ← Linear(512, 256)
│   (256维)           │    Dropout(0.3)
└────────┬────────────┘
         ↓
┌─────────────────────┐
│   LSTM解码器        │  ← 初始化隐藏状态
│   (2层, 256维)      │    逐步生成token
└────────┬────────────┘
         ↓
┌─────────────────────┐
│   Token序列         │  ← Top-K采样
│   [SOS, word, ...]  │    temperature=0.8
└────────┬────────────┘
         ↓
┌─────────────────────┐
│   词汇表映射        │  ← vocab['idx2word']
│   生成报告文本      │    后处理：首字母大写+句号
└─────────────────────┘
❗ 故障排除
🔴 模型文件相关
错误信息	可能原因	解决方案
AI报告功能暂不可用	模型文件缺失	检查iu_best.pth和vocabulary.pkl是否存在
KeyError: 'idx2word'	词汇表格式错误	确保vocabulary.pkl是字典且包含'idx2word'键
size mismatch	超参数不匹配	检查config.py中的参数是否与训练一致
全是<unk>	词汇表大小错误	确认VOCAB_SIZE与词汇表实际大小一致
🟡 系统运行问题
问题	现象	解决方案
登录后跳转	总是返回登录页	检查user_loader回调函数
上传失败	500错误	检查static/uploads目录权限
生成超时	长时间无响应	检查GPU是否可用，或降低图片分辨率
📊 性能优化
🚄 加速推理
python
# 启用GPU加速
import torch
print(f"GPU可用: {torch.cuda.is_available()}")

# 模型预热（在app.py中添加）
with app.app_context():
    # 首次请求预热
    if os.path.exists('static/uploads/sample.jpg'):
        app.report_engine.generate('static/uploads/sample.jpg')
💾 资源管理
bash
# 定期清理旧文件
find static/uploads -type f -mtime +30 -delete

# 数据库备份
cp instance/database.db backups/db_$(date +%Y%m%d).backup
📝 更新日志
v1.2.1 
✅ 添加模型文件完整性检查脚本

✅ 优化词汇表加载错误提示

✅ 增加GPU可用性检测

v1.2.0 
✅ 优化模型推理性能

✅ 添加Top-K采样支持

v1.1.0 
✅ 添加密码找回功能

✅ 增强管理员面板

v1.0.0 
✅ 初始版本发布

✅ 基础功能完成

📞 联系方式
方式	信息
📧 邮箱	syz2136453275@163.com
📦 GitHub	[暂时未定]
📋 问题反馈	[请联系邮箱]
⭐ 如果这个项目对您有帮助，请给个Star ⭐
Copyright © 2026 胸部X光报告生成系统 ```
==================================================
本项目版权所有 © 2026 [孙玉征]
保留所有权利。

1. 本工程仅限 [参考学习] 使用。
2. 未经书面授权，不得复制、分发、修改、商用。
3. 不得去除作者信息，或将本工程宣称为自己原创。
4. 如需其他用途，请联系作者：[孙玉征]
==================================================