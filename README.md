# 胸部 X 光报告生成 · Flask 网页工程

> **架构**：ResNet-101 + Disease-Aware Transformer（与训练工程完全一致）  
> **框架**：Flask + SQLAlchemy + Flask-Login  
> **最后更新**：2026-02-23

---

## 目录

- [项目概述](#项目概述)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [模型文件部署](#模型文件部署)
- [配置说明](#配置说明)
- [系统功能](#系统功能)
- [推理引擎说明](#推理引擎说明)
- [Grad-CAM 热力图](#grad-cam-热力图)
- [诊断工具](#诊断工具)
- [故障排除](#故障排除)

---

## 项目概述

基于 Flask 的胸部 X 光 AI 诊断 Web 系统，集成了**用户管理**、**图像上传**、**AI 报告生成**和 **Grad-CAM 可视化**四大功能模块。

AI 引擎使用 ResNet-101 + Disease-Aware Transformer 架构，模型结构与训练工程完全一致，训练产物（`iu_best.pth` + `vocabulary.pkl`）可直接加载，无需任何转换。

---

## 目录结构

```
webapp_final/
│
├── app.py                          # 主应用（路由、认证、数据库、AI 集成）
├── config.py                       # Flask 配置（路径、模型超参数、Token ID）
├── requirements.txt
│
├── inference_engine/               # AI 推理引擎模块
│   ├── model_definition.py         # 模型定义（与训练工程架构完全一致）
│   └── engine.py                   # MedicalReportEngine（加载、推理、后处理）
│
├── grad_cam_visualize.py           # Grad-CAM 热力图命令行工具
├── diagnose_ai_report.py           # AI 报告诊断工具（测试不同采样策略）
│
├── static/
│   ├── css/style.css
│   └── uploads/                    # 用户上传图像存储目录
│
├── templates/                      # Jinja2 HTML 模板
│   ├── base.html                   # 基础布局
│   ├── index.html                  # 首页
│   ├── login.html / register.html
│   ├── forgot_password.html        # 密码找回（验证码发至控制台）
│   ├── user_profile.html           # 个人中心（上传图像 + 查看报告）
│   ├── upload_image.html
│   ├── admin_login.html
│   ├── admin_profile.html          # 管理员面板（系统统计）
│   ├── manage_users.html           # 用户管理（查看 / 删除）
│   └── manage_images.html          # 图像管理（查看 / 删除全部）
│
└── instance/
    └── database.db                 # SQLite 数据库（首次运行自动创建）
```

---

## 快速开始

### 1. 安装依赖

```bash
# 安装 CUDA 版 PyTorch（推理同样受益于 GPU）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装 Flask 相关依赖
pip install -r requirements.txt
```

### 2. 部署模型文件

将训练工程产出的两个文件复制到本工程**根目录**：

```bash
cp path/to/train_final/output/checkpoints/iu_best.pth    ./
cp path/to/train_final/output/vocabulary.pkl              ./
```

目录应包含：

```
webapp_final/
├── iu_best.pth        ← 必需
├── vocabulary.pkl     ← 必需
├── app.py
└── ...
```

### 3. 配置环境变量（生产环境必须）

```bash
export SECRET_KEY="your-strong-random-key-here"
```

> 不设置时使用代码内置的开发默认值，**不得用于生产**。

### 4. 启动应用

```bash
python app.py
```

首次运行输出：

```
初始管理员账户已创建: 用户名 'admin', 密码 'admin123'
✅ 医疗报告引擎加载成功！
 * Running on http://127.0.0.1:5000
```

若模型文件缺失，引擎仍会启动但报告功能不可用，界面会提示错误。

---

## 模型文件部署

### 文件说明

| 文件 | 来源 | 说明 |
|------|------|------|
| `iu_best.pth` | 训练工程 `output/checkpoints/iu_best.pth` | 模型权重，包含 `vocab_size` 等元信息 |
| `vocabulary.pkl` | 训练工程 `output/vocabulary.pkl` | 词汇表，`idx2word` 映射字典 |

两个文件**必须来自同一次训练**，否则词汇表维度与模型嵌入层不匹配。

### 验证模型文件

```python
import torch, pickle

# 检查 iu_best.pth
ck = torch.load('iu_best.pth', map_location='cpu')
print("模型 vocab_size:", ck['vocab_size'])
print("训练 epoch:", ck['epoch'])
print("val_loss:", ck['val_loss'])

# 检查 vocabulary.pkl
with open('vocabulary.pkl', 'rb') as f:
    raw = pickle.load(f)
vocab = raw['idx2word']
print("词汇表大小:", len(vocab))

# 确认两者一致
assert ck['vocab_size'] == len(vocab), "不匹配！请使用同一训练的配对文件"
print("✅ 文件匹配，可以部署")
```

---

## 配置说明

`config.py` 中的 `Config` 类：

```python
class Config:
    # Flask 基础
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-only-key-CHANGE-IN-PRODUCTION'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///instance/database.db'
    UPLOAD_FOLDER = 'static/uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024   # 16MB 上传限制

    # 模型文件路径（相对于项目根目录）
    MODEL_PATH = os.path.join(BASE_DIR, 'iu_best.pth')
    VOCAB_PATH = os.path.join(BASE_DIR, 'vocabulary.pkl')

    # Transformer 架构参数（必须与训练工程完全一致）
    VOCAB_SIZE   = 3047    # 词汇表大小（以 checkpoint 中保存值为实际准）
    D_MODEL      = 512     # 隐藏维度
    NHEAD        = 8       # 注意力头数
    NUM_LAYERS   = 4       # 编码器 / 解码器层数
    DROPOUT      = 0.1

    # 推理参数
    MAX_REPORT_LEN = 100   # 最大生成长度（含 SOS/EOS）

    # 图像预处理（ImageNet 标准）
    IMG_SIZE = (224, 224)
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD  = [0.229, 0.224, 0.225]

    # Token ID（必须与训练工程一致）
    PAD_TOKEN_ID = 0
    SOS_TOKEN_ID = 1
    EOS_TOKEN_ID = 2
```

> **重要**：`D_MODEL`、`NHEAD`、`NUM_LAYERS`、`DROPOUT`、三个 Token ID 必须与训练工程 `config.py` 完全一致，否则模型加载时会报维度不匹配。

---

## 系统功能

### 路由一览

| 路由 | 方法 | 权限 | 功能 |
|------|------|------|------|
| `/` | GET | 公开 | 首页 |
| `/register` | GET/POST | 公开 | 用户注册 |
| `/login` | GET/POST | 公开 | 用户登录（支持用户名或邮箱） |
| `/logout` | GET | 登录后 | 退出登录 |
| `/user_profile` | GET | 登录后 | 个人中心（查看自己的图像和报告） |
| `/upload_image` | GET/POST | 登录后 | 上传 X 光图像 |
| `/generate_report/<id>` | POST | 登录后 | 为指定图像调用 AI 生成报告 |
| `/uploaded_file/<filename>` | GET | 登录后 | 获取已上传图像文件 |
| `/forgot_password` | GET/POST | 公开 | 密码找回（验证码输出到控制台） |
| `/admin_login` | GET/POST | 公开 | 管理员登录入口 |
| `/admin_profile` | GET | 管理员 | 管理面板（用户数、图像数统计） |
| `/manage_users` | GET | 管理员 | 用户列表 |
| `/delete_user/<id>` | GET | 管理员 | 删除用户（级联删除其图像） |
| `/manage_images` | GET | 管理员 | 全部图像列表 |
| `/delete_image/<id>` | GET | 管理员 | 删除图像 |

### 数据库模型

```python
User:
    id, username (唯一), email (唯一), password_hash
    is_admin: bool
    images: → [ImageModel]（级联删除）

ImageModel:
    id, filename, original_filename
    user_id → User
    upload_date, ai_report (Text, 可为空)
```

### AI 报告生成流程

1. 用户在 `/user_profile` 点击图像旁的"生成 AI 报告"按钮
2. POST 请求到 `/generate_report/<image_id>`
3. 检查文件存在性 → 调用 `report_engine.generate()`
4. 报告文本写入 `ImageModel.ai_report` 字段并提交数据库
5. 重定向回 `/user_profile` 展示结果

生成参数（固定值）：`temperature=0.8`，`top_k=30`，`top_p=0.9`。

### 默认管理员账户

首次运行时自动创建：

| 字段 | 值 |
|------|----|
| 用户名 | `admin` |
| 密码 | `admin123` |

**生产环境请立即修改密码。**

---

## 推理引擎说明

`inference_engine/engine.py` 中的 `MedicalReportEngine` 类：

### 初始化

```python
engine = MedicalReportEngine(config_dict={
    'MODEL_PATH':    'iu_best.pth',
    'VOCAB_PATH':    'vocabulary.pkl',
    'IMG_SIZE':      (224, 224),
    'IMG_MEAN':      [0.485, 0.456, 0.406],
    'IMG_STD':       [0.229, 0.224, 0.225],
    'VOCAB_SIZE':    3047,
    'D_MODEL':       512,
    'NHEAD':         8,
    'NUM_LAYERS':    4,
    'DROPOUT':       0.1,
    'MAX_REPORT_LEN': 100,
    'PAD_TOKEN_ID':  0,
    'SOS_TOKEN_ID':  1,
    'EOS_TOKEN_ID':  2,
}, debug=False)
```

### 核心方法

```python
# 生成单条报告
report: str = engine.generate(
    image_path,
    temperature=0.8,  # 采样温度
    top_k=30,         # Top-K 候选词数
    top_p=0.9,        # Nucleus 采样阈值
)

# 生成多个采样版本（用于观察多样性）
reports: list = engine.generate_multiple(
    image_path, num_samples=3,
    temperature=0.8, top_k=30, top_p=0.9
)

# 获取 14 维疾病概率向量（numpy array）
probs: np.ndarray = engine.get_disease_probs(image_path)
```

### 词汇表加载兼容性

引擎自动兼容以下三种 `vocabulary.pkl` 格式：

- `{'idx2word': {0: '<pad>', ...}, 'word2idx': {...}}` ← 训练工程标准格式
- 带有 `.idx2word` 属性的 `Vocabulary` 对象
- 直接 `{idx: word}` 字典格式

### 报告后处理

生成文本经 `MedicalReportPostProcessor` 自动清理（与训练工程同步）：

| 处理步骤 | 示例 |
|----------|------|
| 去连续重复单词 | `no no cardiomegaly` → `no cardiomegaly` |
| 去重复短语 | `heart size heart size unchanged` → `heart size unchanged` |
| 去数字噪声 | `no 2 cardiomegaly` → `no cardiomegaly` |
| 去末尾无意义词 | `...clear views` → `...clear` |
| 格式化 | 首字母大写，末尾补句号 |

---

## Grad-CAM 热力图

命令行可视化工具，显示模型关注的 X 光区域：

```bash
python grad_cam_visualize.py path/to/xray.jpg
```

输出文件：`gradcam_<图像名>.png`（含原图 / 热力图 / 叠加图三列）

**实现细节：**

- Hook 目标层：`model.encoder.features[7]`（ResNet-101 的 layer4，2048 通道）
- 反传目标：CNN 全局平均池化后特征的 L2 范数（比单点 max 更稳健）
- 模式设置：`model.eval()` 整体（BN 使用训练统计量，热力图稳定），仅对 `encoder` 开启梯度
- 生成报告时同步输出 AI 报告文本和 14 类疾病检测结果（概率 > 0.3 的疾病）

---

## 诊断工具

### AI 报告诊断

```bash
python diagnose_ai_report.py path/to/xray.jpg
```

测试 6 种采样策略并逐一输出报告，用于评估模型质量和调参：

| 策略 | 参数 |
|------|------|
| 贪婪解码（最确定） | `top_k=1` |
| 温度采样（保守） | `T=0.7` |
| 温度采样（标准） | `T=1.0` |
| Top-K 采样 | `K=30, T=0.8` |
| Top-P 采样 | `P=0.9, T=0.8` |
| Top-K + Top-P 组合（推荐） | `K=30, P=0.9, T=0.8` |

同时生成 3 个随机采样版本，统计报告多样性（唯一率）。

---

## 故障排除

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| `AI报告功能暂不可用` | 模型或词汇表文件缺失 | 检查 `iu_best.pth` 和 `vocabulary.pkl` 是否在根目录 |
| `size mismatch` 加载失败 | 架构参数与训练时不一致 | 核对 `config.py` 中 `D_MODEL`、`NHEAD`、`NUM_LAYERS` |
| `vocab_size` 不匹配 | 词汇表文件与模型不配套 | 确保两个文件来自同一次训练 |
| 报告全是 `<unk>` | 词汇表格式不支持 | 引擎已兼容 3 种格式；检查文件是否损坏 |
| `403 Forbidden`（生成报告时） | 图像不属于当前用户 | 正常权限保护，非 bug |
| 图像上传后 AI 按钮无响应 | 图像文件被删除或路径错误 | 检查 `static/uploads/` 目录权限 |
| Grad-CAM 热力图全黑 | 梯度消失 | 确认模型已正确加载权重；检查输入图像预处理是否一致 |
| 密码找回验证码看不到 | 验证码输出到服务器控制台 | 查看运行 `python app.py` 的终端输出 |
| 登录后一直跳回登录页 | `SECRET_KEY` 未设置或会话失效 | 设置 `SECRET_KEY` 环境变量并重启 |

---

> **与训练工程的关系**：本工程的 `inference_engine/model_definition.py` 与训练工程的 `model/cnn_transformer_model.py` 架构代码完全同步。唯一区别是 `CNNEncoder(pretrained=False)`——推理时无需下载 ImageNet 权重，权重由 `iu_best.pth` 的 `model_state_dict` 提供。

```

📞 联系方式

方式 	信息

📧 邮箱	syz2136453275@outlook.com
📦 GitHub	[Maple-Beep]
📋 问题反馈	[请联系邮箱]
⭐ 如果这个项目对您有帮助，请给个Star ⭐

Copyright © 2026 胸部X光报告模型生成 
==================================================

本项目版权所有 © 2026 [孙玉征]
保留所有权利。

1. 本工程仅限 [参考学习] 使用。
2. 未经书面授权，不得复制、分发、修改、商用。
3. 不得去除作者信息，或将本工程宣称为自己原创。
4. 如需其他用途，请联系作者：[孙玉征]

==================================================