#!/usr/bin/env python3
# =============================================================================
# diagnose_ai_report.py - AI报告生成诊断工具
#
# 使用方法：
# python diagnose_ai_report.py <图片路径>
#
# 功能：
# 1. 测试不同的采样参数
# 2. 生成多个报告样本
# 3. 显示详细的调试信息
# =============================================================================

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from inference_engine.engine import MedicalReportEngine


def test_sampling_strategies(engine, image_path):
    """测试不同的采样策略"""
    
    print("\n" + "="*80)
    print("🧪 测试不同的采样策略")
    print("="*80)
    
    strategies = [
        {
            'name': '贪婪解码（原始方法）',
            'params': {'use_sampling': False}
        },
        {
            'name': '温度采样 (T=0.7)',
            'params': {'temperature': 0.7, 'top_k': 0, 'top_p': 0.0, 'use_sampling': True}
        },
        {
            'name': '温度采样 (T=1.0)',
            'params': {'temperature': 1.0, 'top_k': 0, 'top_p': 0.0, 'use_sampling': True}
        },
        {
            'name': 'Top-K采样 (K=30)',
            'params': {'temperature': 0.8, 'top_k': 30, 'top_p': 0.0, 'use_sampling': True}
        },
        {
            'name': 'Top-P采样 (P=0.9)',
            'params': {'temperature': 0.8, 'top_k': 0, 'top_p': 0.9, 'use_sampling': True}
        },
        {
            'name': 'Top-K + Top-P组合（推荐）',
            'params': {'temperature': 0.8, 'top_k': 50, 'top_p': 0.9, 'use_sampling': True}
        },
    ]
    
    for strategy in strategies:
        print(f"\n📊 策略: {strategy['name']}")
        print("-" * 80)
        report = engine.generate(image_path, **strategy['params'])
        print(f"报告: {report}")


def generate_multiple_samples(engine, image_path, num_samples=5):
    """生成多个样本以检查多样性"""
    
    print("\n" + "="*80)
    print(f"🎲 生成 {num_samples} 个不同的报告样本")
    print("="*80)
    
    reports = engine.generate_multiple(
        image_path, 
        num_samples=num_samples,
        temperature=0.8,
        top_k=50,
        top_p=0.9
    )
    
    print("\n生成的报告：")
    for i, report in enumerate(reports, 1):
        print(f"\n样本 {i}:")
        print(f"  {report}")
    
    # 检查唯一性
    unique_reports = set(reports)
    print(f"\n📈 统计信息:")
    print(f"  总样本数: {len(reports)}")
    print(f"  唯一报告数: {len(unique_reports)}")
    print(f"  多样性: {len(unique_reports)/len(reports)*100:.1f}%")
    
    if len(unique_reports) == 1:
        print("\n⚠️  警告: 所有生成的报告都相同！")
        print("   可能的原因:")
        print("   1. 模型过拟合，学习到了固定的模板")
        print("   2. 模型权重可能没有正确加载")
        print("   3. 图像预处理可能有问题")
    elif len(unique_reports) < len(reports) * 0.5:
        print("\n⚠️  警告: 报告多样性较低")
    else:
        print("\n✅ 报告多样性正常")


def main():
    if len(sys.argv) < 2:
        print("使用方法: python diagnose_ai_report.py <图片路径>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"❌ 错误: 图片文件不存在: {image_path}")
        sys.exit(1)
    
    print("🔧 初始化诊断工具...")
    print(f"📁 图片路径: {image_path}")
    
    # 构建配置
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
    
    # 创建引擎（启用调试模式）
    engine = MedicalReportEngine(config_dict=engine_config, debug=True)
    
    if engine.model is None or engine.vocab is None:
        print("❌ 错误: 模型或词汇表未能正确加载")
        sys.exit(1)
    
    # 运行诊断测试
    test_sampling_strategies(engine, image_path)
    generate_multiple_samples(engine, image_path, num_samples=5)
    
    print("\n" + "="*80)
    print("✅ 诊断完成！")
    print("="*80)
    
    print("\n💡 建议：")
    print("1. 如果所有方法都生成相同的报告，问题可能在于模型本身")
    print("2. 如果采样方法生成了不同的报告，可以在app.py中调整采样参数")
    print("3. 推荐使用 temperature=0.8, top_k=50, top_p=0.9 的组合")


if __name__ == '__main__':
    main()
