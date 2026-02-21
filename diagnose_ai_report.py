#!/usr/bin/env python3
# =============================================================================
# diagnose_ai_report.py - AI报告生成诊断工具（修复版）
#
# 使用方法：
# python diagnose_ai_report.py <图片路径>
#
# 功能：
# 1. 测试不同的采样参数
# 2. 生成多个报告样本
# 3. 显示详细的调试信息
# 4. 显示疾病检测结果
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
            'name': '贪婪解码（最确定）',
            'params': {'use_sampling': False}
        },
        {
            'name': '温度采样 (T=0.7, 更保守)',
            'params': {'temperature': 0.7, 'top_k': 0, 'top_p': 0.0, 'use_sampling': True}
        },
        {
            'name': '温度采样 (T=1.0, 标准)',
            'params': {'temperature': 1.0, 'top_k': 0, 'top_p': 0.0, 'use_sampling': True}
        },
        {
            'name': 'Top-K采样 (K=30)',
            'params': {'temperature': 0.8, 'top_k': 30, 'top_p': 0.0, 'use_sampling': True}
        },
        {
            'name': 'Top-P采样 (P=0.9, Nucleus)',
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
        try:
            report = engine.generate(image_path, **strategy['params'])
            print(f"报告: {report}")
        except Exception as e:
            print(f"❌ 生成失败: {str(e)}")


def generate_multiple_samples(engine, image_path, num_samples=5):
    """生成多个样本以检查多样性"""
    
    print("\n" + "="*80)
    print(f"🎲 生成 {num_samples} 个不同的报告样本")
    print("="*80)
    
    try:
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
            print("   2. 温度参数太低或top_k/top_p设置不当")
            print("   3. 图像特征不够明显")
            print("\n   建议:")
            print("   - 提高temperature到1.0-1.2")
            print("   - 调整top_k到20-30")
            print("   - 使用不同的测试图像")
        elif len(unique_reports) < len(reports) * 0.5:
            print("\n⚠️  警告: 报告多样性较低")
            print("   建议适当提高temperature或调整采样参数")
        else:
            print("\n✅ 报告多样性正常")
            
    except Exception as e:
        print(f"❌ 生成多样本失败: {str(e)}")
        import traceback
        traceback.print_exc()


def analyze_disease_detection(engine, image_path):
    """分析疾病检测结果"""
    
    print("\n" + "="*80)
    print("🔬 疾病检测分析")
    print("="*80)
    
    try:
        import torch
        from inference_engine.model_definition import DISEASE_NAMES
        
        # 加载图像
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        tensor = engine.transform(image).unsqueeze(0).to(engine.device)
        
        # 获取疾病预测
        with torch.no_grad():
            feature_map, global_features = engine.model.encoder(tensor)
            disease_logits = engine.model.disease_classifier(global_features)
            disease_probs = torch.sigmoid(disease_logits)[0].cpu().numpy()
        
        print("\n检测到的疾病（概率从高到低）：")
        disease_list = list(zip(DISEASE_NAMES, disease_probs))
        disease_list.sort(key=lambda x: x[1], reverse=True)
        
        found_high = False
        for name, prob in disease_list:
            if prob > 0.3:
                print(f"  🔴 {name}: {prob:.2%} (高概率)")
                found_high = True
            elif prob > 0.15:
                print(f"  🟡 {name}: {prob:.2%} (中等概率)")
        
        if not found_high:
            print("  ✅ 未检测到明显异常")
            print("\n  概率最高的项：")
            for name, prob in disease_list[:3]:
                print(f"     - {name}: {prob:.2%}")
                
    except Exception as e:
        print(f"❌ 疾病检测分析失败: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    if len(sys.argv) < 2:
        print("使用方法: python diagnose_ai_report.py <图片路径>")
        print("\n示例:")
        print("  python diagnose_ai_report.py static/uploads/chest_xray.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"❌ 错误: 图片文件不存在: {image_path}")
        sys.exit(1)
    
    print("="*80)
    print("🔧 AI报告生成诊断工具")
    print("="*80)
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
    print("\n🔧 初始化引擎...")
    engine = MedicalReportEngine(config_dict=engine_config, debug=True)
    
    if engine.model is None or engine.vocab is None:
        print("❌ 错误: 模型或词汇表未能正确加载")
        print("\n请检查:")
        print("  1. iu_best.pth 是否存在于项目根目录")
        print("  2. vocabulary.pkl 是否存在于项目根目录")
        print("  3. 文件是否完整（未损坏）")
        sys.exit(1)
    
    # 运行诊断测试
    try:
        # 1. 测试不同采样策略
        test_sampling_strategies(engine, image_path)
        
        # 2. 生成多个样本
        generate_multiple_samples(engine, image_path, num_samples=5)
        
        # 3. 疾病检测分析
        analyze_disease_detection(engine, image_path)
        
    except Exception as e:
        print(f"\n❌ 诊断过程出错: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ 诊断完成！")
    print("="*80)
    
    print("\n💡 使用建议：")
    print("1. 如果所有方法都生成相同的报告:")
    print("   - 提高temperature到1.0或更高")
    print("   - 尝试使用不同的图像")
    print("   - 检查模型是否过拟合")
    print("\n2. 如果报告质量不佳:")
    print("   - 降低temperature到0.7")
    print("   - 使用top_k=30限制候选词")
    print("   - 确认输入图像是胸部X光")
    print("\n3. 推荐的生成参数:")
    print("   temperature=0.8, top_k=50, top_p=0.9, use_sampling=True")
    print("\n4. 如果疾病检测不准确:")
    print("   - 这是正常的，模型主要用于报告生成")
    print("   - 疾病特征用于辅助报告生成，不是诊断工具")


if __name__ == '__main__':
    main()
