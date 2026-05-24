"""
场景A: 集中式训练（Gradient Boosting Regressor）
在统一全局训练切分上训练单一模型，获取性能上限基准
使用sklearn GradientBoostingRegressor（经验证表现最佳）
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingRegressor

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import (
    load_config, set_seed, setup_logger,
    evaluate_metrics, save_results, format_metrics
)
from src.data_preprocessing import load_centralized_datasets


def main():
    """场景A主函数"""
    
    print("=" * 80)
    print("SCENARIO A: CENTRALIZED TRAINING (Gradient Boosting)")
    print("=" * 80)
    print("\nTraining GradientBoostingRegressor on unified global training split")
    print("This provides the performance upper bound (no privacy constraints)\n")
    
    # ========== 1. 加载配置 ==========
    print("[Step 1/7] Loading configuration...")
    config = load_config("configs/config.yaml")
    
    # 设置随机种子
    seed = config['preprocessing']['random_seed']
    set_seed(seed)
    print(f"          Random seed: {seed}")
    
    # 设置日志
    logger = setup_logger(
        "Centralized_GBR",
        log_file="results/logs/centralized_gbr_training.log",
        console=True
    )
    logger.info("Starting Scenario A: Centralized Training (GradientBoosting)")
    print("          [DONE]\n")
    
    # ========== 2. 加载统一切分数据 ==========
    print("[Step 2/7] Loading unified centralized splits (same as FL global split)...")
    split_data = load_centralized_datasets(config, config_key='scene_c')
    X_train = split_data["X_train"]
    y_train = split_data["y_train"]
    X_val = split_data["X_val"]
    y_val = split_data["y_val"]
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]
    print(f"          Train/Val/Test: {len(X_train)}/{len(X_val)}/{len(X_test)}")
    print(f"          Features: {X_train.shape[1]}")
    print(f"          Train target range: ${y_train.min():,.0f} - ${y_train.max():,.0f}")
    logger.info(f"Loaded unified split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print("          [DONE]\n")
    
    # ========== 3. 数据预处理（已在统一切分流程中完成） ==========
    print("[Step 3/7] Preprocessing data...")
    preprocessor = split_data["preprocessor"]
    X_scaled = split_data["X_train_scaled"]
    y_transformed = split_data["y_train_transformed"]
    X_val_scaled = split_data["X_val_scaled"]
    X_test_scaled = split_data["X_test_scaled"]
    print(f"          Feature scaling: StandardScaler")
    print(f"          Target transform: {config['preprocessing']['target_transform']}")
    print(f"          Scaled X shape: {X_scaled.shape}")
    print(f"          Transformed y shape: {y_transformed.shape}")
    print(f"          Transformed target range: {y_transformed.min():.4f} - {y_transformed.max():.4f}")
    logger.info("Data preprocessing completed")
    print("          [DONE]\n")
    
    # ========== 4. 创建并训练GBR模型 ==========
    print("[Step 4/7] Training Gradient Boosting Regressor...")
    
    # GBR超参数（经过验证的最佳配置）
    gbr_params = {
        'n_estimators': 200,
        'learning_rate': 0.1,
        'max_depth': 5,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'subsample': 0.8,
        'random_state': seed,
        'verbose': 0
    }
    
    print(f"          n_estimators: {gbr_params['n_estimators']}")
    print(f"          learning_rate: {gbr_params['learning_rate']}")
    print(f"          max_depth: {gbr_params['max_depth']}")
    print(f"          subsample: {gbr_params['subsample']}")
    print()
    
    model = GradientBoostingRegressor(**gbr_params)
    model.fit(X_scaled, y_transformed.ravel())
    
    print(f"          Training completed!")
    print(f"          Number of trees: {model.n_estimators_}")
    logger.info(f"GBR model trained with {model.n_estimators_} trees")
    print("          [DONE]\n")
    
    # ========== 5. 在验证集上评估 ==========
    print("[Step 5/7] Evaluating on validation set...")
    
    # 预测（变换空间）
    y_val_pred_transformed = model.predict(X_val_scaled)
    
    print(f"          Validation samples: {len(X_val)}")
    print(f"          Predictions (transformed): {y_val_pred_transformed.min():.4f} - {y_val_pred_transformed.max():.4f}")
    
    # 检查是否超出训练范围
    n_below = (y_val_pred_transformed < y_transformed.min()).sum()
    n_above = (y_val_pred_transformed > y_transformed.max()).sum()
    print(f"          Below training range: {n_below}, Above: {n_above}")
    
    # 反变换到原始空间
    y_val_pred = preprocessor.inverse_transform_target(y_val_pred_transformed).flatten()
    
    # 计算指标
    val_metrics = evaluate_metrics(y_val, y_val_pred, metrics=['MAPE', 'RMSE', 'MAE', 'MPE', 'NRMSE'])
    
    print(f"\n          Validation Results:")
    print(f"          {format_metrics(val_metrics, percentage=True)}")
    logger.info(f"Validation Results: {format_metrics(val_metrics, percentage=True)}")
    print("          [DONE]\n")
    
    # ========== 6. 在测试集上最终评估 ==========
    print("[Step 6/7] Evaluating on test set...")
    
    print(f"          Test samples: {len(X_test)}")
    
    # 预测（变换空间）
    y_test_pred_transformed = model.predict(X_test_scaled)
    
    print(f"          Predictions (transformed): {y_test_pred_transformed.min():.4f} - {y_test_pred_transformed.max():.4f}")
    
    # 检查是否超出训练范围
    n_below = (y_test_pred_transformed < y_transformed.min()).sum()
    n_above = (y_test_pred_transformed > y_transformed.max()).sum()
    print(f"          Below training range: {n_below}, Above: {n_above}")
    
    # 反变换到原始空间
    y_test_pred = preprocessor.inverse_transform_target(y_test_pred_transformed).flatten()
    
    # 计算指标
    metrics = evaluate_metrics(y_test, y_test_pred, metrics=['MAPE', 'RMSE', 'MAE', 'MPE', 'NRMSE'])
    
    print(f"\n          Test Results:")
    print(f"          {format_metrics(metrics, percentage=True)}")
    
    logger.info(f"Final Test Results: {format_metrics(metrics, percentage=True)}")
    print("          [DONE]\n")
    
    # ========== 7. 保存结果 ==========
    print("[Step 7/7] Saving results...")
    
    # 保存模型（使用pickle）
    model_path = Path(config['output']['models_dir']) / "centralized_gbr_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'preprocessor': preprocessor,
            'model_params': gbr_params,
            'training_samples': len(X_train),
            'test_metrics': metrics,
            'val_metrics': val_metrics
        }, f)
    
    print(f"          Model saved: {model_path}")
    logger.info(f"Model saved to {model_path}")
    
    # 保存预测结果
    predictions_df = pd.DataFrame({
        'True_Value': y_test,
        'Predicted_Value': y_test_pred,
        'Absolute_Error': np.abs(y_test - y_test_pred),
        'Percentage_Error': np.abs((y_test - y_test_pred) / y_test * 100)
    })
    predictions_path = Path(config['output']['base_dir']) / "centralized_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"          Predictions saved: {predictions_path}")
    
    # 保存指标
    results = {
        'scenario': 'A_Centralized_GBR',
        'model_type': 'GradientBoostingRegressor',
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test),
        'n_estimators': gbr_params['n_estimators'],
        'learning_rate': gbr_params['learning_rate'],
        'max_depth': gbr_params['max_depth'],
        'val_mape': val_metrics['MAPE'],
        'val_rmse': val_metrics['RMSE'],
        'val_mae': val_metrics['MAE'],
        'val_mpe': val_metrics['MPE'],
        'val_nrmse': val_metrics['NRMSE'],
        'test_mape': metrics['MAPE'],
        'test_rmse': metrics['RMSE'],
        'test_mae': metrics['MAE'],
        'test_mpe': metrics['MPE'],
        'test_nrmse': metrics['NRMSE']
    }
    
    results_path = Path(config['output']['base_dir']) / "centralized_results.csv"
    save_results(results, results_path, format='csv')
    print(f"          Results saved: {results_path}")
    logger.info(f"Results saved to {results_path}")
    print("          [DONE]\n")
    
    # ========== 总结 ==========
    print("=" * 80)
    print("SCENARIO A COMPLETED!")
    print("=" * 80)
    print(f"\nFinal Test Performance:")
    print(f"  MAPE:  {metrics['MAPE']*100:.2f}%")
    print(f"  RMSE:  ${metrics['RMSE']:,.2f}")
    print(f"  MAE:   ${metrics['MAE']:,.2f}")
    print(f"  MPE:   {metrics['MPE']*100:.2f}%")
    print(f"  NRMSE: {metrics['NRMSE']*100:.2f}%")
    print(f"\nValidation Performance:")
    print(f"  MAPE: {val_metrics['MAPE']*100:.2f}%")
    print(f"  RMSE: ${val_metrics['RMSE']:,.2f}")
    print(f"\nThis is the UPPER BOUND (performance with all data, no privacy)")
    print("Model: GradientBoostingRegressor (sklearn)")
    print("=" * 80)
    
    logger.info("=" * 50)
    logger.info("Scenario A completed successfully")
    logger.info(f"Test MAPE: {metrics['MAPE']*100:.2f}%, RMSE: ${metrics['RMSE']:,.2f}")
    logger.info("=" * 50)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Scenario A failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
