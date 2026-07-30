"""
场景A-Prime: 神经网络集中式基线
在统一全局训练切分上训练PyTorch MLP，建立集中式神经网络对照
作为联邦学习场景（B和C）的公平对比基准
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils import (
    load_config, set_seed, setup_logger,
    evaluate_metrics, save_results, format_metrics, get_device,
    compute_mape
)
from src.data_preprocessing import (
    create_dataloader, load_centralized_datasets
)
from src.models import (
    CostEstimationMLP, evaluate_model, save_model, snapshot_model_state
)
from src.federated_learning.mas_agents import (
    apply_mpe_bias_correction, build_prediction_frame
)
from src.experiment_names import experiment_display_name


def parse_args():
    parser = argparse.ArgumentParser(
        description="ANN传统神经网络"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Defaults to preprocessing.random_seed in configs/config.yaml.",
    )
    return parser.parse_args()


def main():
    """ANN传统神经网络主函数"""
    args = parse_args()
    
    print("=" * 80)
    print("ANN传统神经网络")
    print("=" * 80)
    print("\nTraining PyTorch MLP on unified global training split")
    print("Best config: [64,32], dropout=0.1, lr=0.001\n")
    
    # ========== 1. 加载配置 ==========
    print("[Step 1/6] Loading configuration...")
    config = load_config("configs/config.yaml")
    
    seed = args.seed if args.seed is not None else config['preprocessing']['random_seed']
    set_seed(seed)
    print(f"          Random seed: {seed}")
    
    logger = setup_logger(
        "Centralized_NN",
        log_file="results/logs/centralized_nn_training.log",
        console=True
    )
    logger.info("Starting ANN传统神经网络")
    
    device = get_device(config['compute']['device'])
    print(f"          Device: {device}")
    print("          [DONE]\n")
    
    # ========== 2. 加载和预处理数据 ==========
    print("[Step 2/6] Loading and preprocessing data...")
    split_data = load_centralized_datasets(config, config_key='scene_c')
    X_train = split_data["X_train"]
    X_scaled = split_data["X_train_scaled"]
    y_transformed = split_data["y_train_transformed"]
    X_val_scaled = split_data["X_val_scaled"]
    y_val_transformed = split_data["y_val_transformed"]
    X_test_scaled = split_data["X_test_scaled"]
    y_test_transformed = split_data["y_test_transformed"]
    preprocessor = split_data["preprocessor"]

    print(f"          Train/Val/Test samples: {len(split_data['X_train'])}/{len(split_data['X_val'])}/{len(split_data['X_test'])}")
    print(f"          Training scope: {split_data['training_scope']}")
    print(f"          Transformed target range: {y_transformed.min():.4f} - {y_transformed.max():.4f}")
    print(f"          Transform method: power_0.25 (Y = ContAmnt^0.25)")
    print("          [DONE]\n")
    
    # ========== 3. 创建DataLoader ==========
    print("[Step 3/6] Creating DataLoaders...")
    
    # 升级配置（更强的网络架构和训练策略）
    best_config = {
        'hidden_dims': [128, 128, 64, 32],  # 更深的网络结构
        'dropout': 0.1,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,  # 添加 L2 正则化
        'batch_size': 32,
        'epochs': 800,  # 允许训练更久
        'patience': 80  # 增加早停耐心值
    }
    
    train_loader = create_dataloader(
        X_scaled, y_transformed,
        batch_size=best_config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    val_loader = create_dataloader(
        X_val_scaled,
        y_val_transformed,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    print(f"          Batch size: {best_config['batch_size']}")
    print(f"          Validation samples: {len(val_loader.dataset)}")
    print("          [DONE]\n")
    
    # ========== 4. 训练模型 ==========
    print("[Step 4/6] Training model...")
    
    model = CostEstimationMLP(
        input_dim=10,
        hidden_dims=best_config['hidden_dims'],
        output_dim=1,
        activation='gelu',  # 使用 GELU 激活函数
        dropout=best_config['dropout']
    ).to(device)
    
    print(f"          Architecture: 10 -> {best_config['hidden_dims']} -> 1")
    print(f"          Dropout: {best_config['dropout']}")
    print(f"          Parameters: {model.get_num_parameters():,}")
    print(f"          Learning rate: {best_config['learning_rate']}")
    print(f"          Max epochs: {best_config['epochs']}")
    print()
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=best_config['learning_rate'],
        weight_decay=best_config['weight_decay']
    )
    criterion = nn.MSELoss()
    
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10
    )
    
    best_val_mape = float('inf')
    best_epoch = 0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(best_config['epochs']):
        # 训练
        model.train()
        train_losses = []
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # 验证：在原始空间计算 MAPE（与联邦场景 B/C 的 checkpoint 选择对齐）
        _, y_val_pred_trans, y_val_true_trans = evaluate_model(
            model, val_loader, device, return_predictions=True
        )
        y_val_pred_orig = preprocessor.inverse_transform_target(y_val_pred_trans).flatten()
        y_val_true_orig = preprocessor.inverse_transform_target(y_val_true_trans).flatten()
        val_mape = compute_mape(y_val_true_orig, y_val_pred_orig)

        # 更新学习率调度器
        scheduler.step(val_mape)

        # 早停检查（基于原始空间 MAPE，与联邦场景一致）
        if val_mape < best_val_mape:
            best_val_mape = val_mape
            best_epoch = epoch
            patience_counter = 0
            best_model_state = snapshot_model_state(model)
        else:
            patience_counter += 1

        # 打印进度
        if (epoch + 1) % 20 == 0:
            msg = f"Epoch [{epoch+1}/{best_config['epochs']}] Train Loss: {avg_train_loss:.6f}, Val MAPE: {val_mape*100:.2f}%"
            print(f"          {msg}")
            logger.info(msg)
        
        # 早停
        if patience_counter >= best_config['patience']:
            print(f"\n          Early stopping at epoch {epoch+1}")
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"\n          Training completed!")
    print(f"          Best epoch: {best_epoch + 1}")
    print(f"          Best val MAPE: {best_val_mape*100:.2f}%")
    print("          [DONE]\n")
    
    # ========== 5. 在测试集上评估 ==========
    print("[Step 5/6] Evaluating on test set...")
    
    test_loader = create_dataloader(
        X_test_scaled,
        y_test_transformed,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    print(f"          Test samples: {len(test_loader.dataset)}")
    
    # 获取预测
    test_loss, y_pred_log, y_true_log = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        return_predictions=True
    )
    
    print(f"          Predictions (transformed): {y_pred_log.min():.4f} - {y_pred_log.max():.4f}")
    
    # 检查超出范围
    n_below = (y_pred_log < y_transformed.min()).sum()
    n_above = (y_pred_log > y_transformed.max()).sum()
    print(f"          Below range: {n_below}, Above range: {n_above}")
    
    # 反变换到原始空间 (ContAmnt = Y^4)
    y_pred = preprocessor.inverse_transform_target(y_pred_log).flatten()
    y_true = preprocessor.inverse_transform_target(y_true_log).flatten()
    
    # 计算指标
    metrics = evaluate_metrics(y_true, y_pred, metrics=['MAPE', 'RMSE', 'MAE', 'MPE', 'NRMSE', 'R2'])

    _, y_val_pred_log, y_val_true_log = evaluate_model(
        model=model,
        data_loader=val_loader,
        device=device,
        return_predictions=True
    )
    y_val_pred = preprocessor.inverse_transform_target(y_val_pred_log).flatten()
    y_val_true = preprocessor.inverse_transform_target(y_val_true_log).flatten()
    y_pred_corrected, bias_correction_value = apply_mpe_bias_correction(
        val_true=y_val_true,
        val_pred=y_val_pred,
        test_pred=y_pred,
    )
    corrected_metrics = evaluate_metrics(
        y_true,
        y_pred_corrected,
        metrics=['MAPE', 'RMSE', 'MAE', 'MPE', 'NRMSE', 'R2']
    )
    
    print(f"\n          Test Results:")
    print(f"          {format_metrics(metrics, percentage=True)}")
    
    logger.info(f"Final Test Results: {format_metrics(metrics, percentage=True)}")
    print("          [DONE]\n")
    
    # ========== 6. 保存结果 ==========
    print("[Step 6/6] Saving results...")
    
    # 保存模型
    model_path = Path(config['output']['models_dir']) / "centralized_nn_model.pth"
    save_model(model, model_path, model_info={
        'scenario': experiment_display_name("A_prime"),
        'training_samples': len(X_train),
        'training_scope': split_data['training_scope'],
        'best_config': best_config,
        'best_epoch': best_epoch + 1,
        'best_val_mape': best_val_mape,
        'test_metrics': metrics
    })
    print(f"          Model saved: {model_path}")
    
    metadata = split_data["test_df"][["Client"]].reset_index(drop=True)
    predictions_df = build_prediction_frame(
        y_true=y_true,
        y_pred=y_pred,
        metadata=metadata,
        thresholds=config.get("thresholds", {}),
        scenario=experiment_display_name("A_prime"),
    )
    predictions_path = Path(config['output']['base_dir']) / "centralized_nn_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"          Predictions saved: {predictions_path}")

    corrected_predictions_df = build_prediction_frame(
        y_true=y_true,
        y_pred=y_pred_corrected,
        metadata=metadata,
        thresholds=config.get("thresholds", {}),
        scenario=f"{experiment_display_name('A_prime')}（偏差校正）",
    )
    corrected_predictions_path = Path(config['output']['base_dir']) / "centralized_nn_predictions_bias_corrected.csv"
    corrected_predictions_df.to_csv(corrected_predictions_path, index=False)
    print(f"          Bias-corrected predictions saved: {corrected_predictions_path}")
    
    # 保存结果
    results = {
        'scenario': experiment_display_name("A_prime"),
        'model_type': 'PyTorch_MLP',
        'training_samples': len(X_train),
        'training_scope': split_data['training_scope'],
        'local_validation_samples_withheld': len(split_data['local_val_df']),
        'test_samples': len(test_loader.dataset),
        'hidden_dims': str(best_config['hidden_dims']),
        'dropout': best_config['dropout'],
        'learning_rate': best_config['learning_rate'],
        'weight_decay': best_config['weight_decay'],
        'batch_size': best_config['batch_size'],
        'best_epoch': best_epoch + 1,
        'best_val_mape': best_val_mape,
        'test_mape': metrics['MAPE'],
        'test_rmse': metrics['RMSE'],
        'test_mae': metrics['MAE'],
        'test_mpe': metrics['MPE'],
        'test_nrmse': metrics['NRMSE'],
        'test_r2': metrics['R2'],
        'test_mape_corrected': corrected_metrics['MAPE'],
        'test_rmse_corrected': corrected_metrics['RMSE'],
        'test_mae_corrected': corrected_metrics['MAE'],
        'test_mpe_corrected': corrected_metrics['MPE'],
        'test_r2_corrected': corrected_metrics['R2'],
        'bias_correction_value': bias_correction_value,
    }
    
    results_path = Path(config['output']['base_dir']) / "centralized_nn_results.csv"
    save_results(results, results_path, format='csv')
    print(f"          Results saved: {results_path}")
    print("          [DONE]\n")
    
    # ========== 总结 ==========
    print("=" * 80)
    print("ANN传统神经网络 COMPLETED!")
    print("=" * 80)
    print(f"\nNeural Network Configuration:")
    print(f"  Architecture: 10 -> {best_config['hidden_dims']} -> 1")
    print(f"  Dropout: {best_config['dropout']}")
    print(f"  Learning Rate: {best_config['learning_rate']}")
    print(f"  Trained Epochs: {best_epoch + 1}")
    
    print(f"\nFinal Test Performance:")
    print(f"  MAPE:  {metrics['MAPE']*100:.2f}%")
    print(f"  RMSE:  ${metrics['RMSE']:,.2f}")
    print(f"  MAE:   ${metrics['MAE']:,.2f}")
    print(f"  MPE:   {metrics['MPE']*100:.2f}%")
    print(f"  NRMSE: {metrics['NRMSE']*100:.2f}%")
    
    print(f"\nNote:")
    print(f"  Use regenerated paper tables for unified cross-experiment comparison.")
    
    print(f"\nThis is the centralized ANN baseline for federated learning comparison")
    print("=" * 80)
    
    logger.info("=" * 50)
    logger.info("ANN传统神经网络 completed successfully")
    logger.info(f"Best NN Test MAPE: {metrics['MAPE']*100:.2f}%")
    logger.info("=" * 50)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] ANN传统神经网络 failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
