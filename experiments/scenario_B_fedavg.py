"""
场景B: FedAvg基线
使用与场景C完全相同的数据源、划分方式和预处理流程，
固定使用纯FedAvg聚合策略（size_only, fedprox_mu=0），
作为场景C（MAS-FL-LLM）的对照基线。

与场景C的唯一差异：
- 无FedProx近端正则化（mu=0.0）
- 固定size_only聚合策略（无动态切换）
- 无LLM决策（无学习率/epoch动态调整）
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from src.utils import load_config, set_seed, setup_logger, get_device, save_results
from src.data_preprocessing import load_federated_datasets
from src.models import CostEstimationMLP, save_model
from src.federated_learning.mas_agents import LocalAgent, CentralAgent


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="场景B: FedAvg基线"
    )
    parser.add_argument(
        "--num_rounds", type=int, default=None,
        help="联邦学习轮数（默认使用config中的值）"
    )
    parser.add_argument(
        "--fedprox_mu", type=float, default=None,
        help="FedProx正则化系数（默认使用config中的值，0.0表示纯FedAvg）"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="随机种子（默认使用config中的值）"
    )
    parser.add_argument(
        "--strategy", type=str, default=None,
        choices=["size_only", "perf_only", "hybrid", "fairness_clip"],
        help="聚合策略（默认使用config中的值）"
    )
    parser.add_argument(
        "--output_prefix", type=str, default=None,
        help="输出文件前缀（用于消融实验区分不同配置）"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("SCENARIO B: FedAvg BASELINE")
    print("Using same data pipeline as Scenario C for fair comparison")
    print("=" * 80)

    # ========== 1. 加载配置 ==========
    config = load_config("configs/config.yaml")
    scene_b_cfg = config.get('scene_b', {})

    seed = args.seed if args.seed else config['preprocessing']['random_seed']
    set_seed(seed)

    logger = setup_logger(
        "FedAvg",
        log_file="results/logs/fedavg_training.log",
        console=True
    )
    device = get_device(config['compute']['device'])

    # ========== 2. 加载数据（复用场景C的数据加载逻辑） ==========
    print("\n[Step 1] Loading federated datasets (same split as Scenario C)...")
    client_train_sets, client_val_sets, global_val_set, global_test_set, preprocessor = \
        load_federated_datasets(config, config_key='scene_b')

    # 读取训练参数（命令行参数优先级 > config）
    num_rounds = args.num_rounds if args.num_rounds else scene_b_cfg.get('num_rounds', 20)
    batch_size = scene_b_cfg.get('batch_size', 32)
    lr = scene_b_cfg.get('learning_rate', 0.0005)
    local_epochs = scene_b_cfg.get('local_epochs', 20)
    fedprox_mu = args.fedprox_mu if args.fedprox_mu is not None else scene_b_cfg.get('fedprox_mu', 0.0)
    strategy = args.strategy if args.strategy else scene_b_cfg.get('strategy', 'size_only')
    output_prefix = args.output_prefix if args.output_prefix else "scene_B"

    feature_columns = scene_b_cfg.get('data', {}).get('feature_columns', [])
    input_dim = len(feature_columns)

    print(f"  Rounds: {num_rounds}, LR: {lr}, Local Epochs: {local_epochs}")
    print(f"  FedProx mu: {fedprox_mu} ({'pure FedAvg' if fedprox_mu == 0.0 else 'FedProx enabled'})")
    print(f"  Strategy: {strategy}")
    print(f"  Seed: {seed}")
    print(f"  Clients: {list(client_train_sets.keys())}")

    # ========== 3. 创建全局模型 ==========
    print("\n[Step 2] Creating global model...")
    global_model = CostEstimationMLP(
        input_dim=input_dim,
        hidden_dims=[128, 128, 64, 32],
        output_dim=1,
        activation='gelu',
        dropout=0.1
    ).to(device)
    print(f"  Parameters: {global_model.get_num_parameters():,}")

    # ========== 4. 创建客户端Agent ==========
    print("\n[Step 3] Creating client agents...")
    client_agents = {}
    for client_id in client_train_sets.keys():
        agent = LocalAgent(
            client_id=client_id,
            train_dataset=client_train_sets[client_id],
            val_dataset=client_val_sets[client_id],
            config=config,
            device=device,
            input_dim=input_dim,
            fedprox_mu=fedprox_mu,  # 显式传入0.0，确保纯FedAvg
            preprocessor=preprocessor
        )
        client_agents[client_id] = agent
        print(f"  {client_id}: {agent.n_train_samples} train, {agent.n_val_samples} val")

    # ========== 5. 创建CentralAgent ==========
    global_val_loader = DataLoader(global_val_set, batch_size=batch_size, shuffle=False)
    global_test_loader = DataLoader(global_test_set, batch_size=batch_size, shuffle=False)

    central_agent = CentralAgent(
        global_model=global_model,
        client_agents=client_agents,
        global_val_loader=global_val_loader,
        global_test_loader=global_test_loader,
        preprocessor=preprocessor,
        config=config,
        device=device,
        llm_planner=None
    )

    # ========== 6. 运行训练（使用指定策略） ==========
    results = central_agent.run_training(
        num_rounds=num_rounds,
        strategy_name=strategy,
        lr=lr,
        local_epochs=local_epochs,
        verbose=True
    )

    # ========== 7. 保存结果 ==========
    print("\n[Saving] Results and logs...")

    # 保存模型
    model_path = Path(config['output']['models_dir']) / config['output']['model_names']['fedavg']
    save_model(global_model, model_path, model_info={
        'scenario': 'B_FedAvg',
        'num_rounds': num_rounds,
        'best_round': results['best_round'] + 1,
        'test_metrics': results['test_metrics']
    })
    print(f"  Model saved: {model_path}")

    # 保存训练日志（使用output_prefix区分）
    central_agent.save_training_logs(config['output']['logs_dir'], prefix=output_prefix)

    # 保存CSV结果
    test_m = results['test_metrics']
    result_dict = {
        'scenario': 'B_FedAvg',
        'num_rounds': num_rounds,
        'best_round': results['best_round'] + 1,
        'test_mape': test_m['mape'],
        'test_rmse': test_m['rmse'],
        'test_mae': test_m['mae'],
        'test_mpe': test_m.get('mpe', 0),
        'test_nrmse': test_m.get('nrmse', 0),
        'test_r2': test_m.get('r2', 0),
    }
    results_path = Path(config['output']['base_dir']) / "fedavg_results.csv"
    save_results(result_dict, results_path, format='csv')
    print(f"  Results saved: {results_path}")

    # ========== 总结 ==========
    print("\n" + "=" * 80)
    print("SCENARIO B COMPLETED!")
    print("=" * 80)
    print(f"  Best Round: {results['best_round'] + 1}")
    print(f"  Best Val MAPE: {results['best_val_mape'] * 100:.2f}%")
    print(f"  Test MAPE:  {test_m['mape'] * 100:.2f}%")
    print(f"  Test RMSE:  ${test_m['rmse']:,.2f}")
    print(f"  Test MAE:   ${test_m['mae']:,.2f}")
    print(f"  Test R2:    {test_m.get('r2', 0):.4f}")
    print("=" * 80)

    logger.info(f"Scenario B completed: Test MAPE={test_m['mape']*100:.2f}%")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Scenario B failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
