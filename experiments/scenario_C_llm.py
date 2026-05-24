"""
场景C: 多智能体联邦学习 + LLM决策 (MAS-FL-LLM)

阶段1: 基础MAS-FL框架
- 实现等价FedAvg的聚合策略

阶段2: 动态聚合策略
- 支持多种聚合策略：size_only, perf_only, hybrid, fairness_clip

阶段3: LLM智能决策
- 使用DeepSeek API进行策略决策
- LLM根据训练历史自动选择最优策略

使用方法:
    # 阶段2: 使用固定策略
    python experiments/scenario_C_llm.py --strategy hybrid --num_rounds 20 --seed 42
    
    # 阶段2: 比较所有策略
    python experiments/scenario_C_llm.py --compare_strategies --num_rounds 20 --seed 42
    
    # 阶段3: 启用LLM决策
    python experiments/scenario_C_llm.py --use_llm --num_rounds 20 --seed 42
"""

import sys
import argparse
from pathlib import Path
import os

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader

from src.utils import load_config, set_seed, setup_logger, get_device, save_results
from src.data_preprocessing import load_federated_datasets_for_scene_c
from src.models import CostEstimationMLP, save_model
from src.federated_learning.mas_agents import LocalAgent, CentralAgent


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="场景C: 多智能体联邦学习 + LLM决策"
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=20,
        help="联邦学习轮数（默认20轮）"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=["size_only", "perf_only", "hybrid", "fairness_clip"],
        help="聚合策略（不使用--use_llm时生效）"
    )
    parser.add_argument(
        "--compare_strategies",
        action="store_true",
        default=False,
        help="比较所有策略的性能"
    )
    parser.add_argument(
        "--use_llm",
        action="store_true",
        default=False,
        help="启用LLM决策（阶段3）"
    )
    parser.add_argument(
        "--llm_provider",
        type=str,
        choices=["deepseek", "qwen", "openai"],
        help="选择LLM提供商（默认使用config中的default_provider）"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="学习率（默认使用配置文件中的值）"
    )
    parser.add_argument(
        "--local_epochs",
        type=int,
        default=None,
        help="每轮本地训练的epoch数"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="批量大小"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="是否打印详细信息"
    )
    return parser.parse_args()


def create_llm_planner(config, log_dir, provider_override=None):
    """
    创建LLM规划器
    
    Args:
        config: 配置字典
        log_dir: 日志目录
        provider_override: 可选，命令行传入的LLM提供商
        
    Returns:
        LLMPlanner实例
    """
    from src.federated_learning.llm_planner import LLMClient, LLMPlanner
    
    llm_config = config.get('scene_c', {}).get('llm', {})
    providers = llm_config.get('providers', {})
    provider = provider_override or llm_config.get('default_provider') or (list(providers.keys())[0] if providers else None)
    if provider is None or provider not in providers:
        raise ValueError("未找到有效的LLM提供商配置，请检查configs/config.yaml中的scene_c.llm.providers")
    provider_cfg = providers[provider]
    
    # 从环境变量优先读取API Key
    api_key_env = provider_cfg.get('api_key_env')
    api_key = os.getenv(api_key_env, '') if api_key_env else ''
    if not api_key:
        api_key = provider_cfg.get('api_key', '')
    if not api_key:
        raise ValueError(f"未找到{provider}的API Key，请设置环境变量 {api_key_env}")
    
    # 创建LLM客户端
    llm_client = LLMClient(
        api_key=api_key,
        model_name=provider_cfg.get('model_name'),
        base_url=provider_cfg.get('base_url'),
        timeout=llm_config.get('timeout', 60)
    )
    
    # 创建LLM规划器
    llm_planner = LLMPlanner(
        config=config,
        llm_client=llm_client,
        log_dir=log_dir
    )
    
    return llm_planner


def run_with_llm(args, config, client_train_sets, client_val_sets, 
                 global_val_set, global_test_set, preprocessor, logger, device, input_dim):
    """
    运行LLM引导的训练（阶段3）
    """
    print("\n" + "=" * 80)
    print("SCENARIO C: MAS-FL with LLM Decision (Phase 3)")
    print("=" * 80)
    print("\nLLM will analyze training history and choose optimal strategy each round.")
    print()
    
    # 创建全局模型
    global_model = CostEstimationMLP(
        input_dim=input_dim,
        hidden_dims=[128, 128, 64, 32],
        output_dim=1,
        activation='gelu',
        dropout=0.1
    ).to(device)
    
    print(f"[Model] Architecture: 10 -> [128, 128, 64, 32] -> 1")
    print(f"[Model] Parameters: {global_model.get_num_parameters():,}")
    
    # 创建LocalAgent
    client_ids = list(client_train_sets.keys())
    client_agents = {}
    for client_id in client_ids:
        agent = LocalAgent(
            client_id=client_id,
            train_dataset=client_train_sets[client_id],
            val_dataset=client_val_sets[client_id],
            config=config,
            device=device,
            input_dim=input_dim,
            preprocessor=preprocessor
        )
        client_agents[client_id] = agent
        print(f"[Agent] {client_id}: {agent.n_train_samples} train, {agent.n_val_samples} val")
    
    # 创建DataLoader
    global_val_loader = DataLoader(global_val_set, batch_size=args.batch_size, shuffle=False)
    global_test_loader = DataLoader(global_test_set, batch_size=args.batch_size, shuffle=False)
    
    # 创建LLM规划器
    print("\n[LLM] Initializing LLM Planner...")
    llm_planner = create_llm_planner(config, config['output']['logs_dir'], provider_override=args.llm_provider)
    active_provider = args.llm_provider or config['scene_c']['llm'].get('default_provider')
    print(f"[LLM] Provider: {active_provider}")
    
    # 创建中心智能体
    central_agent = CentralAgent(
        global_model=global_model,
        client_agents=client_agents,
        global_val_loader=global_val_loader,
        global_test_loader=global_test_loader,
        preprocessor=preprocessor,
        config=config,
        device=device,
        llm_planner=llm_planner
    )
    
    # 获取训练参数
    scene_c_config = config.get('scene_c', {})
    base_lr = args.learning_rate if args.learning_rate else scene_c_config.get(
        'learning_rate', config['federated_learning']['client']['learning_rate'])
    base_local_epochs = args.local_epochs if args.local_epochs else scene_c_config.get(
        'local_epochs', config['federated_learning']['client']['local_epochs'])
    
    print(f"\n[Training] Base LR: {base_lr}, Base Epochs: {base_local_epochs}")
    print(f"[Training] Rounds: {args.num_rounds}")
    
    logger.info(f"Starting LLM-guided training: {args.num_rounds} rounds")
    
    # 运行LLM引导的训练
    results = central_agent.run_training_with_llm(
        num_rounds=args.num_rounds,
        base_lr=base_lr,
        base_local_epochs=base_local_epochs,
        verbose=args.verbose
    )

    # 偏差校正：在验证集上校准，然后在测试集上应用
    print("\n[Bias Correction] Calibrating on validation set...")
    central_agent.calibrate_bias()

    test_metrics_corrected = central_agent.evaluate_global(
        data_loader=global_test_loader, apply_bias_correction=True
    )
    print(f"\n[Bias Correction] Test Results (after correction):")
    print(f"  MAPE:  {test_metrics_corrected['mape'] * 100:.2f}%")
    print(f"  RMSE:  ${test_metrics_corrected['rmse']:,.2f}")
    print(f"  MAE:   ${test_metrics_corrected['mae']:,.2f}")
    print(f"  MPE:   {test_metrics_corrected['mpe'] * 100:.2f}%")
    print(f"  R2:    {test_metrics_corrected['r2']:.4f}")

    # 将校正后指标附加到结果中
    results['test_metrics_bias_corrected'] = test_metrics_corrected
    results['bias_correction_value'] = central_agent.bias_correction_value
    
    # 保存结果
    # 注意：此时 global_model 已经被 CentralAgent 回滚到 best checkpoint
    print("\n[Saving] Results...")
    print(f"  Saving best checkpoint (Round {results['best_round'] + 1}, Val MAPE: {results['best_val_mape']*100:.2f}%)")
    
    model_path = Path(config['output']['models_dir']) / "scenario_C_llm_model.pt"
    save_model(global_model, model_path, model_info={
        'scenario': 'C_MAS_FL_LLM',
        'phase': 3,
        'num_rounds': args.num_rounds,
        'best_round': results['best_round'] + 1,
        'best_val_mape': results['best_val_mape'],
        'test_metrics': results['test_metrics'],
        'note': 'This model is the best checkpoint based on validation MAPE'
    })
    print(f"  Model saved: {model_path}")
    
    log_paths = central_agent.save_training_logs(config['output']['logs_dir'])
    print(f"  Logs: {config['output']['logs_dir']}")

    test_m = results['test_metrics']
    test_mc = results.get('test_metrics_bias_corrected', test_m)
    result_dict = {
        'scenario': 'C_MAS_FL_LLM',
        'num_rounds': args.num_rounds,
        'best_round': results['best_round'] + 1,
        'test_mape': test_m['mape'],
        'test_rmse': test_m['rmse'],
        'test_mae': test_m['mae'],
        'test_mpe': test_m.get('mpe', 0),
        'test_nrmse': test_m.get('nrmse', 0),
        'test_r2': test_m.get('r2', 0),
        'test_mape_corrected': test_mc['mape'],
        'test_rmse_corrected': test_mc['rmse'],
        'test_mae_corrected': test_mc['mae'],
        'test_mpe_corrected': test_mc.get('mpe', 0),
        'test_r2_corrected': test_mc.get('r2', 0),
        'bias_correction_value': results.get('bias_correction_value', 0),
    }
    results_path = Path(config['output']['base_dir']) / "scenario_c_results.csv"
    save_results(result_dict, results_path, format='csv')
    print(f"  Results saved: {results_path}")
    
    logger.info(f"LLM-guided training completed. Test MAPE: {results['test_metrics']['mape']*100:.2f}%")
    
    return results


def run_single_strategy(strategy_name, config, client_train_sets, client_val_sets,
                        global_val_set, global_test_set, preprocessor, args, logger, device, input_dim):
    """运行单个策略的训练"""
    
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy_name}")
    print(f"{'='*60}")
    
    seed = args.seed if args.seed else config['preprocessing']['random_seed']
    set_seed(seed)
    
    global_model = CostEstimationMLP(
        input_dim=input_dim, hidden_dims=[128, 128, 64, 32],
        output_dim=1, activation='gelu', dropout=0.1
    ).to(device)
    
    client_agents = {}
    for client_id in client_train_sets.keys():
        client_agents[client_id] = LocalAgent(
            client_id=client_id,
            train_dataset=client_train_sets[client_id],
            val_dataset=client_val_sets[client_id],
            config=config, device=device, input_dim=input_dim,
            preprocessor=preprocessor
        )
    
    global_val_loader = DataLoader(global_val_set, batch_size=args.batch_size, shuffle=False)
    global_test_loader = DataLoader(global_test_set, batch_size=args.batch_size, shuffle=False)
    
    central_agent = CentralAgent(
        global_model=global_model,
        client_agents=client_agents,
        global_val_loader=global_val_loader,
        global_test_loader=global_test_loader,
        preprocessor=preprocessor,
        config=config, device=device, llm_planner=None
    )
    
    scene_c_config = config.get('scene_c', {})
    lr = args.learning_rate if args.learning_rate else scene_c_config.get(
        'learning_rate', config['federated_learning']['client']['learning_rate'])
    local_epochs = args.local_epochs if args.local_epochs else scene_c_config.get(
        'local_epochs', config['federated_learning']['client']['local_epochs'])
    
    results = central_agent.run_training(
        num_rounds=args.num_rounds,
        strategy_name=strategy_name,
        lr=lr, local_epochs=local_epochs,
        verbose=args.verbose
    )
    
    central_agent.save_training_logs(config['output']['logs_dir'])
    
    # 注意：此时 global_model 已经被 CentralAgent 回滚到 best checkpoint
    print(f"\n[Saving] Best checkpoint (Round {results['best_round'] + 1}, Val MAPE: {results['best_val_mape']*100:.2f}%)")
    
    model_path = Path(config['output']['models_dir']) / f"scenario_C_{strategy_name}_model.pt"
    save_model(global_model, model_path, model_info={
        'scenario': 'C_MAS_FL', 'phase': 2, 'strategy': strategy_name,
        'best_round': results['best_round'] + 1,
        'best_val_mape': results['best_val_mape'],
        'test_metrics': results['test_metrics'],
        'note': 'This model is the best checkpoint based on validation MAPE'
    })
    
    logger.info(f"Strategy {strategy_name}: Test MAPE={results['test_metrics']['mape']*100:.2f}%")
    
    return results


def main():
    """场景C主函数"""
    args = parse_args()
    
    print("=" * 80)
    print("SCENARIO C: MAS-FL (Multi-Agent System Federated Learning)")
    print("=" * 80)
    
    # 加载配置
    config = load_config("configs/config.yaml")
    scene_c_data = config.get('scene_c', {}).get('data', {})
    feature_columns = scene_c_data.get('feature_columns') or config.get('data', {}).get('feature_columns', [])
    if not feature_columns:
        raise ValueError("feature_columns not configured; cannot determine input_dim")
    input_dim = len(feature_columns)
    seed = args.seed if args.seed else config['preprocessing']['random_seed']
    set_seed(seed)
    
    logger = setup_logger("MAS-FL", log_file="results/logs/scenario_C.log", console=True)
    device = get_device(config['compute']['device'])
    
    print(f"\n[Config] Seed: {seed}, Device: {device}")
    print(f"[Config] Use LLM: {args.use_llm}")
    
    # 加载数据
    print("\n[Data] Loading datasets...")
    client_train_sets, client_val_sets, global_val_set, global_test_set, preprocessor = \
        load_federated_datasets_for_scene_c(config)
    
    total_train = sum(len(d) for d in client_train_sets.values())
    print(f"[Data] Train: {total_train}, Val: {len(global_val_set)}, Test: {len(global_test_set)}")
    
    # 根据模式运行
    if args.use_llm:
        # 阶段3: LLM决策
        results = run_with_llm(
            args, config, client_train_sets, client_val_sets,
            global_val_set, global_test_set, preprocessor, logger, device, input_dim
        )
        
    elif args.compare_strategies:
        # 比较所有策略
        print("\n[Mode] Comparing all strategies...")
        strategies = ["size_only", "perf_only", "hybrid", "fairness_clip"]
        all_results = {}
        
        for strategy in strategies:
            results = run_single_strategy(
                strategy, config, client_train_sets, client_val_sets,
                global_val_set, global_test_set, preprocessor, args, logger, device, input_dim
            )
            all_results[strategy] = results
        
        # 打印比较结果
        print("\n" + "=" * 80)
        print("STRATEGY COMPARISON")
        print("=" * 80)
        print(f"\n{'Strategy':<18} {'Best Round':<12} {'Val MAPE':<12} {'Test MAPE':<12}")
        print("-" * 54)
        
        best_strategy, best_mape = None, float('inf')
        for strategy, results in all_results.items():
            test_mape = results['test_metrics']['mape']
            print(f"{strategy:<18} {results['best_round']+1:<12} "
                  f"{results['best_val_mape']*100:.2f}%{'':<6} {test_mape*100:.2f}%")
            if test_mape < best_mape:
                best_mape, best_strategy = test_mape, strategy
        
        print("-" * 54)
        print(f"\nBest: {best_strategy} (Test MAPE: {best_mape*100:.2f}%)")
        
        return all_results
        
    else:
        # 单策略训练
        strategy = args.strategy if args.strategy else config.get('scene_c', {}).get('default_strategy', 'size_only')
        results = run_single_strategy(
            strategy, config, client_train_sets, client_val_sets,
            global_val_set, global_test_set, preprocessor, args, logger, device, input_dim
        )
    
    # 打印最终结果
    print("\n" + "=" * 80)
    print("COMPLETED!")
    print("=" * 80)
    test_metrics = results['test_metrics']
    print(f"\nTest Results:")
    print(f"  MAPE:  {test_metrics['mape'] * 100:.2f}%")
    print(f"  RMSE:  ${test_metrics['rmse']:,.2f}")
    print(f"  MAE:   ${test_metrics['mae']:,.2f}")
    print(f"  R2:    {test_metrics.get('r2', 0):.4f}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
