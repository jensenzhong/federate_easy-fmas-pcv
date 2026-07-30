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
from src.experiment_names import experiment_display_name


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="多智能体协同联邦学习"
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
        "--temperature",
        type=float,
        default=None,
        help="LLM采样温度；仅在--use_llm时覆盖配置文件中的scene_c.llm.temperature"
    )
    parser.add_argument(
        "--server_optimizer",
        type=str,
        default="fedavg",
        choices=["fedavg", "fedyogi"],
        help="Server-side optimizer for applying the aggregated update."
    )
    parser.add_argument(
        "--server_lr",
        type=float,
        default=0.01,
        help="FedYogi server learning rate."
    )
    parser.add_argument(
        "--server_beta1",
        type=float,
        default=0.9,
        help="FedYogi first-moment decay."
    )
    parser.add_argument(
        "--server_beta2",
        type=float,
        default=0.99,
        help="FedYogi second-moment decay."
    )
    parser.add_argument(
        "--server_tau",
        type=float,
        default=1e-3,
        help="FedYogi denominator stabilizer."
    )
    parser.add_argument(
        "--max_coordinate_step_ratio",
        type=float,
        default=1.0,
        help=(
            "Trust-region ratio for FedYogi coordinates. "
            "1.0 prevents stepping farther than the current weighted-average target; "
            "0 disables the coordinate trust region."
        )
    )
    parser.add_argument(
        "--server_lr_scale",
        type=float,
        default=1.0,
        help="Fixed server learning-rate scale for non-LLM runs."
    )
    parser.add_argument(
        "--update_clip_norm",
        type=float,
        default=None,
        help="Optional L2 clipping norm for server updates."
    )
    parser.add_argument(
        "--strict_no_server_validation",
        action="store_true",
        default=False,
        help="Disable server-side validation during training and use the fixed final-round checkpoint."
    )
    parser.add_argument(
        "--adaptive_mode",
        type=str,
        default="fixed_strategy",
        choices=[
            "fixed_strategy",
            "validation_guided",
            "mas_validation_guided",
            "coherence_guided",
            "llm_generative_coherence",
            "strict_coherence_guided",
            "llm_strict_generative_coherence",
            "llm_validation_preview_generative",
            "validation_preview_gca",
        ],
        help="Adaptive control mode for FedYogi-TR runs."
    )
    parser.add_argument(
        "--candidate_budget",
        type=int,
        default=30,
        help="Maximum number of validation-guided candidates per round."
    )
    parser.add_argument(
        "--weight_grid_step",
        type=float,
        default=0.05,
        help="Client aggregation-weight grid step for candidate generation."
    )
    parser.add_argument(
        "--min_client_weight",
        type=float,
        default=0.05,
        help="Minimum per-client aggregation weight."
    )
    parser.add_argument(
        "--max_client_weight",
        type=float,
        default=0.80,
        help="Maximum per-client aggregation weight."
    )
    parser.add_argument(
        "--selection_epsilon",
        type=float,
        default=0.002,
        help="Minimum validation-score improvement required over conservative candidate."
    )
    parser.add_argument(
        "--llm_score_tolerance",
        type=float,
        default=None,
        help=(
            "Extra validation-score tolerance for MAS candidate choices. "
            "Defaults to selection_epsilon; only applies to mas_validation_guided."
        )
    )
    parser.add_argument(
        "--weight_l1_change_limit",
        type=float,
        default=0.40,
        help="Maximum L1 change in aggregation weights between accepted rounds."
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default=None,
        help="Prefix for result, prediction, and log files."
    )
    parser.add_argument(
        "--method_key",
        type=str,
        default=None,
        help="Canonical method key used for paper-facing names."
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


def create_llm_planner(config, log_dir, provider_override=None, decisions_log_name="scene_C_llm_decisions.jsonl"):
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
        log_dir=log_dir,
        decisions_log_name=decisions_log_name,
    )
    
    return llm_planner


def resolve_method_key(args) -> str:
    if args.method_key:
        return args.method_key
    if args.strict_no_server_validation and args.server_optimizer == "fedyogi":
        return "FEDYOGI_STRICT"
    if args.strict_no_server_validation:
        return "B_STRICT"
    if args.adaptive_mode == "validation_preview_gca":
        return "VP_GCA_FEDYOGI_TR"
    if args.adaptive_mode == "llm_validation_preview_generative":
        return "LLM_VP_GCA_FEDYOGI_TR"
    if args.adaptive_mode == "llm_generative_coherence":
        return "LLM_GCA_FEDYOGI_TR"
    if args.adaptive_mode == "coherence_guided":
        return "COHERENCE_FEDYOGI_TR"
    if args.adaptive_mode == "llm_strict_generative_coherence":
        return "LLM_STRICT_GCA_FEDYOGI_TR"
    if args.adaptive_mode == "strict_coherence_guided":
        return "STRICT_COHERENCE_FEDYOGI_TR"
    if args.adaptive_mode == "mas_validation_guided":
        return "MAS_VG_FEDYOGI_TR"
    if args.adaptive_mode == "validation_guided":
        return "VG_FEDYOGI_TR"
    if args.server_optimizer == "fedyogi" and args.use_llm:
        return "MAS_ADAPTIVE"
    if args.server_optimizer == "fedyogi":
        return "FEDYOGI"
    return "C"


def resolve_output_prefix(args, method_key: str) -> str:
    if args.output_prefix:
        return args.output_prefix
    defaults = {
        "C": "scenario_c",
        "FEDYOGI": "fedyogi",
        "FEDYOGI_STRICT": "fedyogi_strict",
        "B_STRICT": "fedavg_strict",
        "MAS_ADAPTIVE": "mas_adaptive",
        "VG_FEDYOGI_TR": "vg_fedyogi_tr",
        "MAS_VG_FEDYOGI_TR": "mas_vg_fedyogi_tr",
        "COHERENCE_FEDYOGI_TR": "coherence_fedyogi_tr",
        "LLM_GCA_FEDYOGI_TR": "llm_gca_fedyogi_tr",
        "STRICT_COHERENCE_FEDYOGI_TR": "strict_coherence_fedyogi_tr",
        "LLM_STRICT_GCA_FEDYOGI_TR": "llm_strict_gca_fedyogi_tr",
        "VP_GCA_FEDYOGI_TR": "vp_gca_fedyogi_tr",
        "LLM_VP_GCA_FEDYOGI_TR": "llm_vp_gca_fedyogi_tr",
    }
    return defaults.get(method_key, method_key.lower())


def uses_strict_no_server_validation(args) -> bool:
    return bool(args.strict_no_server_validation) or args.adaptive_mode in {
        "strict_coherence_guided",
        "llm_strict_generative_coherence",
    }


def corrected_method_key(method_key: str) -> str:
    return f"{method_key}_bias_corrected"


def build_client_agents(client_train_sets, client_val_sets, config, device, input_dim, preprocessor):
    client_agents = {}
    for client_id in client_train_sets.keys():
        agent = LocalAgent(
            client_id=client_id,
            train_dataset=client_train_sets[client_id],
            val_dataset=client_val_sets[client_id],
            config=config,
            device=device,
            input_dim=input_dim,
            preprocessor=preprocessor,
        )
        client_agents[client_id] = agent
        print(f"[Agent] {client_id}: {agent.n_train_samples} train, {agent.n_val_samples} val")
    return client_agents


def create_central_agent(
    args,
    config,
    client_train_sets,
    client_val_sets,
    global_val_set,
    global_test_set,
    preprocessor,
    device,
    input_dim,
    llm_planner=None,
):
    global_model = CostEstimationMLP(
        input_dim=input_dim,
        hidden_dims=[128, 128, 64, 32],
        output_dim=1,
        activation='gelu',
        dropout=0.1,
    ).to(device)

    print(f"[Model] Architecture: {input_dim} -> [128, 128, 64, 32] -> 1")
    print(f"[Model] Parameters: {global_model.get_num_parameters():,}")

    client_agents = build_client_agents(
        client_train_sets=client_train_sets,
        client_val_sets=client_val_sets,
        config=config,
        device=device,
        input_dim=input_dim,
        preprocessor=preprocessor,
    )
    global_val_loader = DataLoader(global_val_set, batch_size=args.batch_size, shuffle=False)
    global_test_loader = DataLoader(global_test_set, batch_size=args.batch_size, shuffle=False)

    central_agent = CentralAgent(
        global_model=global_model,
        client_agents=client_agents,
        global_val_loader=global_val_loader,
        global_test_loader=global_test_loader,
        preprocessor=preprocessor,
        config=config,
        device=device,
        llm_planner=llm_planner,
        server_optimizer=args.server_optimizer,
        server_lr=args.server_lr,
        server_beta1=args.server_beta1,
        server_beta2=args.server_beta2,
        server_tau=args.server_tau,
        update_clip_norm=args.update_clip_norm,
        max_coordinate_step_ratio=args.max_coordinate_step_ratio,
    )
    return global_model, central_agent, global_val_loader, global_test_loader


def attach_bias_corrected_metrics(central_agent, results, global_test_loader):
    print("\n[Bias Correction] Calibrating on validation set...")
    central_agent.calibrate_bias()

    test_metrics_corrected = central_agent.evaluate_global(
        data_loader=global_test_loader,
        apply_bias_correction=True,
    )
    print("\n[Bias Correction] Test Results (after correction):")
    print(f"  MAPE:  {test_metrics_corrected['mape'] * 100:.2f}%")
    print(f"  RMSE:  ${test_metrics_corrected['rmse']:,.2f}")
    print(f"  MAE:   ${test_metrics_corrected['mae']:,.2f}")
    print(f"  MPE:   {test_metrics_corrected['mpe'] * 100:.2f}%")
    print(f"  R2:    {test_metrics_corrected['r2']:.4f}")

    results['test_metrics_bias_corrected'] = test_metrics_corrected
    results['bias_correction_value'] = central_agent.bias_correction_value
    return results


def save_federated_outputs(
    args,
    config,
    global_model,
    central_agent,
    global_test_loader,
    results,
    method_key: str,
    output_prefix: str,
    logger,
):
    print("\n[Saving] Results...")
    strict_protocol = results.get("checkpoint_policy") == "final_round" or uses_strict_no_server_validation(args)
    if strict_protocol:
        print(
            f"  Saving final-round model (Round {results['best_round'] + 1}, "
            f"client-reported Val MAPE: {results['best_val_mape']*100:.2f}%)"
        )
    else:
        print(
            f"  Saving best checkpoint (Round {results['best_round'] + 1}, "
            f"Val MAPE: {results['best_val_mape']*100:.2f}%)"
        )

    model_path = Path(config['output']['models_dir']) / f"{output_prefix}_model.pt"
    save_model(global_model, model_path, model_info={
        'scenario': experiment_display_name(method_key),
        'method_key': method_key,
        'num_rounds': args.num_rounds,
        'best_round': results['best_round'] + 1,
        'best_val_mape': results['best_val_mape'],
        'server_optimizer': args.server_optimizer,
        'server_lr': args.server_lr,
        'max_coordinate_step_ratio': args.max_coordinate_step_ratio,
        'test_metrics': results['test_metrics'],
        'note': (
            'This model is the fixed final-round checkpoint without server-side validation.'
            if strict_protocol else
            'This model is the best checkpoint based on validation MAPE'
        ),
    })
    print(f"  Model saved: {model_path}")

    central_agent.save_training_logs(config['output']['logs_dir'], prefix=output_prefix)

    test_m = results['test_metrics']
    test_mc = results.get('test_metrics_bias_corrected', test_m)
    result_dict = {
        'scenario': experiment_display_name(method_key),
        'scenario_key': method_key,
        'num_rounds': args.num_rounds,
        'best_round': results['best_round'] + 1,
        'best_val_mape': results['best_val_mape'],
        'server_optimizer': args.server_optimizer,
        'server_lr': args.server_lr,
        'server_beta1': args.server_beta1,
        'server_beta2': args.server_beta2,
        'server_tau': args.server_tau,
        'server_lr_scale': args.server_lr_scale,
        'max_coordinate_step_ratio': args.max_coordinate_step_ratio,
        'update_clip_norm': args.update_clip_norm,
        'adaptive_mode': args.adaptive_mode,
        'strict_no_server_validation': strict_protocol,
        'checkpoint_policy': results.get('checkpoint_policy', 'best_validation'),
        'validation_source': results.get('validation_source', 'server_global'),
        'candidate_budget': args.candidate_budget,
        'weight_grid_step': args.weight_grid_step,
        'min_client_weight': args.min_client_weight,
        'max_client_weight': args.max_client_weight,
        'selection_epsilon': args.selection_epsilon,
        'llm_score_tolerance': args.llm_score_tolerance if args.llm_score_tolerance is not None else args.selection_epsilon,
        'weight_l1_change_limit': args.weight_l1_change_limit,
        'use_llm': args.use_llm,
        'llm_provider': args.llm_provider or config.get('scene_c', {}).get('llm', {}).get('default_provider'),
        'llm_temperature': config.get('scene_c', {}).get('llm', {}).get('temperature'),
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
    results_path = Path(config['output']['base_dir']) / f"{output_prefix}_results.csv"
    save_results(result_dict, results_path, format='csv')
    print(f"  Results saved: {results_path}")

    predictions_path = Path(config['output']['base_dir']) / f"{output_prefix}_predictions.csv"
    central_agent.save_predictions(
        data_loader=global_test_loader,
        output_path=predictions_path,
        scenario=experiment_display_name(method_key),
    )
    corrected_predictions_path = Path(config['output']['base_dir']) / f"{output_prefix}_predictions_bias_corrected.csv"
    central_agent.save_predictions(
        data_loader=global_test_loader,
        output_path=corrected_predictions_path,
        scenario=experiment_display_name(corrected_method_key(method_key)),
        apply_bias_correction=True,
    )

    logger.info(
        f"{experiment_display_name(method_key)} completed. "
        f"Test MAPE: {results['test_metrics']['mape']*100:.2f}%"
    )


def run_with_llm(args, config, client_train_sets, client_val_sets,
                 global_val_set, global_test_set, preprocessor, logger, device, input_dim):
    print("\n" + "=" * 80)
    print(experiment_display_name(resolve_method_key(args)))
    print("=" * 80)
    if args.adaptive_mode == "llm_strict_generative_coherence":
        print("\nLLM will analyze client-reported summaries and update diagnostics only.")
        print("Server-side validation preview is disabled for this mode.")
    else:
        print("\nLLM will analyze validation-only training history and choose each round's controls.")

    if args.temperature is not None:
        config.setdefault('scene_c', {}).setdefault('llm', {})['temperature'] = args.temperature
        print(f"[LLM] Temperature override: {args.temperature}")

    method_key = resolve_method_key(args)
    output_prefix = resolve_output_prefix(args, method_key)
    decisions_log_name = f"{output_prefix}_llm_decisions.jsonl"
    llm_planner = create_llm_planner(
        config,
        config['output']['logs_dir'],
        provider_override=args.llm_provider,
        decisions_log_name=decisions_log_name,
    )
    active_provider = args.llm_provider or config['scene_c']['llm'].get('default_provider')
    print(f"[LLM] Provider: {active_provider}")
    print(f"[LLM] Decision log: {Path(config['output']['logs_dir']) / decisions_log_name}")
    global_model, central_agent, _, global_test_loader = create_central_agent(
        args=args,
        config=config,
        client_train_sets=client_train_sets,
        client_val_sets=client_val_sets,
        global_val_set=global_val_set,
        global_test_set=global_test_set,
        preprocessor=preprocessor,
        device=device,
        input_dim=input_dim,
        llm_planner=llm_planner,
    )

    scene_c_config = config.get('scene_c', {})
    base_lr = args.learning_rate if args.learning_rate else scene_c_config.get(
        'learning_rate', config['federated_learning']['client']['learning_rate'])
    base_local_epochs = args.local_epochs if args.local_epochs else scene_c_config.get(
        'local_epochs', config['federated_learning']['client']['local_epochs'])

    print(f"\n[Training] Base LR: {base_lr}, Base Epochs: {base_local_epochs}")
    print(
        f"[Training] Server Optimizer: {args.server_optimizer}, "
        f"server_lr={args.server_lr}, max_coordinate_step_ratio={args.max_coordinate_step_ratio}"
    )
    print(f"[Training] Rounds: {args.num_rounds}")

    logger.info(f"Starting LLM-guided training: {args.num_rounds} rounds")

    if args.adaptive_mode == "llm_validation_preview_generative":
        results = central_agent.run_training_with_llm_validation_preview_generative(
            num_rounds=args.num_rounds,
            base_lr=base_lr,
            base_local_epochs=base_local_epochs,
            candidate_budget=args.candidate_budget,
            weight_grid_step=args.weight_grid_step,
            min_client_weight=args.min_client_weight,
            max_client_weight=args.max_client_weight,
            selection_epsilon=args.selection_epsilon,
            llm_score_tolerance=args.llm_score_tolerance,
            weight_l1_change_limit=args.weight_l1_change_limit,
            verbose=args.verbose,
        )
    elif args.adaptive_mode == "llm_strict_generative_coherence":
        results = central_agent.run_training_with_llm_strict_generative_coherence(
            num_rounds=args.num_rounds,
            base_lr=base_lr,
            base_local_epochs=base_local_epochs,
            min_client_weight=args.min_client_weight,
            max_client_weight=args.max_client_weight,
            weight_l1_change_limit=args.weight_l1_change_limit,
            verbose=args.verbose,
        )
    elif args.adaptive_mode == "llm_generative_coherence":
        results = central_agent.run_training_with_llm_generative_coherence(
            num_rounds=args.num_rounds,
            base_lr=base_lr,
            base_local_epochs=base_local_epochs,
            min_client_weight=args.min_client_weight,
            max_client_weight=args.max_client_weight,
            weight_l1_change_limit=args.weight_l1_change_limit,
            verbose=args.verbose,
        )
    elif args.adaptive_mode == "mas_validation_guided":
        results = central_agent.run_training_with_mas_validation_guided_adaptation(
            num_rounds=args.num_rounds,
            base_lr=base_lr,
            base_local_epochs=base_local_epochs,
            candidate_budget=args.candidate_budget,
            weight_grid_step=args.weight_grid_step,
            min_client_weight=args.min_client_weight,
            max_client_weight=args.max_client_weight,
            selection_epsilon=args.selection_epsilon,
            llm_score_tolerance=args.llm_score_tolerance,
            weight_l1_change_limit=args.weight_l1_change_limit,
            verbose=args.verbose,
        )
    else:
        results = central_agent.run_training_with_llm(
            num_rounds=args.num_rounds,
            base_lr=base_lr,
            base_local_epochs=base_local_epochs,
            base_server_lr_scale=args.server_lr_scale,
            verbose=args.verbose,
        )

    if uses_strict_no_server_validation(args):
        print("\n[Bias Correction] Skipped for strict no-server-validation protocol.")
    else:
        results = attach_bias_corrected_metrics(central_agent, results, global_test_loader)
    save_federated_outputs(
        args=args,
        config=config,
        global_model=global_model,
        central_agent=central_agent,
        global_test_loader=global_test_loader,
        results=results,
        method_key=method_key,
        output_prefix=output_prefix,
        logger=logger,
    )
    return results


def run_single_strategy(strategy_name, config, client_train_sets, client_val_sets,
                        global_val_set, global_test_set, preprocessor, args, logger, device, input_dim):
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy_name}")
    print(f"Server optimizer: {args.server_optimizer}")
    print(f"{'='*60}")

    seed = args.seed if args.seed else config['preprocessing']['random_seed']
    set_seed(seed)

    if args.adaptive_mode in {
        "mas_validation_guided",
        "llm_generative_coherence",
        "llm_strict_generative_coherence",
        "llm_validation_preview_generative",
    }:
        args.use_llm = True
        return run_with_llm(
            args,
            config,
            client_train_sets,
            client_val_sets,
            global_val_set,
            global_test_set,
            preprocessor,
            logger,
            device,
            input_dim,
        )

    method_key = resolve_method_key(args)
    output_prefix = resolve_output_prefix(args, method_key)
    global_model, central_agent, _, global_test_loader = create_central_agent(
        args=args,
        config=config,
        client_train_sets=client_train_sets,
        client_val_sets=client_val_sets,
        global_val_set=global_val_set,
        global_test_set=global_test_set,
        preprocessor=preprocessor,
        device=device,
        input_dim=input_dim,
        llm_planner=None,
    )

    scene_c_config = config.get('scene_c', {})
    lr = args.learning_rate if args.learning_rate else scene_c_config.get(
        'learning_rate', config['federated_learning']['client']['learning_rate'])
    local_epochs = args.local_epochs if args.local_epochs else scene_c_config.get(
        'local_epochs', config['federated_learning']['client']['local_epochs'])

    if args.adaptive_mode == "strict_coherence_guided":
        results = central_agent.run_training_with_strict_coherence_guided_adaptation(
            num_rounds=args.num_rounds,
            base_lr=lr,
            base_local_epochs=local_epochs,
            min_client_weight=args.min_client_weight,
            max_client_weight=args.max_client_weight,
            verbose=args.verbose,
        )
    elif args.adaptive_mode == "coherence_guided":
        results = central_agent.run_training_with_coherence_guided_adaptation(
            num_rounds=args.num_rounds,
            base_lr=lr,
            base_local_epochs=local_epochs,
            min_client_weight=args.min_client_weight,
            max_client_weight=args.max_client_weight,
            verbose=args.verbose,
        )
    elif args.adaptive_mode == "validation_preview_gca":
        results = central_agent.run_training_with_validation_preview_gca(
            num_rounds=args.num_rounds,
            base_lr=lr,
            base_local_epochs=local_epochs,
            candidate_budget=args.candidate_budget,
            weight_grid_step=args.weight_grid_step,
            min_client_weight=args.min_client_weight,
            max_client_weight=args.max_client_weight,
            selection_epsilon=args.selection_epsilon,
            weight_l1_change_limit=args.weight_l1_change_limit,
            verbose=args.verbose,
        )
    elif args.adaptive_mode == "validation_guided":
        results = central_agent.run_training_with_validation_guided_adaptation(
            num_rounds=args.num_rounds,
            base_lr=lr,
            base_local_epochs=local_epochs,
            candidate_budget=args.candidate_budget,
            weight_grid_step=args.weight_grid_step,
            min_client_weight=args.min_client_weight,
            max_client_weight=args.max_client_weight,
            selection_epsilon=args.selection_epsilon,
            weight_l1_change_limit=args.weight_l1_change_limit,
            verbose=args.verbose,
        )
    elif args.strict_no_server_validation:
        results = central_agent.run_training_strict_final_round(
            num_rounds=args.num_rounds,
            strategy_name=strategy_name,
            lr=lr,
            local_epochs=local_epochs,
            server_lr_scale=args.server_lr_scale,
            verbose=args.verbose
        )
    else:
        results = central_agent.run_training(
            num_rounds=args.num_rounds,
            strategy_name=strategy_name,
            lr=lr,
            local_epochs=local_epochs,
            server_lr_scale=args.server_lr_scale,
            verbose=args.verbose,
        )

    if uses_strict_no_server_validation(args):
        print("\n[Bias Correction] Skipped for strict no-server-validation protocol.")
    else:
        results = attach_bias_corrected_metrics(central_agent, results, global_test_loader)
    save_federated_outputs(
        args=args,
        config=config,
        global_model=global_model,
        central_agent=central_agent,
        global_test_loader=global_test_loader,
        results=results,
        method_key=method_key,
        output_prefix=output_prefix,
        logger=logger,
    )
    return results

def main():
    """场景C主函数"""
    args = parse_args()
    
    print("=" * 80)
    print("多智能体协同联邦学习")
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
    if args.adaptive_mode in {
        "mas_validation_guided",
        "llm_generative_coherence",
        "llm_strict_generative_coherence",
        "llm_validation_preview_generative",
    } and not args.use_llm:
        args.use_llm = True

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
