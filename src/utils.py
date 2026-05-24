"""
工具函数模块
提供数据加载、评估指标计算、日志配置等通用功能
"""

import os
import logging
import random
import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional, Union


def load_config(config_path: str = "configs/config.yaml") -> Dict:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42):
    """
    设置所有随机种子以保证可复现性
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    配置日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径
        level: 日志级别
        console: 是否输出到控制台
        
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除已存在的处理器
    logger.handlers.clear()
    
    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 文件处理器
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    # 控制台处理器
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger


def load_data(
    data_path: Union[str, Path],
    feature_columns: Optional[list] = None,
    target_column: str = "ContAmnt"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    加载数据集
    
    Args:
        data_path: 数据文件路径
        feature_columns: 特征列名列表，None则使用除目标列外的所有列
        target_column: 目标列名
        
    Returns:
        (特征DataFrame, 目标Series)
    """
    # 加载数据
    df = pd.read_csv(data_path)
    
    # 如果没有指定特征列，使用除目标列外的所有列
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]
    
    # 分离特征和目标
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    return X, y


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    计算平均绝对百分比误差 (MAPE)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        epsilon: 防止除零的小值
        
    Returns:
        MAPE值 (0-1之间，需要乘100得到百分比)
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # 避免除以零
    mask = np.abs(y_true) > epsilon
    
    if mask.sum() == 0:
        return float('inf')
    
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))
    return mape


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算均方根误差 (RMSE)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        RMSE值
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算平均绝对误差 (MAE)
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        MAE值
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mae = np.mean(np.abs(y_true - y_pred))
    return mae


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算R²决定系数
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        R²值
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    r2 = 1 - (ss_res / ss_tot)
    return r2


def compute_mpe(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    计算平均百分比误差 (MPE) - 衡量预测偏见
    
    正值表示系统性高估（预测偏高），负值表示系统性低估（预测偏低）
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        epsilon: 防止除零的小值
        
    Returns:
        MPE值 (0-1之间，需要乘100得到百分比)
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # 避免除以零
    mask = np.abs(y_true) > epsilon
    
    if mask.sum() == 0:
        return float('inf')
    
    # Positive MPE means systematic overestimation (y_pred > y_true).
    mpe = np.mean((y_pred[mask] - y_true[mask]) / y_true[mask])
    return mpe


def compute_nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算归一化均方根误差 (NRMSE) - 与尺度无关的误差
    
    通过除以数据范围进行归一化，便于跨数据集比较
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        NRMSE值 (0-1之间，需要乘100得到百分比)
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    rmse = compute_rmse(y_true, y_pred)
    y_range = np.max(y_true) - np.min(y_true)
    
    if y_range == 0:
        return float('inf')
    
    nrmse = rmse / y_range
    return nrmse


def evaluate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: list = ['MAPE', 'RMSE', 'MAE', 'MPE', 'NRMSE']
) -> Dict[str, float]:
    """
    计算多个评估指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        metrics: 要计算的指标列表，默认包含MAPE、RMSE、MAE、MPE、NRMSE
        
    Returns:
        指标字典
    """
    results = {}
    
    metric_functions = {
        'MAPE': compute_mape,
        'RMSE': compute_rmse,
        'MAE': compute_mae,
        'MPE': compute_mpe,
        'NRMSE': compute_nrmse,
        'R2': compute_r2  # 保留R2函数以兼容旧代码，但不在默认列表中
    }
    
    for metric in metrics:
        if metric in metric_functions:
            results[metric] = metric_functions[metric](y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    return results


def get_device(device: str = "auto") -> torch.device:
    """
    获取计算设备
    
    Args:
        device: 设备类型 ('auto', 'cpu', 'cuda', 'cuda:0'等)
        
    Returns:
        torch.device对象
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return torch.device(device)


def save_results(
    results: Dict,
    save_path: Union[str, Path],
    format: str = 'csv'
):
    """
    保存结果到文件
    
    Args:
        results: 结果字典
        save_path: 保存路径
        format: 保存格式 ('csv', 'json')
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if format == 'csv':
        df = pd.DataFrame([results])
        df.to_csv(save_path, index=False)
    elif format == 'json':
        import json
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
    else:
        raise ValueError(f"Unsupported format: {format}")


def format_metrics(metrics: Dict[str, float], percentage: bool = False) -> str:
    """
    格式化输出评估指标
    
    Args:
        metrics: 指标字典
        percentage: 是否显示为百分比（已废弃，自动处理）
        
    Returns:
        格式化的字符串
    """
    lines = []
    for metric, value in metrics.items():
        if metric in ['MAPE', 'MPE', 'NRMSE']:
            # 百分比指标
            lines.append(f"{metric}: {value*100:.2f}%")
        elif metric in ['RMSE', 'MAE']:
            # 美元金额指标
            lines.append(f"{metric}: ${value:,.2f}")
        else:
            # 其他指标（如R2）
            lines.append(f"{metric}: {value:.4f}")
    
    return " | ".join(lines)


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("Testing utils module")
    print("=" * 60)
    
    # 测试随机种子
    set_seed(42)
    print("[PASS] Random seed set successfully")
    
    # 测试日志
    logger = setup_logger("test", console=True)
    logger.info("Logger test")
    print("[PASS] Logger working properly")
    
    # 测试评估指标
    y_true = np.array([1000000, 2000000, 3000000, 4000000, 5000000])
    y_pred = np.array([1100000, 1900000, 3200000, 3800000, 5200000])
    
    metrics = evaluate_metrics(y_true, y_pred)
    print("\nMetrics test:")
    print(format_metrics(metrics, percentage=True))
    print("\n[PASS] All utils functions tested successfully")

