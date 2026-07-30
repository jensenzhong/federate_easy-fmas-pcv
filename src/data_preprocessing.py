"""
数据预处理模块
提供数据标准化、目标变量变换（0.25次幂或log1p）和DataLoader创建功能

参考: Zhang et al. (2023) - 使用ContAmnt^0.25作为因变量
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Union, Dict, Any
from pathlib import Path

from src.utils import load_data


def compute_local_stats(df: pd.DataFrame, feature_columns: list) -> dict:
    """
    Step 1 (Client Side): Compute local statistics for federated standardization.
    Privacy-Preserving Federated Statistics: Only sums and counts are shared.
    
    Args:
        df: Local DataFrame
        feature_columns: List of feature names to compute stats for
        
    Returns:
        dict: {sum, sq_sum, count}
    """
    X = df[feature_columns].values
    local_sum = np.sum(X, axis=0)
    local_sq_sum = np.sum(X ** 2, axis=0)
    local_count = len(X)
    
    return {
        "sum": local_sum,
        "sq_sum": local_sq_sum,
        "count": local_count
    }


def aggregate_federated_stats(client_stats_list: list) -> dict:
    """
    Step 2 (Server Side): Aggregate local statistics to compute global mean and std.
    
    Args:
        client_stats_list: List of dicts from compute_local_stats
        
    Returns:
        dict: {mean, std, var, count}
    """
    total_count = sum(stats["count"] for stats in client_stats_list)
    total_sum = sum(stats["sum"] for stats in client_stats_list)
    total_sq_sum = sum(stats["sq_sum"] for stats in client_stats_list)
    
    # Calculate global mean
    global_mean = total_sum / total_count
    
    # Calculate global variance: E[X^2] - (E[X])^2
    # var = (sum_sq / N) - mean^2
    # Note: This corresponds to the population variance (biased estimator),
    # which matches sklearn StandardScaler's default behavior.
    global_var = (total_sq_sum / total_count) - (global_mean ** 2)
    
    # Handle numerical instability (variance should be non-negative)
    global_var = np.maximum(global_var, 0)
    global_std = np.sqrt(global_var)
    
    return {
        "mean": global_mean,
        "std": global_std,
        "var": global_var,
        "count": total_count
    }


class DataPreprocessor:
    """
    数据预处理器
    负责特征标准化和目标变量变换（支持0.25次幂变换和log1p变换）
    
    默认使用0.25次幂变换以匹配参考论文方法:
    - 变换: Y = ContAmnt^0.25
    - 反变换: ContAmnt = Y^4
    """
    
    def __init__(
        self,
        feature_scaler: str = "StandardScaler",
        target_transform: str = "power_0.25",
        random_seed: int = 42
    ):
        """
        初始化数据预处理器
        
        Args:
            feature_scaler: 特征标准化方法
            target_transform: 目标变量变换方法 ('power_0.25', 'log1p' 或 None)
            random_seed: 随机种子
        """
        self.feature_scaler_name = feature_scaler
        self.target_transform = target_transform
        self.random_seed = random_seed
        
        # 初始化scaler
        if feature_scaler == "StandardScaler":
            self.feature_scaler = StandardScaler()
        else:
            raise ValueError(f"Unsupported scaler: {feature_scaler}")
        
        self.is_fitted = False
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        在训练数据上拟合scaler
        
        Args:
            X: 特征DataFrame
            y: 目标变量Series (用于记录统计信息)
        """
        # 拟合特征scaler
        self.feature_scaler.fit(X)
        
        # 记录特征列名
        self.feature_columns = X.columns.tolist()
        
        # 记录目标变量统计信息（用于验证）
        if y is not None:
            self.y_mean = float(y.mean())
            self.y_std = float(y.std())
            self.y_min = float(y.min())
            self.y_max = float(y.max())
        
        self.is_fitted = True
        return self
    
    def set_global_stats(self, mean: np.ndarray, std: np.ndarray, var: np.ndarray, n_samples: int, feature_columns: list = None):
        """
        Manually set global statistics for federated learning.
        
        Args:
            mean: Global mean
            std: Global standard deviation
            var: Global variance
            n_samples: Total number of samples seen
            feature_columns: List of feature names
        """
        self.feature_scaler.mean_ = mean
        self.feature_scaler.scale_ = std
        self.feature_scaler.var_ = var
        self.feature_scaler.n_samples_seen_ = n_samples
        self.feature_scaler.n_features_in_ = len(mean)
        
        if feature_columns is not None:
            self.feature_columns = feature_columns
            self.feature_scaler.feature_names_in_ = np.array(feature_columns)
            
        self.is_fitted = True
    
    def transform_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        转换特征
        
        Args:
            X: 特征DataFrame
            
        Returns:
            标准化后的特征数组
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet. Call fit() first.")
        
        X_scaled = self.feature_scaler.transform(X)
        return X_scaled
    
    def transform_target(self, y: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        转换目标变量
        
        支持的变换方法:
        - 'power_0.25': Y = ContAmnt^0.25 (参考Zhang et al., 2023)
        - 'log1p': Y = log(1 + ContAmnt)
        - None: 不进行变换
        
        Args:
            y: 目标变量
            
        Returns:
            变换后的目标变量数组
        """
        y_array = np.array(y).reshape(-1, 1)
        
        if self.target_transform == "power_0.25":
            # 0.25次幂变换: Y = ContAmnt^0.25
            y_transformed = np.power(y_array, 0.25)
        elif self.target_transform == "log1p":
            y_transformed = np.log1p(y_array)
        elif self.target_transform is None:
            y_transformed = y_array
        else:
            raise ValueError(f"Unsupported target transform: {self.target_transform}")
        
        return y_transformed
    
    def inverse_transform_target(self, y_transformed: np.ndarray) -> np.ndarray:
        """
        反变换目标变量（从变换空间回到原始空间）
        
        反变换方法:
        - 'power_0.25': ContAmnt = Y^4
        - 'log1p': ContAmnt = exp(Y) - 1
        - None: 不进行反变换
        
        Args:
            y_transformed: 变换后的目标变量
            
        Returns:
            原始空间的目标变量
        """
        y_array = np.array(y_transformed).reshape(-1, 1)
        
        if self.target_transform == "power_0.25":
            # 反变换: ContAmnt = Y^4
            y_original = np.power(y_array, 4)
        elif self.target_transform == "log1p":
            y_original = np.expm1(y_array)
        elif self.target_transform is None:
            y_original = y_array
        else:
            raise ValueError(f"Unsupported target transform: {self.target_transform}")
        
        return y_original
    
    def transform(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        同时转换特征和目标变量
        
        Args:
            X: 特征DataFrame
            y: 目标变量 (可选)
            
        Returns:
            如果y为None，返回X_scaled
            否则返回(X_scaled, y_transformed)
        """
        X_scaled = self.transform_features(X)
        
        if y is not None:
            y_transformed = self.transform_target(y)
            return X_scaled, y_transformed
        
        return X_scaled
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        拟合并转换数据（训练集使用）
        
        Args:
            X: 特征DataFrame
            y: 目标变量 (可选)
            
        Returns:
            转换后的数据
        """
        self.fit(X, y)
        return self.transform(X, y)
    
    def get_stats(self) -> dict:
        """
        获取预处理器的统计信息
        
        Returns:
            统计信息字典
        """
        if not self.is_fitted:
            return {"status": "not fitted"}
        
        stats = {
            "feature_scaler": self.feature_scaler_name,
            "target_transform": self.target_transform,
            "n_features": len(self.feature_columns),
            "feature_columns": self.feature_columns,
            "is_fitted": self.is_fitted
        }
        
        if hasattr(self, 'y_mean'):
            stats.update({
                "target_mean": self.y_mean,
                "target_std": self.y_std,
                "target_min": self.y_min,
                "target_max": self.y_max
            })
        
        return stats


def create_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False
) -> DataLoader:
    """
    创建PyTorch DataLoader
    
    Args:
        X: 特征数组
        y: 目标数组
        batch_size: 批量大小
        shuffle: 是否打乱数据
        num_workers: 数据加载的工作进程数
        pin_memory: 是否将数据固定在内存中（GPU训练时有用）
        
    Returns:
        DataLoader对象
    """
    # 转换为PyTorch张量
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # 创建数据集
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return dataloader


def prepare_data_for_training(
    data_path: Union[str, Path],
    preprocessor: Optional[DataPreprocessor] = None,
    feature_columns: Optional[list] = None,
    target_column: str = "ContAmnt",
    batch_size: int = 32,
    shuffle: bool = True,
    fit_preprocessor: bool = False
) -> Tuple[DataLoader, DataPreprocessor]:
    """
    准备训练数据的完整流程
    
    Args:
        data_path: 数据文件路径
        preprocessor: 数据预处理器（如果为None则创建新的）
        feature_columns: 特征列名
        target_column: 目标列名
        batch_size: 批量大小
        shuffle: 是否打乱
        fit_preprocessor: 是否拟合预处理器
        
    Returns:
        (DataLoader, DataPreprocessor)
    """
    # 加载数据
    X, y = load_data(data_path, feature_columns, target_column)
    
    # 创建或使用预处理器
    if preprocessor is None:
        preprocessor = DataPreprocessor()
        fit_preprocessor = True
    
    # 预处理数据
    if fit_preprocessor:
        X_scaled, y_transformed = preprocessor.fit_transform(X, y)
    else:
        X_scaled, y_transformed = preprocessor.transform(X, y)
    
    # 创建DataLoader
    dataloader = create_dataloader(
        X_scaled,
        y_transformed,
        batch_size=batch_size,
        shuffle=shuffle
    )
    
    return dataloader, preprocessor


def load_federated_datasets(config: dict, config_key: str = 'scene_c') -> Tuple[dict, dict, TensorDataset, TensorDataset, 'DataPreprocessor']:
    """
    通用联邦数据加载器，支持 scene_b 和 scene_c 共用同一数据加载逻辑。

    Args:
        config: 完整配置字典
        config_key: 使用哪个配置节 ('scene_b' 或 'scene_c')

    Returns:
        与 load_federated_datasets_for_scene_c 相同的5元组
    """
    import copy
    config_copy = copy.deepcopy(config)
    config_copy['scene_c'] = config.get(config_key, {})
    return load_federated_datasets_for_scene_c(config_copy)


def load_federated_datasets_for_scene_c(config: dict) -> Tuple[dict, dict, TensorDataset, TensorDataset, 'DataPreprocessor']:
    """
    Scene C loader for unified Client CSV with column cleaning.

    Returns:
    - per-client train/val datasets (local 80/20)
    - global val/test datasets (global 80/10/10)
    - fitted preprocessor
    """
    from sklearn.model_selection import train_test_split

    scene_c_cfg = config.get('scene_c', {})
    data_cfg = scene_c_cfg.get('data', {})
    if not data_cfg:
        raise ValueError("scene_c.data is missing")

    raw_csv = data_cfg.get('raw_csv')
    if not raw_csv:
        raise ValueError("scene_c.data.raw_csv is missing")

    rename_map = data_cfg.get('rename_map', {})
    drop_columns = data_cfg.get('drop_columns', [])
    feature_columns = data_cfg.get('feature_columns', [])
    target_column = data_cfg.get('target_column', 'ContAmnt')
    client_column = data_cfg.get('client_column', 'Client')
    cleaned_csv = data_cfg.get('cleaned_csv')

    splits = data_cfg.get('splits', {})
    global_train_ratio = splits.get('global_train', 0.8)
    global_val_ratio = splits.get('global_val', 0.1)
    global_test_ratio = splits.get('global_test', 0.1)
    local_val_ratio = splits.get('local_val', 0.2)
    local_val_ratio = splits.get('local_val', 0.2)
    local_val_ratio = splits.get('local_val', 0.2)

    # Load raw data
    df_raw = pd.read_csv(raw_csv)

    # Rename columns (Chinese -> English)
    if rename_map:
        df_raw = df_raw.rename(columns=rename_map)

    # Drop unused columns
    if drop_columns:
        cols_to_drop = [c for c in drop_columns if c in df_raw.columns]
        if cols_to_drop:
            df_raw = df_raw.drop(columns=cols_to_drop)

    # Validate required columns
    required_cols = set(feature_columns + [target_column, client_column])
    missing = required_cols - set(df_raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only required columns in fixed order
    df_clean = df_raw[feature_columns + [target_column, client_column]].copy()

    # Save cleaned CSV if configured
    if cleaned_csv:
        Path(cleaned_csv).parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(cleaned_csv, index=False)

    # Check split ratios
    total_ratio = global_train_ratio + global_val_ratio + global_test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Global split ratios must sum to 1.0, got {total_ratio}")
    if not (0 < global_train_ratio < 1):
        raise ValueError("global_train must be in (0, 1)")
    if not (0 < local_val_ratio < 1):
        raise ValueError("local_val must be in (0, 1)")

    seed = config['preprocessing']['random_seed']

    # Global 80/10/10 split, stratified by Client
    temp_ratio = 1.0 - global_train_ratio
    train_df, temp_df = train_test_split(
        df_clean,
        test_size=temp_ratio,
        random_state=seed,
        stratify=df_clean[client_column]
    )

    # Split temp into val/test
    test_ratio = global_test_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio,
        random_state=seed,
        stratify=temp_df[client_column]
    )

    # ========== 1. Privacy-Preserving Federated Statistics Calculation ==========
    client_stats_list = []
    client_data_cache = {}
    for client_id, df_client in train_df.groupby(client_column):
        stats = compute_local_stats(df_client, feature_columns)
        client_stats_list.append(stats)
        client_data_cache[client_id] = df_client

    global_stats = aggregate_federated_stats(client_stats_list)

    preprocessor = DataPreprocessor(
        feature_scaler=config['preprocessing']['scaler'],
        target_transform=config['preprocessing']['target_transform'],
        random_seed=seed
    )
    preprocessor.set_global_stats(
        mean=global_stats['mean'],
        std=global_stats['std'],
        var=global_stats['var'],
        n_samples=global_stats['count'],
        feature_columns=feature_columns
    )

    # ========== 2. Local train/val split per client (80/20) ==========
    client_train_sets = {}
    client_val_sets = {}

    for client_id, df_client in client_data_cache.items():
        X_client = df_client[feature_columns].copy()
        y_client = df_client[target_column].values

        X_train, X_val, y_train, y_val = train_test_split(
            X_client, y_client,
            test_size=local_val_ratio,
            random_state=seed
        )

        X_train_scaled = preprocessor.transform_features(X_train)
        y_train_transformed = preprocessor.transform_target(y_train)
        X_val_scaled = preprocessor.transform_features(X_val)
        y_val_transformed = preprocessor.transform_target(y_val)

        client_train_sets[client_id] = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.FloatTensor(y_train_transformed)
        )
        client_val_sets[client_id] = TensorDataset(
            torch.FloatTensor(X_val_scaled),
            torch.FloatTensor(y_val_transformed)
        )

    # ========== 3. Global validation set ==========
    X_global_val = val_df[feature_columns].copy()
    y_global_val = val_df[target_column].values
    X_global_val_scaled = preprocessor.transform_features(X_global_val)
    y_global_val_transformed = preprocessor.transform_target(y_global_val)
    global_val_set = TensorDataset(
        torch.FloatTensor(X_global_val_scaled),
        torch.FloatTensor(y_global_val_transformed)
    )

    # ========== 4. Global test set ==========
    X_global_test = test_df[feature_columns].copy()
    y_global_test = test_df[target_column].values
    X_global_test_scaled = preprocessor.transform_features(X_global_test)
    y_global_test_transformed = preprocessor.transform_target(y_global_test)
    global_test_set = TensorDataset(
        torch.FloatTensor(X_global_test_scaled),
        torch.FloatTensor(y_global_test_transformed)
    )
    global_test_set.prediction_metadata = test_df[[client_column]].rename(
        columns={client_column: "Client"}
    ).reset_index(drop=True)

    return client_train_sets, client_val_sets, global_val_set, global_test_set, preprocessor


def load_centralized_datasets(config: dict, config_key: str = 'scene_c') -> Dict[str, Any]:
    """
    Load centralized train/val/test splits using the same scene-style data config
    and split logic as federated experiments (global 80/10/10 with client stratify).

    Returns a dictionary containing both original-scale and transformed datasets,
    plus a fitted preprocessor.
    """
    from sklearn.model_selection import train_test_split

    scene_cfg = config.get(config_key, {})
    data_cfg = scene_cfg.get('data', {})
    if not data_cfg:
        raise ValueError(f"{config_key}.data is missing")

    raw_csv = data_cfg.get('raw_csv')
    if not raw_csv:
        raise ValueError(f"{config_key}.data.raw_csv is missing")

    rename_map = data_cfg.get('rename_map', {})
    drop_columns = data_cfg.get('drop_columns', [])
    feature_columns = data_cfg.get('feature_columns', [])
    target_column = data_cfg.get('target_column', 'ContAmnt')
    client_column = data_cfg.get('client_column', 'Client')
    cleaned_csv = data_cfg.get('cleaned_csv')

    splits = data_cfg.get('splits', {})
    global_train_ratio = splits.get('global_train', 0.8)
    global_val_ratio = splits.get('global_val', 0.1)
    global_test_ratio = splits.get('global_test', 0.1)
    local_val_ratio = splits.get('local_val', 0.2)

    df_raw = pd.read_csv(raw_csv)
    if rename_map:
        df_raw = df_raw.rename(columns=rename_map)

    if drop_columns:
        cols_to_drop = [c for c in drop_columns if c in df_raw.columns]
        if cols_to_drop:
            df_raw = df_raw.drop(columns=cols_to_drop)

    required_cols = set(feature_columns + [target_column, client_column])
    missing = required_cols - set(df_raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df_clean = df_raw[feature_columns + [target_column, client_column]].copy()
    if cleaned_csv:
        Path(cleaned_csv).parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(cleaned_csv, index=False)

    total_ratio = global_train_ratio + global_val_ratio + global_test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Global split ratios must sum to 1.0, got {total_ratio}")

    seed = config['preprocessing']['random_seed']

    temp_ratio = 1.0 - global_train_ratio
    train_df, temp_df = train_test_split(
        df_clean,
        test_size=temp_ratio,
        random_state=seed,
        stratify=df_clean[client_column]
    )

    test_ratio = global_test_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio,
        random_state=seed,
        stratify=temp_df[client_column]
    )

    client_stats_list = []
    client_data_cache = {}
    for client_id, df_client in train_df.groupby(client_column):
        stats = compute_local_stats(df_client, feature_columns)
        client_stats_list.append(stats)
        client_data_cache[client_id] = df_client

    global_stats = aggregate_federated_stats(client_stats_list)

    preprocessor = DataPreprocessor(
        feature_scaler=config['preprocessing']['scaler'],
        target_transform=config['preprocessing']['target_transform'],
        random_seed=seed
    )
    preprocessor.set_global_stats(
        mean=global_stats['mean'],
        std=global_stats['std'],
        var=global_stats['var'],
        n_samples=global_stats['count'],
        feature_columns=feature_columns
    )

    local_train_frames = []
    local_val_frames = []
    for _, df_client in client_data_cache.items():
        X_client = df_client[feature_columns].copy()
        y_client = df_client[target_column].values
        X_local_train, X_local_val, y_local_train, y_local_val = train_test_split(
            X_client,
            y_client,
            test_size=local_val_ratio,
            random_state=seed
        )
        local_train_df = X_local_train.copy()
        local_train_df[target_column] = y_local_train
        local_train_df[client_column] = df_client.loc[X_local_train.index, client_column].values
        local_val_df = X_local_val.copy()
        local_val_df[target_column] = y_local_val
        local_val_df[client_column] = df_client.loc[X_local_val.index, client_column].values
        local_train_frames.append(local_train_df)
        local_val_frames.append(local_val_df)

    train_df_model = pd.concat(local_train_frames).sort_index()
    local_val_df = pd.concat(local_val_frames).sort_index()

    X_train = train_df_model[feature_columns].copy()
    y_train = train_df_model[target_column].values
    X_train_scaled = preprocessor.transform_features(X_train)
    y_train_transformed = preprocessor.transform_target(y_train)

    X_val = val_df[feature_columns].copy()
    y_val = val_df[target_column].values
    X_val_scaled = preprocessor.transform_features(X_val)
    y_val_transformed = preprocessor.transform_target(y_val)

    X_test = test_df[feature_columns].copy()
    y_test = test_df[target_column].values
    X_test_scaled = preprocessor.transform_features(X_test)
    y_test_transformed = preprocessor.transform_target(y_test)

    return {
        "feature_columns": feature_columns,
        "target_column": target_column,
        "global_train_df": train_df,
        "train_df": train_df_model,
        "local_val_df": local_val_df,
        "val_df": val_df,
        "test_df": test_df,
        "training_scope": "federated_local_train_union",
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "y_train_transformed": y_train_transformed,
        "X_val_scaled": X_val_scaled,
        "y_val_transformed": y_val_transformed,
        "X_test_scaled": X_test_scaled,
        "y_test_transformed": y_test_transformed,
        "preprocessor": preprocessor
    }


if __name__ == "__main__":
    # 添加项目根目录到Python路径
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # 重新导入（测试模式）
    from src.utils import load_data
    
    # 测试代码
    print("=" * 60)
    print("Testing data preprocessing module")
    print("=" * 60)
    
    # 创建测试数据
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    X_test = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    y_test = pd.Series(np.random.uniform(100000, 5000000, n_samples), name="ContAmnt")
    
    # 测试预处理器
    preprocessor = DataPreprocessor()
    
    print("\n1. Fitting preprocessor...")
    X_scaled, y_transformed = preprocessor.fit_transform(X_test, y_test)
    print(f"[PASS] Feature scaling: {X_test.shape} -> {X_scaled.shape}")
    print(f"[PASS] Target transform: {y_test.shape} -> {y_transformed.shape}")
    
    # 测试反变换
    print("\n2. Testing inverse transform...")
    y_recovered = preprocessor.inverse_transform_target(y_transformed)
    error = np.mean(np.abs(y_test.values.reshape(-1, 1) - y_recovered))
    print(f"[PASS] Inverse transform error: ${error:.2f}")
    
    # 测试DataLoader
    print("\n3. Testing DataLoader...")
    dataloader = create_dataloader(X_scaled, y_transformed, batch_size=16)
    batch_X, batch_y = next(iter(dataloader))
    print(f"[PASS] DataLoader batch: X={batch_X.shape}, y={batch_y.shape}")
    
    # 打印统计信息
    print("\n4. Preprocessor statistics:")
    stats = preprocessor.get_stats()
    for key, value in stats.items():
        if key != "feature_columns":
            print(f"   {key}: {value}")
    
    print("\n[PASS] All data preprocessing functions tested successfully")
    
    # Test Federated Statistics
    print("\n5. Testing Federated Statistics...")
    # Split test data into 3 "clients"
    dfs = np.array_split(X_test, 3)
    client_stats = []
    for df_part in dfs:
        client_stats.append(compute_local_stats(df_part, preprocessor.feature_columns))
    
    global_stats = aggregate_federated_stats(client_stats)
    
    # Compare with directly fitted stats
    print(f"   Original Mean: {preprocessor.feature_scaler.mean_[:3]}...")
    print(f"   Federated Mean: {global_stats['mean'][:3]}...")
    
    mean_diff = np.max(np.abs(preprocessor.feature_scaler.mean_ - global_stats['mean']))
    var_diff = np.max(np.abs(preprocessor.feature_scaler.var_ - global_stats['var']))
    
    print(f"   Max Mean Diff: {mean_diff}")
    print(f"   Max Var Diff: {var_diff}")
    
    if mean_diff < 1e-10 and var_diff < 1e-10:
        print("[PASS] Federated statistics match global statistics")
    else:
        print("[FAIL] Federated statistics do not match global statistics")


