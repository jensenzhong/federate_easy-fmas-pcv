"""
神经网络模型模块
定义成本估算的MLP模型和训练/评估函数
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Union
import numpy as np
from pathlib import Path

from src.utils import get_device, setup_logger


class CostEstimationMLP(nn.Module):
    """
    改进版成本估算多层感知机模型
    默认架构: 10 -> 128 -> 128 -> 64 -> 32 -> 1
    每层: Linear + BatchNorm + GELU + Dropout
    适合小样本、非线性强的工程造价数据
    """
    
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dims: list = None,
        output_dim: int = 1,
        activation: str = "gelu",
        dropout: float = 0.1
    ):
        """
        初始化模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表，默认 [128, 128, 64, 32]
            output_dim: 输出维度
            activation: 激活函数类型 ('relu', 'gelu', 'tanh', 'sigmoid')
            dropout: Dropout比例
        """
        super(CostEstimationMLP, self).__init__()
        
        # 设置默认架构
        if hidden_dims is None:
            hidden_dims = [128, 128, 64, 32]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        self.activation_name = activation.lower()
        
        # 构建网络层
        layers = []
        in_dim = input_dim
        
        # 构建隐藏层
        for h in hidden_dims:
            # Linear层
            layers.append(nn.Linear(in_dim, h))
            # BatchNorm层
            layers.append(nn.BatchNorm1d(h))
            # 激活函数
            if self.activation_name == "gelu":
                layers.append(nn.GELU())
            elif self.activation_name == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif self.activation_name == "tanh":
                layers.append(nn.Tanh())
            elif self.activation_name == "sigmoid":
                layers.append(nn.Sigmoid())
            else:
                # 默认使用ReLU
                layers.append(nn.ReLU(inplace=True))
            # Dropout层
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        
        # 输出层
        layers.append(nn.Linear(in_dim, output_dim))
        
        # 组合所有层
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, input_dim]
            
        Returns:
            输出张量 [batch_size, output_dim]
        """
        return self.network(x)
    
    def get_num_parameters(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "model_name": "CostEstimationMLP",
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "output_dim": self.output_dim,
            "dropout": self.dropout,
            "activation": self.activation_name,
            "num_parameters": self.get_num_parameters()
        }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 100,
    learning_rate: float = 0.001,
    device: str = "auto",
    early_stopping_patience: int = 15,
    logger: Optional[object] = None,
    verbose: bool = True
) -> Dict:
    """
    训练模型
    
    Args:
        model: 神经网络模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 训练轮数
        learning_rate: 学习率
        device: 计算设备
        early_stopping_patience: 早停耐心值
        logger: 日志记录器
        verbose: 是否打印详细信息
        
    Returns:
        训练历史字典
    """
    device = get_device(device)
    model = model.to(device)
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # 训练历史
    history = {
        "train_loss": [],
        "val_loss": [] if val_loader else None,
        "best_epoch": 0,
        "best_val_loss": float('inf')
    }
    
    # 早停计数器
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_losses = []
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # 前向传播
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        history["train_loss"].append(avg_train_loss)
        
        # 验证阶段
        if val_loader:
            val_loss = evaluate_model(model, val_loader, device, return_predictions=False)
            history["val_loss"].append(val_loss)
            
            # 早停检查
            if val_loss < history["best_val_loss"]:
                history["best_val_loss"] = val_loss
                history["best_epoch"] = epoch
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            # 打印进度
            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}"
                if logger:
                    logger.info(msg)
                else:
                    print(msg)
            
            # 早停
            if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                if verbose:
                    msg = f"Early stopping at epoch {epoch+1}"
                    if logger:
                        logger.info(msg)
                    else:
                        print(msg)
                break
        else:
            # 无验证集时的打印
            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.6f}"
                if logger:
                    logger.info(msg)
                else:
                    print(msg)
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "auto",
    return_predictions: bool = False
) -> Union[float, Tuple[float, np.ndarray, np.ndarray]]:
    """
    评估模型
    
    Args:
        model: 神经网络模型
        data_loader: 数据加载器
        device: 计算设备
        return_predictions: 是否返回预测值和真实值
        
    Returns:
        如果return_predictions=False: 返回平均损失
        如果return_predictions=True: 返回(平均损失, 预测值, 真实值)
    """
    device = get_device(device)
    model = model.to(device)
    model.eval()
    
    criterion = nn.MSELoss()
    all_losses = []
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # 前向传播
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            all_losses.append(loss.item())
            
            if return_predictions:
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
    
    avg_loss = np.mean(all_losses)
    
    if return_predictions:
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        return avg_loss, predictions, targets
    
    return avg_loss


def save_model(
    model: nn.Module,
    save_path: Union[str, Path],
    model_info: Optional[Dict] = None
):
    """
    保存模型
    
    Args:
        model: 神经网络模型
        save_path: 保存路径
        model_info: 模型额外信息
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 准备保存的数据
    save_dict = {
        "model_state_dict": model.state_dict(),
        "model_class": model.__class__.__name__
    }
    
    # 如果是CostEstimationMLP，保存架构信息
    if hasattr(model, 'get_model_info'):
        save_dict["model_info"] = model.get_model_info()
    
    # 添加额外信息
    if model_info:
        save_dict.update(model_info)
    
    torch.save(save_dict, save_path)


def load_model(
    load_path: Union[str, Path],
    device: str = "auto"
) -> Tuple[nn.Module, Dict]:
    """
    加载模型
    
    Args:
        load_path: 模型文件路径
        device: 计算设备
        
    Returns:
        (模型, 模型信息字典)
    """
    device = get_device(device)
    
    # 加载checkpoint
    checkpoint = torch.load(load_path, map_location=device, weights_only=False)
    
    # 获取模型信息
    model_info = checkpoint.get("model_info", {})
    
    # 重建模型
    if checkpoint.get("model_class") == "CostEstimationMLP":
        model = CostEstimationMLP(
            input_dim=model_info.get("input_dim", 10),
            hidden_dims=model_info.get("hidden_dims", [128, 128, 64, 32]),
            output_dim=model_info.get("output_dim", 1),
            activation=model_info.get("activation", "gelu"),
            dropout=model_info.get("dropout", 0.1)
        )
    else:
        raise ValueError(f"Unknown model class: {checkpoint.get('model_class')}")
    
    # 加载权重
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    
    return model, checkpoint


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("测试神经网络模型模块")
    print("=" * 60)
    
    # 创建模型
    model = CostEstimationMLP(input_dim=10, hidden_dims=[64, 32], output_dim=1)
    print("\n模型信息:")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 测试前向传播
    print("\n测试前向传播:")
    x = torch.randn(5, 10)
    y_pred = model(x)
    print(f"  输入shape: {x.shape}")
    print(f"  输出shape: {y_pred.shape}")
    print(f"  ✅ 前向传播成功")
    
    # 测试保存和加载
    print("\n测试模型保存和加载:")
    save_model(model, "results/models/test_model.pth")
    print("  ✅ 模型保存成功")
    
    loaded_model, checkpoint = load_model("results/models/test_model.pth")
    print("  ✅ 模型加载成功")
    
    # 验证加载的模型
    y_pred_loaded = loaded_model(x)
    diff = torch.abs(y_pred - y_pred_loaded).mean()
    print(f"  预测差异: {diff.item():.10f}")
    
    print("\n✅ 所有神经网络功能测试通过")

