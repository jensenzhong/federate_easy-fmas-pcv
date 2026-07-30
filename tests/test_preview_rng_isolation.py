import torch
from torch.utils.data import TensorDataset, DataLoader


def _minimal_central_agent():
    from src.federated_learning.mas_agents import CentralAgent, LocalAgent
    from src.models import CostEstimationMLP

    device = torch.device("cpu")
    x = torch.randn(8, 10)
    y = torch.linspace(1.0, 8.0, steps=8).reshape(-1, 1)
    dataset = TensorDataset(x, y)
    config = {
        "federated_learning": {
            "client": {
                "batch_size": 4,
                "learning_rate": 0.001,
                "local_epochs": 1,
            }
        },
        "scene_c": {
            "strategies": [{"name": "size_only"}],
            "llm": {},
        },
    }
    client = LocalAgent(
        client_id="Client 1",
        train_dataset=dataset,
        val_dataset=dataset,
        config=config,
        device=device,
        input_dim=10,
    )
    global_model = CostEstimationMLP(
        input_dim=10,
        hidden_dims=[128, 128, 64, 32],
        output_dim=1,
        activation="gelu",
        dropout=0.1,
    ).to(device)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    return CentralAgent(
        global_model=global_model,
        client_agents={"Client 1": client},
        global_val_loader=loader,
        global_test_loader=loader,
        preprocessor=None,
        config=config,
        device=device,
        server_optimizer="fedavg",
    )


def test_client_validation_preview_preserves_torch_rng_state():
    torch.manual_seed(123)
    agent = _minimal_central_agent()
    state = agent.global_model.state_dict()

    torch.manual_seed(999)
    before = torch.get_rng_state().clone()

    agent._evaluate_state_on_client_vals(state)

    after = torch.get_rng_state()
    assert torch.equal(after, before)
