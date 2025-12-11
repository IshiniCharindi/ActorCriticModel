# src/tlcs/model.py
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn, optim
from tlcs.constants import MODEL_FILE
from tlcs.logger import get_logger
from numpy.typing import NDArray

logger = get_logger(__name__)

class ActorCritic(nn.Module):
    """Shared LSTM base with separate actor and critic heads."""
    def __init__(self, input_dim: int, output_dim: int, width: int, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = width
        self.lstm = nn.LSTM(input_dim, width, num_layers, batch_first=True)
        self.actor_head = nn.Linear(width, output_dim)
        self.critic_head = nn.Linear(width, 1)

    def forward(self, x: Tensor, hidden: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]:
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (T, F) -> (1, T, F)
        batch_size = x.size(0)
        if hidden is None:
            hidden = (
                torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size)
            )
        out, new_hidden = self.lstm(x, hidden)  # out: (B, S, W)
        logits = self.actor_head(out)  # (B, S, A)
        value = self.critic_head(out).squeeze(-1)  # (B, S)
        return logits, value, new_hidden

class Model:
    """Wrapper for ActorCritic and optimizer handling."""
    def __init__(
            self,
            width: int,
            learning_rate: float,
            input_dim: int,
            output_dim: int,
            num_layers: int,
            model_path: Optional[Path] = None,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = ActorCritic(input_dim=input_dim, output_dim=output_dim, width=width, num_layers=num_layers)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        if model_path and (model_path / MODEL_FILE).exists():
            model_file = model_path / MODEL_FILE
            logger.info(f"Loading trained model from {model_file}")
            self.load_model(model_file)

    def predict_one_policy_with_hidden(self, state: np.ndarray, hidden: Optional[Tuple[Tensor, Tensor]]) -> Tuple[np.ndarray, Tuple[Tensor, Tensor]]:
        state_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1, 1, F)
        self.net.eval()
        with torch.no_grad():
            logits, _, new_hidden = self.net(state_t, hidden)
            return logits.squeeze().cpu().numpy().flatten(), new_hidden

    def predict_one_policy(self, state: np.ndarray) -> np.ndarray:
        # For test_mode, no hidden needed
        state_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1, 1, F)
        self.net.eval()
        with torch.no_grad():
            logits, _, _ = self.net(state_t)
            return logits.squeeze().cpu().numpy().flatten()

    def predict_one_value(self, state: np.ndarray) -> float:
        state_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        self.net.eval()
        with torch.no_grad():
            _, value, _ = self.net(state_t)
            return float(value.squeeze())

    def predict_batch_policy(self, states: NDArray) -> NDArray:
        states_t = torch.from_numpy(states.astype(np.float32)).unsqueeze(0)  # (1, T, F)
        self.net.eval()
        with torch.no_grad():
            logits, _, _ = self.net(states_t)
            return logits.squeeze(0).cpu().numpy()

    def predict_batch_value(self, states: NDArray) -> NDArray:
        states_t = torch.from_numpy(states.astype(np.float32)).unsqueeze(0)
        self.net.eval()
        with torch.no_grad():
            _, values, _ = self.net(states_t)
            return values.squeeze(0).cpu().numpy()

    def save_model(self, out_path: Path) -> None:
        out_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), out_path / MODEL_FILE)

    def load_model(self, model_file: Path) -> None:
        checkpoint = torch.load(model_file, map_location="cpu")
        self.net.load_state_dict(checkpoint)