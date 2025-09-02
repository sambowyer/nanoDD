import math
from typing import Optional

import torch
import torch.distributions as D
import torch.nn.functional as F
from torch import nn, Tensor

from dit import DiT


def onehot(x: Tensor, K: int):
    return F.one_hot(x, K).float() if x.ndim == 2 else x.clone()


class MD4(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embed: int,
        n_heads: int,
        n_blocks: int,
        n_cond: int,
        dropout: float,
        T: int,
    ) -> None:
        super().__init__()

        self.T = T
        self.K = vocab_size + 1
        self.net = DiT(self.K, n_embed, n_heads, n_blocks, n_cond, dropout)
        self.eps = 1e-20

        # betas and alpha_bars are 1-indexed, not zero-indexed, to keep indexing simpler
        # beta(0) = undef, beta(1) = 1/T, beta(2) = 1/(T-1), beta(T) = 1
        betas = torch.reciprocal(T - torch.arange(T + 1) + 1)
        betas[0] = 0.0
        alpha_bars = torch.cumprod(1.0 - betas, dim=0)
        alpha_bars[-1] = 0.0 + self.eps # to avoid log(0)
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bars", alpha_bars)

        # Reconstruction loss
        self.recon_loss = -(1-alpha_bars[1]) * math.log(self.K)
        
        # Prior loss
        self.prior_loss = alpha_bars[-1] * math.log(self.K)

    def mul_Qbar(self, x: Tensor, t: Tensor) -> Tensor:
        """Compute q(x_t | x_0) = x_0 @ Qbar_t"""
        y = onehot(x, self.K)
        alpha_bar_t = self.alpha_bars[t]
        y.mul_(alpha_bar_t[:, None, None])
        y[:, :, -1] += (1 - alpha_bar_t)[:, None]
        return y

    def mul_QT(self, x: Tensor, t: Tensor) -> Tensor:
        """Compute q(x_t+1 | x_t) = x_t @ Q_t+1"""
        y = onehot(x, self.K)
        beta_t = self.betas[t][:, None, None]
        z = beta_t * y[:, :, -1:]
        y.mul_(1 - beta_t).add_(z)
        return y

    def forward(self, data: Tensor, t: Optional[Tensor] = None) -> tuple[Tensor, Tensor, dict]:
        """MD4 forward pass implementing Algorithm 1 from MD4.pdf
        
        The key difference from D3PM is that MD4 directly predicts x_0 from x_t
        and computes the loss using the predicted x_0, without needing to compute
        complex posterior distributions.
        """

        # Sample timestep if not provided (NOTE: t cannot be 0)
        t = torch.randint(1, self.T, (data.size(0),), device=data.device) if t is None else t

        # 1. Sample x_t from q(x_t | x_0) where x_0 is the clean data
        q_x_t_given_x_0 = self.mul_Qbar(data, t) # [B, L, V]
        x_t = D.Categorical(probs=q_x_t_given_x_0).sample() # [B, L]

        # 2. Predict x_0 from x_t using the neural network
        logits_predicted_x_0 = self.net(x_t, t.float())
        # Set probability of absorbing state to 0 (as in D3PM)
        logits_predicted_x_0[:, :, -1] = -float("inf")
        log_predicted_x_0 = F.log_softmax(logits_predicted_x_0, dim=-1) # [B, L, V]

        # 3. Compute diffusion loss: alpha'_t/(1-alpha_t) * x_0 * log_pred_x_0 (weighted CE between x_0 and predicted_x_0)
        alpha_prime_t = (self.alpha_bars[t] - self.alpha_bars[t-1]) / (1 / self.T)
        masks = x_t == self.K - 1 # absorbing state

        diffusion_loss = (alpha_prime_t / (1 - self.alpha_bars[t])) * (onehot(data, self.K) * log_predicted_x_0 * masks[:,:,None])[:,:,:-1].sum(dim=(-1,-2)) # [B]
        diffusion_loss = diffusion_loss.mean() / x_t.shape[1]

        # 4. Add reconstruction and prior loss (constants)
        loss = diffusion_loss + self.recon_loss + self.prior_loss

        return log_predicted_x_0, loss, dict(kl=diffusion_loss, ce=self.recon_loss, l_T=loss, bpt=loss / math.log(2))


if __name__ == "__main__":
    B, L, V = 2, 3, 4
    x = torch.randint(0, V, (B, L))
    T = 5
    md4 = MD4(
        vocab_size=V,
        n_embed=8,
        n_cond=4,
        n_heads=1,
        n_blocks=1,
        dropout=0.0,
        T=T,
    )
    output = md4(x)
    print(f"x.shape: {x.shape}")
    print(f"output[0]: {output[0]}")
    print(f"output[0].shape: {output[0].shape}")
    print(f"output[1]: {output[1]}")
    print(f"output[2]: {output[2]}")
