"""Plan2Explore's intrinsic reward, ported from pedropalmav/her-dream.

An ensemble of one-step predictors guesses the next latent state. Where its
members disagree the world model has not pinned the dynamics down yet, so an
explorer rewarded by that disagreement goes where there is still something to
learn (Sekar et al., 2020).
"""

import torch
from torch import nn

import distributions as dists
import networks
from tools import to_f32


class Disagreement(nn.Module):
    """Ensemble of one-step latent predictors; their spread is the reward.

    Each member predicts the next stochastic state from the current feature and
    the action taken there. The ensemble is a probe, not part of the
    representation: its inputs and targets are detached, so training it never
    shapes the world model. Each `MLPHead` initializes its own weights, so the
    members start from independent draws and can disagree.
    """

    def __init__(self, config, feat_size, act_dim, flat_stoch):
        super().__init__()
        self.models = int(config.models)
        if self.models < 2:
            raise ValueError(f"disag.models must be >= 2 for a spread to exist, got {self.models}.")
        self.intr_scale = float(config.intr_scale)
        config.head.shape = [flat_stoch]
        self.heads = nn.ModuleList([networks.MLPHead(config.head, feat_size + act_dim) for _ in range(self.models)])

    def loss(self, feat, action, stoch):
        """One-step prediction loss over replayed sequences.

        `action[:, t]` is the action leading into state t (the replay buffer
        shifts it one step back), so the transition t -> t+1 is driven by
        `action[:, 1:]` and pairs with `feat[:, :-1]`.

        Args:
            feat: (B, T, F) posterior features.
            action: (B, T, A) actions as the buffer returns them.
            stoch: (B, T, S, K) posterior stochastic states.
        """
        inputs = torch.cat([feat[:, :-1], action[:, 1:].to(feat.dtype)], -1).detach()
        # (B, T, S, K) -> (B, T-1, S*K)
        targets = to_f32(stoch[:, 1:].flatten(2)).detach()
        # Summed over members, as DreamerV2 does, so each one gets a full gradient.
        loss = -sum(dists.mse(head(inputs)).log_prob(targets).mean() for head in self.heads)
        return loss, {"disag_pred_err": loss.detach() / self.models}

    @torch.no_grad()
    def intrinsic_reward(self, feat, action):
        """Spread of the ensemble over an imagined rollout.

        Args:
            feat: (N, H, F) imagined features.
            action: (N, H, A) the action taken at each of them.

        Returns:
            ((N, H, 1) reward, metrics).
        """
        inputs = torch.cat([feat, action.to(feat.dtype)], -1)
        # (M, N, H, S*K)
        preds = torch.stack([to_f32(head(inputs)) for head in self.heads])
        # Standard deviation over members, then mean over the predicted dimensions.
        disag = preds.std(0, correction=0).mean(-1, keepdim=True)
        return self.intr_scale * disag, {"disag_raw": disag.mean()}
