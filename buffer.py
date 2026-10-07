import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler


class Buffer:
    def __init__(self, config):
        self.device = torch.device(config.device)
        self.storage_device = torch.device(config.storage_device)
        self.batch_size = int(config.batch_size)
        self.batch_length = int(config.batch_length)
        self.num_eps = 0
        # Sequences per batch that are cut around a rewarded step. With sparse
        # rewards a uniform batch almost never holds one, and the reward model
        # learns to always predict zero.
        self.reward_slices = int(config.get("reward_slices", 0))
        assert 0 <= self.reward_slices < self.batch_size, (self.reward_slices, self.batch_size)
        self._buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=config.max_size, device=self.storage_device, ndim=2),
            sampler=SliceSampler(
                slice_len=self.batch_length + 1, end_key=None, traj_key="episode", truncated_key=None, strict_length=True
            ),
            prefetch=0,
            batch_size=self.batch_size * (self.batch_length + 1),  # +1 for context
        )

    def add_transition(self, data):
        # This is batched data and lifted for storage.
        # (B, ...) -> (B, 1, ...)
        self._buffer.extend(data.unsqueeze(1))

    def sample(self):
        sample_td, info = self._buffer.sample(return_info=True)
        rewarded = self._sample_rewarded()
        if rewarded is not None:
            # Swap the last uniform sequences for the rewarded ones, keeping
            # the batch size the model was compiled for.
            keep = (self.batch_size - self.reward_slices) * (self.batch_length + 1)
            extra_td, extra_index = rewarded
            sample_td = torch.cat([sample_td[:keep], extra_td.to(sample_td.device)])
            info["index"] = tuple(
                torch.cat([ind[:keep], extra.to(ind.device)]) for ind, extra in zip(info["index"], extra_index)
            )
        # The sampler returns a flattened batch of length B*(T+1).
        # (B*(T+1), ...) -> (B, T+1, ...)
        sample_td = sample_td.view(-1, self.batch_length + 1)
        src_dev = sample_td.device
        if src_dev.type == "cpu" and self.device.type == "cuda":
            sample_td = sample_td.pin_memory().to(self.device, non_blocking=True)
        elif src_dev != self.device:
            sample_td = sample_td.to(self.device, non_blocking=True)
        # The initial ones are used only to extract the latent vector
        initial = (sample_td["stoch"][:, 0], sample_td["deter"][:, 0])
        data = sample_td[:, 1:]
        data.set_("action", sample_td["action"][:, :-1])  # action is 1 step back
        index = [ind.view(-1, self.batch_length + 1)[:, 1:] for ind in info["index"]]
        return data, index, initial

    def _sample_rewarded(self):
        """Index sequences that contain a step with nonzero reward.

        Returns the flattened sequences and their (time, env) storage indices,
        in the layout the sampler uses, or None when there is nothing to draw.
        """
        if self.reward_slices == 0 or self._buffer.storage.shape is None:
            return None
        length = self._buffer.storage.shape[0]
        span = self.batch_length + 1
        if length < span:
            return None
        # Scanned on every call so the positions stay right after loading a
        # checkpoint or once the buffer starts overwriting old data.
        reward = self._buffer.storage._storage["reward"][:length].reshape(length, -1)
        hits = torch.nonzero(reward != 0)
        if len(hits) == 0:
            return None
        pick = hits[torch.randint(len(hits), (self.reward_slices,), device=hits.device)]
        time, env = pick[:, 0], pick[:, 1]
        # The first step of a sequence is only context and is not trained on,
        # so the reward has to land somewhere in the remaining batch_length.
        offset = torch.randint(1, span, (self.reward_slices,), device=time.device)
        start = (time - offset).clamp(0, length - span)
        time = (start[:, None] + torch.arange(span, device=start.device)).reshape(-1)
        env = env.repeat_interleave(span)
        return self._buffer.storage.get((time, env)), (time, env)

    def update(self, index, stoch, deter):
        # Flatten the data
        index = [ind.reshape(-1) for ind in index]
        # (B, T, S, K) -> (B*T, S, K)
        stoch = stoch.reshape(-1, *stoch.shape[2:])
        # (B, T, D) -> (B*T, D)
        deter = deter.reshape(-1, *deter.shape[2:])
        # In storage, the length is the first dimension, and the batch (number of environments) is the second dimension.
        n = index[0].shape[0]
        self._buffer[index[1], index[0]] = TensorDict({"stoch": stoch, "deter": deter}, batch_size=(n,))

    def save(self, path):
        self._buffer.dumps(path)

    def load(self, path):
        self._buffer.loads(path)

    def count(self):
        if self._buffer.storage.shape is None:
            return 0
        return self._buffer.storage.shape.numel()
