from torch.optim.lr_scheduler import LinearLR, SequentialLR

def build_linear_scheduler_with_warmup_and_decay(optimizer, n_steps, warmup_ratio=0.1, decay_ratio=0.9):
    n_warmup_steps = int(warmup_ratio * n_steps)
    n_decay_steps = int(decay_ratio * n_steps)

    warmup = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=n_warmup_steps,
    )
    decay = LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.01,
        total_iters=n_decay_steps,
    )
    scheduler = SequentialLR(
        optimizer=optimizer,
        schedulers=[warmup, decay],
        milestones=[n_warmup_steps],
    )

    return scheduler
