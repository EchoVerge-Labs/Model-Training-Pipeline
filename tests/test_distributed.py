"""The multi-rank resume check, run for real: two processes over the gloo backend.

(The trainers use NCCL on the GPUs; gloo exercises the same torch.distributed calls on CPU.)
"""

from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pipeline.common.distributed import assert_ranks_agree_on_resume


def _worker(rank: int, world: int, store: str, steps: list[int], out: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
    try:
        assert_ranks_agree_on_resume(steps[rank], torch.device("cpu"))
        result = "ok"
    except RuntimeError as e:
        result = f"raised: {e}"
    Path(f"{out}/{rank}").write_text(result)
    dist.destroy_process_group()


def _run(tmp_path, steps: list[int]) -> list[str]:
    out = tmp_path / "out"
    out.mkdir()
    mp.spawn(_worker, args=(2, str(tmp_path / "store"), steps, str(out)), nprocs=2, join=True)
    return [(out / str(r)).read_text() for r in range(2)]


def test_ranks_resuming_from_the_same_step_pass(tmp_path):
    assert _run(tmp_path, [3000, 3000]) == ["ok", "ok"]


def test_ranks_that_disagree_all_fail_together_instead_of_hanging(tmp_path):
    results = _run(tmp_path, [3000, 1500])
    assert all(r.startswith("raised: ranks disagree on the resume checkpoint") for r in results)


def test_a_fresh_start_on_every_rank_passes(tmp_path):
    assert _run(tmp_path, [-1, -1]) == ["ok", "ok"]
