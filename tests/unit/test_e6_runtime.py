import asyncio
import threading

import pytest

from coveo_recsys.serving.runtime import BoundedWork, Deadline, ServiceError


@pytest.mark.parametrize(
    "stage", ["input", "state", "representation", "sources", "ranking", "record", "return"]
)
@pytest.mark.parametrize("offset,expired", [(-0.000001, False), (0, True), (0.000001, True)])
def test_stage_and_total_deadlines_are_absolute(stage, offset, expired):
    now = [100.0]
    deadline = Deadline(100.0, timeout_ms=1000, reserve_ms=100, clock=lambda: now[0])
    limit = deadline.stage_limit(stage)
    assert limit == (101.0 if stage in {"record", "return"} else 100.9)
    now[0] = limit + offset
    if expired:
        with pytest.raises(ServiceError, match="backend_timeout"):
            deadline.check(limit)
    else:
        deadline.check(limit)


def test_retry_does_not_extend_absolute_deadline():
    now = [1.0]
    deadline = Deadline(1.0, clock=lambda: now[0])
    now[0] += 0.5
    assert deadline.stage_limit("ranking") == deadline.compute_end == 1.9
    deadline.check(deadline.stage_limit("ranking"))
    now[0] = 1.9
    with pytest.raises(ServiceError):
        deadline.check(deadline.stage_limit("ranking"))


@pytest.mark.parametrize(
    "timeout_ms,reserve_ms",
    [
        (0, 1),
        (-1, 1),
        (True, 1),
        (1000.0, 100),
        (1000, False),
        (1000, 0),
        (1000, -1),
        (1000, 1000),
        (1000, 1001),
        (2**63, 100),
    ],
)
def test_deadline_rejects_invalid_timeout_or_reserve(timeout_ms, reserve_ms):
    with pytest.raises(ValueError):
        Deadline(timeout_ms=timeout_ms, reserve_ms=reserve_ms)


def test_timed_out_work_keeps_slot_until_it_really_finishes():
    async def scenario():
        release = threading.Event()
        pool = BoundedWork(1)
        try:
            with pytest.raises(ServiceError, match="backend_timeout"):
                await pool.run(lambda: release.wait(2), Deadline().clock() + 0.01)
            with pytest.raises(ServiceError, match="overload_rejected"):
                await pool.run(lambda: 42, Deadline().clock() + 1)
        finally:
            release.set()
            pool.close()

    asyncio.run(scenario())


def test_cancelled_work_never_commits_a_late_result():
    async def scenario():
        pool = BoundedWork(1)
        entered, release = threading.Event(), threading.Event()
        results = []

        def calculate():
            entered.set()
            release.wait(2)
            return "late"

        async def request():
            results.append(await pool.run(calculate, Deadline().clock() + 1))

        task = asyncio.create_task(request())
        try:
            while not entered.is_set():
                await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            with pytest.raises(ServiceError, match="overload_rejected"):
                await pool.run(lambda: "wrong slot reuse", Deadline().clock() + 1)
        finally:
            release.set()
            pool.close()
        assert results == []

    asyncio.run(scenario())


def test_early_timer_wakeup_keeps_same_task_and_absolute_limit(monkeypatch):
    from coveo_recsys.serving.runtime import await_until

    async def scenario():
        now = [1.0]
        release = asyncio.Event()
        calls, seen, waits = [], [], []
        real_wait = asyncio.wait

        async def operation():
            calls.append("started")
            await release.wait()
            return 42

        async def premature(tasks, *, timeout):
            seen.append(next(iter(tasks)))
            waits.append(timeout)
            if len(seen) < 3:
                now[0] += 0.001
                await asyncio.sleep(0)
                return set(), tasks
            release.set()
            return await real_wait(tasks, timeout=1)

        monkeypatch.setattr(asyncio, "wait", premature)
        assert await await_until(operation(), 1.010, clock=lambda: now[0]) == 42
        assert calls == ["started"] and len({id(task) for task in seen}) == 1
        assert waits == pytest.approx([0.010, 0.009, 0.008])

    asyncio.run(scenario())


@pytest.mark.parametrize("offset,expired", [(-0.000001, False), (0, True), (0.000001, True)])
def test_finished_task_at_actual_deadline_is_still_timeout(monkeypatch, offset, expired):
    from coveo_recsys.serving.runtime import await_until

    async def scenario():
        now = [1.0]
        real_wait = asyncio.wait

        async def operation():
            return 7

        async def finished(tasks, *, timeout):
            result = await real_wait(tasks, timeout=1)
            now[0] = 2.0 + offset
            return result

        monkeypatch.setattr(asyncio, "wait", finished)
        if expired:
            with pytest.raises(TimeoutError):
                await await_until(operation(), 2.0, clock=lambda: now[0])
        else:
            assert await await_until(operation(), 2.0, clock=lambda: now[0]) == 7

    asyncio.run(scenario())


def test_expired_wait_does_not_start_operation():
    from coveo_recsys.serving.runtime import await_until

    async def scenario():
        calls = []

        async def operation():
            calls.append("late")

        with pytest.raises(TimeoutError):
            await await_until(operation(), 1.0, clock=lambda: 1.0)
        await asyncio.sleep(0)
        assert calls == []

    asyncio.run(scenario())


def test_cancellation_does_not_wait_for_resistant_task_and_consumes_late_error():
    from coveo_recsys.serving.runtime import await_until

    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        errors = []
        asyncio.get_running_loop().set_exception_handler(
            lambda _loop, context: errors.append(context)
        )

        async def operation():
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()
            raise RuntimeError("late failure")

        task = asyncio.create_task(await_until(operation(), 100.0, clock=lambda: 0.0))
        await entered.wait()
        task.cancel()
        for _ in range(4):
            await asyncio.sleep(0)
        try:
            assert task.done(), "Отмена не должна ждать игнорирующую её операцию"
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
        for _ in range(4):
            await asyncio.sleep(0)
        assert errors == []

    asyncio.run(scenario())
