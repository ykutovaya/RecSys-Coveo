"""Монотонные сроки и ограниченный пул вычислений без записи состояния."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore
from typing import TypeVar

T = TypeVar("T")
STAGES = ("input", "state", "representation", "sources", "ranking", "record", "return")


class ServiceError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


async def await_until(
    operation: Awaitable[T], limit: float, *, clock: Callable[[], float] = time.perf_counter
) -> T:
    """Один запуск с абсолютным сроком; ранний таймер asyncio не отменяет работу."""
    task = asyncio.ensure_future(operation)

    def consume(completed: asyncio.Future[T]) -> None:
        if not completed.cancelled():
            completed.exception()

    try:
        while True:
            remaining = limit - clock()
            if remaining <= 0:
                raise TimeoutError
            if task.done():
                return task.result()
            # Windows loop может пробудить таймер раньше perf_counter-срока.
            # asyncio.wait не отменяет task: проверяем действительный остаток.
            await asyncio.wait({task}, timeout=remaining)
    except BaseException:
        task.cancel()
        task.add_done_callback(consume)
        raise


class Deadline:
    def __init__(
        self,
        start: float | None = None,
        *,
        timeout_ms: int = 1000,
        reserve_ms: int = 100,
        clock: Callable[[], float] = time.perf_counter,
    ):
        if (
            type(timeout_ms) is not int
            or type(reserve_ms) is not int
            or not 0 < reserve_ms < timeout_ms <= 2**63 - 1
        ):
            raise ValueError("Нужны целые 0 < reserve_ms < timeout_ms в пределах int64")
        self.timeout_ms, self.reserve_ms = timeout_ms, reserve_ms
        self.clock = clock
        self.start = clock() if start is None else start
        self.end = self.start + timeout_ms / 1000
        self.compute_end = self.start + (timeout_ms - reserve_ms) / 1000

    def stage_limit(self, stage: str) -> float:
        if stage not in STAGES:
            raise ValueError("Неизвестный этап запроса")
        return self.end if stage in {"record", "return"} else self.compute_end

    def check(self, limit: float | None = None) -> None:
        if self.clock() >= (self.end if limit is None else min(limit, self.end)):
            raise ServiceError("backend_timeout")

    @property
    def elapsed_ms(self) -> int:
        return max(0, int((self.clock() - self.start) * 1000))


class BoundedWork:
    """Только чистые операции: позднее завершение не может опубликовать результат.

    Отмена ожидания не освобождает занятый поток. Следующий запрос получает отказ,
    если все потоки ещё работают; скрытой неограниченной очереди у executor нет.
    """

    def __init__(self, workers: int = 4):
        if not 1 <= workers <= 16:
            raise ValueError("Число вычислительных потоков должно быть от 1 до 16")
        self._slots = BoundedSemaphore(workers)
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="coveo-pure")

    async def run(self, operation: Callable[[], T], limit: float) -> T:
        if time.perf_counter() >= limit:
            raise ServiceError("backend_timeout")
        if not self._slots.acquire(blocking=False):
            raise ServiceError("overload_rejected")
        try:
            future = self._pool.submit(operation)
        except BaseException:
            self._slots.release()
            raise
        future.add_done_callback(lambda _: self._slots.release())
        try:
            return await await_until(asyncio.wrap_future(future), limit)
        except TimeoutError as exc:
            raise ServiceError("backend_timeout") from exc

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
