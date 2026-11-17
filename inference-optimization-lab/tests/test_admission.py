import asyncio

import pytest

from qwen_serving_lab.admission import AdmissionLimiter, AdmissionRejected


def test_bounded_admission_rejects_overflow():
    async def check():
        limiter = AdmissionLimiter(max_active=1, max_waiting=0, wait_timeout_s=0.01)
        assert await limiter.acquire() == 0.0
        with pytest.raises(AdmissionRejected, match="queue_full"):
            await limiter.acquire()
        assert limiter.active == 1
        await limiter.release()
        assert limiter.active == 0
    asyncio.run(check())


def test_waiter_gets_released_slot():
    async def check():
        limiter = AdmissionLimiter(1, 1, 1.0)
        await limiter.acquire()
        waiter = asyncio.create_task(limiter.acquire())
        await asyncio.sleep(0.01)
        assert limiter.waiting == 1
        await limiter.release()
        assert await waiter >= 0
        assert limiter.active == 1
        await limiter.release()
    asyncio.run(check())


def test_timeout_cleans_waiting_count():
    async def check():
        limiter = AdmissionLimiter(1, 1, 0.01)
        await limiter.acquire()
        with pytest.raises(AdmissionRejected, match="queue_timeout"):
            await limiter.acquire()
        assert limiter.waiting == 0
        await limiter.release()
    asyncio.run(check())
