from __future__ import annotations

import asyncio
import unittest

from service.runtime import PriorityLane


async def _settle() -> None:
    for _ in range(4):
        await asyncio.sleep(0)


class PriorityLaneTests(unittest.IsolatedAsyncioTestCase):
    async def test_vod_waiter_never_takes_the_lane_before_a_queued_live_waiter(self):
        lane = PriorityLane()
        order: list[str] = []
        await lane.acquire(live=True)  # a live frame holds the lane

        async def live_waiter() -> None:
            await lane.acquire(live=True)
            order.append("live")
            await asyncio.sleep(0.01)
            lane.release()

        async def vod_waiter() -> None:
            await lane.acquire(live=False)
            order.append("vod")
            lane.release()

        # Queue the VOD waiter FIRST so only priority (not arrival order) can let live win.
        vod_task = asyncio.create_task(vod_waiter())
        await _settle()
        live_task = asyncio.create_task(live_waiter())
        await _settle()
        self.assertEqual(lane.live_waiting, 1)

        lane.release()  # the holder lets go
        await asyncio.gather(live_task, vod_task)

        self.assertEqual(order, ["live", "vod"])

    async def test_live_waiters_keep_fifo_order(self):
        lane = PriorityLane()
        order: list[int] = []
        await lane.acquire(live=True)

        async def waiter(index: int) -> None:
            await lane.acquire(live=True)
            order.append(index)
            lane.release()

        tasks = []
        for index in range(1, 4):
            tasks.append(asyncio.create_task(waiter(index)))
            await _settle()  # register each waiter in arrival order

        lane.release()
        await asyncio.gather(*tasks)
        self.assertEqual(order, [1, 2, 3])

    async def test_vod_proceeds_immediately_without_live_demand(self):
        lane = PriorityLane()
        self.assertTrue(await lane.acquire(live=False))
        self.assertTrue(lane.locked())
        lane.release()
        self.assertFalse(lane.locked())

    async def test_live_jumps_ahead_of_a_waiting_vod_frame(self):
        lane = PriorityLane()
        order: list[str] = []
        await lane.acquire(live=False)  # a VOD frame holds the lane

        async def vod_waiter() -> None:
            await lane.acquire(live=False)
            order.append("vod")
            lane.release()

        async def live_waiter() -> None:
            await lane.acquire(live=True)
            order.append("live")
            lane.release()

        vod_task = asyncio.create_task(vod_waiter())
        await _settle()
        live_task = asyncio.create_task(live_waiter())
        await _settle()

        lane.release()  # release the VOD holder; live must win the free lane
        await asyncio.gather(live_task, vod_task)
        self.assertEqual(order, ["live", "vod"])


if __name__ == "__main__":
    unittest.main()
