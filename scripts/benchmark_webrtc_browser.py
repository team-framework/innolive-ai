#!/usr/bin/env python3
"""Measure marked synthetic-camera video in Chrome; never opens a physical camera.

Requires Playwright and an installed Chrome. The Y4M input must contain the 16-bit
frame IDs from benchmark_webrtc.MovingPortrait in its top 256x16 pixels.
Latency is local preview presentation to returned-video presentation, not sensor age.
"""

import argparse
import asyncio
import json
import statistics
from itertools import pairwise
from pathlib import Path

from playwright.async_api import async_playwright

HOOK = """window.__pcs=[];window.__reports=[];const P=RTCPeerConnection;
window.RTCPeerConnection=class extends P {
 constructor(...a){super(...a);__pcs.push(this)}
 createDataChannel(...a){const c=super.createDataChannel(...a);
 c.addEventListener('message',e=>{try{__reports.push({...JSON.parse(e.data),at:performance.now()})}catch{}});return c;}
};"""
OBSERVER = """() => {
 const source=document.querySelector('#source'),result=document.querySelector('#result');
 const c=document.createElement('canvas');c.width=256;c.height=16;
 const x=c.getContext('2d',{willReadFrequently:true});
 const seen=new Map(),rows=[];let lastSource=-1,lastResult=-1;
 window.measurement={rows,done:false,start:performance.now()};
 const read=e=>{x.drawImage(e,0,0,256,16,0,0,256,16);const p=x.getImageData(0,0,256,16).data;let seq=0;
 for(let b=0;b<16;b++){let v=0;for(let y=4;y<12;y++)for(let px=b*16+4;px<b*16+12;px++)v+=p[(y*256+px)*4];
 if(v/64>128)seq|=1<<b;}return seq;};
 const tick=()=>{if(measurement.done)return;const now=performance.now();try{
 if(source.readyState>=2){const seq=read(source);if(seq!==lastSource){lastSource=seq;seen.set(seq,now);}}
 if(result.readyState>=2&&!result.hidden){const seq=read(result);if(seq!==lastResult&&seen.has(seq)){
 lastResult=seq;rows.push({seq,now,latency:now-seen.get(seq)});}}
 }catch{}requestAnimationFrame(tick);};tick();
}"""


def percentile(values, fraction):
    return sorted(values)[min(len(values) - 1, int(len(values) * fraction))]


async def measure(args):
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path=str(args.browser),
            headless=True,
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-video-capture={args.fake_video.resolve()}",
                "--autoplay-policy=no-user-gesture-required",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
            ],
        )
        context = await browser.new_context(
            permissions=["camera"], viewport={"width": 1440, "height": 1050}
        )

        async def one(name, url, barrier):
            page = await context.new_page()
            sid = None
            try:
                await page.add_init_script(HOOK)
                await page.goto(url, wait_until="networkidle")
                await page.get_by_role("button", name="새 세션", exact=True).click()
                await page.wait_for_timeout(750)
                sid = await page.locator("#session").input_value()
                await page.get_by_role("button", name="카메라 시작", exact=True).click()
                await page.wait_for_function(
                    "document.querySelector('#status').textContent==='합성 중'", timeout=60000
                )
                await page.wait_for_timeout(args.warmup * 1000)
                await barrier.wait()
                await page.evaluate(OBSERVER)
                before = await page.evaluate(
                    "async()=>Array.from((await __pcs[0].getStats()).values())"
                )
                await page.wait_for_timeout(args.seconds * 1000)
                rows = await page.evaluate("measurement.done=true;measurement.rows")
                status = await page.locator("#status").inner_text()
                assert status == "합성 중" and len(rows) > 50, (name, status, len(rows))
                after = await page.evaluate(
                    "async()=>Array.from((await __pcs[0].getStats()).values())"
                )
                reports = await page.evaluate("__reports.filter(r=>r.at>=measurement.start)")
                intervals = [b["now"] - a["now"] for a, b in pairwise(rows)]
                latency = [r["latency"] for r in rows]
                frames = [r for r in reports if r.get("type") == "frame"]
                result = {
                    "url": url,
                    "seconds": args.seconds,
                    "frames": len(rows),
                    "fps": (len(rows) - 1) * 1000 / (rows[-1]["now"] - rows[0]["now"]),
                    "preview_to_output_p50_ms": statistics.median(latency),
                    "preview_to_output_p95_ms": percentile(latency, 0.95),
                    "interval_p95_ms": percentile(intervals, 0.95),
                    "gaps_over_100ms": sum(v > 100 for v in intervals),
                    "server_p95_ms": percentile([r["server_ms"] for r in frames], 0.95),
                    "server_max_ms": max(r["server_ms"] for r in frames),
                    "stats_before": before,
                    "stats_after": after,
                    "samples": rows,
                    "reports": reports,
                }
                for field in ("freezeCount", "totalFreezesDuration", "packetsLost"):
                    result[field] = sum(
                        s.get(field, 0) for s in after if s["type"] == "inbound-rtp"
                    ) - sum(s.get(field, 0) for s in before if s["type"] == "inbound-rtp")
                results[name] = result
                print(
                    name,
                    {
                        k: v
                        for k, v in result.items()
                        if k not in {"stats_before", "stats_after", "samples", "reports"}
                    },
                    flush=True,
                )
                await page.screenshot(path=str(args.output / f"{name}.png"), full_page=True)
                (args.output / "result.json").write_text(
                    json.dumps(results, indent=2, ensure_ascii=False)
                )
                # Keep every fake camera alive until all concurrent measurements
                # are collected; stopping one can interrupt Chrome's shared source.
                await barrier.wait()
            except BaseException:
                await barrier.abort()
                raise
            finally:
                await page.close()
                if sid:
                    await context.request.delete(url + "/api/sessions/" + sid, max_retries=2)

        for target in args.url:
            name, url = target.split("=", 1)
            barrier = asyncio.Barrier(args.sessions)
            await asyncio.gather(
                *(
                    one(
                        name if args.sessions == 1 else f"{name}-s{index + 1}",
                        url.rstrip("/"),
                        barrier,
                    )
                    for index in range(args.sessions)
                )
            )
        await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", action="append", required=True, help="label=https://host:port")
    parser.add_argument("--fake-video", type=Path, required=True)
    parser.add_argument("--browser", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=1, choices=range(1, 9))
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--warmup", type=float, default=5)
    parser.add_argument("--output", type=Path, default=Path("reports/webrtc/browser"))
    args = parser.parse_args()
    if not 10 <= args.seconds <= 300 or not 0 <= args.warmup <= 30:
        parser.error("seconds must be 10..300 and warmup 0..30")
    if not args.fake_video.is_file() or not args.browser.is_file():
        parser.error("fake video and browser paths must exist")
    for target in args.url:
        label, separator, url = target.partition("=")
        if (
            not separator
            or not label.replace("-", "").replace("_", "").isalnum()
            or not url.startswith("https://")
        ):
            parser.error("url must be label=https://host:port")
    asyncio.run(measure(args))
