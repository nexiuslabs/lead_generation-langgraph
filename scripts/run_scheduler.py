import os
import asyncio
import logging


async def main_async():
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger
        from zoneinfo import ZoneInfo
    except Exception as e:
        logging.error(
            "APScheduler not installed or import failed: %s. Install 'apscheduler' to use the scheduler.",
            e,
        )
        return

    from run_nightly import run_all

    logging.basicConfig(level=logging.INFO)
    tz = ZoneInfo("Asia/Singapore")
    cron = os.getenv("SCHED_START_CRON", "0 1 * * *")
    try:
        minute, hour, dom, month, dow = cron.split()
    except Exception:
        minute, hour, dom, month, dow = "0", "1", "*", "*", "*"

    async def job():
        try:
            await run_all()
        except Exception as exc:
            logging.exception("nightly run failed: %s", exc)

    sched = AsyncIOScheduler(timezone=tz)
    cron_job = sched.add_job(
        job,
        CronTrigger(minute=minute, hour=hour, day=dom, month=month, day_of_week=dow),
    )
    # Start the scheduler inside a running event loop
    sched.start()
    try:
        # After start(), next_run_time is available
        logging.info(
            "scheduler: cron=%s %s/%s next_run=%s",
            cron,
            tz.key if hasattr(tz, "key") else str(tz),
            tz,
            getattr(cron_job, "next_run_time", None),
        )
    except Exception:
        pass

    # Keep the loop alive forever
    stop = asyncio.Event()
    await stop.wait()


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass
