import os
import asyncio
import logging


def main():
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
    job = sched.add_job(
        job,
        CronTrigger(minute=minute, hour=hour, day=dom, month=month, day_of_week=dow),
    )
    # Log schedule details and next run time
    try:
        logging.info(
            "scheduler: cron=%s %s/%s next_run=%s",
            cron,
            tz.key if hasattr(tz, "key") else str(tz),
            tz,
            getattr(job, "next_run_time", None),
        )
    except Exception:
        pass
    sched.start()
    try:
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
