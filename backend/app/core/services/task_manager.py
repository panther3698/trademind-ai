import asyncio
import signal
import logging
from datetime import datetime, time as dt_time, timedelta
import psutil

IST_OFFSET = timedelta(hours=5, minutes=30)

class TaskManager:
    def __init__(self):
        self.tasks = {}
        self.health = {}
        self.restart_count = {}
        self.shutdown_event = asyncio.Event()
        self.market_open = dt_time(9, 15)
        self.market_close = dt_time(15, 30)
        self.start_time = datetime.now()
        self.resource_stats = {}
        self.registered_tasks = {}

    def register_task(self, name: str, coro, priority: str, market_hours_only: bool = False):
        """Register a task with the TaskManager"""
        self.registered_tasks[name] = {
            'coro': coro,
            'priority': priority,
            'market_hours_only': market_hours_only
        }
        logging.info(f"Registered task: {name} with priority {priority}")

    def get_health(self):
        """Get health status of all tasks"""
        return self.health.copy()

    def get_active_tasks(self):
        """Get list of active task names"""
        return list(self.tasks.keys())

    def get_resource_usage(self):
        """Get current resource usage"""
        try:
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent
            }
        except:
            return {'cpu_percent': 0, 'memory_percent': 0}

    def is_market_hours(self):
        now = (datetime.utcnow() + IST_OFFSET).time()
        return self.market_open <= now <= self.market_close

    def get_uptime(self):
        return (datetime.now() - self.start_time).total_seconds()

    async def start(self):
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
            except NotImplementedError:
                pass  # Not available on Windows event loop
        await self.schedule_tasks()

    async def schedule_tasks(self):
        """Schedule all registered tasks"""
        for name, task_info in self.registered_tasks.items():
            self.tasks[name] = asyncio.create_task(
                self.run_task(name, task_info['coro'], task_info['priority'], task_info['market_hours_only'])
            )

    async def run_task(self, name, coro, priority, market_hours_only=False):
        self.restart_count.setdefault(name, 0)
        while not self.shutdown_event.is_set():
            try:
                # Skip market_hours_only tasks during off-hours
                if market_hours_only and not self.is_market_hours():
                    await asyncio.sleep(60)  # Check every minute
                    continue
                
                interval = self.get_interval(priority)
                start = datetime.now()
                await coro()
                elapsed = (datetime.now() - start).total_seconds()
                self.health[name] = {'last_run': datetime.now().isoformat(), 'status': 'ok', 'last_duration': elapsed}
                self.resource_stats[name] = self.get_resource_usage()
            except Exception as e:
                logging.error(f"Task {name} failed: {e}")
                self.health[name] = {'last_run': datetime.now().isoformat(), 'status': f'error: {e}'}
                self.restart_count[name] += 1
                if priority == 'critical':
                    logging.warning(f"Restarting critical task {name}")
            await asyncio.sleep(self.get_interval(priority))

    def get_interval(self, priority):
        if self.is_market_hours():
            if priority == 'critical':
                return 5
            elif priority == 'important':
                return 30
            else:
                return 60
        else:
            if priority == 'critical':
                return 60
            elif priority == 'important':
                return 300
            else:
                return 3600  # Pause nice-to-have tasks

    async def shutdown(self):
        self.shutdown_event.set()
        for task in self.tasks.values():
            task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        logging.info("All background tasks shut down gracefully.")

    # --- Task stubs to be replaced with actual logic ---
    async def market_monitor(self):
        # TODO: Use shared cache, fetch only if needed
        pass

    async def regime_monitor(self):
        pass

    async def news_monitor(self):
        pass

    async def signal_generation(self):
        pass 