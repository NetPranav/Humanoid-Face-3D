"""
Live Execution Log Monitor & Real-Time Sync Daemon.
Pipes running background task progress, Kaggle dataset creation status,
and training metrics directly into view/execution_log.md.
"""
import os
import sys
import time
import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXECUTION_LOG = PROJECT_ROOT / "view" / "execution_log.md"


def get_latest_upload_progress() -> str:
    """Extracts latest upload percentage and speed from task logs."""
    task_dir = Path("/Users/pranav/.gemini/antigravity-ide/brain/e7dc96bd-2100-432f-8969-3fd1f51b4e6e/.system_generated/tasks")
    if not task_dir.exists():
        return "No active background tasks."

    log_files = sorted(list(task_dir.glob("task-*.log")), key=lambda p: p.stat().st_mtime, reverse=True)
    if not log_files:
        return "No active background tasks."

    latest_log = log_files[0]
    try:
        lines = latest_log.read_text(errors='ignore').strip().split('\n')
        for l in reversed(lines):
            match = re.search(r'(\d+%\s*\|.*?\])', l)
            if match:
                return f"`{latest_log.name}`: {match.group(1).strip()}"
            if "Upload successful" in l or "Your private Dataset" in l:
                return f"`{latest_log.name}`: {l.strip()}"
    except Exception as e:
        return f"Reading task log: {e}"

    return "Task active, awaiting chunk progress..."


def check_active_kernel() -> str:
    """Queries Kaggle CLI for running kernel status."""
    try:
        env = os.environ.copy()
        env['KAGGLE_API_TOKEN'] = 'KGAT_2ef9ab9c57c5ca109e7af862a81c6b21'
        res = subprocess.run(
            ['kaggle', 'kernels', 'status', 'nightshowdown/phase-2-5-geometry-preprocessing-1024'],
            capture_output=True, text=True, env=env, timeout=15
        )
        if res.returncode == 0:
            return res.stdout.strip()
        return res.stderr.strip()
    except Exception as e:
        return str(e)


def update_log_file():
    """Reads execution_log.md and updates the active status section."""
    if not EXECUTION_LOG.exists():
        return

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    kernel_status = check_active_kernel()

    content = EXECUTION_LOG.read_text()
    
    # Append kernel status update if status changed or every few minutes
    new_event = f"`[{timestamp}]` ☁️ Kaggle Cloud Status: {kernel_status}\n"
    
    if kernel_status and (kernel_status not in content or "RUNNING" in kernel_status):
        with open(EXECUTION_LOG, "a") as f:
            f.write(new_event)


if __name__ == '__main__':
    update_log_file()

