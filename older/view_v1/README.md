# Humanoid-Face-3D: View & Live Monitoring Dashboard

This folder contains human-readable logs, CLI run outputs, and job statuses for all Kaggle cloud runs and local pipeline executions.

---

## Active Logs & Reports
* **[view/execution_log.md](execution_log.md):** Chronological log of cloud jobs, CLI commands, training status, and checkpoint saves.
* **[view/latest_job_status.json](latest_job_status.json):** Structured JSON metadata for active Kaggle background jobs.

---

## How to Inspect Progress
1. Keep `view/execution_log.md` open in your editor.
2. The agent automatically writes real-time loss metrics, epoch progress, ETA, and download links into this folder during execution.
