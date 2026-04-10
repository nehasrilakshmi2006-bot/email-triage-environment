---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Email Triage OpenEnv

RL environment where an AI agent triages emails by priority, category, and routing.

## API
- `POST /reset` — start new episode
- `POST /step` — submit triage action
- `GET /state` — get episode state
- `GET /tasks` — list tasks

## Tasks
- `task_easy` — basic priority/category classification
- `task_medium` — adds routing decisions
- `task_hard` — security, legal, HR nuanced cases

## Action
```json
{"priority": "urgent", "category": "technical", "route": "engineering"}
```

## Reward: 0.0 – 1.0
- 0.4 for correct priority
- 0.3 for correct category
- 0.3 for correct route
