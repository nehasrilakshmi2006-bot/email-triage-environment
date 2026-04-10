"""
Email Triage OpenEnv — Single-file server (flat layout, no subfolders needed)
"""
import sys, traceback, random, uuid
from typing import Any, Dict
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# ── Email data ────────────────────────────────────────────────────────────────

EMAILS = {
    "task_easy": [
        {"id":"e1","subject":"URGENT: Server is down in production",
         "body":"Our main production server crashed. Customers cannot access the app.",
         "sender":"ops@company.com",
         "expected_priority":"urgent","expected_category":"technical","expected_route":"engineering"},
        {"id":"e2","subject":"Happy Birthday wishes",
         "body":"Hey, just wanted to wish you a happy birthday!",
         "sender":"friend@personal.com",
         "expected_priority":"low","expected_category":"personal","expected_route":"no_action"},
        {"id":"e3","subject":"Invoice #1042 due today",
         "body":"Please find Invoice #1042 for $5000 due today. Kindly process payment.",
         "sender":"billing@vendor.com",
         "expected_priority":"high","expected_category":"billing","expected_route":"finance"},
    ],
    "task_medium": [
        {"id":"m1","subject":"Re: Partnership Proposal",
         "body":"We are interested in a co-marketing deal worth $50K. Waiting for your response before end of quarter.",
         "sender":"partnerships@bigcorp.com",
         "expected_priority":"high","expected_category":"business","expected_route":"sales"},
        {"id":"m2","subject":"New feature request from enterprise client",
         "body":"Acme Corp ($120K/year) wants custom SSO integration or they may cancel next month.",
         "sender":"account-manager@company.com",
         "expected_priority":"urgent","expected_category":"product","expected_route":"engineering"},
        {"id":"m3","subject":"Quarterly newsletter",
         "body":"Here is our Q1 newsletter. Industry updates, team achievements, upcoming events.",
         "sender":"newsletter@industry.org",
         "expected_priority":"low","expected_category":"newsletter","expected_route":"no_action"},
    ],
    "task_hard": [
        {"id":"h1","subject":"Re: Re: Data breach concern",
         "body":"Suspicious logins from multiple countries. Third time following up. Concerned about data exposure.",
         "sender":"worried.customer@gmail.com",
         "expected_priority":"urgent","expected_category":"security","expected_route":"security_team"},
        {"id":"h2","subject":"Interview feedback + offer discussion",
         "body":"Completed all 5 rounds. Have another offer expiring Friday. Senior engineer role. Team is interested.",
         "sender":"candidate@gmail.com",
         "expected_priority":"high","expected_category":"hr","expected_route":"hr_team"},
        {"id":"h3","subject":"Fwd: Legal notice regarding IP dispute",
         "body":"Legal notice from attorneys re: IP claim. Action required within 14 days or default judgment. From CEO.",
         "sender":"ceo-assistant@company.com",
         "expected_priority":"urgent","expected_category":"legal","expected_route":"legal_team"},
    ],
}

TASKS_META = [
    {"id":"task_easy",  "name":"Basic Email Prioritization","description":"Classify simple emails by priority and category.","difficulty":"easy"},
    {"id":"task_medium","name":"Email Routing",              "description":"Classify moderately complex emails and route to correct team.","difficulty":"medium"},
    {"id":"task_hard",  "name":"Complex Multi-Factor Triage","description":"Handle nuanced emails: security, legal, HR scenarios.","difficulty":"hard"},
]

PRIORITY_ORDER = ["low","medium","high","urgent"]

# ── Environment state (module-level, single session) ──────────────────────────

_state: Dict[str, Any] = {
    "episode_id": None,
    "task_id": None,
    "email_queue": [],
    "current_index": 0,
    "step_count": 0,
    "total_reward": 0.0,
    "done": False,
}


def _reset(task_id: str = "task_easy") -> dict:
    if task_id not in EMAILS:
        task_id = "task_easy"
    queue = list(EMAILS[task_id])
    random.shuffle(queue)
    _state.update({
        "episode_id":    str(uuid.uuid4()),
        "task_id":       task_id,
        "email_queue":   queue,
        "current_index": 0,
        "step_count":    0,
        "total_reward":  0.0,
        "done":          False,
    })
    task_meta = next(t for t in TASKS_META if t["id"] == task_id)
    email = queue[0]
    return {
        "episode_id": _state["episode_id"],
        "step": 0,
        "observation": {
            "done": False,
            "reward": 0.0,
            "metadata": {
                "task_id":          task_id,
                "task_name":        task_meta["name"],
                "task_description": task_meta["description"],
                "email": {
                    "subject": email["subject"],
                    "body":    email["body"],
                    "sender":  email["sender"],
                },
                "emails_remaining": len(queue),
                "valid_priorities": ["urgent","high","medium","low"],
                "valid_categories": ["technical","billing","personal","business","product",
                                     "newsletter","security","hr","legal","other"],
                "valid_routes":     ["engineering","finance","sales","hr_team","legal_team",
                                     "security_team","no_action","management"],
            },
        },
    }


def _grade(action: dict, email: dict) -> tuple:
    priority = str(action.get("priority","")).lower().strip()
    category = str(action.get("category","")).lower().strip()
    route    = str(action.get("route","")).lower().strip()
    score = 0.0
    parts = []

    # Priority — 0.4 weight, partial credit for adjacent
    if priority == email["expected_priority"]:
        score += 0.4
        parts.append(f"✓ priority={priority}")
    else:
        try:
            ei = PRIORITY_ORDER.index(email["expected_priority"])
            gi = PRIORITY_ORDER.index(priority)
            if abs(ei - gi) == 1:
                score += 0.2
                parts.append(f"~ priority={priority} (expected {email['expected_priority']})")
            else:
                parts.append(f"✗ priority={priority} (expected {email['expected_priority']})")
        except ValueError:
            parts.append(f"✗ priority={priority} invalid")

    # Category — 0.3 weight
    if category == email["expected_category"]:
        score += 0.3
        parts.append(f"✓ category={category}")
    else:
        parts.append(f"✗ category={category} (expected {email['expected_category']})")

    # Route — 0.3 weight
    if route == email["expected_route"]:
        score += 0.3
        parts.append(f"✓ route={route}")
    else:
        parts.append(f"✗ route={route} (expected {email['expected_route']})")

    return round(min(max(score, 0.0), 1.0), 3), " | ".join(parts)


def _step(action: dict) -> dict:
    if _state["done"]:
        return {"observation":{"done":True,"reward":0.0,"metadata":{}},"reward":0.0,"done":True,"info":{"error":"Episode done, call /reset"}}

    idx   = _state["current_index"]
    email = _state["email_queue"][idx]
    reward, feedback = _grade(action, email)

    _state["step_count"]   += 1
    _state["total_reward"] += reward
    _state["current_index"] += 1

    done = _state["current_index"] >= len(_state["email_queue"])
    _state["done"] = done

    if not done:
        next_email = _state["email_queue"][_state["current_index"]]
        next_meta = {
            "email": {"subject": next_email["subject"], "body": next_email["body"], "sender": next_email["sender"]},
            "emails_remaining": len(_state["email_queue"]) - _state["current_index"],
        }
    else:
        next_meta = {}

    return {
        "observation": {
            "done":   done,
            "reward": reward,
            "metadata": next_meta,
        },
        "reward": reward,
        "done":   done,
        "info": {
            "feedback":         feedback,
            "step":             _state["step_count"],
            "cumulative_reward": round(_state["total_reward"], 4),
            "average_reward":   round(_state["total_reward"] / _state["step_count"], 4),
        },
    }


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Email Triage OpenEnv", version="1.0.0")


async def _safe_json(request: Request) -> dict:
    try:
        body = await request.body()
        if not body or body.strip() in (b"", b"null"):
            return {}
        return await request.json()
    except Exception:
        return {}


@app.get("/")
async def root():
    return {"name":"Email Triage OpenEnv","version":"1.0.0","status":"running",
            "endpoints":["/reset","/step","/state","/tasks","/health"]}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/reset")
async def reset(request: Request):
    try:
        body    = await _safe_json(request)
        task_id = body.get("task_id") or body.get("extra_data", {}).get("task_id") or "task_easy"
        return JSONResponse(content=_reset(task_id))
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/step")
async def step(request: Request):
    try:
        body   = await _safe_json(request)
        action = body.get("action") or {k: body[k] for k in ("priority","category","route") if k in body}
        if not action:
            action = {"priority":"medium","category":"other","route":"management"}
        return JSONResponse(content=_step(action))
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/state")
async def state():
    return JSONResponse(content={
        "episode_id":  _state["episode_id"],
        "task_id":     _state["task_id"],
        "step_count":  _state["step_count"],
        "total_reward": round(_state["total_reward"], 4),
        "done":        _state["done"],
        "emails_done": _state["current_index"],
        "emails_total": len(_state["email_queue"]),
    })


@app.get("/tasks")
async def tasks():
    return JSONResponse(content={"tasks": TASKS_META})


@app.exception_handler(Exception)
async def _err(request: Request, exc: Exception):
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"error": str(exc)})
