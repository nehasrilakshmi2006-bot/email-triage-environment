#!/usr/bin/env python3
"""
inference.py — Email Triage OpenEnv Baseline Agent
Prints [START], [STEP], [END] logs to stdout.
"""
import os, sys, json, time, requests
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy-key")

TASKS = ["task_easy", "task_medium", "task_hard"]

SYSTEM_PROMPT = """You are an expert email triage assistant.
Classify each email with EXACT fields:
- priority: one of [urgent, high, medium, low]
- category: one of [technical, billing, personal, business, product, newsletter, security, hr, legal, other]
- route:    one of [engineering, finance, sales, hr_team, legal_team, security_team, no_action, management]
Respond ONLY with valid JSON. No explanation. No markdown.
Example: {"priority": "urgent", "category": "technical", "route": "engineering"}"""

FALLBACK = {"priority": "medium", "category": "other", "route": "management"}
VALID_P  = {"urgent","high","medium","low"}
VALID_C  = {"technical","billing","personal","business","product","newsletter","security","hr","legal","other"}
VALID_R  = {"engineering","finance","sales","hr_team","legal_team","security_team","no_action","management"}


def call_llm(subject, body, sender):
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":f"From: {sender}\nSubject: {subject}\nBody: {body}\n\nReturn JSON."},
            ],
            max_tokens=80, temperature=0.0,
        )
        content = resp.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"): content = content[4:]
        action = json.loads(content.strip())
        action["priority"] = str(action.get("priority","")).lower().strip()
        action["category"] = str(action.get("category","")).lower().strip()
        action["route"]    = str(action.get("route","")).lower().strip()
        if action["priority"] not in VALID_P: action["priority"] = FALLBACK["priority"]
        if action["category"] not in VALID_C: action["category"] = FALLBACK["category"]
        if action["route"]    not in VALID_R: action["route"]    = FALLBACK["route"]
        return action
    except Exception as e:
        print(f"LLM error: {e}", file=sys.stderr)
        return dict(FALLBACK)


def run_task(task_id):
    obs      = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=30).json()
    ep_id    = obs.get("episode_id","?")
    task_obs = obs.get("observation",{}).get("metadata",{})
    email    = task_obs.get("email",{})

    print(json.dumps({"type":"[START]","task_id":task_id,"task_name":task_obs.get("task_name",task_id),"episode_id":ep_id,"timestamp":time.time()}), flush=True)

    step_num=0; total=0.0; done=False
    while not done:
        step_num += 1
        action = call_llm(email.get("subject",""), email.get("body",""), email.get("sender",""))
        r      = requests.post(f"{ENV_BASE_URL}/step", json={"action": action}, timeout=30).json()
        reward = float(r.get("reward",0.0))
        done   = bool(r.get("done",True))
        info   = r.get("info",{})
        total += reward

        print(json.dumps({"type":"[STEP]","task_id":task_id,"episode_id":ep_id,"step":step_num,"action":action,"reward":round(reward,4),"cumulative_reward":round(total,4),"done":done,"feedback":info.get("feedback","")}), flush=True)

        if not done:
            email = r.get("observation",{}).get("metadata",{}).get("email", email)

    state = requests.get(f"{ENV_BASE_URL}/state", timeout=30).json()
    avg   = state.get("total_reward", total) / max(step_num,1)
    print(json.dumps({"type":"[END]","task_id":task_id,"episode_id":ep_id,"total_steps":step_num,"total_reward":round(total,4),"average_reward":round(avg,4),"score":round(avg,4)}), flush=True)
    return avg


def main():
    print(f"model={MODEL_NAME} env={ENV_BASE_URL}", file=sys.stderr, flush=True)
    for attempt in range(12):
        try:
            if requests.get(f"{ENV_BASE_URL}/health", timeout=5).status_code == 200:
                break
        except Exception:
            pass
        time.sleep(5)

    scores = {}
    for task_id in TASKS:
        try:
            scores[task_id] = run_task(task_id)
        except Exception as e:
            print(f"FAILED {task_id}: {e}", file=sys.stderr)
            scores[task_id] = 0.0

    overall = sum(scores.values()) / len(scores)
    print(f"OVERALL: {overall:.4f}", file=sys.stderr, flush=True)

if __name__ == "__main__":
    main()
