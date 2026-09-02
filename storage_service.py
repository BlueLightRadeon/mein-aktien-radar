import os
import json
import base64
import urllib.request
import streamlit as st

PORTFOLIO_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_portfolio.json")

def sync_portfolio_to_github(items, cash):
    """Sichert das Depot NUR bei Upload/Änderung auf GitHub."""
    token = str(st.secrets.get("GITHUB_TOKEN", "")).strip()
    repo = str(st.secrets.get("GITHUB_REPO", "")).strip()
    if not token or not repo:
        return
    try:
        url = f"https://api.github.com/repos/{repo}/contents/saved_portfolio.json"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Streamlit-Stock-Radar"
        }
        sha = None
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode())
                sha = data.get("sha")
        except Exception:
            pass

        content_str = json.dumps({"items": items, "cash": float(cash)}, indent=2, ensure_ascii=False)
        content_b64 = base64.b64encode(content_str.encode()).decode()

        payload = {
            "message": "Update saved_portfolio.json",
            "content": content_b64
        }
        if sha:
            payload["sha"] = sha

        req_put = urllib.request.Request(
            url, 
            data=json.dumps(payload).encode(), 
            headers=headers, 
            method="PUT"
        )
        with urllib.request.urlopen(req_put, timeout=4.0) as resp:
            pass
    except Exception:
        pass

def delete_portfolio_from_github():
    token = str(st.secrets.get("GITHUB_TOKEN", "")).strip()
    repo = str(st.secrets.get("GITHUB_REPO", "")).strip()
    if not token or not repo:
        return
    try:
        url = f"https://api.github.com/repos/{repo}/contents/saved_portfolio.json"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Streamlit-Stock-Radar"
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode())
            sha = data.get("sha")
        if sha:
            payload = {
                "message": "Delete saved_portfolio.json",
                "sha": sha
            }
            req_del = urllib.request.Request(
                url, 
                data=json.dumps(payload).encode(), 
                headers=headers, 
                method="DELETE"
            )
            with urllib.request.urlopen(req_del, timeout=4.0) as resp:
                pass
    except Exception:
        pass

def load_saved_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("items", []), float(data.get("cash", 0.0))
        except Exception:
            pass

    token = str(st.secrets.get("GITHUB_TOKEN", "")).strip()
    repo = str(st.secrets.get("GITHUB_REPO", "")).strip()
    if token and repo:
        try:
            url = f"https://api.github.com/repos/{repo}/contents/saved_portfolio.json"
            headers = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Streamlit-Stock-Radar"
            }
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                res_data = json.loads(resp.read().decode())
                content_b64 = res_data.get("content", "")
                if content_b64:
                    content_str = base64.b64decode(content_b64.replace("\n", "")).decode("utf-8")
                    parsed = json.loads(content_str)
                    items = parsed.get("items", [])
                    cash = float(parsed.get("cash", 0.0))
                    try:
                        with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
                            json.dump({"items": items, "cash": cash}, f, indent=2, ensure_ascii=False)
                    except Exception:
                        pass
                    return items, cash
        except Exception:
            pass

    return [], 0.0

def save_saved_portfolio(items, cash, sync_github=True):
    try:
        with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
            json.dump({"items": items, "cash": float(cash)}, f, indent=2, ensure_ascii=False)
    except Exception:
        pass
    if sync_github:
        sync_portfolio_to_github(items, cash)

def delete_saved_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            os.remove(PORTFOLIO_FILE)
        except Exception:
            pass
    delete_portfolio_from_github()
