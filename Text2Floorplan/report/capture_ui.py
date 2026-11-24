"""Capture real screenshots of the browser editor for the report deck.

Starts headless Chrome, opens the running editor (python -m web_editor.server),
loads a generated plan JSON through the editor's own "Open JSON" input, and
saves 1920x1080 screenshots of the split, 2D and 3D views to report/ui/.
Usage: .venv/bin/python report/capture_ui.py <plan.json> "<prompt text>"
"""
import base64
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import websocket

ROOT = Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT = 9333
plan, prompt = Path(sys.argv[1]).resolve(), sys.argv[2]
out = ROOT / "ui"
out.mkdir(exist_ok=True)
profile = tempfile.mkdtemp(prefix="t2f-chrome-")
chrome = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}",
                           f"--user-data-dir={profile}", "--window-size=1920,1080",
                           "--hide-scrollbars", "about:blank"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(50):
        try:
            targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json/list"))
            page = next(t for t in targets if t["type"] == "page")
            break
        except Exception:
            time.sleep(0.2)
    ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=60)
    ids = itertools.count(1)

    def call(method, **params):
        message_id = next(ids)
        ws.send(json.dumps({"id": message_id, "method": method, "params": params}))
        while True:
            reply = json.loads(ws.recv())
            if reply.get("id") == message_id:
                if "error" in reply:
                    raise RuntimeError(reply["error"])
                return reply.get("result", {})

    def js(expression):
        return call("Runtime.evaluate", expression=expression, awaitPromise=True,
                    returnByValue=True).get("result", {}).get("value")

    def shot(name):
        time.sleep(2.5)
        data = call("Page.captureScreenshot", format="png")["data"]
        (out / name).write_bytes(base64.b64decode(data))
        print("saved", out / name)

    call("Emulation.setDeviceMetricsOverride", width=1920, height=1080, deviceScaleFactor=1, mobile=False)
    call("Page.enable")
    call("Page.navigate", url="http://127.0.0.1:8765/en")
    time.sleep(3)
    root = call("DOM.getDocument")["root"]["nodeId"]
    node = call("DOM.querySelector", nodeId=root, selector="#file")["nodeId"]
    call("DOM.setFileInputFiles", nodeId=node, files=[str(plan)])
    js("document.getElementById('file').dispatchEvent(new Event('change', {bubbles: true}))")
    time.sleep(4)
    js(f"document.getElementById('prompt').value = {json.dumps(prompt)}")
    shot("ui_split.png")
    js("document.getElementById('view-plan').click()")
    shot("ui_2d.png")
    js("document.getElementById('view-model').click()")
    shot("ui_3d.png")
    ws.close()
finally:
    chrome.terminate()
    chrome.wait(timeout=10)
    shutil.rmtree(profile, ignore_errors=True)
