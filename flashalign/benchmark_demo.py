# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""Self-contained, offline playback. Only explicitly reviewed benign pairs export."""
import json
from pathlib import Path

from .benchmark import SCHEMA, digest, read_json, read_jsonl


def build_demo(run_path, case_id, output, *, repetition=0, note=""):
    if not isinstance(note, str) or len(note) > 2000:
        raise ValueError("Demo note must be text of at most 2000 characters")
    run_path = Path(run_path)
    run = read_json(run_path / "run.json")
    rows = [r for r in read_jsonl(run_path / "records.jsonl")
            if r["case_id"] == case_id and r["phase"] == "measured" and r["repetition"] == repetition]
    if len(rows) != 2 or {r["mode"] for r in rows} != {"off", "profile"}:
        raise ValueError("Select one complete off/profile pair")
    if any(r["status"] != "ok" or not r.get("demo_safe") or r["group"] != "safe" for r in rows):
        raise ValueError("Demo export requires a completed, explicitly reviewed benign pair")
    if rows[0]["messages"] != rows[1]["messages"] or rows[0]["case_sha256"] != rows[1]["case_sha256"]:
        raise ValueError("The pair does not contain identical prompts")
    # Reasoning text remains private. Visible text uses its actual capture time;
    # a thinking marker preserves that interval without distributing reasoning.
    public_rows = [{"mode": r["mode"], "order": r["order"], "metrics": r["metrics"],
                    "events": [{"at_s": e["at_s"], **({"content": e["content"]} if e.get("content") else {}),
                                **({"thinking": True} if e.get("reasoning") else {})}
                               for e in r["events"] if e.get("content") or e.get("reasoning")]}
                   for r in rows]
    payload = {"schema": SCHEMA, "case_id": case_id, "created_utc": run["created_utc"],
               "manifest": run["manifest"], "request_settings": run["request_settings"],
               "messages": rows[0]["messages"], "rows": public_rows,
               "warmups": run["warmups"], "repetition": repetition,
               "profile_scale": run["scale"],
               "note": note,
               "recording_pair_sha256": digest(rows)}
    # Safe even if model text contains </script> or HTML. DOM writes use textContent.
    data = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    html = TEMPLATE.replace("__RECORDING_JSON__", data).replace("__LOGO_DATA__", LOGO_DATA)
    output = Path(output)
    fd = output.open("x", encoding="utf-8")
    try:
        output.chmod(0o600)
        fd.write(html)
    finally:
        fd.close()
    return payload


# Original FlashAlign SVG artwork, copied from assets/brand/flashalign-readme.svg.
# Kept inline so a pip-installed client can export a self-contained demo offline.
LOGO_DATA = "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4MDAiIGhlaWdodD0iMTgwIiB2aWV3Qm94PSIwIDAgODAwIDE4MCIgcm9sZT0iaW1nIiBhcmlhLWxhYmVsbGVkYnk9InRpdGxlIGRlc2MiPjx0aXRsZSBpZD0idGl0bGUiPkZsYXNoQWxpZ24gYnkgQWV2b25peCBSZXNlYXJjaDwvdGl0bGU+PGRlc2MgaWQ9ImRlc2MiPlRocmVlIHNlcGFyYXRlZCBhbmd1bGFyIHN0cm9rZXMgZm9ybSBhIGxpZ2h0bmluZyBib2x0LiBPdXRsaW5lZCBGTEFTSEFMSUdOIHdvcmRtYXJrLiBPbmUgdHJhbnNwYXJlbnQgbG9nbyBpbWFnZSBmb3IgcmVzcG9uc2l2ZSBNYXJrZG93biByZWFkZXJzLjwvZGVzYz48ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg3Ny44NjAgMjIpIiBzdHJva2U9IiMzODQ2NDAiIHN0cm9rZS13aWR0aD0iMC45IiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNCAxMikgc2NhbGUoMC45KSI+PGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEwIC0yKSI+PHBhdGggZmlsbD0iI0ZGQjg3NSIgZD0iTTczIDRIMTE2TDg0IDQwSDQ2WiIvPjxwYXRoIGZpbGw9IiNGNTkxODciIGQ9Ik00MCA0OEgxMDhMODAgNzRIMjVaIi8+PHBhdGggZmlsbD0iIzc2RENDQiIgZD0iTTMyIDgySDcyTDI5IDEyMEw0NCA5MEgyNFoiLz48L2c+PC9nPjxwYXRoIGZpbGw9IiNGRkYxREQiIGQ9Ik0xNTYuMDY5IDkxVjM5Ljg4SDE5MS45NjVWNDkuNzEySDE2Ny45NjNWNjEuMTQ2SDE5MC42NjlWNzAuODg2SDE2Ny45NjNWOTFaTTE5OS43NzMgOTFWMzkuODhIMjExLjY2N1Y4MS4xNjhIMjM0Ljg0MlY5MVpNMjM4LjU5MiA5MSAyNTcuMDI0IDM5Ljg4SDI3MC45MjNMMjg5LjM1NSA5MUgyNzcuMDkyTDI3My41ODcgODAuODYySDI1NC4zMDhMMjUwLjgwMyA5MVpNMjU3LjU4OSA3MS4yODlIMjcwLjM1OUwyNjQgNTIuNTg5Wk0zMTYuNDc5IDkyLjE1MlEzMDkuNzQ2IDkyLjE1MiAzMDQuOTM5IDg5LjkyUTMwMC4xMzIgODcuNjg4IDI5Ny40NDcgODMuNjM0UTI5NC43NjEgNzkuNTgxIDI5NC4zODEgNzQuMDg2TDMwNi4zMDQgNzMuNDk4UTMwNi43MDQgNzYuMzgxIDMwNy45NjYgNzguMzk3UTMwOS4yMjcgODAuNDEzIDMxMS4zNjUgODEuNDQ0UTMxMy41MDQgODIuNDc1IDMxNi42MjMgODIuNDc1UTMxOS4xODYgODIuNDc1IDMyMC45OCA4MS44NzZRMzIyLjc3NSA4MS4yNzcgMzIzLjcxOCA4MC4wNDdRMzI0LjY2MSA3OC44MTggMzI0LjY2MSA3Ny4wMDNRMzI0LjY2MSA3NS4zNTMgMzIzLjg1NiA3NC4wODdRMzIzLjA1MSA3Mi44MjEgMzIwLjc0OSA3MS43OTJRMzE4LjQ0NiA3MC43NjIgMzEzLjkzMyA2OS43NjlRMzA3LjI1MSA2OC4zMDggMzAzLjE2MyA2Ni40MzFRMjk5LjA3NSA2NC41NTMgMjk3LjIxNiA2MS42NVEyOTUuMzU3IDU4Ljc0NyAyOTUuMzU3IDU0LjMwOVEyOTUuMzU3IDQ5LjY3OCAyOTcuNjk0IDQ2LjE2OFEzMDAuMDMxIDQyLjY1OSAzMDQuNDE4IDQwLjY5NFEzMDguODA0IDM4LjcyOCAzMTQuOTQxIDM4LjcyOFEzMjEuNDM4IDM4LjcyOCAzMjUuODg1IDQwLjk4MlEzMzAuMzMyIDQzLjIzNSAzMzIuODM3IDQ3LjE2NlEzMzUuMzQzIDUxLjA5OCAzMzUuODY3IDU2LjI0NEwzMjQuMDQ4IDU2Ljc4UTMyMy44MjMgNTQuMjAyIDMyMi42NjQgNTIuMzM0UTMyMS41MDUgNTAuNDY3IDMxOS41NCA0OS40MzZRMzE3LjU3NiA0OC40MDUgMzE0Ljc5NyA0OC40MDVRMzExLjI4OSA0OC40MDUgMzA5LjMxMSA0OS45MjFRMzA3LjMzMiA1MS40MzcgMzA3LjMzMiA1My45MTdRMzA3LjMzMiA1NS43MjMgMzA4LjI0MSA1Ni45MzRRMzA5LjE0OSA1OC4xNDUgMzExLjMyOSA1OC45OTZRMzEzLjUxIDU5Ljg0NyAzMTcuNDA2IDYwLjY0NVEzMjQuNjQ0IDYyLjAyMiAzMjguODY0IDY0LjI0M1EzMzMuMDg1IDY2LjQ2NSAzMzQuODYgNjkuNTAxUTMzNi42MzYgNzIuNTM2IDMzNi42MzYgNzYuNDY4UTMzNi42MzYgODEuMjk0IDMzNC4yMDIgODQuODNRMzMxLjc2OSA4OC4zNjUgMzI3LjI0NCA5MC4yNThRMzIyLjcyIDkyLjE1MiAzMTYuNDc5IDkyLjE1MlpNMzQ1LjQyOSA5MVYzOS44OEgzNTcuMzIzVjYwLjM1NEgzNzUuNDU5VjM5Ljg4SDM4Ny4zNTNWOTFIMzc1LjQ1OVY3MC4xODZIMzU3LjMyM1Y5MVpNMzkzLjc1MiA5MSA0MTIuMTg0IDM5Ljg4SDQyNi4wODNMNDQ0LjUxNSA5MUg0MzIuMjUyTDQyOC43NDcgODAuODYySDQwOS40NjhMNDA1Ljk2MyA5MVpNNDEyLjc0OSA3MS4yODlINDI1LjUxOUw0MTkuMTYgNTIuNTg5Wk00NTAuOTA5IDkxVjM5Ljg4SDQ2Mi44MDNWODEuMTY4SDQ4NS45NzhWOTFaTTQ5My40NjEgOTFWMzkuODhINTA1LjM1NVY5MVpNNTM3LjE2MiA5Mi4xNTJRNTI5Ljc1NSA5Mi4xNTIgNTI0LjQzNSA4OC43NzRRNTE5LjExNiA4NS4zOTYgNTE2LjI2IDc5LjQwOVE1MTMuNDA1IDczLjQyMyA1MTMuNDA1IDY1LjUxMlE1MTMuNDA1IDU3LjgwOCA1MTYuMzIyIDUxLjc2UTUxOS4yNCA0NS43MTIgNTI0LjcyNSA0Mi4yMlE1MzAuMjEgMzguNzI4IDUzNy45MjggMzguNzI4UTU0NC42NyAzOC43MjggNTQ5LjI5NyA0MC45MzRRNTUzLjkyMyA0My4xNCA1NTYuNzE4IDQ3LjIxNVE1NTkuNTEzIDUxLjI5MSA1NjAuNjU3IDU2Ljg3Mkw1NDguMzE5IDU3LjQwOFE1NDcuNTkgNTMuMjY5IDU0NS4xOCA1MC45MTVRNTQyLjc2OSA0OC41NiA1MzguMDMyIDQ4LjU2UTUzMy45MSA0OC41NiA1MzEuMTU2IDUwLjc2NlE1MjguNDAxIDUyLjk3MiA1MjcuMDIgNTYuNzgxUTUyNS42MzkgNjAuNTkgNTI1LjYzOSA2NS41MTJRNTI1LjYzOSA3MC40NjMgNTI2Ljk3OCA3NC4yNTFRNTI4LjMxNyA3OC4wNCA1MzEuMTA4IDgwLjE4UTUzMy44OTkgODIuMzIgNTM4LjIzOSA4Mi4zMlE1NDEuNDI3IDgyLjMyIDU0My43MjggODEuMDk3UTU0Ni4wMyA3OS44NzUgNTQ3LjM4OSA3Ny42ODlRNTQ4Ljc0OCA3NS41MDMgNTQ5LjAyOCA3Mi42NjNINTM4LjAyVjY0LjI3MUg1NjAuODAxVjkxSDU1My42MDFMNTUzLjE4MyA4My45NlE1NTIuNDA2IDg1LjQyNyA1NTEuMjEzIDg2LjY5M1E1NDguNzkxIDg5LjI2MyA1NDUuMTY0IDkwLjcwOFE1NDEuNTM3IDkyLjE1MiA1MzcuMTYyIDkyLjE1MlpNNTY5LjIwNSA5MVYzOS44OEg1ODEuOTI5TDYwMS4yOTEgNzMuMTk2VjM5Ljg4SDYxMy4xODVWOTFINjAwLjM0MUw1ODEuMDk5IDU4Ljk2MlY5MVoiLz48L2c+PC9zdmc+Cg=="

TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'; base-uri 'none'; form-action 'none'">
<title>FlashAlign | Recorded off/on comparison</title>
<style>
:root{color-scheme:dark;--ink:#fff1dd;--dim:#afa99f;--line:#353833;--teal:#76dccb;--amber:#ffb875;--coral:#f59187}
*{box-sizing:border-box}body{margin:0;background:#101311;color:var(--ink);font:16px/1.5 system-ui,sans-serif}
main{max-width:1440px;margin:auto;padding:36px 42px 28px}header{display:flex;justify-content:space-between;align-items:center;gap:24px}
.logo{width:335px;max-width:50vw;height:auto}.recorded{font-size:12px;letter-spacing:.16em;color:var(--teal);text-transform:uppercase}
h1{font-size:36px;font-weight:550;letter-spacing:-.025em;margin:32px 0 8px}.deck{color:var(--dim);margin:0 0 24px;max-width:960px}
.meta{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}.tag{background:#1c221e;border:1px solid var(--line);padding:5px 10px;border-radius:5px;font-size:12px}
.qualification{border:1px solid var(--amber);padding:10px 14px;color:var(--amber);font-size:13px}
.prompt{border-left:3px solid var(--amber);padding:12px 18px;background:#1a1e1b;white-space:pre-wrap;margin:18px 0 24px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.panel{border:1px solid var(--line);background:#171c18;border-radius:12px;overflow:hidden}
.panel.on{border-color:#457d71}.panel-head{padding:15px 20px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center}
h2{font-size:17px;margin:0;font-weight:650}.on h2{color:var(--teal)}.status{font-size:12px;color:var(--dim)}
.answer{padding:22px;min-height:280px;max-height:480px;overflow-y:auto;white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.65 system-ui,sans-serif}
.stats{padding:14px 20px;border-top:1px solid var(--line);display:flex;gap:24px;flex-wrap:wrap;background:#131814}
.stat b{display:block;font-variant-numeric:tabular-nums;font-size:20px;font-weight:550}.stat span{font-size:11px;color:var(--dim)}
.controls{display:flex;align-items:center;gap:10px;margin:22px 0 12px;flex-wrap:wrap}button,select{font:inherit;color:var(--ink);border:1px solid #565d54;background:#232a24;border-radius:6px;padding:7px 14px;cursor:pointer}
button:hover{border-color:var(--teal)}input[type=range]{accent-color:var(--teal);flex:1;min-width:140px}#time{font-variant-numeric:tabular-nums;color:var(--dim);font-size:13px;min-width:110px}
footer{color:var(--dim);font-size:11px;border-top:1px solid var(--line);padding-top:14px;margin-top:18px}details{margin-top:12px;color:var(--dim);font-size:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere}
@media(max-width:760px){main{padding:20px}h1{font-size:27px}.grid{grid-template-columns:1fr}.logo{width:240px;max-width:62vw}.answer{min-height:200px}.recorded{font-size:9px}.stats{gap:16px}}
</style></head><body><main>
<header><img class="logo" alt="FlashAlign lightning bolt and wordmark" src="data:image/svg+xml;base64,__LOGO_DATA__"><span class="recorded">Recorded comparison</span></header>
<h1>ReAlign models instantly, on demand.</h1><p class="deck">Two requests to the same running engine, shown side by side. Only the signed alignment mode changes.</p>
<p class="qualification" id="qualification" hidden></p>
<div class="meta" id="meta"></div><div class="prompt" id="prompt"></div>
<div class="grid"><section class="panel"><div class="panel-head"><h2>FlashAlign off</h2><span class="status" id="off-status">Ready</span></div><div class="answer" id="off-answer"></div><div class="stats" id="off-stats"></div></section>
<section class="panel on"><div class="panel-head"><h2>FlashAlign on</h2><span class="status" id="profile-status">Ready</span></div><div class="answer" id="profile-answer"></div><div class="stats" id="profile-stats"></div></section></div>
<div class="controls"><button id="play">Play</button><button id="reset">Replay</button><select id="speed" aria-label="Playback speed"><option value="1">1× actual timing</option><option value="2">2× playback</option><option value="4">4× playback</option></select><input type="range" id="seek" min="0" step="0.01" aria-label="Playback position"><span id="time"></span></div>
<footer id="disclosure"></footer><details><summary>Recording provenance and generation settings</summary><pre id="provenance"></pre></details>
</main><script id="recording" type="application/json">__RECORDING_JSON__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('recording').textContent), byId=id=>document.getElementById(id);
const duration=Math.max(...data.rows.map(r=>r.metrics.total_s)), range=byId('seek');range.max=duration;
let elapsed=0,playing=false,previous=null;
function text(id,value){byId(id).textContent=value}
function tag(value){const el=document.createElement('span');el.className='tag';el.textContent=value;byId('meta').append(el)}
tag(data.manifest.model.id);tag(data.manifest.engine.name+' '+data.manifest.engine.version);tag(data.manifest.model.quantization);
tag('Speculative decoding '+(data.manifest.speculation.enabled?'on':'off'));tag('Profile: '+data.manifest.profile.name+' · scale '+data.profile_scale);
text('prompt',data.messages.map(m=>m.role.toUpperCase()+': '+m.content).join('\n\n'));
text('qualification',data.note||'');byId('qualification').hidden=!data.note;
text('disclosure','Recorded sequentially, not a simultaneous load test. Requests are replayed from their own start times. '+data.warmups+' warmup pair(s); cache state was not controlled. Sample '+(data.repetition+1)+'. TTFT includes reasoning, network and queue time. Output token count may include reasoning. No synthetic text or token counts.');
text('provenance',JSON.stringify({recorded_utc:data.created_utc,recording_pair_sha256:data.recording_pair_sha256,case_id:data.case_id,manifest:data.manifest,request_settings:data.request_settings},null,2));
function stat(parent,value,label){const node=document.createElement('div');node.className='stat';const b=document.createElement('b'),s=document.createElement('span');b.textContent=value;s.textContent=label;node.append(b,s);parent.append(node)}
function format(n,unit){return n==null?'Not reported':n.toFixed(3)+unit}
function render(){range.value=elapsed;text('time',elapsed.toFixed(1)+' / '+duration.toFixed(1)+'s');
for(const row of data.rows){let answer='',thinking=false;for(const event of row.events){if(event.at_s>elapsed)break;if(event.content)answer+=event.content;if(event.thinking)thinking=true}
const done=elapsed>=row.metrics.total_s;text(row.mode+'-answer',answer);const pane=byId(row.mode+'-answer');pane.scrollTop=pane.scrollHeight;
text(row.mode+'-status',done?'Complete':answer?'Responding':thinking?'Thinking':elapsed>0?'Waiting':'Ready');
const box=byId(row.mode+'-stats');box.replaceChildren();stat(box,elapsed>=row.metrics.ttft_s&&row.metrics.ttft_s!=null?format(row.metrics.ttft_s,'s'):'—','Time to first token');
stat(box,done&&row.metrics.completion_tokens!=null?String(row.metrics.completion_tokens):'—','Output tokens, reported');
stat(box,done?format(row.metrics.total_s,'s'):'—','Total request time');
if(done&&row.metrics.engine_decode_tps!=null)stat(box,row.metrics.engine_decode_tps.toFixed(1),'Decode tok/s, engine timing');
else if(done&&row.metrics.stream_decode_tps_estimate!=null)stat(box,row.metrics.stream_decode_tps_estimate.toFixed(1),'Decode tok/s, stream estimate');
}}
function toggle(){playing=!playing;if(playing&&elapsed>=duration)elapsed=0;previous=null;byId('play').textContent=playing?'Pause':'Play'}
byId('play').onclick=toggle;byId('reset').onclick=()=>{elapsed=0;playing=true;previous=null;byId('play').textContent='Pause';render()};
range.oninput=()=>{elapsed=Number(range.value);render()};
function frame(now){if(playing){if(previous!==null)elapsed=Math.min(duration,elapsed+(now-previous)/1000*Number(byId('speed').value));previous=now;if(elapsed>=duration){playing=false;byId('play').textContent='Play'}render()}requestAnimationFrame(frame)}
window.flashalignPlayback={duration,setTime:s=>{playing=false;elapsed=Math.max(0,Math.min(duration,s));render()},play:()=>{if(!playing)toggle()},data};
render();requestAnimationFrame(frame);if(new URLSearchParams(location.search).get('autoplay')==='1')toggle();
</script></body></html>'''
