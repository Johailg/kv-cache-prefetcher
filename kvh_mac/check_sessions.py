import sys
from transformers import AutoTokenizer
try:
    from sessions import build_sessions
except ImportError:
    from kvh.sessions import build_sessions

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
s = build_sessions(sys.argv[1], tok, 2048, 6, 200, "cpu")
print("SESSIONS BUILT:", len(s))
assert len(s) > 0, "FATAL: build_sessions returned 0"

x = s[0]
print("type:", type(x).__name__)
if hasattr(x, "_fields"): print("fields:", x._fields)
elif isinstance(x, dict): print("keys:", list(x.keys()))
else: print("attrs:", [a for a in dir(x) if not a.startswith("_")][:20])
