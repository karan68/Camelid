import sys, json
def send(o):
    sys.stdout.write(json.dumps(o) + "\n"); sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    msg = json.loads(line)
    m, i = msg.get("method"), msg.get("id")
    if m == "initialize":
        send({"jsonrpc":"2.0","id":i,"result":{"protocolVersion":"2024-11-05"}})
    elif m == "tools/list":
        send({"jsonrpc":"2.0","id":i,"result":{"tools":[
            {"name":"lookup_part","description":"Look up a part number in the parts database and return its record.",
             "inputSchema":{"type":"object","properties":{"part":{"type":"string"}},"required":["part"]}}
        ]}})
    elif m == "tools/call":
        part = msg.get("params",{}).get("arguments",{}).get("part","?")
        send({"jsonrpc":"2.0","id":i,"result":{"content":[{"type":"text",
              "text":"part " + part + ": name=Flux Capacitor Bracket, qty_on_hand=42, bin=B7"}]}})
