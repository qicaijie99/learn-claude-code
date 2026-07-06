"""
Execute the 3 MCP steps directly against the mock code in code.py.
Safe import: use importlib to avoid name collision with stdlib 'code'.
"""
import sys, json, importlib.util

# Load code.py via importlib to avoid collision with stdlib 'code'
spec = importlib.util.spec_from_file_location("agent_code", "code.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Grab the symbols we need
connect_mcp      = mod.connect_mcp
mcp_clients      = mod.mcp_clients
assemble_tool_pool = mod.assemble_tool_pool
normalize_mcp_name = mod.normalize_mcp_name

def clear_mcp():
    """Clear all MCP connections (no disconnect_mcp exists)."""
    mcp_clients.clear()

def show_pool():
    """Print all MCP tools in current tool pool."""
    tools, handlers = assemble_tool_pool()
    mcp_tools = [t for t in tools if t['name'].startswith('mcp__')]
    print(f"  → MCP tools in pool: {[t['name'] for t in mcp_tools]}")
    print(f"  → Total pool size: {len(tools)} tools")
    return tools, handlers

# ===== STEP 1: Connect to docs → search "authentication" → show tools =====
print("=" * 70)
print("STEP 1: Connect to 'docs' server, search, show tools")
print("=" * 70)

clear_mcp()
r1 = connect_mcp('docs')
print(f"connect_mcp('docs') => {r1}")

dc = mcp_clients.get('docs')
if dc:
    sr = dc.call_tool('search', {'query': 'authentication'})
    print(f"docs.search(query='authentication') => {sr}")
    vr = dc.call_tool('get_version', {})
    print(f"docs.get_version() => {vr}")

show_pool()

# ===== STEP 2: Connect to deploy → trigger a deployment → show tools =====
print("\n" + "=" * 70)
print("STEP 2: Connect to 'deploy' server, trigger deployment, show tools")
print("=" * 70)

clear_mcp()
r2 = connect_mcp('deploy')
print(f"connect_mcp('deploy') => {r2}")

dc2 = mcp_clients.get('deploy')
if dc2:
    tr = dc2.call_tool('trigger', {'service': 'api-gateway'})
    print(f"deploy.trigger(service='api-gateway') => {tr}")
    st = dc2.call_tool('status', {'service': 'api-gateway'})
    print(f"deploy.status(service='api-gateway') => {st}")

show_pool()

# ===== STEP 3: Connect BOTH servers → list all tools =====
print("\n" + "=" * 70)
print("STEP 3: Connect both servers and list ALL available tools")
print("=" * 70)

clear_mcp()
r3a = connect_mcp('docs')
r3b = connect_mcp('deploy')
print(f"connect_mcp('docs')  => {r3a}")
print(f"connect_mcp('deploy') => {r3b}")
print(f"Connected clients: {list(mcp_clients.keys())}")

tools_all, handlers_all = show_pool()

# Print full details of MCP tools
mcp_all = [t for t in tools_all if t['name'].startswith('mcp__')]
print(f"\nAll MCP tool details:")
for t in mcp_all:
    print(f"  \033[33m{t['name']}\033[0m")
    print(f"    Description: {t.get('description', '')}")
    print(f"    Schema: {json.dumps(t.get('input_schema', {}), indent=6)}")

# Summary
non_mcp_count = len([t for t in tools_all if not t['name'].startswith('mcp__')])
print(f"\n{'='*70}")
print(f"SUMMARY:")
print(f"  Builtin tools:  {non_mcp_count}")
print(f"  MCP tools:      {len(mcp_all)}")
print(f"  Total pool:     {len(tools_all)}")
print(f"{'='*70}")
print("Done! All 3 steps executed successfully. meow!")
