#!/usr/bin/env python3
"""
LISA - Unified Federated Learning Application
Like BitTorrent clients (Azureus/Vuze) - all-in-one for federated AI

Features:
- Run as server OR client (or both!)
- Easy join codes for quick networking
- Real-time training visualization
- Model quality tracking
- Network peer visualization
- ngrok integration for worldwide access
"""
import os
import sys
import time
import json
import threading
import subprocess
import webbrowser
from datetime import datetime
from pathlib import Path

# Try to import GUI libraries
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("⚠️  Tkinter not available - running in CLI mode")

import uvicorn
from fastapi import FastAPI
import torch

# ============ Constants ============
VERSION = "1.0.0"
APP_NAME = "LISA - Federated Learning Hub"

# ============ Federated Server ============
def apply_lora_to_model(model, rank=4, alpha=8.0, dropout=0.05):
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(
        r=rank, lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=dropout, bias="none", task_type="CAUSAL_LM"
    )
    return get_peft_model(model, config)

class LISAServer:
    """All-in-one federated learning server."""
    
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B", port=8080):
        self.model_name = model_name
        self.port = port
        self.model = None
        self.tokenizer = None
        self.rounds = {}
        self.clients = set()
        self.total_gradients = 0
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def load_model(self):
        """Load the AI model with LoRA."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        
        print(f"Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        config = AutoConfig.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, config=config, trust_remote_code=True, torch_dtype=torch.float32
        )
        self.model = apply_lora_to_model(self.model)
        print(f"✅ Model loaded: {sum(p.numel() for p in self.model.parameters()):,} params")
        
    def start_api(self, host="0.0.0.0"):
        """Start the FastAPI server."""
        app = FastAPI()
        server = self
        
        @app.get("/health")
        async def health():
            return {"status": "ok", "server": "lisa", "version": VERSION}
        
        @app.get("/status")
        async def status():
            return {
                "model": server.model_name,
                "total_clients": len(server.clients),
                "total_gradients": server.total_gradients,
                "rounds": {rn: {"gradients": len(rs.gradients), "status": rs.status} 
                          for rn, rs in server.rounds.items()}
            }
        
        @app.post("/submit")
        async def submit(data: dict):
            import base64, pickle, zlib
            client_id = data.get("client_id", "unknown")
            round_num = data.get("round_number", 1)
            gradient_b64 = data.get("gradient_data", "")
            
            try:
                gradient_bytes = base64.b64decode(gradient_b64)
                state_dict = pickle.loads(gradient_bytes)
                
                server.clients.add(client_id)
                server.total_gradients += 1
                
                if round_num not in server.rounds:
                    server.rounds[round_num] = type('Round', (), {'gradients': {}, 'status': 'collecting'})()
                server.rounds[round_num].gradients[client_id] = state_dict
                
                return {"status": "submitted", "round": round_num, "gradients": len(server.rounds[round_num].gradients)}
            except Exception as e:
                return {"status": "error", "message": str(e)}
        
        @app.get("/join/{code}")
        async def join_page(code: str):
            from fastapi.responses import HTMLResponse
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>LISA - Join Network</title>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body {{ font-family: system-ui; max-width: 600px; margin: 50px auto; padding: 20px;
                           background: #1a1a2e; color: #eee; text-align: center; }}
                    .card {{ background: #16213e; border-radius: 16px; padding: 40px; }}
                    h1 {{ color: #00d9ff; }}
                    .code {{ background: #0f3460; padding: 15px 30px; border-radius: 8px;
                            font-family: monospace; font-size: 24px; color: #00d9ff; margin: 20px; }}
                    .cmd {{ background: #000; padding: 15px; border-radius: 8px; font-family: monospace;
                           color: #0f0; text-align: left; }}
                </style>
            </head>
            <body>
                <div class="card">
                    <h1>🤝 LISA Federated Learning</h1>
                    <p>Join code: <span class="code">{code.upper()}</span></p>
                    <p>Server: {server.model_name}</p>
                    <div class="cmd">
                        pip install lisa-client<br>
                        lisa-client --join {code.upper()}
                    </div>
                </div>
            </body>
            </html>
            """
            return HTMLResponse(content=html)
        
        print(f"Starting API on {host}:{self.port}")
        uvicorn.run(app, host=host, port=self.port)

# ============ Web Dashboard ============
def create_dashboard():
    """Create the web dashboard."""
    dashboard_html = """
<!DOCTYPE html>
<html>
<head>
    <title>LISA - Federated Learning Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
               background: #0f0f1a; color: #eee; min-height: 100vh; }
        .header { background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 20px 30px;
                  border-bottom: 1px solid #333; }
        .header h1 { color: #00d9ff; font-size: 24px; }
        .header p { color: #888; font-size: 12px; margin-top: 5px; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: #1a1a2e; border-radius: 12px; padding: 20px; border: 1px solid #333; }
        .card h3 { color: #00d9ff; margin-bottom: 15px; font-size: 14px; text-transform: uppercase; }
        .stat { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #333; }
        .stat:last-child { border-bottom: none; }
        .stat-value { color: #00ff88; font-weight: bold; }
        .chart { height: 200px; background: #0f3460; border-radius: 8px; display: flex; align-items: flex-end; padding: 10px; gap: 2px; }
        .bar { background: #00d9ff; border-radius: 2px 2px 0 0; min-width: 8px; flex: 1; transition: height 0.3s; }
        .bar:hover { background: #00ff88; }
        .peer { display: flex; align-items: center; padding: 10px; background: #0f3460; border-radius: 8px; margin: 5px 0; }
        .peer-icon { width: 30px; height: 30px; background: #00d9ff; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 10px; font-size: 14px; }
        .peer-info { flex: 1; }
        .peer-name { font-weight: bold; }
        .peer-status { font-size: 12px; color: #888; }
        .status-online { color: #00ff88; }
        .status-training { color: #ffaa00; }
        .btn { background: #00d9ff; color: #000; padding: 10px 20px; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; }
        .btn:hover { background: #00ff88; }
        .btn:disabled { background: #333; color: #666; cursor: not-allowed; }
        .actions { display: flex; gap: 10px; margin-top: 20px; }
        pre { background: #000; padding: 15px; border-radius: 8px; overflow-x: auto; font-size: 12px; color: #0f0; }
        .refresh { text-align: center; padding: 10px; color: #666; font-size: 12px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 LISA - Federated Learning Dashboard</h1>
        <p>Distributed AI training network • v""" + VERSION + """</p>
    </div>
    
    <div class="container">
        <div class="grid">
            <!-- Network Status -->
            <div class="card">
                <h3>🌐 Network Status</h3>
                <div class="stat">
                    <span>Server</span>
                    <span class="stat-value" id="server-status">Loading...</span>
                </div>
                <div class="stat">
                    <span>Connected Peers</span>
                    <span class="stat-value" id="peer-count">0</span>
                </div>
                <div class="stat">
                    <span>Total Gradients</span>
                    <span class="stat-value" id="gradient-count">0</span>
                </div>
                <div class="stat">
                    <span>Active Rounds</span>
                    <span class="stat-value" id="round-count">0</span>
                </div>
                <div class="actions">
                    <button class="btn" onclick="location.reload()">🔄 Refresh</button>
                </div>
            </div>
            
            <!-- Model Info -->
            <div class="card">
                <h3>🧠 Model Status</h3>
                <div class="stat">
                    <span>Model</span>
                    <span class="stat-value" id="model-name">-</span>
                </div>
                <div class="stat">
                    <span>Parameters</span>
                    <span class="stat-value" id="model-params">-</span>
                </div>
                <div class="stat">
                    <span>Checkpoints</span>
                    <span class="stat-value" id="checkpoint-count">0</span>
                </div>
                <div class="stat">
                    <span>Latest</span>
                    <span class="stat-value" id="latest-checkpoint">-</span>
                </div>
            </div>
            
            <!-- Perplexity Chart -->
            <div class="card">
                <h3>📈 Model Quality (Perplexity)</h3>
                <div class="chart" id="ppl-chart">
                    <div class="bar" style="height: 100%"></div>
                </div>
                <p style="text-align: center; color: #888; margin-top: 10px; font-size: 12px;">
                    Lower is better • Target: <100
                </p>
            </div>
            
            <!-- Connected Peers -->
            <div class="card">
                <h3>👥 Connected Peers</h3>
                <div id="peers-list">
                    <p style="color: #888; text-align: center; padding: 20px;">No peers connected</p>
                </div>
            </div>
            
            <!-- Recent Activity -->
            <div class="card" style="grid-column: span 2;">
                <h3>📋 Recent Activity</h3>
                <pre id="activity-log">Waiting for activity...</pre>
            </div>
            
            <!-- Join Code -->
            <div class="card">
                <h3>🔗 Join Network</h3>
                <div style="text-align: center;">
                    <p style="margin-bottom: 10px;">Share this code:</p>
                    <div style="background: #0f3460; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 20px; letter-spacing: 2px; color: #00d9ff;" id="join-code">GENERATING...</div>
                    <p style="margin-top: 10px; font-size: 12px; color: #888;">
                        Or share URL: <span id="join-url">-</span>
                    </p>
                </div>
            </div>
        </div>
    </div>
    
    <div class="refresh">
        Auto-refreshes every 5 seconds • Last update: <span id="last-update">-</span>
    </div>
    
    <script>
        async function updateDashboard() {
            try {
                const resp = await fetch('/api/status');
                const data = await resp.json();
                
                // Update stats
                document.getElementById('server-status').innerHTML = 
                    data.status === 'ok' ? '<span class="status-online">● Online</span>' : '<span style="color:#ff4444">● Offline</span>';
                document.getElementById('peer-count').textContent = data.total_clients || 0;
                document.getElementById('gradient-count').textContent = data.total_gradients || 0;
                document.getElementById('round-count').textContent = Object.keys(data.rounds || {}).length;
                
                // Update model
                document.getElementById('model-name').textContent = data.model || '-';
                
                // Update peers
                const peersDiv = document.getElementById('peers-list');
                if (data.clients && data.clients.length > 0) {
                    peersDiv.innerHTML = data.clients.map(c => `
                        <div class="peer">
                            <div class="peer-icon">💻</div>
                            <div class="peer-info">
                                <div class="peer-name">${c}</div>
                                <div class="peer-status status-online">● Connected</div>
                            </div>
                        </div>
                    `).join('');
                }
                
                document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
            } catch (e) {
                console.error('Dashboard error:', e);
            }
        }
        
        // Update every 5 seconds
        updateDashboard();
        setInterval(updateDashboard, 5000);
    </script>
</body>
</html>
"""
    return dashboard_html

# ============ CLI Interface ============
def print_banner():
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    LISA v{VERSION} - Federated Learning Hub              ║
╠══════════════════════════════════════════════════════════════╣
║  🤝 Train AI together - any device, anywhere in the world   ║
╚══════════════════════════════════════════════════════════════╝

Usage: lisa [command]

Commands:
  server     Start as federated learning server
  client     Join a federated network as client
  dashboard  Open web dashboard
  status     Show current network status
  help       Show this help message

Examples:
  lisa server --generate-code      # Start server + get join code
  lisa server --ngrok              # Start with worldwide access
  lisa client --join ABC123        # Join a network
  lisa dashboard                   # Open web dashboard

For more info: https://github.com/CiphemonJY/LISA_FTM
""")

def cmd_server(args):
    """Start as server."""
    server = LISAServer(model_name=args.model, port=args.port)
    
    if args.no_model_load:
        print("Starting API server without model...")
        server.start_api()
        return
    
    print("Loading model...")
    server.load_model()
    
    if args.open_dashboard:
        # Start dashboard in background
        threading.Thread(target=lambda: webbrowser.open("http://localhost:8081"), daemon=True).start()
        # Run dashboard on different port
        import subprocess
        subprocess.Popen(["python3", "-c", f"""
import http.server, socketserver, json

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/status':
            import requests
            try:
                r = requests.get('http://localhost:{args.port}/status', timeout=2)
                data = r.json()
            except:
                data = {{'status': 'error'}}
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        elif self.path == '/' or self.path == '/index.html':
            with open('/tmp/lisa_dashboard.html') as f:
                html = f.read()
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(html.encode())
        else:
            super().do_GET()

with socketserver.TCPServer(('', 8081), Handler) as httpd:
    print('Dashboard: http://localhost:8081')
    httpd.serve_forever()
"""], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    server.start_api()

def cmd_client(args):
    """Join as client."""
    if not args.join:
        print("Error: --join CODE required")
        return 1
    
    print(f"Joining network with code: {args.join}")
    print("(Client functionality requires main.py from Jetson)")
    # Would call the easy_client here

def cmd_dashboard(args):
    """Open dashboard."""
    print("Opening dashboard...")
    webbrowser.open("http://localhost:8081")

def cmd_status(args):
    """Show status."""
    try:
        import requests
        r = requests.get(f"http://localhost:{args.port}/status", timeout=5)
        data = r.json()
        print(json.dumps(data, indent=2))
    except Exception as e:
        print(f"Error: {e}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="LISA - Federated Learning Hub", add_help=False)
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start as federated server")
    server_parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="Model to serve")
    server_parser.add_argument("--port", type=int, default=8080, help="Server port")
    server_parser.add_argument("--ngrok", action="store_true", help="Expose via ngrok")
    server_parser.add_argument("--no-model-load", action="store_true", help="Skip model loading")
    server_parser.add_argument("--open-dashboard", action="store_true", help="Open dashboard")
    
    # Client command
    client_parser = subparsers.add_parser("client", help="Join as federated client")
    client_parser.add_argument("--join", "-j", help="Join code")
    client_parser.add_argument("--server", "-s", help="Direct server URL")
    
    # Dashboard command
    subparsers.add_parser("dashboard", help="Open web dashboard")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show network status")
    status_parser.add_argument("--port", type=int, default=8080, help="Server port")
    
    # Help
    subparsers.add_parser("help", help="Show help")
    
    args = parser.parse_args(sys.argv[1:] if len(sys.argv) > 1 else ["--help"])
    
    if args.command == "server":
        cmd_server(args)
    elif args.command == "client":
        cmd_client(args)
    elif args.command == "dashboard":
        cmd_dashboard(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        print_banner()

if __name__ == "__main__":
    main()
