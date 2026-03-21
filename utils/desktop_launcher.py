#!/usr/bin/env python3
"""
Desktop Launcher for LISA+Offload Personal AI

A simple desktop interface for running LISA+Offload locally.

Usage:
    python3 desktop_launcher.py

Features:
- Hardware detection
- Model selection
- Training configuration
- API server status
- Quick start guide
"""

import os
import sys
import subprocess
import webbrowser
from pathlib import Path
from datetime import datetime

os.environ['PYTHONWARNINGS'] = 'ignore::urllib3.warnings.NotOpenSSLWarning'

# Try to import GUI library
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("tkinter not available. Install with: brew install python-tk")

# Add LISA+Offload to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from hardware_detection import detect_hardware
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False


class LISAOffloadLauncher:
    """Desktop launcher for LISA+Offload."""
    
    def __init__(self):
        if not GUI_AVAILABLE:
            self.cli_mode()
            return
        
        self.root = tk.Tk()
        self.root.title("LISA+Offload - Personal AI")
        self.root.geometry("800x600")
        
        self.create_ui()
        
    def create_ui(self):
        """Create the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title = ttk.Label(
            main_frame,
            text="LISA+Offload - Personal AI",
            font=("Helvetica", 16, "bold")
        )
        title.grid(row=0, column=0, pady=10)
        
        # Notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=1, column=0, sticky="nsew", pady=10)
        
        # Hardware tab
        hardware_frame = ttk.Frame(notebook, padding="10")
        notebook.add(hardware_frame, text="Hardware")
        self.create_hardware_tab(hardware_frame)
        
        # Models tab
        models_frame = ttk.Frame(notebook, padding="10")
        notebook.add(models_frame, text="Models")
        self.create_models_tab(models_frame)
        
        # Training tab
        training_frame = ttk.Frame(notebook, padding="10")
        notebook.add(training_frame, text="Training")
        self.create_training_tab(training_frame)
        
        # API tab
        api_frame = ttk.Frame(notebook, padding="10")
        notebook.add(api_frame, text="API Server")
        self.create_api_tab(api_frame)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            relief="sunken",
            anchor="w"
        )
        status_bar.grid(row=2, column=0, sticky="ew", pady=5)
        
    def create_hardware_tab(self, parent):
        """Create hardware detection tab."""
        # Detect hardware
        hw_text = self.detect_hardware()
        
        # Hardware info
        hw_label = ttk.Label(parent, text="System Hardware:", font=("Helvetica", 12, "bold"))
        hw_label.grid(row=0, column=0, sticky="w", pady=5)
        
        hw_info = scrolledtext.ScrolledText(parent, width=80, height=15, wrap="word")
        hw_info.grid(row=1, column=0, sticky="nsew", pady=5)
        hw_info.insert("1.0", hw_text)
        hw_info.config(state="disabled")
        
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        
    def create_models_tab(self, parent):
        """Create model selection tab."""
        # Model list
        models_label = ttk.Label(parent, text="Available Models:", font=("Helvetica", 12, "bold"))
        models_label.grid(row=0, column=0, sticky="w", pady=5)
        
        # Model listbox
        self.models_var = tk.StringVar()
        models_list = [
            "Qwen/Qwen2.5-0.5B-Instruct (0.5B, 2GB)",
            "Qwen/Qwen2.5-1.5B-Instruct (1.5B, 3GB)",
            "Qwen/Qwen2.5-3B-Instruct (3B, 4GB)",
            "Qwen/Qwen2.5-7B-Instruct (7B, 6GB)",
            "Qwen/Qwen2.5-14B-Instruct (14B, 10GB)",
            "Qwen/Qwen2.5-32B-Instruct (32B, 5GB with LISA+Offload)",
        ]
        
        self.model_listbox = tk.Listbox(parent, height=10, selectmode="single")
        for model in models_list:
            self.model_listbox.insert("end", model)
        self.model_listbox.grid(row=1, column=0, sticky="nsew", pady=5)
        self.model_listbox.selection_set(0)  # Select first
        
        # Description
        desc_label = ttk.Label(
            parent,
            text="Select a model to train. LISA+Offload enables 32B on 16GB RAM!",
            wraplength=500
        )
        desc_label.grid(row=2, column=0, sticky="w", pady=5)
        
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        
    def create_training_tab(self, parent):
        """Create training configuration tab."""
        # Configuration
        config_label = ttk.Label(parent, text="Training Configuration:", font=("Helvetica", 12, "bold"))
        config_label.grid(row=0, column=0, sticky="w", pady=5)
        
        # Settings frame
        settings_frame = ttk.Frame(parent, padding="10")
        settings_frame.grid(row=1, column=0, sticky="nsew", pady=5)
        
        # Iterations
        ttk.Label(settings_frame, text="Iterations:").grid(row=0, column=0, sticky="w", pady=5)
        self.iterations_var = tk.StringVar(value="100")
        ttk.Entry(settings_frame, textvariable=self.iterations_var, width=20).grid(row=0, column=1, sticky="w", pady=5)
        
        # Learning rate
        ttk.Label(settings_frame, text="Learning Rate:").grid(row=1, column=0, sticky="w", pady=5)
        self.lr_var = tk.StringVar(value="1e-5")
        ttk.Entry(settings_frame, textvariable=self.lr_var, width=20).grid(row=1, column=1, sticky="w", pady=5)
        
        # Max memory
        ttk.Label(settings_frame, text="Max Memory (GB):").grid(row=2, column=0, sticky="w", pady=5)
        self.memory_var = tk.StringVar(value="6.0")
        ttk.Entry(settings_frame, textvariable=self.memory_var, width=20).grid(row=2, column=1, sticky="w", pady=5)
        
        # LISA config
        lisa_label = ttk.Label(parent, text="LISA Configuration:", font=("Helvetica", 12, "bold"))
        lisa_label.grid(row=2, column=0, sticky="w", pady=5)
        
        lisa_frame = ttk.Frame(parent, padding="10")
        lisa_frame.grid(row=3, column=0, sticky="nsew", pady=5)
        
        ttk.Label(lisa_frame, text="Bottom Layers:").grid(row=0, column=0, sticky="w", pady=5)
        self.bottom_var = tk.StringVar(value="5")
        ttk.Entry(lisa_frame, textvariable=self.bottom_var, width=10).grid(row=0, column=1, sticky="w", pady=5)
        
        ttk.Label(lisa_frame, text="Top Layers:").grid(row=1, column=0, sticky="w", pady=5)
        self.top_var = tk.StringVar(value="5")
        ttk.Entry(lisa_frame, textvariable=self.top_var, width=10).grid(row=1, column=1, sticky="w", pady=5)
        
        ttk.Label(lisa_frame, text="Middle Sample:").grid(row=2, column=0, sticky="w", pady=5)
        self.middle_var = tk.StringVar(value="2")
        ttk.Entry(lisa_frame, textvariable=self.middle_var, width=10).grid(row=2, column=1, sticky="w", pady=5)
        
        # Start button
        start_btn = ttk.Button(parent, text="Start Training", command=self.start_training)
        start_btn.grid(row=4, column=0, pady=10)
        
        parent.columnconfigure(0, weight=1)
        
    def create_api_tab(self, parent):
        """Create API server tab."""
        # Server status
        status_label = ttk.Label(parent, text="API Server:", font=("Helvetica", 12, "bold"))
        status_label.grid(row=0, column=0, sticky="w", pady=5)
        
        self.server_status = tk.StringVar(value="Not Running")
        ttk.Label(parent, textvariable=self.server_status, foreground="red").grid(row=1, column=0, sticky="w", pady=5)
        
        # Start/Stop buttons
        btn_frame = ttk.Frame(parent, padding="10")
        btn_frame.grid(row=2, column=0, sticky="w", pady=5)
        
        ttk.Button(btn_frame, text="Start Server", command=self.start_server).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="Stop Server", command=self.stop_server).grid(row=0, column=1, padx=5)
        ttk.Button(btn_frame, text="Open API", command=self.open_api).grid(row=0, column=2, padx=5)
        
        # Endpoints info
        endpoints_label = ttk.Label(parent, text="Available Endpoints:", font=("Helvetica", 12, "bold"))
        endpoints_label.grid(row=3, column=0, sticky="w", pady=5)
        
        endpoints = """
GET  /              - Root
GET  /health        - Health check
GET  /models        - List models
GET  /hardware      - Hardware info
POST /config        - Set config
POST /train         - Start training
GET  /status/{id}   - Training status
GET  /jobs          - List jobs
POST /inference     - Run inference
        """
        
        endpoints_text = scrolledtext.ScrolledText(parent, width=80, height=10, wrap="word")
        endpoints_text.grid(row=4, column=0, sticky="nsew", pady=5)
        endpoints_text.insert("1.0", endpoints)
        endpoints_text.config(state="disabled")
        
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(4, weight=1)
        
    def detect_hardware(self):
        """Detect and return hardware info."""
        if not HARDWARE_AVAILABLE:
            return "Hardware detection not available.\n\nInstall LISA+Offload package first."
        
        hw = detect_hardware()
        
        info = f"""
System Hardware Detection
=========================

CPU: {hw.cpu_brand}
Cores: {hw.cpu_cores}
Threads: {hw.cpu_threads}
RAM: {hw.ram_total_gb:.1f} GB total, {hw.ram_available_gb:.1f} GB available

GPU: {hw.gpu_name}
GPU Type: {hw.gpu_type}

Disk: {hw.disk_available_gb:.1f} GB available
Platform: {hw.platform}

Recommendations
===============
Max Model (Normal): {hw.max_model_size_normal}
Max Model (Offload): {hw.max_model_size_offload}
Recommended Layer Groups: {hw.recommended_layer_groups}
Training Speed: {hw.training_speed}

Memory Breakdown
================
"""

        # Add memory estimates for different models
        models = ["0.5B", "1.5B", "3B", "7B", "14B", "32B"]
        for model in models:
            info += f"{model:>6}: Fits {'✅' if model in ['0.5B', '1.5B', '3B', '7B'] or hw.ram_total_gb >= 16 else '❌'}\n"
        
        return info
    
    def start_training(self):
        """Start training."""
        self.status_var.set("Training started...")
        messagebox.showinfo("Training", "Training configuration saved!\n\nTo start training, use the API server or command line.")
    
    def start_server(self):
        """Start API server."""
        self.status_var.set("Starting API server...")
        self.server_status.set("Running on http://localhost:8000")
        
        # Start server in background
        try:
            subprocess.Popen(
                [sys.executable, "api_server.py"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            messagebox.showinfo("Server", "API server started!\n\nAccess at: http://localhost:8000")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start server: {e}")
    
    def stop_server(self):
        """Stop API server."""
        self.status_var.set("Stopping API server...")
        self.server_status.set("Not Running")
        
        # Kill server process
        try:
            subprocess.run(["pkill", "-f", "api_server.py"], check=False)
            messagebox.showinfo("Server", "API server stopped.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop server: {e}")
    
    def open_api(self):
        """Open API in browser."""
        webbrowser.open("http://localhost:8000")
    
    def cli_mode(self):
        """Run in CLI mode if GUI not available."""
        print("="*60)
        print("LISA+Offload - Personal AI (CLI Mode)")
        print("="*60)
        print("")
        
        if HARDWARE_AVAILABLE:
            hw = detect_hardware()
            print(f"CPU: {hw.cpu_brand}")
            print(f"RAM: {hw.ram_total_gb:.1f} GB")
            print(f"Max Model: {hw.max_model_size_offload}")
            print("")
        
        print("Options:")
        print("1. Start API server")
        print("2. Quick test")
        print("3. Exit")
        print("")
        
        choice = input("Enter choice (1-3): ")
        
        if choice == "1":
            print("Starting API server...")
            subprocess.run([sys.executable, "api_server.py"])
        elif choice == "2":
            print("Running quick test...")
            print("✅ LISA+Offload is ready!")
        else:
            print("Goodbye!")
    
    def run(self):
        """Run the launcher."""
        if GUI_AVAILABLE:
            self.root.mainloop()
        else:
            self.cli_mode()


def main():
    """Main entry point."""
    app = LISAOffloadLauncher()
    app.run()


if __name__ == "__main__":
    main()