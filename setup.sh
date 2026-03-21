#!/bin/bash
# Platform-aware setup for LISA + AutoResearch
# Works on: macOS, Linux, Windows (Git Bash/WSL)

set -e

echo "========================================"
echo "LISA + AutoResearch Setup"
echo "========================================"
echo ""

# Detect platform
detect_platform() {
    case "$(uname -s)" in
        Darwin*)    echo "macos" ;;
        Linux*)     echo "linux" ;;
        CYGWIN*)    echo "windows" ;;
        MINGW*)     echo "windows" ;;
        MSYS*)      echo "windows" ;;
        *)          echo "unknown" ;;
    esac
}

PLATFORM=$(detect_platform)
echo "Detected platform: $PLATFORM"
echo ""

# Check dependencies
echo "Checking dependencies..."
python3 -c "import mlx" 2>/dev/null || {
    echo "Installing mlx..."
    pip install mlx mlx-lm transformers
}

python3 -c "import transformers" 2>/dev/null || {
    echo "Installing transformers..."
    pip install transformers huggingface_hub
}

echo "✅ Dependencies installed"
echo ""

# Create directories
echo "Creating directories..."
mkdir -p "$HOME/lisa-autoresearch/logs/autoresearch"
mkdir -p "$HOME/lisa-autoresearch/logs/training"
mkdir -p "$HOME/lisa-autoresearch/adapters"
mkdir -p "$HOME/lisa-autoresearch/mlx_data"
echo "✅ Directories created"
echo ""

# Copy scripts to home directory
echo "Installing scripts..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cp -r "$SCRIPT_DIR"/* "$HOME/lisa-autoresearch/" 2>/dev/null || true
echo "✅ Scripts installed"
echo ""

# Prepare example data if available
if [ -f "$HOME/lisa-autoresearch/example_data.jsonl" ]; then
    echo "Preparing example data..."
    cd "$HOME/lisa-autoresearch"
    python3 prepare_data.py --input example_data.jsonl --output mlx_data/ 2>/dev/null || {
        echo "⚠️ Could not prepare example data (will prepare on first run)"
    }
    echo "✅ Example data prepared"
    echo ""
fi

# Platform-specific setup
case "$PLATFORM" in
    macos)
        echo "Setting up macOS LaunchAgents..."
        
        # Create LaunchAgent for autoresearch
        cat > "$HOME/Library/LaunchAgents/com.lisa.autoresearch.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.lisa.autoresearch</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>$HOME/lisa-autoresearch/nightly_autoresearch.sh</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>2</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>$HOME/lisa-autoresearch/logs/autoresearch/launchd.log</string>
    <key>StandardErrorPath</key>
    <string>$HOME/lisa-autoresearch/logs/autoresearch/launchd.log</string>
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
PLIST
        
        # Create LaunchAgent for weekly retrain
        cat > "$HOME/Library/LaunchAgents/com.lisa.weekly-retrain.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.lisa.weekly-retrain</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>$HOME/lisa-autoresearch/weekly_retrain.sh</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Weekday</key>
        <integer>0</integer>
        <key>Hour</key>
        <integer>4</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>$HOME/lisa-autoresearch/logs/training/launchd.log</string>
    <key>StandardErrorPath</key>
    <string>$HOME/lisa-autoresearch/logs/training/launchd.log</string>
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
PLIST
        
        # Load LaunchAgents
        launchctl load "$HOME/Library/LaunchAgents/com.lisa.autoresearch.plist" 2>/dev/null || true
        launchctl load "$HOME/Library/LaunchAgents/com.lisa.weekly-retrain.plist" 2>/dev/null || true
        echo "✅ LaunchAgents created and loaded"
        ;;
        
    linux)
        echo "Setting up Linux cron jobs..."
        
        # Add cron jobs
        (crontab -l 2>/dev/null | grep -v "lisa-autoresearch"; echo "0 2 * * * $HOME/lisa-autoresearch/nightly_autoresearch.sh >> $HOME/lisa-autoresearch/logs/autoresearch/cron.log 2>&1") | crontab -
        (crontab -l 2>/dev/null | grep -v "weekly-retrain"; echo "0 4 * * 0 $HOME/lisa-autoresearch/weekly_retrain.sh >> $HOME/lisa-autoresearch/logs/training/cron.log 2>&1") | crontab -
        echo "✅ Cron jobs created"
        ;;
        
    windows)
        echo "========================================"
        echo "Windows Detected"
        echo "========================================"
        echo ""
        echo "For best results, use WSL (Windows Subsystem for Linux)"
        echo ""
        echo "Option 1: WSL (Recommended)"
        echo "  - Install WSL: wsl --install"
        echo "  - Run this script inside WSL"
        echo ""
        echo "Option 2: Git Bash"
        echo "  - Scripts will work, but no auto-scheduling"
        echo "  - Run manually: ./nightly_autoresearch.sh"
        echo ""
        echo "Option 3: Task Scheduler"
        echo "  - Run setup_windows.bat"
        echo ""
        ;;
        
    unknown)
        echo "⚠️ Unknown platform. Manual setup required."
        echo "You can still run the scripts manually:"
        echo "  python3 train_qwen7b.py --iters 500"
        ;;
esac

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Install location: ~/lisa-autoresearch/"
echo ""
echo "To test immediately:"
echo "  cd ~/lisa-autoresearch"
echo "  python3 train_qwen7b.py --iters 50"
echo ""
echo "To run autoresearch manually:"
echo "  cd ~/lisa-autoresearch"
echo "  ./nightly_autoresearch.sh"
echo ""
echo "To run weekly retrain manually:"
echo "  cd ~/lisa-autoresearch"
echo "  ./weekly_retrain.sh"
echo ""