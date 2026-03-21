#!/bin/bash
# Setup LISA + AutoResearch on macOS

set -e

echo "========================================"
echo "LISA + AutoResearch Setup"
echo "========================================"
echo ""

# Check dependencies
echo "Checking dependencies..."
python3 -c "import mlx" 2>/dev/null || {
    echo "Installing mlx..."
    pip install mlx mlx-lm transformers
}

echo "✅ Dependencies installed"
echo ""

# Create directories
echo "Creating directories..."
mkdir -p ~/lisa-autoresearch/logs/autoresearch
mkdir -p ~/lisa-autoresearch/logs/training
mkdir -p ~/lisa-autoresearch/adapters
mkdir -p ~/lisa-autoresearch/mlx_data
echo "✅ Directories created"
echo ""

# Prepare example data
echo "Preparing example data..."
if [ -f "example_data.jsonl" ]; then
    python3 prepare_data.py --input example_data.jsonl --output mlx_data/
    echo "✅ Example data prepared"
else
    echo "⚠️ example_data.jsonl not found, skipping"
fi
echo ""

# Create LaunchAgent for autoresearch
echo "Creating LaunchAgent for nightly autoresearch..."
cat > ~/Library/LaunchAgents/com.lisa.autoresearch.plist << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.lisa.autoresearch</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>~/lisa-autoresearch/nightly_autoresearch.sh</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>2</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>~/lisa-autoresearch/logs/autoresearch/launchd.log</string>
    <key>StandardErrorPath</key>
    <string>~/lisa-autoresearch/logs/autoresearch/launchd.log</string>
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
PLIST
echo "✅ AutoResearch LaunchAgent created"

# Create LaunchAgent for weekly retrain
echo "Creating LaunchAgent for weekly retraining..."
cat > ~/Library/LaunchAgents/com.lisa.weekly-retrain.plist << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.lisa.weekly-retrain</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>~/lisa-autoresearch/weekly_retrain.sh</string>
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
    <string>~/lisa-autoresearch/logs/training/launchd.log</string>
    <key>StandardErrorPath</key>
    <string>~/lisa-autoresearch/logs/training/launchd.log</string>
    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
PLIST
echo "✅ Weekly Retrain LaunchAgent created"
echo ""

# Load LaunchAgents
echo "Loading LaunchAgents..."
launchctl load ~/Library/LaunchAgents/com.lisa.autoresearch.plist 2>/dev/null || true
launchctl load ~/Library/LaunchAgents/com.lisa.weekly-retrain.plist 2>/dev/null || true
echo "✅ LaunchAgents loaded"
echo ""

# Summary
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Schedule:"
echo "  • AutoResearch: Nightly at 2 AM"
echo "  • Weekly Retrain: Sundays at 4 AM"
echo ""
echo "To test immediately:"
echo "  python3 train_qwen7b.py --iters 50"
echo ""
echo "To run autoresearch manually:"
echo "  ./nightly_autoresearch.sh"
echo ""
echo "To run weekly retrain manually:"
echo "  ./weekly_retrain.sh"
echo ""