#!/bin/bash
# LISA+LCSB Cron Runner - Autonomous progress tracker
# Runs every 15 minutes

LOG="/tmp/lisa_lcsb_cron.log"
SSH="ssh jetson@10.0.0.145"

echo "=== $(date) ===" >> $LOG

# Check Jetson status
MEM=$($SSH "free -h | grep Mem:" 2>/dev/null)
PROC=$($SSH "ps aux | grep python | grep -v grep | wc -l" 2>/dev/null)

echo "Memory: $MEM" >> $LOG
echo "Python processes: $PROC" >> $LOG

# Check for running scripts
RUNNING=$($SSH "ps aux | grep -E 'lisa_lcsb|test_.*lora' | grep python | grep -v grep" 2>/dev/null)
if [ -n "$RUNNING" ]; then
    echo "Running scripts found:" >> $LOG
    echo "$RUNNING" >> $LOG
fi

# Check for recent output files
OUTFILES=$($SSH "ls -lt /tmp/*.txt /tmp/*.log 2>/dev/null | head -5" 2>/dev/null)
if [ -n "$OUTFILES" ]; then
    echo "Recent output files:" >> $LOG
    echo "$OUTFILES" >> $LOG
fi

# Check if there's a success indicator
SUCCESS=$($SSH "grep -l 'SUCCESS' /tmp/*.txt 2>/dev/null" 2>/dev/null)
if [ -n "$SUCCESS" ]; then
    echo "SUCCESS found in: $SUCCESS" >> $LOG
    for f in $SUCCESS; do
        echo "Last 5 lines of $f:" >> $LOG
        $SSH "tail -5 $f" >> $LOG 2>/dev/null
    done
fi

echo "---" >> $LOG
