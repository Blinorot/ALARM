#!/usr/bin/env bash

while true; do
    echo "Running script at $(date)"
    python3 mmsu.py
    EXIT_CODE=$?

    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "Script exited successfully (code 0). Stopping."
        break
    else
        echo "Script failed (code $EXIT_CODE). Waiting 1 hour before retry."
        sleep 1200
    fi
done
