#!/bin/bash

SCRIPT_PATH="live_detection.py"
PID_FILE="/tmp/intrudex_daemon.pid"
LOG_FILE="logs/intrudex_daemon.log"

clear
echo ""
echo " ___         _                    _              "     
echo "|_ _| _ __  | |_ _ __  _   _   __| |  ___ __  __ "
echo " | | | '_ \| __|| '__|| | | | / _\` | / _ \\\ \/ / "
echo " | | | | | || |_ | |  | |_| || (_| ||  __/ >  <  "
echo "|___||_| |_| \\__||_|   \\__,_| \\__,_| \\___|/_/\\_\\ "
echo ""
echo "       an IDS powered by machine learning"
echo "                   by Pablo SB"
echo ""
echo ""
echo -e "\033[32mAvailable options:\033[0m"
echo "  1. start      - Start the Intrudex IDS."
echo "  2. stop       - Stop Intrudex."
echo "  3. status     - Show the Intrudex daemon status."
echo "  4. restart    - Restart Intrudex."
echo "  5. exit       - Exit Intrudex script."
echo ""


while true; do

    read -e -p "$(echo -e "\033[33mSelect an option (1, 2, 3, 4, 5): \033[0m")" option

    start_daemon() {
        echo "Starting the Intrudex daemon..."

        # Check required Python libraries and install if necessary
        required_libraries=(
            numpy scapy joblib pandas threading yaml sklearn xgboost imblearn logging
        )

        for lib in "${required_libraries[@]}"; do
            if ! python3 -c "import $lib" &>/dev/null; then
                echo "$lib not found, installing..."
                pip3 install "$lib"
            fi
        done

        # Create logs directory if not exists
        LOG_DIR="logs"
        if [ ! -d "$LOG_DIR" ]; then
            mkdir "$LOG_DIR"
        fi

        # Run the daemon in the background
        nohup python3 "$SCRIPT_PATH" > "$LOG_FILE" 2>&1 &
        DAEMON_PID=$!
        echo "$DAEMON_PID" > "$PID_FILE"
        echo "Daemon started with PID $DAEMON_PID. Log at $LOG_FILE"
    }

    stop_daemon() {
        if [[ -f "$PID_FILE" ]]; then
            pkill -f "$SCRIPT_PATH"
            rm -f "$PID_FILE"
            echo "Daemon stopped."
        else
            echo "No daemon is running."
        fi
    }

    status_daemon() {
        if [[ -f "$PID_FILE" ]] && ps -p $(cat "$PID_FILE") > /dev/null 2>&1; then
            echo "The Intrudex daemon is running with PID $(cat $PID_FILE)."
        else
            echo "The Intrudex daemon is not running."
        fi
    }

    case "$option" in
        1)
            start_daemon
            ;;
        2)
            stop_daemon
            ;;
        3)
            status_daemon
            ;;
        4)
            stop_daemon
            start_daemon
            ;;
        5)
            echo "Exiting Intrudex script. Goodbye!"
            exit 0
            ;;
        *)
            echo "Invalid option. Please select 1, 2, 3, 4, or 5."
            ;;
    esac
done