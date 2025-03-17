#!/bin/bash

# Master script to run all installation scripts and log their outputs
# Created: $(date +"%Y-%m-%d %H:%M:%S")

# Function to print section headers
print_header() {
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

# Create a timestamp for the log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="installation_log-${TIMESTAMP}.dat"

# Ensure all scripts are executable
chmod +x install_python2.7.18.sh
chmod +x install_gnuplot_4.6.6.sh
chmod +x install_mctdh_related_packages.sh

# Start logging
echo "Installation started at: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run Python 2.7.18 installation
print_header "Installing Python 2.7.18" | tee -a "$LOG_FILE"
echo "Running install_python2.7.18.sh..." | tee -a "$LOG_FILE"
./install_python2.7.18.sh 2>&1 | tee -a "$LOG_FILE"

# Check if Python installation was successful
if command -v python2.7 >/dev/null 2>&1 && python2.7 --version 2>&1 | grep -q "2.7.18"; then
    echo "Python 2.7.18 installation SUCCESSFUL!" | tee -a "$LOG_FILE"
else
    echo "Python 2.7.18 installation may have FAILED. Check the log for details." | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "-------------------------------------------------------------" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run Gnuplot 4.6.6 installation
print_header "Installing Gnuplot 4.6.6" | tee -a "$LOG_FILE"
echo "Running install_gnuplot_4.6.6.sh..." | tee -a "$LOG_FILE"
./install_gnuplot_4.6.6.sh 2>&1 | tee -a "$LOG_FILE"

# Check if Gnuplot installation was successful
if command -v gnuplot >/dev/null 2>&1 && gnuplot --version | grep -q "4.6.6"; then
    echo "Gnuplot 4.6.6 installation SUCCESSFUL!" | tee -a "$LOG_FILE"
else
    echo "Gnuplot 4.6.6 installation may have FAILED. Check the log for details." | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "-------------------------------------------------------------" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run MCTDH related packages installation
print_header "Installing MCTDH Related Packages" | tee -a "$LOG_FILE"
echo "Running install_mctdh_related_packages.sh..." | tee -a "$LOG_FILE"
./install_mctdh_related_packages.sh 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "-------------------------------------------------------------" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Installation summary
print_header "Installation Summary" | tee -a "$LOG_FILE"

# Check Python
echo "Python 2.7 status:" | tee -a "$LOG_FILE"
if command -v python2.7 >/dev/null 2>&1; then
    python2.7 --version 2>&1 | tee -a "$LOG_FILE"
else
    echo "Python 2.7 is NOT installed or not in PATH" | tee -a "$LOG_FILE"
fi

# Check pip2
echo "Pip2 status:" | tee -a "$LOG_FILE"
if command -v pip2 >/dev/null 2>&1; then
    pip2 --version | tee -a "$LOG_FILE"
else
    echo "Pip2 is NOT installed or not in PATH" | tee -a "$LOG_FILE"
fi

# Check Gnuplot
echo "Gnuplot status:" | tee -a "$LOG_FILE"
if command -v gnuplot >/dev/null 2>&1; then
    gnuplot --version | tee -a "$LOG_FILE"
else
    echo "Gnuplot is NOT installed or not in PATH" | tee -a "$LOG_FILE"
fi

# Finish
echo "" | tee -a "$LOG_FILE"
echo "Installation completed at: $(date)" | tee -a "$LOG_FILE"
echo "Log file saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

print_header "Next Steps" | tee -a "$LOG_FILE"
echo "1. Review the log file for any errors: $LOG_FILE" | tee -a "$LOG_FILE"
echo "2. Verify installations by running:" | tee -a "$LOG_FILE"
echo "   - python2.7 --version" | tee -a "$LOG_FILE"
echo "   - pip2 --version" | tee -a "$LOG_FILE"
echo "   - gnuplot --version" | tee -a "$LOG_FILE"
echo "3. If any installation failed, check the log file for details and try running the specific script manually." | tee -a "$LOG_FILE"
