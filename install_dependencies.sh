#!/bin/bash

# System Benchmark Tool - Dependencies Installation Script
# This script installs all required dependencies for the system benchmark tool

set -e

echo "System Benchmark Tool - Installing Dependencies"
echo "=============================================="

# Check if running as root for system-wide installation
if [[ $EUID -eq 0 ]]; then
    echo "Running as root - installing system-wide packages"
    SUDO=""
else
    echo "Running as user - using sudo for system packages"
    SUDO="sudo"
fi

# Detect package manager
if command -v apt-get &> /dev/null; then
    PKG_MANAGER="apt-get"
    UPDATE_CMD="$SUDO apt-get update"
    INSTALL_CMD="$SUDO apt-get install -y"
elif command -v yum &> /dev/null; then
    PKG_MANAGER="yum"
    UPDATE_CMD="$SUDO yum update -y"
    INSTALL_CMD="$SUDO yum install -y"
elif command -v dnf &> /dev/null; then
    PKG_MANAGER="dnf"
    UPDATE_CMD="$SUDO dnf update -y"
    INSTALL_CMD="$SUDO dnf install -y"
elif command -v pacman &> /dev/null; then
    PKG_MANAGER="pacman"
    UPDATE_CMD="$SUDO pacman -Sy"
    INSTALL_CMD="$SUDO pacman -S --noconfirm"
else
    echo "Error: Unsupported package manager. Please install dependencies manually."
    exit 1
fi

echo "Detected package manager: $PKG_MANAGER"

# Update package lists
echo "Updating package lists..."
$UPDATE_CMD

# Install system dependencies
echo "Installing system dependencies..."

case $PKG_MANAGER in
    "apt-get")
        $INSTALL_CMD python3 python3-pip python3-venv python3-dev build-essential
        ;;
    "yum"|"dnf")
        $INSTALL_CMD python3 python3-pip python3-venv python3-devel gcc gcc-c++ make
        ;;
    "pacman")
        $INSTALL_CMD python python-pip base-devel
        ;;
esac

export VD=/tmp/.venv
python3 -m venv $VD
. $VD/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install psutil

# Verify installation
echo "Verifying installation..."
python3 -c "import psutil; print(f'psutil version: {psutil.__version__}')"

# Make the benchmark script executable
chmod +x system_benchmark.py

echo ""
echo "Installation completed successfully!"
echo ""
echo "Usage examples:"
echo "  # Run full benchmark"
echo "  python3 system_benchmark.py"
echo ""
echo "  # Run CPU benchmark only (30 seconds)"
echo "  python3 system_benchmark.py --cpu-only --cpu-duration 30"
echo ""
echo "  # Run memory benchmark only (2GB test)"
echo "  python3 system_benchmark.py --memory-only --memory-size 2.0"
echo ""
echo "  # Run disk benchmark only (5GB test)"
echo "  python3 system_benchmark.py --disk-only --disk-size 5.0"
echo ""
echo "  # Custom test with all components"
echo "  python3 system_benchmark.py --cpu-duration 60 --memory-size 2.0 --disk-size 3.0 --output my_results.json"