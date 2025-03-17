#!/bin/bash

# Script to install gnuplot 4.6.6 from source
set -e  # Exit immediately if a command exits with a non-zero status

echo "============================================================"
echo "  Installing Gnuplot 4.6.6"
echo "============================================================"

REQUIRED_VERSION="4.6.6"
GNUPLOT_SOURCE="gnuplot-${REQUIRED_VERSION}.tar.gz"
# Primary URL (SourceForge)
GNUPLOT_URL="https://sourceforge.net/projects/gnuplot/files/gnuplot/${REQUIRED_VERSION}/${GNUPLOT_SOURCE}/download"
# Backup URL (SourceForge mirror)
GNUPLOT_BACKUP_URL="https://downloads.sourceforge.net/project/gnuplot/gnuplot/${REQUIRED_VERSION}/${GNUPLOT_SOURCE}"

# Install build dependencies
echo "Installing build dependencies..."
sudo apt-get install -y \
    build-essential \
    libx11-dev \
    libreadline-dev \
    zlib1g-dev \
    libcairo2-dev \
    libpango1.0-dev \
    liblua5.3-dev \
    libgd-dev

# Create a temporary directory for building
BUILD_DIR=$(mktemp -d)
cd "$BUILD_DIR"

echo "Downloading gnuplot ${REQUIRED_VERSION}..."
if ! wget -O "$GNUPLOT_SOURCE" "$GNUPLOT_URL"; then
    echo "Primary download failed, trying backup URL..."
    if ! wget -O "$GNUPLOT_SOURCE" "$GNUPLOT_BACKUP_URL"; then
        echo "Backup download failed, trying direct URL..."
        wget -O "$GNUPLOT_SOURCE" "https://sourceforge.net/projects/gnuplot/files/gnuplot/4.6.6/gnuplot-4.6.6.tar.gz/download"
    fi
fi

# Verify the download was successful
if [ ! -s "$GNUPLOT_SOURCE" ]; then
    echo "Failed to download gnuplot source. Please check your internet connection."
    exit 1
fi

echo "Extracting source..."
tar xzf "$GNUPLOT_SOURCE"
cd "gnuplot-${REQUIRED_VERSION}"

echo "Configuring build..."
./configure --with-readline=gnu \
           --with-cairo \
           --with-lua \
           --disable-wxwidgets

echo "Building gnuplot..."
make -j$(nproc)

echo "Installing gnuplot..."
sudo make install
sudo ldconfig

# Clean up
cd /
rm -rf "$BUILD_DIR"

# Verify installation
if command -v gnuplot >/dev/null 2>&1; then
    GNUPLOT_VERSION=$(gnuplot --version | awk '{print $2}')
    echo "Successfully installed gnuplot version: $GNUPLOT_VERSION"
else
    echo "Error: gnuplot installation failed"
    exit 1
fi

echo "Gnuplot 4.6.6 installation completed successfully!"
