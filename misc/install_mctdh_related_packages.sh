#!/bin/bash

# MCTDH Required Packages Installation Script
# This script installs all required dependencies for MCTDH

set -e  # Exit immediately if a command exits with a non-zero status

# Function to print section headers
print_header() {
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Create a Python virtual environment
setup_python_venv() {
    print_header "Setting up Python Virtual Environment"
    
    if ! command_exists python2; then
        echo "Python 2 is required but not found. Installing..."
        sudo apt-get install -y python2.7 python2.7-dev
        
        # Create a symbolic link if needed
        if [ ! -f "/usr/bin/python" ]; then
            echo "Creating python symbolic link to python2..."
            sudo ln -s /usr/bin/python2.7 /usr/bin/python
        fi
    else
        echo "Found Python 2: $(python2 --version 2>&1)"
    fi
    
    # Install pip for Python 2 if needed
    if ! command_exists pip2; then
        echo "Installing pip for Python 2..."
        sudo apt-get install -y python-pip || {
            echo "Could not install python-pip package, trying alternative method..."
            curl https://bootstrap.pypa.io/pip/2.7/get-pip.py -o get-pip.py
            sudo python2.7 get-pip.py
            rm get-pip.py
        }
    fi
    
    # Create a virtual environment directory if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "Creating Python virtual environment..."
        # For Python 2, we need to install virtualenv first
        if ! command_exists virtualenv; then
            sudo apt-get install -y python-virtualenv || sudo pip2 install virtualenv
        fi
        virtualenv -p python2.7 venv
    else
        echo "Virtual environment already exists."
    fi
    
    # Activate the virtual environment
    echo "Activating virtual environment..."
    source venv/bin/activate
    
    # Create requirements.txt if it doesn't exist
    if [ ! -f "requirements.txt" ]; then
        echo "Creating requirements.txt file..."
        cat > requirements.txt << EOF
numpy>=1.16.6,<2.0.0
matplotlib>=2.2.5,<3.0.0
scipy>=1.2.3,<2.0.0
EOF
    fi
    
    # Install Python dependencies
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
    
    echo "Python environment setup complete."
}

# Install required system packages
install_system_packages() {
    print_header "Installing System Packages"
    
    # Update package lists
    echo "Updating package lists..."
    sudo apt-get update
    
    # Install required packages
    echo "Installing required packages..."
    sudo apt-get install -y \
        build-essential \
        gfortran \
        gcc \
        g++ \
        make \
        bash \
        texlive-full \
        texlive-latex-extra \
        texlive-science \
        texlive-fonts-recommended \
        texlive-font-utils \
        xdvik-ja \
        dvipng \
        firefox
    
    echo "System packages installation complete."
}

# Check Fortran compiler version
check_fortran_version() {
    print_header "Checking Fortran Compiler Version"
    
    if command_exists gfortran; then
        GFORTRAN_VERSION=$(gfortran --version | head -n 1 | awk '{print $NF}')
        GFORTRAN_MAJOR=$(echo $GFORTRAN_VERSION | cut -d. -f1)
        
        echo "Found gfortran version: $GFORTRAN_VERSION"
        
        if [ "$GFORTRAN_MAJOR" -lt 6 ]; then
            echo "Warning: MCTDH requires gfortran 6.1 or higher."
            echo "Current version is $GFORTRAN_VERSION."
            echo "Consider upgrading your gfortran installation."
            
            # Offer to install a newer version if available
            read -p "Would you like to try installing a newer version of gfortran? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                sudo apt-get install -y software-properties-common
                sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
                sudo apt-get update
                sudo apt-get install -y gfortran-8
                sudo update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-8 60
                
                # Check the new version
                GFORTRAN_VERSION=$(gfortran --version | head -n 1 | awk '{print $NF}')
                echo "Upgraded gfortran version: $GFORTRAN_VERSION"
            fi
        else
            echo "gfortran version is compatible with MCTDH requirements."
        fi
    else
        echo "gfortran is not installed. Installing..."
        sudo apt-get install -y gfortran
        
        # Check the installed version
        GFORTRAN_VERSION=$(gfortran --version | head -n 1 | awk '{print $NF}')
        echo "Installed gfortran version: $GFORTRAN_VERSION"
    fi
}

# Check bash version
check_bash_version() {
    print_header "Checking Bash Version"
    
    BASH_VERSION_STRING=$(bash --version | head -n 1)
    BASH_VERSION=$(echo $BASH_VERSION_STRING | grep -oP 'version \K[0-9]+\.[0-9]+')
    BASH_MAJOR=$(echo $BASH_VERSION | cut -d. -f1)
    
    echo "Found bash version: $BASH_VERSION"
    
    if [ "$BASH_MAJOR" -lt 3 ]; then
        echo "Warning: MCTDH recommends bash 3.0 or higher."
        echo "Current version is $BASH_VERSION."
        echo "Consider upgrading your bash installation."
    else
        echo "bash version is compatible with MCTDH requirements."
    fi
    
    # Check if bash is in /bin
    if [ ! -f "/bin/bash" ]; then
        echo "Warning: bash is not located in /bin."
        echo "This might cause issues with MCTDH."
        echo "Current bash location: $(which bash)"
    else
        echo "bash is correctly located in /bin."
    fi
}

# Check LaTeX installation
check_latex() {
    print_header "Checking LaTeX Installation"
    
    if command_exists latex; then
        echo "LaTeX is installed."
        
        # Check for required LaTeX packages
        echo "Checking for required LaTeX packages..."
        
        # Create a temporary LaTeX file to test packages
        cat > test_latex.tex << EOF
\\documentclass{article}
\\usepackage{amsmath}
\\usepackage{fancyhdr}
\\begin{document}
Test document
\\end{document}
EOF
        
        # Try to compile the test file
        if latex -interaction=nonstopmode test_latex.tex > /dev/null 2>&1; then
            echo "Required LaTeX packages are installed."
        else
            echo "Some required LaTeX packages might be missing."
            echo "Installing additional LaTeX packages..."
            sudo apt-get install -y texlive-latex-extra texlive-science
        fi
        
        # Clean up test files
        rm -f test_latex.tex test_latex.aux test_latex.log test_latex.dvi
    else
        echo "LaTeX is not installed. Installing..."
        sudo apt-get install -y texlive texlive-latex-extra texlive-science
    fi
    
    # Check for xdvi and dvips
    if ! command_exists xdvi; then
        echo "xdvi is not installed. Installing..."
        sudo apt-get install -y xdvik-ja
    else
        echo "xdvi is installed."
    fi
    
    if ! command_exists dvips; then
        echo "dvips is not installed. Installing..."
        sudo apt-get install -y texlive-font-utils
    else
        echo "dvips is installed."
    fi
}

# Check gnuplot installation
check_gnuplot() {
    print_header "Installing Gnuplot 4.6.6"
    
    REQUIRED_VERSION="4.6.6"
    GNUPLOT_SOURCE="gnuplot-${REQUIRED_VERSION}.tar.gz"
    GNUPLOT_URL="https://sourceforge.net/projects/gnuplot/files/gnuplot/${REQUIRED_VERSION}/${GNUPLOT_SOURCE}"
    
    # Install build dependencies
    echo "Installing build dependencies..."
    sudo apt-get install -y \
        libx11-dev \
        libreadline-dev \
        zlib1g-dev \
        libcairo2-dev \
        libpango1.0-dev \
        libwxgtk3.0-gtk3-dev \
        liblua5.3-dev \
        libgd-dev
    
    # Create a temporary directory for building
    BUILD_DIR=$(mktemp -d)
    cd "$BUILD_DIR"
    
    echo "Downloading gnuplot ${REQUIRED_VERSION}..."
    wget "$GNUPLOT_URL"
    
    echo "Extracting source..."
    tar xzf "$GNUPLOT_SOURCE"
    cd "gnuplot-${REQUIRED_VERSION}"
    
    echo "Configuring build..."
    ./configure --with-readline=gnu \
               --with-cairo \
               --with-wx \
               --with-lua \
               --with-gd
    
    echo "Building gnuplot..."
    make -j$(nproc)
    
    echo "Installing gnuplot..."
    sudo make install
    sudo ldconfig
    
    # Clean up
    cd /
    rm -rf "$BUILD_DIR"
    
    # Verify installation
    if command_exists gnuplot; then
        GNUPLOT_VERSION=$(gnuplot --version | awk '{print $2}')
        echo "Successfully installed gnuplot version: $GNUPLOT_VERSION"
    else
        echo "Error: gnuplot installation failed"
        exit 1
    fi
}

# Main execution
main() {
    print_header "MCTDH Required Packages Installation"
    
    # Check if running as root
    if [ "$(id -u)" -eq 0 ]; then
        echo "This script should not be run as root."
        exit 1
    fi
    
    # Install system packages
    install_system_packages
    
    # Check compiler versions
    check_fortran_version
    check_bash_version
    
    # Check LaTeX installation
    check_latex
    
    # Check gnuplot installation
    check_gnuplot
    
    # Setup Python environment
    setup_python_venv
    
    print_header "Installation Complete"
    echo "All required packages for MCTDH have been installed."
    echo "To use the Python environment, run: source venv/bin/activate"
    echo "To install MCTDH, please follow the instructions in the MCTDH_DIR/install/README file."
}

# Run the main function
main