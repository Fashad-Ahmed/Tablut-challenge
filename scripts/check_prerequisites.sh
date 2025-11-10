#!/bin/bash
# Check prerequisites for running the Tablut agent

echo "Checking prerequisites..."

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo "✓ Python $PYTHON_VERSION found"
else
    echo "✗ Python 3 not found. Install from https://www.python.org/"
    exit 1
fi

# Check Java
if command -v java &> /dev/null; then
    JAVA_VERSION=$(java -version 2>&1 | head -n 1)
    if [[ "$JAVA_VERSION" == *"Unable to locate"* ]] || [[ "$JAVA_VERSION" == *"No Java runtime"* ]]; then
        echo "✗ Java Runtime not installed"
        echo ""
        echo "To install Java on macOS:"
        echo "  1. Install Homebrew (if not installed): /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        echo "  2. Install Java: brew install openjdk@17"
        echo "  3. Link Java: sudo ln -sfn /opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-17.jdk"
        echo ""
        echo "Or download from: https://www.oracle.com/java/technologies/downloads/"
        exit 1
    else
        echo "✓ Java found: $JAVA_VERSION"
    fi
else
    echo "✗ Java not found in PATH"
    echo ""
    echo "To install Java on macOS:"
    echo "  1. Install Homebrew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    echo "  2. Install Java: brew install openjdk@17"
    echo "  3. Link Java: sudo ln -sfn /opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-17.jdk"
    exit 1
fi

# Check if Java server exists
if [ -f "Tablut/Executables/Server.jar" ]; then
    echo "✓ Java server found (Tablut/Executables/Server.jar)"
else
    echo "✗ Java server not found at Tablut/Executables/Server.jar"
    echo "  Make sure the Tablut project is in the correct location"
fi

# Check Python dependencies
if [ -d "venv" ]; then
    echo "✓ Virtual environment found"
    source venv/bin/activate
    if python -c "import torch, numpy, gymnasium" 2>/dev/null; then
        echo "✓ Python dependencies installed"
    else
        echo "✗ Python dependencies missing"
        echo "  Run: pip install -r requirements.txt"
    fi
else
    echo "⚠ Virtual environment not found"
    echo "  Create with: python3 -m venv venv"
fi

echo ""
echo "Prerequisites check complete!"

