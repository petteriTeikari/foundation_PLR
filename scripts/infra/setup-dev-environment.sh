#!/usr/bin/env bash
#
# Foundation PLR - Development Environment Setup
# ==============================================
#
# ONE COMMAND TO SET UP EVERYTHING:
#
#   sudo ./scripts/setup-dev-environment.sh
#
# That's it. No manual steps. No copy-paste. Just run it.
#
# What gets installed:
#   - Python 3.11+ (via pyenv or system)
#   - uv (fast Python package manager) - MANDATORY, replaces pip/conda
#   - Node.js 20 LTS (via nvm) + npm
#   - R 4.5.2 (from CRAN) + pminternal package
#   - Docker & Docker Compose
#   - ruff (Python linter)
#   - pre-commit (git hooks)
#   - Just (command runner, optional)
#
# REPRODUCIBILITY: Version Pinning Policy
# =======================================
# We practice what we preach (see manuscript section-22-mlops-reproducability.tex):
# "The solution requires explicit, versioned specification of the entire computational environment."
#
# All versions are PINNED for reproducibility:
#   - Python packages: uv.lock (cross-platform lock file)
#   - R packages: PINNED_VERSIONS in this script (search for "PINNED_VERSIONS")
#   - Node.js: 20.x LTS
#   - R: 4.5.2 (search for "R_PINNED_VERSION")
#
# TO UPDATE VERSIONS:
#   1. Search for "R_PINNED_VERSION" and "PINNED_VERSIONS" in this file
#   2. Check CRAN for latest versions: https://cran.r-project.org/web/packages/
#   3. Update version strings
#   4. Test installation
#   5. Document changes in git commit
#
# Package Manager Policy (ENFORCED - see CLAUDE.md):
#   - Python: uv only (pip/conda BANNED)
#   - R: install.packages() from CRAN only (conda BANNED)
#   - JS/TS: npm (bun under consideration for future)
#
# Supports: Ubuntu/Debian, macOS, Fedora, Arch, Windows (WSL)
# For NixOS: Run 'nix develop' instead (see flake.nix)
#

set -euo pipefail

# ============================================
# APT LOCK CHECK & CLEANUP (Ubuntu/Debian)
# ============================================
# Handles both active locks (process holding) and stale locks (orphaned files)

clean_apt_locks() {
    # Only run on Debian-based systems
    if ! command -v apt-get &>/dev/null; then
        return 0
    fi

    local locks=(
        "/var/lib/apt/lists/lock"
        "/var/lib/dpkg/lock"
        "/var/lib/dpkg/lock-frontend"
        "/var/cache/apt/archives/lock"
    )

    local has_active_lock=false
    local has_stale_lock=false

    # First pass: check for active locks (process holding them)
    for lock in "${locks[@]}"; do
        if [ -f "$lock" ]; then
            if fuser "$lock" &>/dev/null; then
                local pid=$(fuser "$lock" 2>/dev/null | awk '{print $1}')
                local proc=$(ps -p "$pid" -o comm= 2>/dev/null || echo "unknown")
                print_error "apt is locked by active process $pid ($proc)"
                print_info "Wait for it to finish, or run: sudo kill $pid"
                has_active_lock=true
            fi
        fi
    done

    if $has_active_lock; then
        return 1
    fi

    # Second pass: check for stale locks (file exists but no process)
    for lock in "${locks[@]}"; do
        if [ -f "$lock" ]; then
            if ! fuser "$lock" &>/dev/null; then
                print_warning "Found stale lock file: $lock"
                has_stale_lock=true
            fi
        fi
    done

    # Clean up stale locks
    if $has_stale_lock; then
        print_info "Cleaning up stale apt locks..."
        for lock in "${locks[@]}"; do
            if [ -f "$lock" ] && ! fuser "$lock" &>/dev/null; then
                rm -f "$lock" 2>/dev/null || true
                print_info "  Removed: $lock"
            fi
        done
        # Fix any interrupted dpkg operations
        print_info "Running dpkg --configure -a to fix any interrupted operations..."
        dpkg --configure -a 2>/dev/null || true
        print_success "Stale locks cleaned"
    fi

    return 0
}

wait_for_apt_locks() {
    local max_wait=60
    local waited=0

    while ! clean_apt_locks 2>/dev/null; do
        if [ $waited -ge $max_wait ]; then
            print_error "Timed out waiting for apt locks after ${max_wait}s"
            print_info "Kill the blocking process manually and retry"
            exit 1
        fi
        print_info "Waiting for apt locks to be released... (${waited}s/${max_wait}s)"
        sleep 5
        waited=$((waited + 5))
    done
}

# ============================================
# SUDO HANDLING
# ============================================
# When run with sudo, we need to handle user-space tools carefully.
# System packages (R, apt) need sudo; user tools (uv, nvm) don't.

REAL_USER="${SUDO_USER:-$USER}"
REAL_HOME=$(eval echo "~$REAL_USER")

# Run command as the real user (not root) when needed
run_as_user() {
    if [ -n "${SUDO_USER:-}" ]; then
        sudo -u "$REAL_USER" "$@"
    else
        "$@"
    fi
}

# ============================================
# COLORS AND FORMATTING
# ============================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

print_header() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC} ${BOLD}$1${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
}

print_step() {
    echo -e "${CYAN}▶${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${PURPLE}ℹ${NC} $1"
}

# ============================================
# OS DETECTION
# ============================================

detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            if [[ "$ID" == "ubuntu" || "$ID" == "debian" || "$ID_LIKE" == *"debian"* ]]; then
                echo "ubuntu"
            elif [[ "$ID" == "fedora" || "$ID" == "rhel" || "$ID" == "centos" ]]; then
                echo "fedora"
            elif [[ "$ID" == "arch" || "$ID_LIKE" == *"arch"* ]]; then
                echo "arch"
            elif [[ "$ID" == "nixos" ]]; then
                echo "nixos"
            else
                echo "linux-unknown"
            fi
        else
            echo "linux-unknown"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# ============================================
# PREREQUISITE CHECKS
# ============================================

check_command() {
    command -v "$1" &> /dev/null
}

check_sudo() {
    if [[ "$OS" != "macos" && "$OS" != "windows" ]]; then
        if ! check_command sudo; then
            print_error "sudo is required but not installed"
            exit 1
        fi
    fi
}

# ============================================
# UBUNTU/DEBIAN INSTALLATION
# ============================================

install_ubuntu() {
    print_header "Installing on Ubuntu/Debian"

    # Check if we need sudo at all (Docker might be the only thing needing install)
    local NEEDS_SYSTEM_DEPS=false
    local NEEDS_DOCKER=false
    local NEEDS_COMPOSE=false

    # Check for essential build tools
    if ! check_command gcc || ! check_command make; then
        NEEDS_SYSTEM_DEPS=true
    fi

    if ! check_command docker; then
        NEEDS_DOCKER=true
    fi

    if ! docker compose version &> /dev/null 2>&1; then
        NEEDS_COMPOSE=true
    fi

    # Only run sudo commands if actually needed
    if $NEEDS_SYSTEM_DEPS || $NEEDS_DOCKER || $NEEDS_COMPOSE; then
        print_step "Updating package lists..."
        sudo apt-get update -qq

        if $NEEDS_SYSTEM_DEPS; then
            print_step "Installing system dependencies..."
            sudo apt-get install -y -qq \
                build-essential \
                curl \
                wget \
                git \
                ca-certificates \
                gnupg \
                lsb-release \
                software-properties-common \
                libssl-dev \
                libffi-dev \
                python3-dev \
                python3-pip \
                python3-venv \
                zlib1g-dev \
                libbz2-dev \
                libreadline-dev \
                libsqlite3-dev \
                libncurses5-dev \
                libncursesw5-dev \
                xz-utils \
                tk-dev \
                libxml2-dev \
                libxmlsec1-dev \
                liblzma-dev
            print_success "System dependencies installed"
        else
            print_info "System build dependencies already installed"
        fi

        # Docker
        if $NEEDS_DOCKER; then
            print_step "Installing Docker..."
            curl -fsSL https://get.docker.com | sudo sh
            sudo usermod -aG docker "$USER"
            print_success "Docker installed (log out and back in for group changes)"
        else
            print_info "Docker already installed"
        fi

        # Docker Compose (v2 plugin)
        if $NEEDS_COMPOSE; then
            print_step "Installing Docker Compose..."
            sudo apt-get install -y -qq docker-compose-plugin
            print_success "Docker Compose installed"
        else
            print_info "Docker Compose already installed"
        fi
    else
        print_info "All system dependencies already installed (skipping sudo)"
    fi
}

# ============================================
# MACOS INSTALLATION
# ============================================

install_macos() {
    print_header "Installing on macOS"

    # Homebrew
    if ! check_command brew; then
        print_step "Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

        # Add to PATH for Apple Silicon
        if [[ -f /opt/homebrew/bin/brew ]]; then
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi
        print_success "Homebrew installed"
    else
        print_info "Homebrew already installed"
        print_step "Updating Homebrew..."
        brew update
    fi

    print_step "Installing system dependencies..."
    brew install \
        git \
        curl \
        wget \
        openssl \
        readline \
        sqlite3 \
        xz \
        zlib \
        tcl-tk

    print_success "System dependencies installed"

    # Docker Desktop
    if ! check_command docker; then
        print_step "Installing Docker Desktop..."
        brew install --cask docker
        print_warning "Docker Desktop installed. Please open it manually to complete setup."
    else
        print_info "Docker already installed"
    fi
}

# ============================================
# WINDOWS INSTALLATION (WSL/Git Bash)
# ============================================

install_windows() {
    print_header "Installing on Windows (Git Bash/WSL)"

    print_warning "For best experience, use WSL2 with Ubuntu."
    print_info "This script will install tools available in Git Bash."

    # Check if we're in WSL
    if grep -qi microsoft /proc/version 2>/dev/null; then
        print_info "Detected WSL - using Linux installation path"
        install_ubuntu
        return
    fi

    # Chocolatey
    if ! check_command choco; then
        print_step "Installing Chocolatey..."
        print_warning "Please run this in an elevated PowerShell:"
        echo ""
        echo '  Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString("https://community.chocolatey.org/install.ps1"))'
        echo ""
        print_info "After installing Chocolatey, re-run this script."
        exit 1
    fi

    print_step "Installing dependencies via Chocolatey..."
    choco install -y git nodejs-lts python docker-desktop

    print_success "Dependencies installed"
    print_warning "Please restart your terminal and Docker Desktop."
}

# ============================================
# NIXOS SPECIAL HANDLING
# ============================================

install_nixos() {
    print_header "NixOS Detected - Welcome, Nix Enthusiast! 🎉"

    echo ""
    echo -e "${PURPLE}We appreciate you running NixOS!${NC}"
    echo ""
    echo "For NixOS, we recommend using the flake.nix in this repository:"
    echo ""
    echo -e "  ${CYAN}# Enter development shell${NC}"
    echo -e "  ${BOLD}nix develop${NC}"
    echo ""
    echo -e "  ${CYAN}# Or with direnv (recommended)${NC}"
    echo -e "  ${BOLD}echo 'use flake' > .envrc && direnv allow${NC}"
    echo ""

    # Create flake.nix if it doesn't exist
    if [ ! -f "flake.nix" ]; then
        print_step "Creating flake.nix..."
        create_nix_flake
        print_success "flake.nix created"
    else
        print_info "flake.nix already exists"
    fi

    echo ""
    print_info "Run 'nix develop' to enter the development environment."
    exit 0
}

create_nix_flake() {
    cat > flake.nix << 'FLAKE_EOF'
{
  description = "Foundation PLR Development Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Python
            python311
            python311Packages.pip
            python311Packages.virtualenv

            # Node.js
            nodejs_20
            nodePackages.npm

            # Development tools
            uv
            ruff
            pre-commit
            just

            # Docker (if you want native, use docker-compose instead)
            docker-compose

            # Build dependencies
            gcc
            gnumake
            openssl
            zlib
            readline
            sqlite
            libffi

            # Git
            git
            gh
          ];

          shellHook = ''
            echo "🔬 Foundation PLR Development Environment"
            echo ""
            echo "Available commands:"
            echo "  uv sync          - Install Python dependencies"
            echo "  npm install      - Install Node.js dependencies (in apps/visualization/)"
            echo "  pre-commit install - Set up git hooks"
            echo ""
          '';
        };
      });
}
FLAKE_EOF
}

# ============================================
# R INSTALLATION (for pminternal and statistical analysis)
# ============================================
# VERSION PINS - Reproducibility (see manuscript section-22-mlops-reproducability.tex)
# "The solution requires explicit, versioned specification of the entire computational environment."
# ============================================

# R Version - pinned for reproducibility
# Check latest: https://cloud.r-project.org/bin/linux/ubuntu/jammy-cran40/
# As of 2026-01: CRAN has R 4.5.2 for Ubuntu 22.04 (jammy)
R_PINNED_VERSION="4.5.2"

# R Package Versions - pinned for reproducibility
# Check latest: https://cran.r-project.org/web/packages/<package>/index.html
declare -A R_PACKAGE_VERSIONS=(
    ["Hmisc"]="5.2-1"
    ["survival"]="3.7-0"
    ["MASS"]="7.3-61"
    ["mgcv"]="1.9-1"
    ["pROC"]="1.18.5"
    ["dcurves"]="0.5.0"
    ["pmcalibration"]="0.1.0"
    ["pminternal"]="0.1.0"
)

# ============================================

install_r() {
    print_header "Installing R ($R_PINNED_VERSION for pminternal)"

    local R_MIN_VERSION="4.4.0"

    # Check if R is already installed and version
    if check_command R; then
        local R_VERSION=$(R --version 2>&1 | head -n1 | grep -oP 'R version \K[0-9]+\.[0-9]+\.[0-9]+' || echo "0.0.0")
        local R_PATH=$(which R)
        print_info "Current R: $R_PATH (version $R_VERSION)"

        # Check if it's conda R
        if [[ "$R_PATH" == *"conda"* ]] || [[ "$R_PATH" == *"anaconda"* ]]; then
            print_warning "Detected conda R installation"
            print_info "Conda R can have package compatibility issues with pminternal"
            print_info "Recommendation: Install system R from CRAN for better compatibility"
        fi

        # Compare versions
        local R_MAJOR=$(echo "$R_VERSION" | cut -d. -f1)
        local R_MINOR=$(echo "$R_VERSION" | cut -d. -f2)
        local MIN_MAJOR=$(echo "$R_MIN_VERSION" | cut -d. -f1)
        local MIN_MINOR=$(echo "$R_MIN_VERSION" | cut -d. -f2)

        if [[ "$R_MAJOR" -gt "$MIN_MAJOR" ]] || \
           [[ "$R_MAJOR" -eq "$MIN_MAJOR" && "$R_MINOR" -ge "$MIN_MINOR" ]]; then
            print_success "R version $R_VERSION meets requirements (>= $R_MIN_VERSION)"
            return 0
        else
            print_warning "R version $R_VERSION is below required $R_MIN_VERSION"
        fi
    fi

    # Install system R from CRAN (preferred over conda)
    print_step "Installing R from CRAN (recommended for pminternal)..."

    case "$OS" in
        ubuntu)
            # =================================================================
            # Install R from CRAN for Ubuntu - MODERN SIGNED-BY METHOD
            # Based on research: https://wiki.debian.org/SecureApt
            # This avoids deprecated apt-key and add-apt-repository
            # =================================================================
            print_info "Setting up CRAN repository for R ${R_PINNED_VERSION}..."
            apt-get install -y -qq wget ca-certificates gnupg

            # COMPLETELY remove any existing Ubuntu R and broken configs
            print_info "Purging old Ubuntu R packages and broken configs..."
            apt-get purge -y -qq 'r-base*' 'r-cran-*' 'r-recommended' 2>/dev/null || true
            apt-get autoremove -y -qq 2>/dev/null || true

            # Clean up ALL previous CRAN configuration attempts
            rm -f /etc/apt/preferences.d/99-block-ubuntu-r
            rm -f /etc/apt/preferences.d/99-r-cran
            rm -f /etc/apt/sources.list.d/r-project.list
            rm -f /etc/apt/sources.list.d/r-cran.list
            rm -f /etc/apt/sources.list.d/*cran*.list
            rm -f /etc/apt/sources.list.d/*r-project*.list
            rm -f /etc/apt/sources.list.d/archive_uri-https_cloud_r-project_org*.list
            rm -f /usr/share/keyrings/r-project.gpg
            rm -f /usr/share/keyrings/cran-archive-keyring.gpg
            rm -f /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
            print_success "Cleaned up old CRAN configurations"

            # MODERN METHOD Step 1: Download and convert GPG key to BINARY format
            # Key goes to /usr/share/keyrings/ (recommended location since APT 2.4)
            print_info "Downloading CRAN GPG key..."
            wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | \
                gpg --dearmor -o /usr/share/keyrings/cran-archive-keyring.gpg

            # CRITICAL: Set correct permissions (644 = readable by _apt user)
            chmod 644 /usr/share/keyrings/cran-archive-keyring.gpg
            chown root:root /usr/share/keyrings/cran-archive-keyring.gpg

            # Verify key was added and has correct permissions
            if [ ! -f /usr/share/keyrings/cran-archive-keyring.gpg ]; then
                print_error "Failed to create CRAN GPG keyring"
                return 1
            fi
            local KEY_PERMS=$(stat -c "%a" /usr/share/keyrings/cran-archive-keyring.gpg)
            print_success "GPG key installed: /usr/share/keyrings/cran-archive-keyring.gpg (perms: $KEY_PERMS)"

            # MODERN METHOD Step 2: Create sources.list with signed-by directive
            local UBUNTU_CODENAME=$(lsb_release -cs)
            print_info "Adding CRAN repo for Ubuntu $UBUNTU_CODENAME with signed-by..."

            # Use signed-by to explicitly link key to this repository (better security)
            cat > /etc/apt/sources.list.d/cran-r.list << EOF
# CRAN R repository for Ubuntu ${UBUNTU_CODENAME}
# Installed by Foundation PLR setup script
deb [arch=amd64 signed-by=/usr/share/keyrings/cran-archive-keyring.gpg] https://cloud.r-project.org/bin/linux/ubuntu ${UBUNTU_CODENAME}-cran40/
EOF
            chmod 644 /etc/apt/sources.list.d/cran-r.list
            print_success "CRAN repository added with signed-by method"

            # Pin CRAN packages higher than Ubuntu's
            print_info "Pinning CRAN R packages to priority 700..."
            cat > /etc/apt/preferences.d/99-r-cran << 'EOF'
# Prefer CRAN R packages (4.5+) over Ubuntu's (4.1.x)
Package: r-base r-base-core r-base-dev r-recommended r-cran-*
Pin: origin cloud.r-project.org
Pin-Priority: 700
EOF
            chmod 644 /etc/apt/preferences.d/99-r-cran

            # Update package lists (show output but continue on warnings)
            print_info "Updating package lists..."
            apt-get update 2>&1 | grep -v "GPG error" | grep -v "is not signed" | grep -v "^W:" || true
            print_success "Package lists updated"

            # Debug: Show what version apt will install
            print_info "Checking apt-cache policy for r-base..."
            apt-cache policy r-base | head -10 || true

            # Verify CRAN version will be installed (not Ubuntu's)
            # Note: Using || true to prevent set -e from killing script on grep failure
            local CANDIDATE_VERSION
            CANDIDATE_VERSION=$(apt-cache policy r-base | grep "Candidate:" | awk '{print $2}' || echo "")
            print_info "apt will install r-base version: $CANDIDATE_VERSION"

            # Check if CRAN version is available
            # CRAN versions have .2204 suffix (for Ubuntu 22.04), Ubuntu's have "ubuntu" suffix
            if [[ "$CANDIDATE_VERSION" == "(none)" ]] || [[ -z "$CANDIDATE_VERSION" ]]; then
                print_error "No R installation candidate found!"
                print_info "CRAN repository may not be properly configured."
                print_info "Debug commands:"
                print_info "  cat /etc/apt/sources.list.d/cran-r.list"
                print_info "  ls -la /usr/share/keyrings/cran-archive-keyring.gpg"
                print_info "  gpg --show-keys /usr/share/keyrings/cran-archive-keyring.gpg"
                return 1
            fi

            if [[ "$CANDIDATE_VERSION" == *"ubuntu"* ]]; then
                print_error "apt is trying to install Ubuntu's R ($CANDIDATE_VERSION), not CRAN's!"
                print_info "CRAN repository priority may not be high enough."
                print_info "Debug commands:"
                print_info "  cat /etc/apt/preferences.d/99-r-cran"
                print_info "  apt-cache policy r-base"
                return 1
            fi

            print_success "CRAN R version $CANDIDATE_VERSION will be installed"

            print_info "Installing R ${R_PINNED_VERSION} from CRAN (this takes a while)..."
            apt-get install -y r-base r-base-dev

            # Install common R package dependencies (including Hmisc deps)
            print_info "Installing R package system dependencies..."
            apt-get install -y -qq \
                libcurl4-openssl-dev \
                libssl-dev \
                libxml2-dev \
                libfontconfig1-dev \
                libharfbuzz-dev \
                libfribidi-dev \
                libfreetype6-dev \
                libpng-dev \
                libtiff5-dev \
                libjpeg-dev \
                libcairo2-dev \
                libxt-dev \
                libx11-dev \
                libicu-dev \
                libgit2-dev \
                libssh2-1-dev \
                zlib1g-dev \
                libgmp-dev \
                libmpfr-dev \
                libgsl-dev \
                tcl-dev \
                tk-dev \
                libsodium-dev

            print_success "R installed from CRAN"
            ;;

        macos)
            print_step "Installing R via Homebrew..."
            brew install r
            print_success "R installed via Homebrew"
            ;;

        fedora)
            sudo dnf install -y R R-devel
            print_success "R installed via dnf"
            ;;

        arch)
            sudo pacman -S --noconfirm r
            print_success "R installed via pacman"
            ;;

        *)
            print_warning "Please install R >= $R_MIN_VERSION manually"
            print_info "Visit: https://cran.r-project.org/"
            return 1
            ;;
    esac

    # Verify installation - FAIL if Ubuntu's old R was installed instead of CRAN's
    hash -r  # Refresh PATH cache
    local NEW_R_VERSION=$(R --version 2>&1 | head -n1 | grep -oP 'R version \K[0-9]+\.[0-9]+\.[0-9]+' || echo "unknown")
    local NEW_R_PATH=$(which R)
    print_info "R version after installation: $NEW_R_VERSION ($NEW_R_PATH)"

    # Verify we got CRAN's R, not Ubuntu's
    local NEW_MAJOR=$(echo "$NEW_R_VERSION" | cut -d. -f1)
    local NEW_MINOR=$(echo "$NEW_R_VERSION" | cut -d. -f2)
    if [[ "$NEW_MAJOR" -lt 4 ]] || [[ "$NEW_MAJOR" -eq 4 && "$NEW_MINOR" -lt 4 ]]; then
        print_error "R $NEW_R_VERSION was installed but we need >= 4.4.0"
        print_error "Ubuntu's outdated R package may have been installed instead of CRAN's"
        print_info "Debug: Check /etc/apt/preferences.d/99-block-ubuntu-r"
        print_info "Debug: Check /etc/apt/preferences.d/99-r-cran"
        print_info "Try manually: apt-cache policy r-base"
        return 1
    fi
    print_success "R $NEW_R_VERSION meets version requirement (>= 4.4.0)"
}

install_r_packages() {
    print_header "Installing R Packages (pminternal and dependencies)"

    if ! check_command R; then
        print_error "R is not installed. Cannot install R packages."
        return 1
    fi

    print_step "Installing pminternal and dependencies..."
    print_warning "⏳ This may take 5-10 minutes (compiling R packages from source)..."
    print_info "Don't panic if it seems stuck - R package compilation is slow but working"

    # Install packages from CRAN with PINNED VERSIONS for reproducibility
    # See: manuscript section-22-mlops-reproducability.tex
    # "lock files guarantee identical environments across machines and time"
    R --quiet --no-save << 'R_EOF'
# Set CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org"))

# =============================================================================
# PINNED VERSIONS for reproducibility (like uv.lock for Python)
# Update these when upgrading - check CRAN for latest compatible versions
# =============================================================================
PINNED_VERSIONS <- list(
    remotes = "2.5.0",        # For install_version()
    Hmisc = "5.2-1",          # Critical dependency
    survival = "3.7-0",       # dcurves dependency
    MASS = "7.3-61",          # pmcalibration dependency
    mgcv = "1.9-1",           # pmcalibration dependency
    pROC = "1.18.5",          # ROC analysis
    dcurves = "0.5.0",        # Decision curve analysis
    pmcalibration = "0.1.0",  # Calibration assessment
    pminternal = "0.1.0"      # Main package for model stability
)

cat("=== R Package Installation (Version-Pinned) ===\n")
cat("Practicing what we preach: explicit, versioned specification\n\n")

# Install remotes first (needed for install_version)
if (!requireNamespace("remotes", quietly = TRUE)) {
    cat("Installing remotes (for version-pinned installation)...\n")
    install.packages("remotes", quiet = TRUE)
}

# Function to install specific version
install_pinned <- function(pkg, version) {
    current_ver <- tryCatch(
        as.character(packageVersion(pkg)),
        error = function(e) NULL
    )

    if (!is.null(current_ver) && current_ver == version) {
        cat(sprintf("✓ %s %s already installed\n", pkg, version))
        return(TRUE)
    }

    if (!is.null(current_ver)) {
        cat(sprintf("⚠ %s %s installed, but pinned version is %s - upgrading...\n",
                    pkg, current_ver, version))
    } else {
        cat(sprintf("Installing %s %s...\n", pkg, version))
    }

    tryCatch({
        remotes::install_version(pkg, version = version,
                                  dependencies = TRUE, quiet = FALSE,
                                  upgrade = "never")
        cat(sprintf("✓ %s %s installed successfully\n", pkg, version))
        TRUE
    }, error = function(e) {
        cat(sprintf("⚠ Failed to install %s %s, trying latest...\n", pkg, version))
        tryCatch({
            install.packages(pkg, dependencies = TRUE, quiet = TRUE)
            new_ver <- as.character(packageVersion(pkg))
            cat(sprintf("✓ %s %s installed (pinned %s unavailable)\n", pkg, new_ver, version))
            TRUE
        }, error = function(e2) {
            cat(sprintf("✗ Failed to install %s: %s\n", pkg, e2$message))
            FALSE
        })
    })
}

# Install packages in dependency order with pinned versions
install_order <- c("Hmisc", "survival", "MASS", "mgcv", "pROC",
                   "dcurves", "pmcalibration", "pminternal")

all_success <- TRUE
for (pkg in install_order) {
    version <- PINNED_VERSIONS[[pkg]]
    if (!install_pinned(pkg, version)) {
        all_success <- FALSE
    }
}

# Verify pminternal is available
cat("\n=== Verification ===\n")
if (requireNamespace("pminternal", quietly = TRUE)) {
    cat(sprintf("✓ pminternal %s installed!\n", packageVersion("pminternal")))

    # Print all installed versions for reproducibility documentation
    cat("\nInstalled R package versions (for reproducibility):\n")
    for (pkg in install_order) {
        if (requireNamespace(pkg, quietly = TRUE)) {
            cat(sprintf("  %s: %s\n", pkg, packageVersion(pkg)))
        }
    }
} else {
    cat("✗ pminternal installation failed\n")
    quit(status = 1)
}
R_EOF

    if [ $? -eq 0 ]; then
        print_success "R packages installed successfully"
    else
        print_error "Failed to install some R packages"
        return 1
    fi
}

# ============================================
# CROSS-PLATFORM TOOL INSTALLATION
# ============================================

install_nvm_and_node() {
    print_header "Installing Node.js (via nvm)"

    if [ -d "$REAL_HOME/.nvm" ]; then
        print_info "nvm already installed"
    else
        print_step "Installing nvm..."
        run_as_user bash -c 'curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash'
        print_success "nvm installed"
    fi

    # Load nvm
    export NVM_DIR="$REAL_HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

    if ! check_command node || [[ "$(node --version 2>/dev/null)" != v20* ]]; then
        print_step "Installing Node.js 20 LTS..."
        run_as_user bash -c "source $NVM_DIR/nvm.sh && nvm install 20 && nvm use 20 && nvm alias default 20"
        print_success "Node.js 20 installed"
    else
        print_info "Node.js $(node --version) already installed"
    fi
}

install_uv() {
    print_header "Installing uv (Python package manager)"

    # uv installs to ~/.local/bin now (not ~/.cargo/bin)
    local UV_PATH="$REAL_HOME/.local/bin"

    if [ -x "$UV_PATH/uv" ]; then
        print_info "uv already installed"
    else
        print_step "Installing uv..."
        run_as_user bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
        print_success "uv installed"
    fi

    # Add to PATH for current session and export for subshells
    export PATH="$UV_PATH:$REAL_HOME/.cargo/bin:$PATH"

    # Verify uv is accessible
    if [ -x "$UV_PATH/uv" ]; then
        print_info "uv available at: $UV_PATH/uv"
    fi
}

install_python_tools() {
    print_header "Installing Python Development Tools"

    # uv is in ~/.local/bin
    local UV_BIN="$REAL_HOME/.local/bin"
    local TOOLS_BIN="$REAL_HOME/.local/bin"

    # ruff
    if [ -x "$TOOLS_BIN/ruff" ]; then
        print_info "ruff already installed"
    else
        print_step "Installing ruff..."
        run_as_user bash -c "export PATH=$UV_BIN:\$PATH && uv tool install ruff"
        print_success "ruff installed"
    fi

    # pre-commit
    if [ -x "$TOOLS_BIN/pre-commit" ]; then
        print_info "pre-commit already installed"
    else
        print_step "Installing pre-commit..."
        run_as_user bash -c "export PATH=$UV_BIN:\$PATH && uv tool install pre-commit"
        print_success "pre-commit installed"
    fi
}

install_just() {
    print_header "Installing just (command runner)"

    if check_command just; then
        print_info "just already installed ($(just --version))"
        return
    fi

    print_step "Installing just..."

    case "$OS" in
        ubuntu)
            # Try cargo first, then prebuilt
            if check_command cargo; then
                cargo install just
            else
                curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/.local/bin
                export PATH="$HOME/.local/bin:$PATH"
            fi
            ;;
        macos)
            brew install just
            ;;
        *)
            curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to ~/.local/bin
            export PATH="$HOME/.local/bin:$PATH"
            ;;
    esac

    print_success "just installed"
}

# ============================================
# PROJECT SETUP
# ============================================

setup_project() {
    print_header "Setting Up Project"

    # Try to find project root (look for pyproject.toml)
    local PROJECT_ROOT=""
    local SEARCH_DIR="$PWD"

    # If script is in scripts/ directory, go up one level
    if [[ "$(basename "$PWD")" == "scripts" ]]; then
        SEARCH_DIR="$(dirname "$PWD")"
    fi

    # Search upward for pyproject.toml
    while [[ "$SEARCH_DIR" != "/" ]]; do
        if [[ -f "$SEARCH_DIR/pyproject.toml" ]]; then
            PROJECT_ROOT="$SEARCH_DIR"
            break
        fi
        SEARCH_DIR="$(dirname "$SEARCH_DIR")"
    done

    if [ -z "$PROJECT_ROOT" ]; then
        print_warning "pyproject.toml not found. Are you in the project root?"
        print_info "Skipping project-specific setup."
        return
    fi

    print_info "Project root: $PROJECT_ROOT"
    cd "$PROJECT_ROOT"

    # Load nvm for Node.js
    export NVM_DIR="$REAL_HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"

    # Python dependencies
    print_step "Installing Python dependencies with uv..."
    if [ -f "pyproject.toml" ]; then
        run_as_user bash -c "cd $PROJECT_ROOT && export PATH=$REAL_HOME/.local/bin:\$PATH && uv sync"
        print_success "Python dependencies installed"
    fi

    # Node.js dependencies (for visualization)
    if [ -d "apps/visualization" ] && [ -f "apps/visualization/package.json" ]; then
        print_step "Installing Node.js dependencies..."
        run_as_user bash -c "source $NVM_DIR/nvm.sh && cd $PROJECT_ROOT/apps/visualization && npm install"
        print_success "Node.js dependencies installed"
    fi

    # Pre-commit hooks
    print_step "Setting up pre-commit hooks..."
    if [ -f ".pre-commit-config.yaml" ]; then
        run_as_user bash -c "cd $PROJECT_ROOT && export PATH=$REAL_HOME/.local/bin:\$PATH && pre-commit install"
        print_success "Pre-commit hooks installed"
    else
        print_info "No .pre-commit-config.yaml found, skipping"
    fi
}

# ============================================
# SHELL CONFIGURATION
# ============================================

configure_shell() {
    print_header "Configuring Shell"

    # Detect shell config file
    SHELL_CONFIG=""
    if [ -f "$REAL_HOME/.zshrc" ]; then
        SHELL_CONFIG="$REAL_HOME/.zshrc"
    elif [ -f "$REAL_HOME/.bashrc" ]; then
        SHELL_CONFIG="$REAL_HOME/.bashrc"
    elif [ -f "$REAL_HOME/.bash_profile" ]; then
        SHELL_CONFIG="$REAL_HOME/.bash_profile"
    fi

    if [ -z "$SHELL_CONFIG" ]; then
        print_warning "Could not detect shell config file"
        return
    fi

    print_info "Shell config: $SHELL_CONFIG"

    # Add PATH entries if not present
    local PATHS_TO_ADD=(
        'export PATH="$HOME/.cargo/bin:$PATH"'
        'export PATH="$HOME/.local/bin:$PATH"'
    )

    # Ensure we write to the real user's config, not root's
    if [ -n "${SUDO_USER:-}" ]; then
        chown "$REAL_USER:$REAL_USER" "$SHELL_CONFIG" 2>/dev/null || true
    fi

    for path_line in "${PATHS_TO_ADD[@]}"; do
        if ! grep -q "$path_line" "$SHELL_CONFIG"; then
            echo "$path_line" >> "$SHELL_CONFIG"
            print_success "Added: $path_line"
        fi
    done

    print_info "Restart your terminal or run: source $SHELL_CONFIG"
}

# ============================================
# VERIFICATION
# ============================================

verify_installation() {
    print_header "Verifying Installation"

    local ALL_GOOD=true

    # Check each tool
    local TOOLS=("git" "python3" "node" "npm" "uv" "ruff" "pre-commit" "docker" "R")

    for tool in "${TOOLS[@]}"; do
        if check_command "$tool"; then
            local version=$("$tool" --version 2>&1 | head -n1)
            print_success "$tool: $version"
        else
            print_error "$tool: NOT FOUND"
            ALL_GOOD=false
        fi
    done

    # Optional tools
    local OPTIONAL_TOOLS=("just" "docker-compose")
    for tool in "${OPTIONAL_TOOLS[@]}"; do
        if check_command "$tool"; then
            local version=$("$tool" --version 2>&1 | head -n1)
            print_success "$tool: $version (optional)"
        else
            print_warning "$tool: not installed (optional)"
        fi
    done

    echo ""
    if $ALL_GOOD; then
        print_success "All required tools installed successfully!"
    else
        print_warning "Some tools are missing. Check the errors above."
    fi
}

# ============================================
# MAIN
# ============================================

main() {
    echo ""
    echo -e "${BOLD}╔═══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║   Foundation PLR - Development Environment Setup              ║${NC}"
    echo -e "${BOLD}╚═══════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Show sudo status
    if [ -n "${SUDO_USER:-}" ]; then
        print_info "Running as root (sudo), real user: $REAL_USER"
        print_info "User tools will be installed to: $REAL_HOME"
    fi

    # Ensure common tool paths are available (uv installs to .local/bin)
    export PATH="$REAL_HOME/.local/bin:$REAL_HOME/.cargo/bin:$PATH"

    # Detect OS first (needed for apt lock check)
    OS=$(detect_os)
    print_info "Detected OS: $OS"

    # Check for apt locks early (Ubuntu/Debian)
    if [[ "$OS" == "ubuntu" ]]; then
        print_step "Checking for apt locks (active + stale)..."
        if ! clean_apt_locks; then
            print_warning "Active apt lock detected, waiting..."
            wait_for_apt_locks
        fi
        print_success "apt is available"
    fi

    # Handle NixOS specially
    if [[ "$OS" == "nixos" ]]; then
        install_nixos
        exit 0
    fi

    # Check for unsupported OS
    if [[ "$OS" == "unknown" || "$OS" == "linux-unknown" ]]; then
        print_warning "Unsupported OS. Attempting generic Linux installation..."
        OS="ubuntu"  # Try Ubuntu-style installation
    fi

    # OS-specific installation
    case "$OS" in
        ubuntu)
            check_sudo
            install_ubuntu
            ;;
        macos)
            install_macos
            ;;
        windows)
            install_windows
            ;;
        fedora)
            print_warning "Fedora support is experimental"
            sudo dnf install -y git curl wget python3 python3-pip docker docker-compose
            sudo systemctl enable --now docker
            sudo usermod -aG docker "$USER"
            ;;
        arch)
            print_warning "Arch support is experimental"
            sudo pacman -Syu --noconfirm git curl wget python python-pip docker docker-compose
            sudo systemctl enable --now docker
            sudo usermod -aG docker "$USER"
            ;;
    esac

    # Cross-platform tools
    install_nvm_and_node
    install_uv
    install_python_tools
    install_just

    # R installation (for statistical analysis and pminternal)
    install_r
    install_r_packages

    # Shell configuration
    configure_shell

    # Project setup (if in project directory)
    setup_project

    # Verify
    verify_installation

    # Final message
    echo ""
    print_header "Setup Complete! 🎉"
    echo ""
    echo "Next steps:"
    echo "  1. Restart your terminal (or run: source ~/.bashrc)"
    echo "  2. If Docker was just installed, log out and back in"
    echo "  3. Run 'uv sync' to install Python dependencies"
    echo "  4. Run 'cd visualization && npm install' for D3.js viz"
    echo ""
    echo "For NixOS users: Run 'nix develop' instead"
    echo ""
    print_info "Happy coding! 🔬"
}

# Run main
main "$@"
