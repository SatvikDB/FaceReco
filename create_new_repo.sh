#!/bin/bash

# YuNet Face Detection - New Repository Setup Script
echo "🎭 Setting up new YuNet Face Detection repository..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if git is installed
if ! command -v git &> /dev/null; then
    print_error "Git is not installed. Please install Git first."
    exit 1
fi

# Get repository name from user
echo -e "${BLUE}📝 Enter the name for your new repository:${NC}"
read -p "Repository name (default: yunet-face-detection): " REPO_NAME
REPO_NAME=${REPO_NAME:-yunet-face-detection}

# Create new directory for the clean repository
NEW_REPO_DIR="../${REPO_NAME}"

print_info "Creating new repository directory: $NEW_REPO_DIR"

# Create new directory
mkdir -p "$NEW_REPO_DIR"

# Copy essential files to new repository
print_info "Copying project files..."

# Core Python files
cp *.py "$NEW_REPO_DIR/"
print_status "Copied Python scripts"

# Documentation
cp README_NEW.md "$NEW_REPO_DIR/README.md"
cp CAMERA_SETUP.md "$NEW_REPO_DIR/"
cp LICENSE.txt "$NEW_REPO_DIR/"
print_status "Copied documentation"

# Configuration files
cp requirements.txt "$NEW_REPO_DIR/"
cp .gitignore_new "$NEW_REPO_DIR/.gitignore"
print_status "Copied configuration files"

# Copy directories
cp -r img/ "$NEW_REPO_DIR/"
cp -r models/ "$NEW_REPO_DIR/"
print_status "Copied assets and models"

# Remove unnecessary files from new repo
cd "$NEW_REPO_DIR"
rm -f README_NEW.md .gitignore_new
rm -f create_new_repo.sh

# Initialize git repository
print_info "Initializing Git repository..."
git init
git add .
git commit -m "🎉 Initial commit: YuNet Face Detection Project

✨ Features:
- Real-time face detection with camera
- Image processing capabilities  
- Multiple detection backends (YuNet, Haar cascade)
- macOS optimized camera handling
- Comprehensive documentation and examples

🚀 Quick start: python3 camera_face_detection.py"

print_status "Git repository initialized with initial commit"

# Instructions for GitHub upload
echo ""
echo -e "${BLUE}🚀 Next steps to upload to GitHub:${NC}"
echo ""
echo "1. Create a new repository on GitHub:"
echo "   - Go to https://github.com/new"
echo "   - Repository name: $REPO_NAME"
echo "   - Description: 'YuNet Face Detection with real-time camera support'"
echo "   - Make it public or private as desired"
echo "   - Don't initialize with README (we already have one)"
echo ""
echo "2. Connect and push to GitHub:"
echo "   cd $NEW_REPO_DIR"
echo "   git remote add origin https://github.com/YOUR_USERNAME/$REPO_NAME.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. Or use GitHub CLI (if installed):"
echo "   cd $NEW_REPO_DIR"
echo "   gh repo create $REPO_NAME --public --source=. --remote=origin --push"
echo ""

print_status "Repository prepared successfully!"
print_info "New repository location: $NEW_REPO_DIR"

# Show repository contents
echo ""
echo -e "${BLUE}📁 Repository contents:${NC}"
ls -la "$NEW_REPO_DIR"

echo ""
print_status "Setup complete! Your YuNet Face Detection project is ready for GitHub! 🎉"