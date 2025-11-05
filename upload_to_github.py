#!/usr/bin/env python3
"""
Upload FaceReco project to GitHub
"""
import subprocess
import sys
import os
import webbrowser

def run_command(command, description, cwd=None):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, cwd=cwd)
        print(f"✅ {description} completed successfully")
        if result.stdout.strip():
            print(f"   Output: {result.stdout.strip()}")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        return False, e.stderr

def main():
    print("🚀 Uploading FaceReco to GitHub")
    print("=" * 40)
    
    # Check if FaceReco directory exists
    repo_path = "../FaceReco"
    if not os.path.exists(repo_path):
        print(f"❌ Repository directory not found: {repo_path}")
        print("Please run the create_new_repo.sh script first")
        return
    
    print(f"✅ Found repository at: {os.path.abspath(repo_path)}")
    
    # Get GitHub username
    print("\n📝 GitHub Setup")
    print("-" * 20)
    
    # Try to get GitHub username from git config
    success, github_user = run_command("git config --global github.user", "Getting GitHub username")
    
    if not success or not github_user.strip():
        print("GitHub username not found in git config.")
        github_user = input("Enter your GitHub username: ").strip()
        if not github_user:
            print("❌ GitHub username is required")
            return
    else:
        github_user = github_user.strip()
        print(f"Using GitHub username: {github_user}")
    
    # Repository details
    repo_name = "FaceReco"
    repo_url = f"https://github.com/{github_user}/{repo_name}.git"
    
    print(f"\n🎯 Repository Details:")
    print(f"   Name: {repo_name}")
    print(f"   URL: {repo_url}")
    print(f"   Path: {os.path.abspath(repo_path)}")
    
    # Step 1: Open GitHub to create repository
    print(f"\n📋 Step 1: Create Repository on GitHub")
    print("-" * 40)
    print("I'll open GitHub in your browser to create the repository.")
    print("Please:")
    print("1. Create a new repository named 'FaceReco'")
    print("2. Add description: 'YuNet Face Detection with real-time camera support'")
    print("3. Make it Public (recommended)")
    print("4. DON'T initialize with README, .gitignore, or license")
    print("5. Click 'Create repository'")
    
    input("\nPress Enter to open GitHub in your browser...")
    
    try:
        webbrowser.open("https://github.com/new")
        print("✅ Opened GitHub in browser")
    except:
        print("❌ Could not open browser. Please go to: https://github.com/new")
    
    input("\nAfter creating the repository on GitHub, press Enter to continue...")
    
    # Step 2: Add remote and push
    print(f"\n🔗 Step 2: Connect and Upload")
    print("-" * 40)
    
    # Add remote origin
    success, _ = run_command(f"git remote add origin {repo_url}", 
                           f"Adding remote origin", cwd=repo_path)
    
    if not success:
        # Remote might already exist, try to set URL
        success, _ = run_command(f"git remote set-url origin {repo_url}", 
                               f"Setting remote URL", cwd=repo_path)
    
    if not success:
        print("❌ Failed to set remote origin")
        return
    
    # Set main branch
    success, _ = run_command("git branch -M main", "Setting main branch", cwd=repo_path)
    
    # Push to GitHub
    print("\n🚀 Pushing to GitHub...")
    success, output = run_command("git push -u origin main", "Pushing to GitHub", cwd=repo_path)
    
    if success:
        print("\n🎉 SUCCESS! Your repository has been uploaded to GitHub!")
        print(f"🔗 Repository URL: https://github.com/{github_user}/{repo_name}")
        print(f"📱 Clone command: git clone {repo_url}")
        
        # Open the repository in browser
        try:
            webbrowser.open(f"https://github.com/{github_user}/{repo_name}")
            print("✅ Opened your repository in browser")
        except:
            print("❌ Could not open browser automatically")
        
        print("\n📋 Your repository includes:")
        print("   ✅ Real-time face detection with camera")
        print("   ✅ Image processing capabilities")
        print("   ✅ Comprehensive documentation")
        print("   ✅ Multiple detection methods")
        print("   ✅ Easy setup and installation")
        
        print(f"\n🌟 Share your project: https://github.com/{github_user}/{repo_name}")
        
    else:
        print("\n❌ Upload failed. Common issues:")
        print("1. Repository not created on GitHub yet")
        print("2. Wrong GitHub username")
        print("3. Authentication required")
        print("\nTry running: git push -u origin main")
        print("You may need to enter your GitHub credentials")

if __name__ == "__main__":
    main()