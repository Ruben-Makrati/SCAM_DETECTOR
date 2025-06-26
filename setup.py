#!/usr/bin/env python
"""
Setup script for Scam Offers Detection System
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during {description}: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Scam Offers Detection System")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("✗ Failed to install dependencies")
        sys.exit(1)
    
    # Run Django migrations
    if not run_command("python manage.py makemigrations", "Creating database migrations"):
        print("✗ Failed to create migrations")
        sys.exit(1)
    
    if not run_command("python manage.py migrate", "Applying database migrations"):
        print("✗ Failed to apply migrations")
        sys.exit(1)
    
    # Train the model
    if not run_command("python manage.py train_model --save-model", "Training machine learning model"):
        print("⚠ Warning: Model training failed, but you can still use the system")
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nTo start the development server, run:")
    print("  python manage.py runserver")
    print("\nThe application will be available at: http://127.0.0.1:8000/")
    print("\nOptional: Create a superuser account:")
    print("  python manage.py createsuperuser")

if __name__ == "__main__":
    main() 