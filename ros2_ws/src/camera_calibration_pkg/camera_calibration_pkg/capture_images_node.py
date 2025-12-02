#!/usr/bin/env python3

import os
import sys
import tty
import termios
import subprocess

CAMERA_URL = "http://172.27.96.1:8000/frame"
SAVE_DIR = "../images"

# Create images directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

frame_count = 0

def capture_frame():
    """Capture a frame using curl"""
    global frame_count
    frame_count += 1
    
    filename = os.path.join(SAVE_DIR, f"frame_{frame_count:04d}.jpg")
    
    print(f"\n{'='*60}")
    print(f"[SPACEBAR pressed] Capturing frame #{frame_count}")
    print(f"{'='*60}")
    print(f"Saving to: {os.path.abspath(filename)}")
    
    # Run curl command
    cmd = ["curl", CAMERA_URL, "-o", filename, "-s"]  # -s for silent mode
    print(f"Running: curl {CAMERA_URL} -o {filename}")
    
    result = subprocess.run(cmd)
    
    # Check if file was created
    if os.path.exists(filename):
        size = os.path.getsize(filename)
        print(f"\n✓ SUCCESS! Frame #{frame_count} saved")
        print(f"  File: {filename}")
        print(f"  Size: {size} bytes")
    else:
        print(f"\n✗ FAILED! Frame #{frame_count} was not saved")
    
    print(f"{'='*60}")
    print("Press SPACEBAR to capture, 'q' to quit")

def getch():
    """Get a single character from standard input (no echo, no enter needed)"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def main():
    print("=" * 60)
    print("Simple Image Capture - SPACEBAR mode")
    print("=" * 60)
    print(f"Camera URL: {CAMERA_URL}")
    print(f"Save directory: {os.path.abspath(SAVE_DIR)}")
    print()
    print("Instructions:")
    print("  Press SPACEBAR to capture a frame (no ENTER needed)")
    print("  Press 'q' to quit")
    print("=" * 60)
    print("\nReady! Press SPACEBAR to start capturing...")
    print()
    
    try:
        while True:
            char = getch()
            
            # Check for spacebar (space character)
            if char == ' ':
                print("\n>>> SPACEBAR detected! <<<")
                capture_frame()
            # Check for 'q' to quit
            elif char == 'q' or char == 'Q':
                print("\n\n'q' pressed - Quitting...")
                break
            # Check for Ctrl+C (character code 3)
            elif ord(char) == 3:
                print("\n\nCtrl+C detected - Quitting...")
                break
            else:
                # Ignore other keys
                pass
                
    except KeyboardInterrupt:
        print("\n\nKeyboard interrupt - Quitting...")
    
    print(f"\nTotal frames captured: {frame_count}")
    print("Goodbye!")

if __name__ == "__main__":
    main()