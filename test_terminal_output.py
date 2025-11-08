#!/usr/bin/env python3
"""
Test script to verify terminal output capture is working
"""
import time
import sys

print("=" * 60)
print("Testing Terminal Output Capture")
print("=" * 60)
print()

# Test various types of output
print("✅ Normal output - this should appear in green")
print("❌ ERROR: This should appear in red")
print("⚠️  WARNING: This should appear in yellow")
print("ℹ️  INFO: This should appear in blue")
print()

# Test continuous output
print("Starting continuous output test...")
for i in range(5):
    print(f"Line {i+1}: Testing terminal output at {time.strftime('%H:%M:%S')}")
    time.sleep(1)

print()
print("✨ Terminal output test complete!")
print("If you can see this on the web dashboard, the feature is working!")
