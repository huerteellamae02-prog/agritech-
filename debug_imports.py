import sys
import os

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current directory: {current_dir}")

# Add current directory to Python path
sys.path.append(current_dir)

# Try importing
try:
    from model.predict import predict_crop
    print("Successfully imported predict_crop")
except Exception as e:
    print(f"Import error: {str(e)}")

# Print Python path
print("\nPython path:")
for path in sys.path:
    print(path)