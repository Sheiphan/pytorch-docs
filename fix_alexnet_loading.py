"""
This script demonstrates how to load AlexNet with pretrained weights
while avoiding SSL certificate verification issues on macOS.
"""

import torch
from torchvision.models import alexnet
from torchvision.models.alexnet import AlexNet_Weights

# Method 1: Use the installed certificates (recommended if you ran install_certificates.py)
# This should work now that we've installed the proper certificates
try:
    print("Attempting to load AlexNet with default SSL context...")
    weights = AlexNet_Weights.DEFAULT
    model = alexnet(weights=weights)
    print("Success! AlexNet loaded with pretrained weights.")
except Exception as e:
    print(f"Error with default SSL context: {e}")
    
    # Method 2: Temporarily disable SSL verification
    print("\nFalling back to disabled SSL verification...")
    import ssl
    old_https_context = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    
    try:
        weights = AlexNet_Weights.DEFAULT
        model = alexnet(weights=weights)
        print("Success! AlexNet loaded with pretrained weights using disabled SSL verification.")
    except Exception as e:
        print(f"Error with disabled SSL verification: {e}")
    finally:
        # Restore the default SSL context
        ssl._create_default_https_context = old_https_context

print("\nTo fix this issue in your Jupyter notebook, add these lines before loading the model:")
print("```python")
print("# Fix SSL certificate issues")
print("import ssl")
print("ssl._create_default_https_context = ssl._create_unverified_context")
print("```") 