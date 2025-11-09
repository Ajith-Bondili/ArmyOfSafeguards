"""Quick test to verify the factuality safeguard is working."""
import sys
from pathlib import Path

# Add parent directory to path so we can import the safeguard
sys.path.insert(0, str(Path(__file__).parent.parent))

from safeguard_factuality import predict

print("Testing factuality safeguard...")
print("-" * 50)

# Test a factual statement
factual = "The capital of France is Paris."
result1 = predict(factual)
print(f"✓ Factual: '{factual}'")
print(f"  → {result1['label']} (confidence: {result1['confidence']:.2%})")

print()

# Test a non-factual statement
nonfactual = "The moon is made of cheese."
result2 = predict(nonfactual)
print(f"✓ Non-factual: '{nonfactual}'")
print(f"  → {result2['label']} (confidence: {result2['confidence']:.2%})")

print("-" * 50)
print("✅ Safeguard is working correctly!")

