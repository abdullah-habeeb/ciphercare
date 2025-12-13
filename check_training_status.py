"""
Quick status check script for Hospital D training.
Run this tomorrow to see if training completed.
"""
import os
import numpy as np
from datetime import datetime

print("="*60)
print("Hospital D Training Status Check")
print("="*60)
print(f"Check time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Check if checkpoint exists
checkpoint_path = "src/hospital_d/train/checkpoints/best_model_geriatric.pth"

if os.path.exists(checkpoint_path):
    size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    mod_time = datetime.fromtimestamp(os.path.getmtime(checkpoint_path))
    
    print("✅ TRAINING COMPLETED!")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Size: {size_mb:.2f} MB")
    print(f"   Last modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Try to load and check data
    try:
        X_test = np.load('src/hospital_d/data/X_real_test_fixed.npy')
        Y_test = np.load('src/hospital_d/data/Y_real_test.npy')
        print(f"\n📊 Test Data Ready:")
        print(f"   Samples: {len(X_test)}")
        print(f"   Positive labels: {Y_test.sum(axis=0)}")
        
        print("\n🎯 Next Steps:")
        print("   1. Run: python evaluate_hospital_d.py")
        print("   2. Check final Test AUROC")
        print("   3. Update API to use trained model")
        print("   4. Test FL with trained weights")
        
    except Exception as e:
        print(f"\n⚠️  Data check failed: {e}")
        
else:
    print("⏳ TRAINING STILL IN PROGRESS")
    print(f"   Expected checkpoint: {checkpoint_path}")
    print(f"   Status: Not found yet")
    
    # Check if training process is still running
    print("\n💡 To check if training is running:")
    print("   Windows: tasklist | findstr python")
    print("   Check for: train_geriatric.py")
    
    print("\n⏰ Estimated completion:")
    print("   Started: ~22:00 (11 Dec 2025)")
    print("   Expected: ~02:00-04:00 (12 Dec 2025)")
    print("   Duration: 4-6 hours on CPU")

print("\n" + "="*60)
