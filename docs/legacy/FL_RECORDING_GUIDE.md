# 🎬 FL Training Recording Guide

**Purpose**: Record FL training session for hackathon demo  
**Duration**: ~5-10 minutes  
**Tools**: Windows built-in screen recorder or OBS Studio

---

## 📋 Pre-Recording Checklist

### 1. Prepare Terminals
Open 3 PowerShell terminals:
- **Terminal 1**: FL Server
- **Terminal 2**: Hospital A Client
- **Terminal 3**: Hospital D Client

### 2. Test Commands (Don't Record Yet!)
```powershell
# Terminal 1
cd C:\Users\aishw\codered5
python fl_server_enhanced.py

# Terminal 2 (in new window)
cd C:\Users\aishw\codered5
python run_hospital_a_client_enhanced.py

# Terminal 3 (in new window)
cd C:\Users\aishw\codered5
python run_hospital_d_client_enhanced.py
```

**Stop all processes** after testing (Ctrl+C)

### 3. Arrange Windows
- Position terminals side-by-side or stacked
- Make font size readable (14-16pt)
- Clear terminal history (`cls` command)

---

## 🎥 Recording Script (5-10 minutes)

### Part 1: Start FL Server (1 min)
**Terminal 1**:
```powershell
python fl_server_enhanced.py
```

**Wait for**:
```
============================================================
Enhanced Federated Learning Server
FedProx + Fairness Weighting + Differential Privacy
============================================================

✓ Loaded relevance scores for 5 hospitals
✓ FedProxFairness strategy initialized
  - Blockchain audit: Enabled

Starting FL Server...
Waiting for clients to connect on 0.0.0.0:8080
============================================================
```

**Pause recording briefly** (you can edit this out)

---

### Part 2: Start Hospital A Client (1 min)
**Terminal 2**:
```powershell
python run_hospital_a_client_enhanced.py
```

**Wait for**:
```
============================================================
Hospital A - Enhanced FL Client
ECG Classification (PTB-XL)
============================================================

Device: cpu

Loading data...
✓ Training samples: 1000
✓ Validation samples: 200

Initializing model...
✓ Model parameters: 5,631,717

✓ DP Config: DPConfig(epsilon=5.0, delta=1e-05, max_grad_norm=1.0, num_rounds=5)

✓ Hospital A client initialized
  - Training samples: 1000
  - Validation samples: 200
  - DP enabled: ε=5.0, δ=1e-5

============================================================
Connecting to FL server...
============================================================
```

**Show it connecting to server**

---

### Part 3: Start Hospital D Client (1 min)
**Terminal 3**:
```powershell
python run_hospital_d_client_enhanced.py
```

**Wait for similar output**

---

### Part 4: Watch FL Round 1 (2-3 min)
**Focus on Terminal 1 (Server)**:

You should see:
```
============================================================
Round 1: Aggregating 2 clients
============================================================
  A: AUROC=0.72, samples=1,000, weight=0.5875
  D: AUROC=0.68, samples=500, weight=0.3595

Normalized weights:
  A: 0.6203 (62.0%)
  D: 0.3797 (38.0%)

✓ Aggregation complete. Log saved to: fl_results/round_1_aggregation.json
✓ Blockchain audit updated (Block #2)
```

**Switch to Terminal 2 (Hospital A)**:
```
============================================================
Hospital A - Round 1 Training
============================================================
FedProx µ: 0.01
DP: ε=5.0, δ=1e-5

  Batch 0: loss=0.6931, grad_norm=1.2345, σ=0.000081

✓ Training complete: avg_loss=0.6543
✓ AUROC: 0.7234
```

**Switch to Terminal 3 (Hospital D)** - similar output

---

### Part 5: Watch Rounds 2-3 (Optional, 2-3 min each)
- Let it run for 2-3 rounds total
- Show AUROC improving each round
- Show blockchain audit blocks incrementing

---

### Part 6: Stop and Show Results (1 min)
**After 2-3 rounds, press Ctrl+C on all terminals**

**Show final blockchain audit**:
```powershell
python -c "from fl_utils.blockchain_audit import BlockchainAuditLog; audit = BlockchainAuditLog('fl_results/blockchain_audit'); audit.verify_chain(); print(audit.get_summary())"
```

**Expected output**:
```
✓ Chain verified: 8 blocks
{
  "total_blocks": 8,
  "block_types": {
    "FL_ROUND": 3,
    "DP_GUARANTEE": 4
  },
  "chain_verified": true
}
```

---

## 🎬 Recording Tips

### Screen Recording Tools

**Option A: Windows Game Bar** (Built-in)
1. Press `Win + G`
2. Click "Capture" → "Record"
3. Press `Win + Alt + R` to stop

**Option B: OBS Studio** (Better quality)
1. Download from obsproject.com
2. Add "Window Capture" source
3. Click "Start Recording"

### Recording Settings
- **Resolution**: 1920x1080 (Full HD)
- **Frame Rate**: 30 FPS
- **Audio**: Optional (you can narrate live or add voiceover later)

### What to Capture
- ✅ All 3 terminal windows
- ✅ Server aggregation output
- ✅ Client training progress
- ✅ Blockchain audit verification
- ❌ Don't show errors or restarts (edit them out)

---

## 📝 Narration Script (Optional)

If adding voiceover:

**Part 1 (Server start)**:
> "First, we start the federated learning server with our enhanced FedProxFairness strategy. Notice it loads the domain relevance matrix for all 5 hospitals and enables blockchain audit logging."

**Part 2 (Clients connect)**:
> "Now Hospital A and Hospital D connect to the server. Each client has differential privacy enabled with epsilon equals 5.0. Hospital A has 1,000 ECG samples, Hospital D has 500 synthetic ECG samples."

**Part 3 (Training)**:
> "Round 1 begins. Watch the server compute fairness weights: Hospital A gets 62% weight, Hospital D gets 38%. This is based on their AUROC scores, sample counts, and domain relevance. Notice the blockchain audit is updated with each round."

**Part 4 (Results)**:
> "After 3 rounds, we verify the blockchain audit chain. All blocks are cryptographically linked and verified. This provides immutable proof of privacy-preserving federated learning."

---

## 🎯 Demo Day Usage

### During Presentation
1. **Show recording** during "Live FL Training" section (Slide 4)
2. **Pause at key moments**:
   - Server initialization
   - Fairness weights computation
   - Blockchain audit update
3. **Narrate over recording** or use pre-recorded audio

### Backup Plan
If recording fails, you can:
- Show static screenshots from recording
- Walk through code instead
- Show pre-generated log files

---

## 📊 What to Highlight in Recording

### Key Moments to Capture
1. ✅ Server: "Blockchain audit: Enabled"
2. ✅ Server: "Loaded relevance scores for 5 hospitals"
3. ✅ Client A: "DP enabled: ε=5.0, δ=1e-5"
4. ✅ Server: Fairness weights (A: 62%, D: 38%)
5. ✅ Server: "Blockchain audit updated (Block #2)"
6. ✅ Client: Training with DP noise (σ values)
7. ✅ Final: Chain verification success

### Metrics to Point Out
- **Privacy**: ε=5.0, δ=1e-5
- **Fairness**: Weights based on AUROC + samples + relevance
- **FedProx**: µ=0.01 proximal term
- **Blockchain**: Immutable audit trail

---

## 🚨 Troubleshooting

### If Server Won't Start
```powershell
# Check if port 8080 is in use
netstat -ano | findstr :8080

# Kill process if needed
taskkill /PID <process_id> /F
```

### If Clients Won't Connect
- Make sure server is running first
- Check firewall settings
- Use `127.0.0.1:8080` instead of `0.0.0.0:8080`

### If Training is Too Slow
- Reduce subset size in clients (line ~180)
- Use fewer FL rounds (change config to 2-3)
- Skip recording, use screenshots instead

---

## ✅ Post-Recording Checklist

- [ ] Recording saved successfully
- [ ] Video shows all key moments
- [ ] Audio is clear (if narrated)
- [ ] No errors visible in recording
- [ ] File size reasonable (<500MB)
- [ ] Backed up to cloud (Google Drive, etc.)

---

## 🎁 Bonus: Create GIF Highlights

For social media/README:
1. Use https://ezgif.com/video-to-gif
2. Upload 10-15 second clips
3. Create GIFs showing:
   - Server starting
   - Fairness weights computation
   - Blockchain audit update

---

**Ready to Record!** 🎬

**Estimated Time**: 10-15 minutes total  
**Output**: 5-10 minute demo video  
**Format**: MP4 (H.264)  
**Usage**: Hackathon presentation Slide 4

Good luck! 🚀
