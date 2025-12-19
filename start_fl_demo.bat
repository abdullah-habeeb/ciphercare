@echo off
REM FL Training Demo - Quick Start Script
REM This script helps you record FL training for hackathon demo

echo ============================================================
echo FL Training Demo - Quick Start
echo ============================================================
echo.
echo This will help you record FL training for your hackathon demo.
echo.
echo INSTRUCTIONS:
echo 1. Open 3 PowerShell windows
echo 2. Run commands in this order:
echo.
echo    Terminal 1 (Server):
echo    python fl_server_enhanced.py
echo.
echo    Terminal 2 (Hospital A):
echo    python run_hospital_a_client_enhanced.py
echo.
echo    Terminal 3 (Hospital D):
echo    python run_hospital_d_client_enhanced.py
echo.
echo 3. Start screen recording BEFORE running commands
echo 4. Let it run for 2-3 FL rounds
echo 5. Press Ctrl+C to stop all terminals
echo 6. Verify blockchain: python -c "from fl_utils.blockchain_audit import BlockchainAuditLog; audit = BlockchainAuditLog('fl_results/blockchain_audit'); audit.verify_chain()"
echo.
echo ============================================================
echo.
echo Ready to start? 
echo.
choice /C YN /M "Do you want to test the server now"

if errorlevel 2 goto end
if errorlevel 1 goto runserver

:runserver
echo.
echo Starting FL Server...
echo.
python fl_server_enhanced.py

:end
echo.
echo ============================================================
echo See FL_RECORDING_GUIDE.md for detailed instructions
echo ============================================================
pause
