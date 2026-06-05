# ubx-ray

.venv\\Scripts\\activate

uvicorn app:app --reload

uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Set-ExecutionPolicy RemoteSigned -Scope CurrentUser



\# 상태 / 제어

Get-Service     ubxray

Restart-Service ubxray            # 코드 변경 후 적용

Stop-Service    ubxray

Start-Service   ubxray



\# 로그 실시간 보기

Get-Content C:\\ClaudeWorkspace\\ubx-ray\\logs\\ubxray.out.log -Wait -Tail 50



\# 환경변수 / 명령 수정 (GUI)

C:\\ClaudeWorkspace\\ubx-ray\\tools\\nssm\\nssm.exe edit ubxray



\# 제거

.\\install\_service.ps1 -Uninstall

