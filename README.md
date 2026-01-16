# ubx-ray
 
.venv\Scripts\activate

uvicorn app:app --reload

uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Set-ExecutionPolicy RemoteSigned -Scope CurrentUser