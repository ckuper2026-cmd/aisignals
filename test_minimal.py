"""Minimal test app to verify FastAPI routing works"""

from fastapi import FastAPI
import uvicorn
import os

app = FastAPI(title="Test Trading App")

@app.get("/")
async def root():
    return {"status": "working", "message": "Root endpoint OK"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/api/signals")
async def signals():
    return {"signals": [], "message": "Signals endpoint OK"}

@app.get("/api/portfolio")
async def portfolio():
    return {"portfolio": {}, "message": "Portfolio endpoint OK"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting test app on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")