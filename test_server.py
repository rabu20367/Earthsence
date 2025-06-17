from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello, EarthSense AI!"}

if __name__ == "__main__":
    uvicorn.run("test_server:app", host="127.0.0.1", port=8000, reload=True)
