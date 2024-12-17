from fastapi import FastAPI
from dotenv import load_dotenv
from src.routes.routes import router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Dhvanilab AI",
    description="Dhvanilab AI API",
    version="0.1.0",
)

# Load environment variables
load_dotenv()

# Include the router
app.include_router(router)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://ai.dhvanilab.com"],
    allow_credentials=True,
    allow_methods=["*"],  # Fixed: Use list of HTTP methods instead of origins
    allow_headers=["*"],  # Fixed: Use list of allowed headers
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)