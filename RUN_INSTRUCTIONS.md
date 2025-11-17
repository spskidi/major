# How to Run the Project

This is a full-stack application with a Python FastAPI backend and a React frontend.

## Prerequisites

- **Python 3.8+** (Python 3.12 is being used based on the venv)
- **Node.js 14+** and npm
- **OpenWeatherMap API Key** (free at https://openweathermap.org/api)

## Step 1: Backend Setup

### 1.1 Navigate to Backend Directory
```bash
cd weather-app/backend
```

### 1.2 Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
.\venv\Scripts\activate.bat
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

### 1.3 Install Dependencies (if not already installed)
```bash
pip install -r requirements.txt
```

### 1.4 Create Environment File

Create a `.env` file in the `weather-app/backend` directory:

```bash
# Create .env file
echo OPENWEATHER_API_KEY=your_api_key_here > .env
```

Or manually create a `.env` file with:
```
OPENWEATHER_API_KEY=your_actual_api_key_here
```

**To get an API key:**
1. Go to https://openweathermap.org/api
2. Sign up for a free account
3. Get your API key from the dashboard
4. Replace `your_actual_api_key_here` with your actual key

### 1.5 Run the Backend Server

```bash
uvicorn main:app --reload
```

Or if you're in the backend directory:
```bash
python -m uvicorn main:app --reload
```

The backend will be available at: **http://localhost:8000**

You can also access:
- API Documentation: **http://localhost:8000/docs** (Swagger UI)
- Alternative Docs: **http://localhost:8000/redoc**

## Step 2: Frontend Setup

### 2.1 Open a New Terminal Window

Keep the backend running and open a new terminal.

### 2.2 Navigate to Frontend Directory
```bash
cd weather-app/frontend
```

### 2.3 Install Dependencies
```bash
npm install
```

### 2.4 Start the Frontend Development Server
```bash
npm start
```

The frontend will be available at: **http://localhost:3000**

The browser should automatically open. If not, manually navigate to `http://localhost:3000`

## Step 3: Verify Everything is Working

1. **Backend**: Check http://localhost:8000/docs - you should see the API documentation
2. **Frontend**: Check http://localhost:3000 - you should see the application dashboard
3. **WebSocket**: The frontend should connect to the backend WebSocket at `ws://localhost:8000/ws`

## Troubleshooting

### Backend Issues

**Port 8000 already in use:**
```bash
# Use a different port
uvicorn main:app --reload --port 8001
```
Then update the frontend API config if needed.

**Missing API Key Error:**
- Make sure the `.env` file exists in `weather-app/backend/`
- Verify the API key is correct
- Restart the backend server after creating/updating `.env`

**Module not found errors:**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### Frontend Issues

**Port 3000 already in use:**
- The terminal will prompt you to use a different port (usually 3001)
- Or manually specify: `PORT=3001 npm start`

**Module not found errors:**
```bash
# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

**WebSocket connection errors:**
- Make sure the backend is running on port 8000
- Check browser console for connection errors
- Verify CORS settings in `main.py` allow `http://localhost:3000`

## Running Both Services

### Option 1: Two Separate Terminals (Recommended)
- Terminal 1: Backend (`cd weather-app/backend && uvicorn main:app --reload`)
- Terminal 2: Frontend (`cd weather-app/frontend && npm start`)

### Option 2: Background Processes (Windows PowerShell)
```powershell
# Start backend in background
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd weather-app/backend; .\venv\Scripts\Activate.ps1; uvicorn main:app --reload"

# Start frontend in background
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd weather-app/frontend; npm start"
```

## Quick Start Commands Summary

**Backend:**
```bash
cd weather-app/backend
.\venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt  # If needed
# Create .env file with OPENWEATHER_API_KEY
uvicorn main:app --reload
```

**Frontend:**
```bash
cd weather-app/frontend
npm install  # If needed
npm start
```

## Project Structure

```
weather-app/
├── backend/          # FastAPI Python backend
│   ├── main.py      # Main application file
│   ├── requirements.txt
│   └── .env         # Environment variables (create this)
└── frontend/        # React frontend
    ├── src/
    └── package.json
```

## API Endpoints

Once running, you can access:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **WebSocket**: ws://localhost:8000/ws

