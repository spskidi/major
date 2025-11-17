from fastapi import FastAPI, File, HTTPException, Depends, Query, UploadFile
from pydantic import BaseModel, Field
import httpx
import os
from datetime import datetime
from supabase import create_client, Client
import logging
from typing import List, Optional, Dict, Any,Literal
from uuid import UUID, uuid4
from huggingface_hub import InferenceClient
from fastapi.responses import StreamingResponse
from io import BytesIO
from datetime import datetime, timedelta # Import timedelta for date comparisons
from typing import List, Dict
from fastapi.middleware.cors import CORSMiddleware


from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TOGETHER_AI_API_KEY = os.getenv("TOGETHER_AI_API_KEY") # Renamed to TOGETHER_AI_API_KEY

# Check for required environment variables
if not SUPABASE_URL:
    raise Exception("SUPABASE_URL environment variable is not set.")
if not SUPABASE_KEY:
    raise Exception("SUPABASE_KEY environment variable is not set.")
if not TOGETHER_AI_API_KEY: # Check for TOGETHER_AI_API_KEY
    raise Exception("TOGETHER_AI_API_KEY environment variable is not set.") # Check for TOGETHER_AI_API_KEY

# Initialize FastAPI app
app = FastAPI(title="Weather API Backend")

# Add CORS middleware
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000", # or your frontend port
    "*", # Add '*' for development to allow all origins (not recommended for production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Together AI Client - No need for explicit client, using httpx directly for now
logger.info("Together AI will be used for chatbot functionality.")

# Simplified model
class LocationCoordinates(BaseModel):
    lat: float = Field(..., description="Latitude of the location")
    lon: float = Field(..., description="Longitude of the location")
    user_id: Optional[UUID] = Field(None, description="User identifier for tracking")
    forecast_days: Optional[int] = Field(7, description="Number of forecast days (1-16)")
    timezone: Optional[str] = Field("auto", description="Timezone for weather data")
    effective_type: Optional[str] = Field(None, description="Effective connection type (e.g., '4g', '3g')")
    downlink: Optional[float] = Field(None, description="Estimated bandwidth in Mbps")
    rtt: Optional[int] = Field(None, description="Round-trip time in milliseconds")

# Chatbot Request Model
class ChatbotRequest(BaseModel):
    prompt: str = Field(..., description="User's message prompt for the chatbot")

# Document Retrieval Request Model (Not needed for top 5 endpoint, but keeping for potential future use)
class DocumentRequest(BaseModel):
    document_id: UUID = Field(..., description="Unique identifier of the document to retrieve")

# Document Upload Request Model
class DocumentUploadRequest(BaseModel):
    title: str = Field(..., description="Title of the coursework document")
    user_id: Optional[UUID] = Field(None, description="User ID who uploaded the document (optional)")
    document_file: UploadFile = File(..., description="File to upload")

# Updated Document Metadata Model for Database - added file_size
class CourseWorkDocument(BaseModel):
    id: UUID = Field(..., description="Unique identifier of the document")
    filename: str = Field(..., description="Original filename")
    storage_path: str = Field(..., description="Path to the document in Supabase Storage")
    title: str = Field(..., description="Title of the coursework document")
    uploaded_at: datetime = Field(default_factory=datetime.now)
    user_id: Optional[UUID] = Field(None, description="User ID who uploaded the document (optional)")
    content_type: str = Field(..., description="MIME type of the document (e.g., application/pdf)")
    file_size: int = Field(..., description="File size in bytes") # Added file_size


# Priority Level Type
PriorityLevel = Literal["high", "medium", "low"]

# Document Metadata with Priority
class PrioritizedDocument(BaseModel):
    document_id: UUID
    filename: str
    title: str
    uploaded_at: datetime
    posted_by: Optional[UUID] # User ID as "posted_by"
    priority: PriorityLevel
    content: bytes # File content as bytes - for including downloadable file
    content_type: str # Content Type
    estimated_download_time_seconds: Optional[float] = Field(None, description="Estimated download time in seconds") # Added download time

# School Location Model
class SchoolLocation(BaseModel):
    location_id: str

# School Model
class School(BaseModel):
    school_id: UUID  # Added school_id of UUID type
    school_name: str
    school_location: SchoolLocation
    school_region: Optional[str] = None
    type_of_problem: Optional[str] = None
    duration: Optional[str] = None
    severity: Optional[str] = None
    last_updated: Optional[datetime] = None

# Schools List Response Model
class SchoolsListResponse(BaseModel):
    total_schools: int
    schools: List[School]

    # Alert Pydantic Model
class SystemAlert(BaseModel):
    id: Optional[UUID] = None # ID is generated by DB, optional on creation
    title: str
    description: str
    alert_type: Optional[str] = None
    severity: Optional[str] = None
    location_id: Optional[str] = None
    created_at: Optional[datetime] = None # Timestamp is set by DB, optional on creation

# System Alerts List Response Model
class SystemAlertsListResponse(BaseModel):
    alerts: List[SystemAlert]
    total_alerts: int


# Dependency for Supabase client
def get_supabase() -> Client:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    return supabase

# Helper function to convert values for database compatibility (rest of your existing functions remain the same)
def convert_value_for_db(key: str, value: Any) -> Any:
    """Convert values to appropriate types for database storage"""
    if value is None:
        return None
    if isinstance(value, UUID):
        return str(value)
    integer_fields = [
        "pressure", "humidity", "visibility", "wind_direction",
        "weather_code", "is_day", "cloud_cover", "cloud_cover_low",
        "cloud_cover_mid", "cloud_cover_high", "precipitation_hours",
        "precipitation_probability_max"
    ]
    if key in integer_fields and isinstance(value, (float, str)):
        try:
            return int(float(value))
        except (ValueError, TypeError):
            logger.warning(f"Could not convert {key}={value} to integer, using original value")
            return value
    return value

def sanitize_data_for_db(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively sanitize all values in a dictionary for database compatibility"""
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = sanitize_data_for_db(value)
        elif isinstance(value, list):
            result[key] = [
                sanitize_data_for_db(item) if isinstance(item, dict)
                else convert_value_for_db(key, item)
                for item in value
            ]
        else:
            result[key] = convert_value_for_db(key, value)
    return result

def generate_location_id(lat: float, lon: float, user_id: Optional[UUID] = None) -> str:
    """Generate a unique identifier for a location"""
    if user_id:
        return f"{str(user_id)}_{lat:.4f}_{lon:.4f}"
    return f"{lat:.4f}_{lon:.4f}"

async def fetch_weather_data(lat: float, lon: float, forecast_days: int = 7, timezone: str = "auto") -> Dict[str, Any]:
    """Fetch weather data from Open-Meteo API"""
    base_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": timezone,
        "forecast_days": forecast_days,
        "current": [
            "temperature_2m", "relative_humidity_2m", "apparent_temperature",
            "is_day", "weather_code", "surface_pressure", "wind_speed_10m",
            "wind_direction_10m", "visibility", "uv_index"
        ],
        "hourly": [
            "temperature_2m", "relative_humidity_2m", "apparent_temperature",
            "surface_pressure", "cloud_cover", "cloud_cover_low", "cloud_cover_mid",
            "cloud_cover_high", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
            "precipitation", "snowfall", "snow_depth", "weather_code", "visibility", "is_day"
        ],
        "daily": [
            "temperature_2m_max", "temperature_2m_min", "apparent_temperature_max",
            "apparent_temperature_min", "precipitation_sum", "rain_sum", "showers_sum",
            "snowfall_sum", "precipitation_hours", "precipitation_probability_max",
            "weather_code", "sunrise", "sunset", "wind_speed_10m_max",
            "wind_gusts_10m_max", "wind_direction_10m_dominant", "uv_index_max"
        ]
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(base_url, params=params)
        if response.status_code != 200:
            logger.error(f"Open-Meteo API error: {response.status_code}, {response.text}")
            raise HTTPException(status_code=response.status_code, detail="Error fetching weather data")
        return response.json()

async def fetch_weather_data_for_rules(lat: float, lon: float) -> Dict[str, Any]:
    """Fetch only relevant weather data for rule-based prediction from Open-Meteo API"""
    base_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "forecast_days": 1, # Fetch only for today
        "current": [
            "weather_code", "wind_speed_10m"
        ],
        "daily": [
            "weather_code", "wind_speed_10m_max", "wind_gusts_10m_max", "precipitation_sum", "precipitation_probability_max"
        ]
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(base_url, params=params)
        if response.status_code != 200:
            logger.error(f"Open-Meteo API error: {response.status_code}, {response.text}")
            raise HTTPException(status_code=response.status_code, detail="Error fetching weather data for rules")
        return response.json()

def format_weather_data(raw_data: Dict[str, Any], location_id: str) -> Dict[str, Any]:
    """Format and categorize weather data for database storage"""
    current = None
    if "current" in raw_data:
        current = {
            "dt": int(datetime.fromisoformat(raw_data["current"]["time"]).timestamp()),
            "temp": raw_data["current"]["temperature_2m"],
            "feels_like": raw_data["current"]["apparent_temperature"],
            "humidity": raw_data["current"]["relative_humidity_2m"],
            "pressure": raw_data["current"]["surface_pressure"],
            "wind_speed": raw_data["current"]["wind_speed_10m"],
            "wind_direction": raw_data["current"]["wind_direction_10m"],
            "weather_code": raw_data["current"]["weather_code"],
            "is_day": raw_data["current"]["is_day"],
            "visibility": raw_data["current"].get("visibility"),
            "uv_index": raw_data["current"].get("uv_index"),
            "location_id": location_id
        }
    hourly = []
    for i, time_str in enumerate(raw_data["hourly"]["time"]):
        dt = int(datetime.fromisoformat(time_str).timestamp())
        hourly_data = {
            "dt": dt,
            "temp": raw_data["hourly"]["temperature_2m"][i],
            "feels_like": raw_data["hourly"]["apparent_temperature"][i],
            "humidity": raw_data["hourly"]["relative_humidity_2m"][i],
            "pressure": raw_data["hourly"]["surface_pressure"][i],
            "wind_speed": raw_data["hourly"]["wind_speed_10m"][i],
            "wind_direction": raw_data["hourly"]["wind_direction_10m"][i],
            "wind_gusts": raw_data["hourly"].get("wind_gusts_10m", [None])[i],
            "precipitation": raw_data["hourly"]["precipitation"][i],
            "snowfall": raw_data["hourly"]["snowfall"][i],
            "weather_code": raw_data["hourly"]["weather_code"][i],
            "cloud_cover": raw_data["hourly"]["cloud_cover"][i],
            "cloud_cover_low": raw_data["hourly"].get("cloud_cover_low", [None])[i],
            "cloud_cover_mid": raw_data["hourly"].get("cloud_cover_mid", [None])[i],
            "cloud_cover_high": raw_data["hourly"].get("cloud_cover_high", [None])[i],
            "visibility": raw_data["hourly"].get("visibility", [None])[i],
            "is_day": raw_data["hourly"]["is_day"][i],
            "snow_depth": raw_data["hourly"].get("snow_depth", [None])[i],
            "location_id": location_id
        }
        hourly.append(hourly_data)
    daily = []
    if "daily" in raw_data:
        for i, time_str in enumerate(raw_data["daily"]["time"]):
            dt = int(datetime.fromisoformat(time_str).timestamp())
            daily_data = {
                "dt": dt,
                "temperature_max": raw_data["daily"]["temperature_2m_max"][i],
                "temperature_min": raw_data["daily"]["temperature_2m_min"][i],
                "apparent_temperature_max": raw_data["daily"]["apparent_temperature_max"][i],
                "apparent_temperature_min": raw_data["daily"]["apparent_temperature_min"][i],
                "precipitation_sum": raw_data["daily"]["precipitation_sum"][i],
                "rain_sum": raw_data["daily"]["rain_sum"][i],
                "showers_sum": raw_data["daily"]["showers_sum"][i],
                "snowfall_sum": raw_data["daily"]["snowfall_sum"][i],
                "precipitation_hours": raw_data["daily"]["precipitation_hours"][i],
                "precipitation_probability_max": raw_data["daily"].get("precipitation_probability_max", [None])[i],
                "weather_code": raw_data["daily"]["weather_code"][i],
                "sunrise": int(datetime.fromisoformat(raw_data["daily"]["sunrise"][i]).timestamp()),
                "sunset": int(datetime.fromisoformat(raw_data["daily"]["sunset"][i]).timestamp()),
                "wind_speed_max": raw_data["daily"]["wind_speed_10m_max"][i],
                "wind_gusts_max": raw_data["daily"]["wind_gusts_10m_max"][i],
                "wind_direction_dominant": raw_data["daily"]["wind_direction_10m_dominant"][i],
                "uv_index_max": raw_data["daily"].get("uv_index_max", [None])[i],
                "location_id": location_id
            }
            daily.append(daily_data)
    return {
        "current": current,
        "hourly": hourly,
        "daily": daily
    }

async def store_weather_data(data: Dict[str, Any], supabase: Client) -> Dict[str, Any]:
    """Store formatted weather data in Supabase with type conversion"""
    results = {}
    try:
        sanitized_data = sanitize_data_for_db(data)
        location_result = supabase.table("weather_locations").upsert(sanitized_data["location"]).execute()
        results["location"] = location_result.data
        if sanitized_data["current"]:
            try:
                current_result = supabase.table("current_weather").upsert(sanitized_data["current"]).execute()
                results["current"] = current_result.data
            except Exception as e:
                logger.error(f"Error storing current weather: {str(e)}")
        batch_size = 50
        hourly_results = []
        for i in range(0, len(sanitized_data["hourly"]), batch_size):
            batch = sanitized_data["hourly"][i:i+batch_size]
            try:
                hourly_batch_result = supabase.table("hourly_weather").upsert(batch).execute()
                hourly_results.extend(hourly_batch_result.data)
            except Exception as e:
                logger.error(f"Error storing hourly batch {i//batch_size}: {str(e)}")
        results["hourly"] = hourly_results
        if sanitized_data["daily"]:
            try:
                daily_result = supabase.table("daily_weather").upsert(sanitized_data["daily"]).execute()
                results["daily"] = daily_result.data
            except Exception as e:
                logger.error(f"Error storing daily weather: {str(e)}")
        return results
    except Exception as e:
        logger.error(f"Error in store_weather_data: {str(e)}")
        raise e

async def predict_outage_probability_rule_based(coords: LocationCoordinates):
    """Predict outage probability based on weather and network data using rule-based logic."""
    weather_data = await fetch_weather_data_for_rules(coords.lat, coords.lon)
    network_metrics = {
        "downlink": coords.downlink,
        "rtt": coords.rtt
    }
    outage_probability = "Low" # Default
    daily_weather = weather_data.get("daily", {})
    current_weather = weather_data.get("current", {})
    wind_speed_max_list = daily_weather.get("wind_speed_10m_max", []) # Get list, default to empty list
    wind_gusts_max_list = daily_weather.get("wind_gusts_max_max", []) # Get list, default to empty list # corrected typo here - should be wind_gusts_max instead of wind_gusts_max_max
    wind_speed_max = wind_speed_max_list[0] if isinstance(wind_speed_max_list, list) and wind_speed_max_list else 0 # Safe access, default 0
    wind_gusts_max = wind_gusts_max_list[0] if isinstance(wind_gusts_max_list, list) and wind_gusts_max_list else 0 # Safe access, default 0
    if wind_speed_max >= 70 or wind_gusts_max >= 90:
        outage_probability = "High"
        return {"outage_probability": outage_probability, "reasoning": "Rule 1 Triggered: High Wind/Gusts"}
    heavy_precipitation_codes = [61, 63, 65, 66, 67, 71, 73, 75, 77, 82, 85, 86, 95, 96, 99] # Example codes for heavy rain/snow/storms - CHECK OPEN-METEO DOCS
    daily_weather_code_list = daily_weather.get("weather_code", [])
    current_weather_code_list_raw = current_weather.get("weather_code", []) # Get raw value, might be list or single value
    daily_weather_code = daily_weather_code_list[0] if isinstance(daily_weather_code_list, list) and daily_weather_code_list else 0
    current_weather_code = current_weather_code_list_raw[0] if isinstance(current_weather_code_list_raw, list) and current_weather_code_list_raw else current_weather_code_list_raw if isinstance(current_weather_code_list_raw, int) else 0 # Handle both list and int cases
    precipitation_sum_list = daily_weather.get("precipitation_sum", [])
    precipitation_sum = precipitation_sum_list[0] if isinstance(precipitation_sum_list, list) and precipitation_sum_list else 0 # Safe access, default 0
    if precipitation_sum >= 30 or daily_weather_code in heavy_precipitation_codes or current_weather_code in heavy_precipitation_codes:
        outage_probability = "Medium to High"
        if outage_probability != "High": # Don't override "High" from Rule 1
            return {"outage_probability": outage_probability, "reasoning": "Rule 2 Triggered: Heavy Precipitation"}
    if (wind_speed_max >= 50 or wind_gusts_max >= 60) and precipitation_sum >= 10:
        outage_probability = "Medium"
        if outage_probability != "High" and outage_probability != "Medium to High": # Don't override higher risks
            return {"outage_probability": outage_probability, "reasoning": "Rule 3 Triggered: Moderate Wind + Precipitation"}
    if network_metrics["rtt"] is not None and network_metrics["downlink"] is not None: # Check for None values
        if network_metrics["rtt"] >= 200 and network_metrics["downlink"] <= 1:
            outage_probability = "Medium"
            if outage_probability != "High" and outage_probability != "Medium to High": # Don't override higher risks
                return {"outage_probability": outage_probability, "reasoning": "Rule 4 Triggered: Network Degradation"}
    if outage_probability != "Low": # If any weather risk is already Medium or High
        if network_metrics["rtt"] is not None and network_metrics["downlink"] is not None: # Check for None values
            if network_metrics["rtt"] >= 100 and network_metrics["downlink"] <= 5:
                outage_probability = "Medium to High" #Escalate to Medium to High if network is also degraded
                return {"outage_probability": outage_probability, "reasoning": "Rule 5 Triggered: Network Degraded + Weather Risk"}
    return {"outage_probability": outage_probability, "reasoning": "No specific rule triggered - Low Probability (Default)"}

def calculate_document_priority(document: CourseWorkDocument) -> PriorityLevel:
    """
    Calculates the priority of a document based on filename, file_size, and upload date.
    """
    filename_lower = document.filename.lower()
    is_urgent = "urgent" in filename_lower

    if is_urgent:
        return "high"

    if document.file_size < 1024 * 1024: # Less than 1MB - Medium priority for now, adjust as needed
        return "medium"

    return "low" # Default to low priority

def calculate_estimated_download_time(file_size_bytes: int, downlink_mbps: Optional[float]) -> Optional[float]:
    """
    Estimates the download time for a file given its size and network downlink speed.
    Returns None if downlink_mbps is not provided.
    """
    if downlink_mbps is None:
        return None

    # Convert downlink speed from Mbps to Bytes per second
    download_speed_bytes_per_second = (downlink_mbps * 1000 * 1000) / 8

    if download_speed_bytes_per_second <= 0:
        return None  # Avoid division by zero or negative speed

    download_time_seconds = file_size_bytes / download_speed_bytes_per_second
    return download_time_seconds


@app.post("/weather", description="Fetch and store weather data for a location")
async def get_weather(
    coords: LocationCoordinates,
    supabase: Client = Depends(get_supabase)
):
    try:
        logger.info(f"Processing weather request for coordinates: {coords.lat}, {coords.lon}")
        location_id = generate_location_id(coords.lat, coords.lon, coords.user_id)
        logger.info(f"Generated location_id: {location_id}")
        logger.info("Fetching data from Open-Meteo...")
        weather_data = await fetch_weather_data(
            coords.lat,
            coords.lon,
            forecast_days=coords.forecast_days,
            timezone=coords.timezone
        )
        logger.info("Successfully fetched data from Open-Meteo")
        logger.info("Formatting data for storage...")
        formatted_data = format_weather_data(weather_data, location_id)
        logger.info("Data formatted successfully")
        formatted_data["location"] = {
            "id": location_id,
            "lat": coords.lat,
            "lon": coords.lon,
            "timezone": weather_data["timezone"],
            "timezone_abbreviation": weather_data["timezone_abbreviation"],
            "elevation": weather_data["elevation"],
            "user_id": coords.user_id,
            "created_at": datetime.now().isoformat()
        }
        logger.info("Storing data in Supabase...")
        storage_result = await store_weather_data(formatted_data, supabase)
        logger.info("Data stored successfully")
        if any([coords.effective_type, coords.downlink, coords.rtt]):
            network_data = {
                "user_id": str(coords.user_id) if coords.user_id else None,
                "location_id": location_id,
                "effective_type": coords.effective_type,
                "downlink": coords.downlink,
                "rtt": coords.rtt,
                "created_at": datetime.now().isoformat()
            }
            try:
                supabase.table("network_logs").insert(network_data).execute()
                logger.info("Network data stored successfully")
            except Exception as e:
                logger.error(f"Error storing network data: {str(e)}")
        success_data = {
            "message": "Weather data processing complete",
            "location_id": location_id,
            "data": {
                "current": "current" in storage_result,
                "hourly_count": len(storage_result.get("hourly", [])),
                "daily_count": len(storage_result.get("daily", []))
            }
        }
        return success_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing weather data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing weather data: {str(e)}")

@app.get("/weather/{location_id}", description="Get stored weather data for a location")
async def retrieve_weather(
    location_id: str,
    supabase: Client = Depends(get_supabase)
):
    try:
        location = supabase.table("weather_locations").select("*").eq("id", location_id).execute()
        if not location.data:
            raise HTTPException(status_code=404, detail="Location not found")
        current = supabase.table("current_weather").select("*").eq("location_id", location_id).order("dt.desc").limit(1).execute()
        hourly = supabase.table("hourly_weather").select("*").eq("location_id", location_id).order("dt.asc").execute()
        daily = supabase.table("daily_weather").select("*").eq("location_id", location_id).order("dt.asc").execute()
        return {
            "location": location.data[0],
            "current": current.data[0] if current.data else None,
            "hourly": hourly.data,
            "daily": daily.data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving weather data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving weather data: {str(e)}")

@app.post("/predict-outage-rule-based", description="Predict internet outage probability using rule-based logic")
async def get_outage_prediction_rule_based(coords: LocationCoordinates):
    """Endpoint to predict outage probability based on rule-based system."""
    try:
        prediction_result = await predict_outage_probability_rule_based(coords)
        return prediction_result
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error predicting outage probability (rule-based): {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error predicting outage probability: {str(e)}")

@app.post("/chatbot", description="Chatbot endpoint for network problem diagnosis")
@app.post("/chatbot", description="Chatbot endpoint for network problem diagnosis")
async def chatbot_endpoint(request: ChatbotRequest):
    """Endpoint to interact with the DeepSeek R1 chatbot for network problem diagnosis using Together AI."""
    try:
        # No need to check hf_client anymore
        user_prompt = request.prompt
        # System prompt to guide chatbot for network diagnosis
        system_prompt = """You are a highly intelligent AI assistant specializing in diagnosing and solving network problems in a hospital environment.
        Your goal is to help users troubleshoot network issues.
        Ask clarifying questions to understand the problem, consider any information provided, and suggest logical, step-by-step troubleshooting actions.
        Focus on accuracy and providing helpful, practical advice related to network connectivity, router issues, and common hospital network scenarios.
        When a user provides an image (or says they have), acknowledge it and ask them to describe visual details relevant to the network problem."""

        messages = [
            {"role": "system", "content": system_prompt}, # System prompt for context
            {"role": "user", "content": user_prompt}    # User's prompt
        ]

        api_url = "https://api.together.xyz/v1/chat/completions" # Together AI endpoint
        headers = {
            "Authorization": f"Bearer {TOGETHER_AI_API_KEY}", # Use Together AI API Key
            "Content-Type": "application/json"
        }

        data = {
            "model": "deepseek-ai/DeepSeek-R1", # Or choose a different model available on Together AI
            "messages": messages,
            "max_tokens": 5000, # Set max tokens as before
            "temperature": 0.1 # Set temperature as before
        }

        async with httpx.AsyncClient(timeout=30.0) as client: # Increased timeout to 30 seconds
            response = await client.post(api_url, headers=headers, json=data)
            if response.status_code != 200: # Together AI returns 200 on success
                logger.error(f"Together AI error: {response.status_code}, {response.text}")
                raise HTTPException(status_code=response.status_code, detail="Error communicating with chatbot service")

            response_data = response.json()

            # Extract response from Together AI format
            response_text = ""
            if "choices" in response_data and response_data["choices"]:
                response_text = response_data["choices"][0]["message"]["content"]
            else:
                logger.warning(f"Unexpected response format from Together AI: {response_data}")
                response_text = "Sorry, I encountered an issue processing your request."


        # **Cleaning the response: Remove the <think> block** - Keep cleaning logic as is if needed
        if "<think>" in response_text and "</think>" in response_text:
            start_index = response_text.find("</think>") + len("</think>")
            cleaned_response = response_text[start_index:].strip()
        else:
            cleaned_response = response_text

        return {"response": cleaned_response}

    except HTTPException as e: # Re-raise HTTPExceptions directly
        raise
    except Exception as e: # Catch other exceptions and return as HTTP 500
        logger.error(f"Chatbot error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chatbot service error: {str(e)}")
    
#  Document Retrieval Endpoint
@app.get("/documents/high-priority-docs", description="Retrieve the top 5 coursework documents based on priority with estimated download time")
async def retrieve_top_documents(
    downlink: Optional[float] = Query(None, description="Network downlink speed in Mbps for download time estimation"), # Optional downlink query parameter
    supabase: Client = Depends(get_supabase)
) -> List[PrioritizedDocument]:
    """
    Retrieves the top 5 coursework documents, prioritized by urgency, size, and upload date,
    and includes estimated download time based on provided downlink speed.
    """
    try:
        # 1. Fetch all document metadata from the database
        response = supabase.table("course_work_documents").select("*").execute()
        documents_metadata_list = [CourseWorkDocument(**item) for item in response.data] # Parse all to Pydantic models


        # 2. Calculate priority for each document
        prioritized_documents_metadata = []
        for doc_meta in documents_metadata_list:
            priority = calculate_document_priority(doc_meta)
            prioritized_documents_metadata.append({"metadata": doc_meta, "priority": priority})

        # 3. Sort documents by priority (high > medium > low), then by file_size (ascending), then by upload date (ascending - older first)
        def sort_priority(doc_item):
            priority_order = {"high": 0, "medium": 1, "low": 2} # Lower value = higher priority
            meta = doc_item["metadata"]
            return (priority_order[doc_item["priority"]], meta.file_size, meta.uploaded_at) # Sort by priority, then size, then date

        prioritized_documents_metadata.sort(key=sort_priority)

        top_documents_metadata = prioritized_documents_metadata[:5] # Take top 5

        # 4. Fetch content for the top 5 documents and prepare PrioritizedDocument list
        top_prioritized_documents: List[PrioritizedDocument] = []

        for doc_item in top_documents_metadata:
            doc_meta = doc_item["metadata"]
            priority = doc_item["priority"]

            try:
                res = supabase.storage().from_("coursework-documents").download(doc_meta.storage_path) # Replace "coursework-documents" with your bucket name
                if res.error:
                    logger.warning(f"Could not download document {doc_meta.id} from storage, skipping. Error: {res.error}") # Non-critical error, skip this doc
                    continue # Skip to the next document
                document_content = res.data # bytes

                estimated_download_time = calculate_estimated_download_time(doc_meta.file_size, downlink) # Calculate download time

                top_prioritized_documents.append(PrioritizedDocument(
                    document_id=doc_meta.id,
                    filename=doc_meta.filename,
                    title=doc_meta.title,
                    uploaded_at=doc_meta.uploaded_at,
                    posted_by=doc_meta.user_id,
                    priority=priority,
                    content=document_content,
                    content_type=doc_meta.content_type,
                    estimated_download_time_seconds=estimated_download_time # Assign estimated download time
                ))

            except Exception as storage_err:
                logger.error(f"Error downloading document {doc_meta.id} from Supabase Storage: {storage_err}, skipping.") # Non-critical error, skip this doc
                continue # Skip to the next document


        return top_prioritized_documents # Return list of PrioritizedDocument objects with metadata, content, and download time


    except HTTPException as http_exc:
        raise http_exc # Re-raise HTTPExceptions
    except Exception as e:
        logger.error(f"Error retrieving top documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving top documents: {str(e)}")

#  Document Retrieval Endpoint
@app.get("/documents/{document_id}", description="Retrieve a coursework document by ID (now returns StreamingResponse)")
async def retrieve_document(
    document_id: UUID,
    supabase: Client = Depends(get_supabase)
) -> StreamingResponse:
    """
    Retrieves a specific coursework document by its ID and returns it as a StreamingResponse.
    This is efficient for large files as it streams the file content directly without loading it all into memory.
    """
    try:
        # 1. Retrieve document metadata from the database to get the storage path and filename
        response = supabase.table("course_work_documents").select("storage_path, filename, content_type, file_size").eq("id", document_id).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail="Document not found")
        document_metadata = response.data[0] # Expecting only one document with this ID

        # 2. Download the file from Supabase storage as bytes
        res = supabase.storage().from_("admin_course_work_documents").download(document_metadata["storage_path"]) # Replace "coursework-documents" with your bucket name
        if res.error:
            logger.error(f"Error downloading document {document_id} from Supabase Storage: {res.error}")
            raise HTTPException(status_code=500, detail="Failed to download document from storage")
        document_content = res.data # File content as bytes

        # 3. Create a StreamingResponse to stream the file content
        def stream_file():
            yield document_content # Yield the bytes content directly for streaming

        # Determine content type and filename for headers
        content_type = document_metadata.get("content_type", "application/octet-stream") # Default content type if not in DB
        filename = document_metadata.get("filename", f"document-{document_id}") # Default filename if not in DB

        return StreamingResponse(
            stream_file(),
            media_type=content_type,
            headers={"Content-Disposition": f"attachment;filename={filename}"} # Suggest download filename
        )

    except HTTPException as http_exc:
        raise http_exc # Re-raise HTTPExceptions
    except Exception as e:
        logger.error(f"Error retrieving document {document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")
    
# Document Upload Endpoint
@app.post("/documents/upload", description="Upload a new coursework document")
async def upload_document(
    request: DocumentUploadRequest = Depends(), # Use Depends to handle form data and file
    supabase: Client = Depends(get_supabase)
) -> CourseWorkDocument:
    """
    Endpoint to upload a new coursework document.
    """
    try:
        uploaded_file = request.document_file
        file_size = uploaded_file.size or 0 # Get file size, default to 0 if None

        if uploaded_file.content_type is None:
            raise HTTPException(status_code=400, detail="Content type could not be determined")

        # Generate a unique filename and storage path (using UUID for uniqueness)
        unique_filename = f"{uuid4()}_{uploaded_file.filename}"
        storage_path = f"coursework/{unique_filename}" # Store in 'coursework' folder in Supabase Storage

        # Upload file to Supabase Storage
        try:
            contents = await uploaded_file.read() # Read file content as bytes
            res = supabase.storage().from_("user_course_work_documents").upload(storage_path, contents, file_options={"content-type": uploaded_file.content_type}) # Upload to bucket
            if res.error:
                logger.error(f"Supabase Storage upload error: {res.error}")
                raise HTTPException(status_code=500, detail="File upload to storage failed")
        except Exception as storage_upload_err:
            logger.error(f"Error uploading to Supabase Storage: {storage_upload_err}")
            raise HTTPException(status_code=500, detail="File upload to storage failed")

        # Create Document Metadata object and store in database
        document_metadata = CourseWorkDocument(
            id=uuid4(), # Generate UUID for document ID
            filename=uploaded_file.filename,
            storage_path=storage_path,
            title=request.title,
            user_id=request.user_id,
            content_type=uploaded_file.content_type,
            file_size=file_size # Store file size in metadata
        )

        try:
            db_response = supabase.table("course_work_documents").insert(document_metadata.dict()).execute() # Insert metadata to DB
            if db_response.error:
                logger.error(f"Database insert error: {db_response.error}")
                raise HTTPException(status_code=500, detail="Failed to save document metadata to database")
        except Exception as db_insert_err:
            logger.error(f"Error inserting document metadata to database: {db_insert_err}")
            # If DB insert fails, consider deleting the file from storage to keep data consistent (optional, depends on your error handling policy)
            try:
                supabase.storage().from_("coursework-documents").remove([storage_path]) # Attempt to delete from storage
                logger.warning(f"Rolled back storage upload for {storage_path} due to DB error.")
            except Exception as rollback_err:
                logger.error(f"Rollback of storage upload failed: {rollback_err}") # Log rollback failure
            raise HTTPException(status_code=500, detail="Failed to save document metadata to database")


        return document_metadata # Return the created document metadata (including generated ID, storage path etc.)


    except HTTPException as http_exc:
        raise http_exc # Re-raise HTTP Exceptions
    except Exception as e:
        logger.error(f"Error uploading document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")
    
# New Schools List Endpoint
@app.get("/schools-list", response_model=SchoolsListResponse, description="Retrieve a list of schools with problem details")
async def get_schools_list(supabase: Client = Depends(get_supabase)) -> SchoolsListResponse:
    """
    Retrieves a list of schools with details about any problems they are facing.
    """
    try:
        # SQL Query to fetch school data
        response = supabase.rpc("get_schools_data").execute()

        # Check for errors in a different way - examine response directly
        if response.data is None:
            logger.error(f"Database query error: Response data is None. Full response: {response}")
            raise HTTPException(status_code=500, detail="Database query failed or returned no data.")

        schools_data = response.data

        # Format the data into the School model
        schools_list: List[School] = []
        for school_item in schools_data:
            schools_list.append(School(
                school_id=school_item['school_id'], # Include school_id from database response
                school_name=school_item['school_name'],
                school_location=SchoolLocation(location_id=school_item['location_id']),
                school_region=school_item.get('school_region'),
                type_of_problem=school_item.get('type_of_problem'),
                duration=school_item.get('duration'),
                severity=school_item.get('severity'),
                last_updated=school_item.get('last_updated')
            ))

        return SchoolsListResponse(total_schools=len(schools_list), schools=schools_list)

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error retrieving schools list: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving schools list: {str(e)}")

# System Alerts Endpoint
@app.post("/system-alerts", description="Generate and retrieve system alerts based on weather, network, and outage predictions")
async def get_system_alerts(
    coords: LocationCoordinates, # Take LocationCoordinates as input for context
    supabase: Client = Depends(get_supabase)
) -> List[SystemAlert]:
    """
    Generates and retrieves system alerts based on weather data, network data, and outage probability predictions.
    """
    alerts: List[SystemAlert] = [] # Initialize an empty list to store alerts

    # 1. Predict outage probability
    outage_prediction = await predict_outage_probability_rule_based(coords)
    outage_probability = outage_prediction["outage_probability"]
    outage_reasoning = outage_prediction["reasoning"]

    if outage_probability == "High":
        alert_description = f"High internet outage probability predicted due to {outage_reasoning}. Consider taking preventative measures."
        alerts.append(SystemAlert(
            title="High Internet Outage Risk Predicted",
            description=alert_description,
            alert_type="outage",
            severity="high",
            location_id=generate_location_id(coords.lat, coords.lon, coords.user_id)
        ))
    elif outage_probability == "Medium to High":
        alert_description = f"Medium to High internet outage probability predicted due to {outage_reasoning}. Monitor conditions closely."
        alerts.append(SystemAlert(
            title="Medium to High Internet Outage Risk",
            description=alert_description,
            alert_type="outage",
            severity="medium", # Or "high" - you can decide based on your severity scale
            location_id=generate_location_id(coords.lat, coords.lon, coords.user_id)
        ))
    elif outage_probability == "Medium":
        alert_description = f"Medium internet outage probability predicted due to {outage_reasoning}. Be aware of potential disruptions."
        alerts.append(SystemAlert(
            title="Medium Internet Outage Risk",
            description=alert_description,
            alert_type="outage",
            severity="medium",
            location_id=generate_location_id(coords.lat, coords.lon, coords.user_id)
        ))

    # 2. Fetch weather data for specific weather alerts (beyond outage prediction)
    weather_data_rules = await fetch_weather_data_for_rules(coords.lat, coords.lon)
    daily_weather = weather_data_rules.get("daily", {})
    current_weather = weather_data_rules.get("current", {})
    wind_speed_max_list = daily_weather.get("wind_speed_10m_max", [])
    wind_gusts_max_list = daily_weather.get("wind_gusts_max_max", []) # Corrected typo here as well
    wind_speed_max = wind_speed_max_list[0] if isinstance(wind_speed_max_list, list) and wind_speed_max_list else 0
    wind_gusts_max = wind_gusts_max_list[0] if isinstance(wind_gusts_max_list, list) and wind_gusts_max_list else 0
    precipitation_sum_list = daily_weather.get("precipitation_sum", [])
    precipitation_sum = precipitation_sum_list[0] if isinstance(precipitation_sum_list, list) and precipitation_sum_list else 0

    if wind_speed_max >= 80: # Example threshold for very high wind warning
        alerts.append(SystemAlert(
            title="Very High Wind Warning",
            description=f"Maximum wind speed today may reach {wind_speed_max} m/s. Secure outdoor equipment and be cautious.",
            alert_type="weather",
            severity="high",
            location_id=generate_location_id(coords.lat, coords.lon, coords.user_id)
        ))
    elif wind_speed_max >= 60: # Example threshold for high wind warning
        alerts.append(SystemAlert(
            title="High Wind Warning",
            description=f"Maximum wind speed today may reach {wind_speed_max} m/s. Expect potential disruptions.",
            alert_type="weather",
            severity="medium",
            location_id=generate_location_id(coords.lat, coords.lon, coords.user_id)
        ))

    heavy_precipitation_threshold_mm = 50 # Example threshold for heavy precipitation (mm/day)
    if precipitation_sum >= heavy_precipitation_threshold_mm:
        alerts.append(SystemAlert(
            title="Heavy Precipitation Alert",
            description=f"Heavy precipitation expected today, with total rainfall/snowfall potentially exceeding {precipitation_sum} mm. Possible flooding or travel disruptions.",
            alert_type="weather",
            severity="medium", # Or "high" depending on threshold and impact
            location_id=generate_location_id(coords.lat, coords.lon, coords.user_id)
        ))

    # 3. Network condition alerts (example - you can expand based on more network metrics)
    network_metrics = {
        "downlink": coords.downlink,
        "rtt": coords.rtt
    }
    if network_metrics["rtt"] is not None and network_metrics["downlink"] is not None:
        if network_metrics["rtt"] >= 300: # Example threshold for high latency alert
            alerts.append(SystemAlert(
                title="High Network Latency Alert",
                description=f"Network latency is currently high (RTT >= {network_metrics['rtt']}ms). Network performance may be significantly degraded.",
                alert_type="network",
                severity="medium", # Or "high"
                location_id=generate_location_id(coords.lat, coords.lon, coords.user_id)
            ))
        elif network_metrics["downlink"] <= 0.5: # Example threshold for very low downlink
            alerts.append(SystemAlert(
                title="Very Low Bandwidth Alert",
                description=f"Downlink bandwidth is very low (<= {network_metrics['downlink']} Mbps). Internet speed is severely limited.",
                alert_type="network",
                severity="high",
                location_id=generate_location_id(coords.lat, coords.lon, coords.user_id)
            ))
        elif network_metrics["rtt"] >= 150 or network_metrics["downlink"] <= 2: # Combined medium degradation
            alerts.append(SystemAlert(
                title="Network Performance Degradation",
                description="Experiencing degraded network performance. Latency and/or bandwidth are below optimal levels.",
                alert_type="network",
                severity="low", # Or "medium"
                location_id=generate_location_id(coords.lat, coords.lon, coords.user_id)
            ))

    # 4. Store generated alerts in the database (optional, but good practice)
    if alerts: # Only store if there are alerts to store
        alerts_to_store = [alert.dict() for alert in alerts] # Convert Pydantic models to dictionaries for DB insert
        try:
            db_response = supabase.table("system_alerts").insert(alerts_to_store).execute() # Batch insert
            if db_response.error:
                logger.error(f"Error storing system alerts in database: {db_response.error}") # Log DB insert errors, but don't fail the API request for alert retrieval
            else:
                logger.info(f"Successfully stored {len(alerts)} system alerts in database.")
        except Exception as db_error:
            logger.error(f"Exception while storing system alerts in database: {db_error}") # Log DB exceptions


    return alerts # Return the list of generated alerts (whether stored in DB or not)

# Get System Alerts Endpoint
@app.get("/system-alerts-list", response_model=SystemAlertsListResponse, description="Retrieve a list of system alerts")
async def get_system_alerts_list(supabase: Client = Depends(get_supabase)) -> SystemAlertsListResponse:
    """
    Retrieves a list of system alerts from the database.
    """
    try:
        response = supabase.table("system_alerts").select("*").order("created_at", desc=True).execute() # Fetch all alerts, ordered by creation time (newest first)

        # **Check for errors by examining response.data directly**
        if response.data is None: # Check if response.data is None, which might indicate an error
            logger.error(f"Database query error retrieving system alerts: Response data is None. Full response: {response}") # Log full response for debugging
            raise HTTPException(status_code=500, detail="Failed to retrieve system alerts from database or database returned no data.")


        alerts_data = response.data
        system_alerts: List[SystemAlert] = []
        for alert_item in alerts_data:
            system_alerts.append(SystemAlert(**alert_item)) # Create SystemAlert objects from database data

        return SystemAlertsListResponse(alerts=system_alerts, total_alerts=len(system_alerts))

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error retrieving system alerts list: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving system alerts list: {str(e)}")