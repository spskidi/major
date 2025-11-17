
![Starting_soon_Screen_13](https://github.com/user-attachments/assets/d648675e-4f85-4215-abe4-46311e54b09a)

# Hackathon Core API Documentation

This document provides API documentation for interacting with the Lonelist Team Core AI Integrated APIS

**Base URL:** `https://core-api-xurt.onrender.com`

## Endpoints

### 1. Fetch and Store Weather Data for a Location (`/weather`)

*   **Endpoint:** `/weather`
*   **Method:** `POST`
*   **Description:** This endpoint fetches weather data from the Open-Meteo API for a given location and stores it in a Supabase database. It also optionally stores network performance metrics if provided.

#### Request Body (`application/json`)

```json
{
  "lat": 40.7128,        
  "lon": -74.0060,      
  "user_id": "optional-uuid-string", // optional
  "forecast_days": 7,   
  "timezone": "auto",   
  "effective_type": "4g",
  "downlink": 10.5,      
  "rtt": 50              
}
```

#### Response Body (`application/json`)

```json

{
  "message": "Weather data fetched and stored successfully",
  "location_id": "40.7128_-74.0060",
  "stored_data_details": {
    "latitude": 40.7128,
    "longitude": -74.0060,
    "forecast_days": 7
  }
}

```

### 2. Retrieve Stored Weather Data for a Location (`/weather/{location_id}`)
- **Endpoint:** `/weather/{location_id}`
- **Method:** `GET`
- **Description:** Retrieves weather data stored in the database for a specific location ID.

#### Path Parameters
- `location_id` (string): Required. The unique identifier of the location for which to retrieve weather data. This`location_id` is returned in the response of the `/weather` endpoint.

#### Request Body (`application/json`)


```json
{
  "location_id": "40.7128_-74.0060",
  "weather_data": {
    "current": {
      "dt": 1709474400,
      "temp": 10.2,
      "feels_like": 9.5,
      "humidity": 67,
      "pressure": 1017,
      "wind_speed": 11.2,
      "wind_direction": 240,
      "weather_code": 3,
      "is_day": 1,
      "visibility": 10,
      "uv_index": 3,
      "location_id": "40.7128_-74.0060"
    },
    "hourly": [
      {
        "dt": 1709474400,
        "temp": 10.2,
        "feels_like": 9.5,
        "humidity": 67,
        "pressure": 1017,
        "wind_speed": 11.2,
        "wind_direction": 240,
        "wind_gusts": 15.8,
        "precipitation": 0.0,
        "snowfall": 0.0,
        "weather_code": 3,
        "cloud_cover": 75,
        "cloud_cover_low": 75,
        "cloud_cover_mid": 0,
        "cloud_cover_high": 0,
        "visibility": 10,
        "is_day": 1,
        "snow_depth": 0,
        "location_id": "40.7128_-74.0060"
      },
      {
        "dt": 1709478000,
        "temp": 9.8,
        "feels_like": 9.1,
        "humidity": 69,
        "pressure": 1017,
        "wind_speed": 10.8,
        "wind_direction": 241,
        "wind_gusts": 15.1,
        "precipitation": 0.0,
        "snowfall": 0.0,
        "weather_code": 3,
        "cloud_cover": 74,
        "cloud_cover_low": 74,
        "cloud_cover_mid": 0,
        "cloud_cover_high": 0,
        "visibility": 10,
        "is_day": 1,
        "snow_depth": 0,
        "location_id": "40.7128_-74.0060"
      },
      // ... (rest of hourly data) ...
    ],
    "daily": [
      {
        "dt": 1709443200,
        "temperature_max": 12.5,
        "temperature_min": 7.8,
        "apparent_temperature_max": 11.8,
        "apparent_temperature_min": 6.9,
        "precipitation_sum": 0.0,
        "rain_sum": 0.0,
        "showers_sum": 0.0,
        "snowfall_sum": 0.0,
        "precipitation_hours": 0.0,
        "precipitation_probability_max": 2,
        "weather_code": 1,
        "sunrise": 1709468182,
        "sunset": 1709508231,
        "wind_speed_max": 17.3,
        "wind_gusts_max": 24.5,
        "wind_direction_dominant": 255,
        "uv_index_max": 4,
        "location_id": "40.7128_-74.0060"
      },
      // ... (rest of daily data) ...
    ]
  }
}
```

### 3. Predict Internet Outage Probability (Rule-Based) (`/predict-outage-rule-based`)

- **Endpoint:** `/predict-outage-rule-based`
- **Method:** `POST`
- **Description:** Predicts the probability of an internet outage at a given location based on weather conditions and network performance metrics using a rule-based system.

#### Request Body (`application/json`)

```json
{
  "lat": 40.7128,       
  "lon": -74.0060,       
  "effective_type": "4g",
  "downlink": 5.2,      
  "rtt": 150             
}
```
#### Response Body (`application/json`)

```json

{
    "outage_probability": "Low",
    "reasoning": "No specific rule triggered - Low Probability (Default)"
}
```
### 4. Network Problem Diagnosis Chatbot (`/chatbot`)
- **Endpoint:** `/chatbot`
- **Method:** `POST`
- **Description:** This endpoint provides a chatbot interface powered by `DeepSeek R1` to assist with network problem diagnosis. It takes a user's text prompt and returns a chatbot response aimed at troubleshooting network issues.
#### Request Body (`application/json`)
```json
{
    "prompt": "We have a connection problem in our hospital, the network is very slow."
}
```
#### Response Body (`application/json`)
```json
{
    "response": "Thanks for reaching out! I understand you are experiencing network issues..."
}
```
### 5. Retrieve Coursework Document (`/documents/{document_id}`)

- **Endpoint:** `/documents/{document_id}`
- **Method:** `GET`
- **Description:** Retrieves a coursework document file from Supabase Storage based on its unique document ID. The document is returned as a file download.

#### Path Parameters

- `document_id` (UUID): **Required.** The unique identifier (UUID) of the coursework document to retrieve.

#### Response

- **Response Type:** File Download (e.g., application/pdf, application/vnd.openxmlformats-officedocument.wordprocessingml.document, text/plain, etc.)

- **Headers:**
- `Content-Type`:  Indicates the MIME type of the document (e.g., `application/pdf`).
- `Content-Disposition`:  Set to `attachment;filename="{document_filename}"` to suggest a filename for download.

- **Status Codes:**
- `200 OK`: Document retrieved successfully. The response body is the document file content.
- `404 Not Found`: Document with the given `document_id` was not found.
- `500 Internal Server Error`:  An error occurred during document retrieval (e.g., database error, storage error).

### 6. Retrieve Top 5 Prioritized Coursework Documents (`/documents/high-priority-docs`)

- **Endpoint:** `/documents/high-priority-docs`
- **Method:** `GET`
- **Description:** Retrieves the top 5 coursework documents from Supabase Storage, prioritized based on urgency (keyword "urgent" in filename), file size (smaller files prioritized), and upload date (older files prioritized).  Returns a list of documents with their metadata, priority level, and downloadable file content.

#### Query Parameters

- `downlink` (float, optional): Network downlink speed in Mbps. If provided, the response will include an estimated download time for each document based on this speed. If not provided, `estimated_download_time_seconds` will be `null` for each document.

#### Response Body (`application/json`)

```json
[
  {
    "document_id": "uuid-of-document-1",
    "filename": "urgent_assignment_older.pdf",
    "title": "Urgent Older Assignment",
    "uploaded_at": "2024-01-15T10:00:00Z",
    "posted_by": "uuid-of-user-1",
    "priority": "high",
    "content_type": "application/pdf",
    "content": "Base64 encoded file content..." 
  },
  {
    "document_id": "uuid-of-document-2",
    "filename": "quiz_smaller_size.docx",
    "title": "Smaller Size Quiz",
    "uploaded_at": "2024-02-20T14:30:00Z",
    "posted_by": "uuid-of-user-2",
    "priority": "medium",
    "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "content": "Base64 encoded file content..."
  },
 { "... (up to 5 document entries) ..." },
]
```

### 7. Retrieve List of Schools with Problem Details (`/schools-list`)

-   **Endpoint:** `/schools-list`
-   **Method:** `GET`
-   **Description:** Retrieves a list of schools with details about any current network/infrastructure problems they are facing.

#### Response Body (`application/json`)

```json
{
  "total_schools": 5,
  "schools": [
    {
      "school_id": "uuid-of-school-1",
      "school_name": "Springfield Elementary",
      "school_location": {
        "location_id": "34.0522_-118.2437"
      },
      "school_region": "Central",
      "type_of_problem": "Network Outage",
      "duration": "1 day",
      "severity": "Medium",
      "last_updated": "2024-03-02T18:03:01.294937+00:00"
    },
    {
      "school_id": "uuid-of-school-2",
      "school_name": "Northwood High",
      "school_location": {
        "location_id": "34.0689_-118.1554"
      },
      "school_region": "North",
      "type_of_problem": "Power Fluctuation",
      "duration": "4 hours",
      "severity": "Low",
      "last_updated": "2024-03-02T19:03:01.294937+00:00"
    },
    {
      "...": "..."
    }
  ]
}
```

### 8. Upload Coursework Document (`/documents/upload`)
- **Endpoint:**` /documents/upload`
- **Method:** `POST`
- **Description:** Uploads a new coursework document to Supabase Storage and saves its metadata to the database.
- **Request Body:** (`multipart/form-data`)
#### Form Data Fields:
- **title (text):** Required. Title of the coursework document.
- **user_id** (text, optional): User ID who is uploading the document (optional).
- **document_file (file):** Required. The file to upload.

#### Response Body (`application/json`)
```json

{
  "id": "uuid-of-uploaded-document",
  "filename": "uploaded_document_name.pdf",
  "storage_path": "coursework/uuid_uploaded_document_name.pdf",
  "title": "Document Title Provided",
  "user_id": "optional-uuid-string",
  "content_type": "application/pdf",
  "file_size": 12345,
  "uploaded_at": "2024-03-02T19:30:00.123456+00:00"
}
```

### 9. Generate System Alerts (`/system-alerts`)
- **Endpoint:** `/system-alerts`
- **Method:** `POST`
- **Description:** Generates system alerts based on weather data, network performance, and predicted internet outage probability for a given location.

#### Request Body (`application/json`)
```json
{
  "lat": 40.7128,
  "lon": -74.0060,
  "downlink": 8.5,
  "rtt": 100
}

```
#### Response Body (`application/json`)
```json

[
  {
    "id": "uuid-of-alert-1",
    "title": "High Internet Outage Risk Predicted",
    "description": "High internet outage probability predicted due to weather conditions. Consider taking preventative measures.",
    "alert_type": "outage",
    "severity": "high",
    "location_id": "34.0522_-118.2437",
    "created_at": "2024-03-02T20:00:00.123456+00:00"
  },
  {
    "id": "uuid-of-alert-2",
    "title": "High Wind Warning",
    "description": "Maximum wind speed today may reach 70 m/s. Expect potential disruptions.",
    "alert_type": "weather",
    "severity": "medium",
    "location_id": "34.0522_-118.2437",
    "created_at": "2024-03-02T20:00:00.123456+00:00"
  },
  {
    "...": "..."
  }
]

```

### 10. Retrieve List of System Alerts (`/system-alerts-list`)
- **Endpoint:** `/system-alerts-list`
- **Method:**` GET`
- **Description:** Retrieves a list of system alerts from the database, ordered by creation time (newest first).

#### Response Body (`application/json`)

```json

{
  "alerts": [
    {
      "id": "uuid-of-alert-1",
      "title": "Network priority changed",
      "description": "Over the next 24 hours, the network will only prioritize essential content. Critical systems will remain operational.",
      "alert_type": "network",
      "severity": "medium",
      "location_id": null,
      "created_at": "2024-03-02T20:15:00.123456+00:00"
    },
    {
      "id": "uuid-of-alert-2",
      "title": "Alternative systems ready",
      "description": "Offline mode is set up and ready to use. Log in from the main application by selecting \"Offline Mode\".",
      "alert_type": "system",
      "severity": "low",
      "location_id": null,
      "created_at": "2024-03-02T20:10:00.123456+00:00"
    },
    {
      "...": "..."
    }
  ],
  "total_alerts": 4
}
```
