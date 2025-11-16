# -*- coding: utf-8 -*-
"""
MongoDB Atlas database integration for DAQ system.
"""
from datetime import datetime
from typing import Any, Dict, Optional, Union

import pandas as pd
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

# MongoDB Atlas connection URI
MONGODB_URI = (
    "mongodb+srv://lanqing_yuan:WdXq8PrrHj78zjYy@freec.f96deaj."
    "mongodb.net/?appName=FreeC"
)
DB_NAME = "WashU_Astroparticle_Detector"
COLLECTION_NAME = "measurement"


def _get_collection():
    """Get the MongoDB collection for measurements."""
    client = MongoClient(MONGODB_URI, server_api=ServerApi('1'))
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    return collection


def get_next_number() -> str:
    """
    Get the next available measurement number from the database.
    
    Returns:
        str: 8-digit zero-padded number string (e.g., "00000001")
    """
    collection = _get_collection()
    
    # Find the document with the highest number
    result = collection.find_one(
        sort=[("number", -1)],
        projection={"number": 1}
    )
    
    if result and "number" in result:
        # Extract the integer from the zero-padded string
        last_number = int(result["number"])
        next_number = last_number + 1
    else:
        # No documents yet, start from 1
        next_number = 1
    
    # Return as 8-digit zero-padded string
    return str(next_number).zfill(8)


def generate_filename(
    number: str, device: str, measurement_type: str
) -> str:
    """
    Generate a filename for the measurement data file.
    
    Args:
        number: 8-digit zero-padded measurement number
        device: Device name
        measurement_type: Type of measurement ("sweep" or "timestream")
    
    Returns:
        str: Filename in format: <number>-<device>-<type>.h5
    """
    return f"{number}-{device}-{measurement_type}.h5"


def insert_measurement(document: Dict[str, Any]) -> str:
    """
    Insert a measurement document into the MongoDB collection.
    
    Args:
        document: Dictionary containing measurement metadata
    
    Returns:
        str: The inserted document's ID as a string
    """
    collection = _get_collection()
    
    # Ensure UTC time is present
    if "utc_time" not in document:
        document["utc_time"] = datetime.utcnow().isoformat()
    
    result = collection.insert_one(document)
    return str(result.inserted_id)


def select_runs(
    device: Optional[str] = None,
    filter_name: Optional[str] = None,
    notes: Optional[str] = None,
    measurement_type: Optional[str] = None,
    start_time: Optional[Union[datetime, str]] = None,
    end_time: Optional[Union[datetime, str]] = None,
    string_match: str = "exact",
    **kwargs
) -> pd.DataFrame:
    """
    Query the database for measurement runs matching specified criteria.
    
    Args:
        device: Device name to filter by
        filter_name: Filter name to filter by
        notes: Notes text to filter by
        measurement_type: Measurement type (e.g., "sweep", "timestream")
        start_time: Start time for time range (datetime or ISO string)
        end_time: End time for time range (datetime or ISO string)
        string_match: Matching mode for string fields ("exact" or "regex")
        **kwargs: Additional field filters (e.g., output_port=1, amp=0.1)
    
    Returns:
        pd.DataFrame: DataFrame containing all matching documents with all
            fields. Includes "number" as identifier and "file" as location.
    """
    collection = _get_collection()
    query: Dict[str, Any] = {}
    
    # Helper function to add string field to query
    def add_string_filter(
        field: str, value: Optional[str], query_dict: Dict[str, Any]
    ):
        if value is not None:
            if string_match == "regex":
                query_dict[field] = {"$regex": value, "$options": "i"}
            else:  # exact match
                query_dict[field] = value
    
    # Add string field filters
    add_string_filter("device", device, query)
    add_string_filter("filter", filter_name, query)
    add_string_filter("notes", notes, query)
    add_string_filter("type", measurement_type, query)
    
    # Handle time range filtering
    if start_time is not None or end_time is not None:
        time_query: Dict[str, Any] = {}
        
        if start_time is not None:
            if isinstance(start_time, str):
                start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            else:
                start_dt = start_time
            time_query["$gte"] = start_dt.isoformat()
        
        if end_time is not None:
            if isinstance(end_time, str):
                end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
            else:
                end_dt = end_time
            time_query["$lte"] = end_dt.isoformat()
        
        query["utc_time"] = time_query
    
    # Handle additional kwargs filters
    for key, value in kwargs.items():
        if value is not None:
            # For string values, apply matching mode
            if isinstance(value, str):
                add_string_filter(key, value, query)
            else:
                query[key] = value
    
    # Execute query
    cursor = collection.find(query)
    results = list(cursor)
    
    # Handle empty results
    if not results:
        return pd.DataFrame()
    
    # Convert ObjectId to string for all documents
    for doc in results:
        if "_id" in doc:
            doc["_id"] = str(doc["_id"])
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    return df


def list_devices() -> pd.DataFrame:
    """
    List all unique device names recorded in the database with their counts.
    
    Returns:
        pd.DataFrame: DataFrame with columns ['device', 'count'] containing
            all unique device names and the number of measurements for each.
            Sorted by count in descending order.
    """
    collection = _get_collection()
    
    # Use aggregation pipeline to count measurements per device
    pipeline = [
        {"$match": {"device": {"$exists": True, "$ne": None}}},
        {"$group": {"_id": "$device", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$project": {"_id": 0, "device": "$_id", "count": 1}}
    ]
    
    results = list(collection.aggregate(pipeline))
    
    # Handle empty results
    if not results:
        return pd.DataFrame(columns=["device", "count"])
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    return df

