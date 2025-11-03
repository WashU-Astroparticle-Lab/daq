# -*- coding: utf-8 -*-
"""
MongoDB Atlas database integration for DAQ system.
"""
from datetime import datetime
from typing import Any, Dict, Optional
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

