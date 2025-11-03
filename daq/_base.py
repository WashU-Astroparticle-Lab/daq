# -*- coding: utf-8 -*-

from datetime import datetime
import os
from typing import Any, Dict, Optional

import h5py
import numpy as np

from presto.hardware import AdcMode, DacMode
from presto.utils import get_sourcecode

from .db import get_next_number, insert_measurement, generate_filename
from .utils import get_data_folder


class Base:
    """
    Base class for measurements
    """

    DAC_CURRENT: int = 32_000
    """μA -- Change to increase or decrease DAC analog output range"""
    ADC_ATTENUATION: float = 0.0  # dB
    """dB -- Change to increase or decrease ADC analog input range"""
    DC_PARAMS: dict = {
        "adc_mode": AdcMode.Mixed,
        "dac_mode": DacMode.Mixed,
    }
    """Parameters to configure the data converters (ADC and DAC)"""

    def _save(self, script_path: str, save_filename: Optional[str] = None) -> str:
        script_path = os.path.realpath(script_path)
        
        # Determine measurement type from script filename
        script_basename = os.path.basename(script_path)
        script_filename = os.path.splitext(script_basename)[0]
        measurement_type = script_filename  # e.g., "sweep" or "timestream"
        
        # Get device, filter, and notes from instance if available
        device = getattr(self, "device", None)
        filter_name = getattr(self, "filter", None)
        notes = getattr(self, "notes", None)
        
        # Validate required fields for database
        if device is None:
            raise ValueError(
                "device parameter is required for database logging"
            )
        
        # Get next number from database
        number = get_next_number()
        
        # Generate filename if not provided
        if save_filename is None:
            data_folder = get_data_folder()
            os.makedirs(data_folder, exist_ok=True)
            save_basename = generate_filename(
                number, device, measurement_type
            )
            save_path = os.path.join(data_folder, save_basename)
        else:
            save_path = os.path.realpath(save_filename)
        
        # Save h5 file
        source_code = get_sourcecode(script_path)
        with h5py.File(save_path, "w") as h5f:
            dt = h5py.string_dtype(encoding="utf-8")
            ds = h5f.create_dataset("source_code", (len(source_code),), dt)
            for ii, line in enumerate(source_code):
                ds[ii] = line
            
            for attribute in self.__dict__:
                try:
                    if attribute.startswith("_"):
                        continue
                    if attribute in ["jpa_params", "clear"]:
                        h5f.attrs[attribute] = str(self.__dict__[attribute])
                    elif np.isscalar(self.__dict__[attribute]):
                        h5f.attrs[attribute] = self.__dict__[attribute]
                    else:
                        h5f.create_dataset(
                            attribute, data=self.__dict__[attribute]
                        )
                except Exception as err:
                    print(f"WARN: unable to save {attribute}: {err}")
        
        print(f"Data saved to: {save_path}")
        
        # Build MongoDB document
        document = self._build_document(
            number=number,
            measurement_type=measurement_type,
            file_path=save_path,
            device=device,
            filter_name=filter_name,
            notes=notes,
        )
        
        # Insert into MongoDB
        try:
            doc_id = insert_measurement(document)
            print(f"Document inserted to MongoDB with ID: {doc_id}")
        except Exception as e:
            print(f"WARN: Failed to insert to MongoDB: {e}")
        
        return save_path
    
    def _build_document(
        self,
        number: str,
        measurement_type: str,
        file_path: str,
        device: str,
        filter_name: Optional[str],
        notes: Optional[str],
    ) -> Dict[str, Any]:
        """Build MongoDB document from measurement data."""
        document: Dict[str, Any] = {
            "utc_time": datetime.utcnow().isoformat(),
            "number": number,
            "type": measurement_type,
            "file": file_path,
            "device": device,
            "filter": filter_name,
            "notes": notes,
        }
        
        # Add all measurement parameters from __init__
        for attribute in self.__dict__:
            if attribute.startswith("_"):
                continue
            
            # Skip metadata fields already added
            if attribute in ["device", "filter", "notes"]:
                continue
            
            # Skip data arrays (large datasets)
            if attribute in [
                "freq_arr", "resp_arr", "pixel_i", "pixel_q",
                "lsb", "usb", "freqs_usb", "freqs_lsb"
            ]:
                continue
            
            value = self.__dict__[attribute]
            
            # Convert numpy types to Python native types
            if isinstance(value, np.ndarray):
                document[attribute] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                document[attribute] = value.item()
            elif np.isscalar(value) or value is None:
                document[attribute] = value
            else:
                # For other types, try to convert to string
                try:
                    document[attribute] = str(value)
                except Exception:
                    pass
        
        return document