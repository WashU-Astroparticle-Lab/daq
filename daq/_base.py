# -*- coding: utf-8 -*-

from datetime import datetime
import os
from typing import Optional

import h5py
import numpy as np

from presto.hardware import AdcMode, DacMode
from presto.utils import get_sourcecode


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
        script_path = os.path.realpath(script_path)  # full path of current script

        if save_filename is None:
            current_dir, script_basename = os.path.split(script_path)
            script_filename = os.path.splitext(script_basename)[0]  # name of current script
            timestamp = datetime.now().isoformat(timespec="seconds")  # current date and time
            save_basename = f"{script_filename:s}_{timestamp:s}.h5"  # name of save file
            save_path = os.path.join(current_dir, "data", save_basename)  # full path of save file
        else:
            save_path = os.path.realpath(save_filename)

        source_code = get_sourcecode(
            script_path
        )  # save also the sourcecode of the script for future reference
        with h5py.File(save_path, "w") as h5f:
            dt = h5py.string_dtype(encoding="utf-8")
            ds = h5f.create_dataset("source_code", (len(source_code),), dt)
            for ii, line in enumerate(source_code):
                ds[ii] = line

            for attribute in self.__dict__:
                try:
                    if attribute.startswith("_"):
                        # don't save private attributes
                        continue
                    if attribute in ["jpa_params", "clear"]:
                        h5f.attrs[attribute] = str(self.__dict__[attribute])
                    elif np.isscalar(self.__dict__[attribute]):
                        h5f.attrs[attribute] = self.__dict__[attribute]
                    else:
                        h5f.create_dataset(attribute, data=self.__dict__[attribute])
                except Exception as err:
                    print(f"WARN: unable to save {attribute}: {err}")
        print(f"Data saved to: {save_path}")
        return save_path