# -*- coding: utf-8 -*-
"""
TimeStream measurement class for acquiring time-domain data with multiple frequencies.
"""
from typing import List, Optional, Union

import h5py
import numpy as np
import numpy.typing as npt

from presto import lockin
from presto.utils import untwist_downconversion
from daq.utils import get_presto_address, get_presto_port
from daq._base import Base

FloatAny = Union[float, List[float], npt.NDArray[np.floating]]


class TimeStream(Base):
    def __init__(
        self,
        lo_freq: float,
        if_freqs: FloatAny,
        if_freqs_in: FloatAny,
        df: float,
        pixel_counts: int,
        amp: FloatAny,
        phases_i: FloatAny,
        phases_q: FloatAny,
        output_port: int,
        input_port: int,
        dither: bool = True,
    ) -> None:
        self.lo_freq = lo_freq
        self.if_freqs = np.asarray(if_freqs, dtype=np.float64)
        self.if_freqs_in = np.asarray(if_freqs_in, dtype=np.float64)
        self.df = df  # modified after tuning
        self.pixel_counts = pixel_counts
        self.amp = np.asarray(amp, dtype=np.float64)
        self.phases_i = np.asarray(phases_i, dtype=np.float64)
        self.phases_q = np.asarray(phases_q, dtype=np.float64)
        self.output_port = output_port
        self.input_port = input_port
        self.dither = dither

        # Data arrays - set by run method
        self.freq_arr = None
        self.pixel_i = None
        self.pixel_q = None
        self.lsb = None
        self.usb = None
        self.freqs_usb = None
        self.freqs_lsb = None

    def run(
        self,
        presto_address: Optional[str] = None,
        presto_port: Optional[int] = None,
        ext_ref_clk: bool = False,
        save_filename: Optional[str] = None,
    ) -> str:
        if presto_address is None:
            presto_address = get_presto_address()
        if presto_port is None:
            presto_port = get_presto_port()
        with lockin.Lockin(
            address=presto_address,
            port=presto_port,
            ext_ref_clk=ext_ref_clk,
            **self.DC_PARAMS,
        ) as lck:
            lck.hardware.set_adc_attenuation(self.input_port, self.ADC_ATTENUATION)
            lck.hardware.set_dac_current(self.output_port, self.DAC_CURRENT)
            lck.hardware.set_inv_sinc(self.output_port, 0)
            lck.hardware.configure_mixer(
                self.lo_freq, 
                out_ports=self.output_port, 
                in_ports=self.input_port
            )
            lck.set_dither(self.dither, self.output_port)

            _, self.df = lck.tune(0.0, self.df)
            lck.set_df(self.df)

            # Configure output group
            og = lck.add_output_group(self.output_port, len(self.if_freqs))
            og.set_frequencies(self.if_freqs)
            og.set_amplitudes(self.amp)
            og.set_phases(self.phases_i, self.phases_q)

            # Configure input group
            ig = lck.add_input_group(self.input_port, len(self.if_freqs_in))
            ig.set_frequencies(self.if_freqs_in)

            lck.apply_settings()

            # Acquire data
            pixel_dict = lck.get_pixels(self.pixel_counts)
            
            self.freq_arr, self.pixel_i, self.pixel_q = pixel_dict[self.input_port]
            self.lsb, self.usb = untwist_downconversion(self.pixel_i, self.pixel_q)

            # Calculate frequency arrays
            self.freqs_usb = self.lo_freq + self.if_freqs
            self.freqs_lsb = self.lo_freq - self.if_freqs

            # Mute outputs at the end
            og.set_amplitudes(0.0)
            lck.apply_settings()

        return self.save(save_filename=save_filename)

    def save(self, save_filename: Optional[str] = None) -> str:
        return super()._save(__file__, save_filename=save_filename)

    @classmethod
    def load(cls, load_filename: str) -> "TimeStream":
        with h5py.File(load_filename, "r") as h5f:
            lo_freq = float(h5f.attrs["lo_freq"])  # type: ignore
            df = float(h5f.attrs["df"])  # type: ignore
            pixel_counts = int(h5f.attrs["pixel_counts"])  # type: ignore
            output_port = int(h5f.attrs["output_port"])  # type: ignore
            input_port = int(h5f.attrs["input_port"])  # type: ignore
            dither = bool(h5f.attrs["dither"])  # type: ignore

            if_freqs: npt.NDArray[np.float64] = h5f["if_freqs"][()]  # type: ignore
            if_freqs_in: npt.NDArray[np.float64] = h5f["if_freqs_in"][()]  # type: ignore
            amp: npt.NDArray[np.float64] = h5f["amp"][()]  # type: ignore
            phases_i: npt.NDArray[np.float64] = h5f["phases_i"][()]  # type: ignore
            phases_q: npt.NDArray[np.float64] = h5f["phases_q"][()]  # type: ignore

            # Load data arrays if they exist
            freq_arr = h5f["freq_arr"][()] if "freq_arr" in h5f else None  # type: ignore
            pixel_i = h5f["pixel_i"][()] if "pixel_i" in h5f else None  # type: ignore
            pixel_q = h5f["pixel_q"][()] if "pixel_q" in h5f else None  # type: ignore
            lsb = h5f["lsb"][()] if "lsb" in h5f else None  # type: ignore
            usb = h5f["usb"][()] if "usb" in h5f else None  # type: ignore
            freqs_usb = h5f["freqs_usb"][()] if "freqs_usb" in h5f else None  # type: ignore
            freqs_lsb = h5f["freqs_lsb"][()] if "freqs_lsb" in h5f else None  # type: ignore

        self = cls(
            lo_freq=lo_freq,
            if_freqs=if_freqs,
            if_freqs_in=if_freqs_in,
            df=df,
            pixel_counts=pixel_counts,
            amp=amp,
            phases_i=phases_i,
            phases_q=phases_q,
            output_port=output_port,
            input_port=input_port,
            dither=dither,
        )

        # Restore data arrays
        self.freq_arr = freq_arr
        self.pixel_i = pixel_i
        self.pixel_q = pixel_q
        self.lsb = lsb
        self.usb = usb
        self.freqs_usb = freqs_usb
        self.freqs_lsb = freqs_lsb

        return self

    def analyze(
        self, 
        sideband: str = "usb", 
        num_samples: Optional[int] = None, 
        title: Optional[str] = None,
        show_iq: bool = True
    ):
        """
        Plot the timestream data.
        
        Parameters:
        -----------
        sideband : str, optional
            Which sideband to plot: "usb" (upper) or "lsb" (lower). Default is "usb".
        num_samples : int, optional
            Number of samples to plot. If None, plots all samples.
        title : str, optional
            Title for the plot.
        show_iq : bool, optional
            If True, show I and Q streams instead of phase and power. Default is False.
        """
        if self.usb is None or self.lsb is None:
            raise RuntimeError("No data available. Run the measurement first.")

        import matplotlib.pyplot as plt

        # Select data based on sideband
        if sideband.lower() == "usb":
            data = self.usb
            freqs = self.freqs_usb
            sideband_label = "USB"
        elif sideband.lower() == "lsb":
            data = self.lsb
            freqs = self.freqs_lsb
            sideband_label = "LSB"
        else:
            raise ValueError("sideband must be 'usb' or 'lsb'")

        # Limit number of samples if specified
        if num_samples is not None:
            data = data[:num_samples]

        # Create time axis
        time_axis = np.arange(data.shape[0]) / self.df * 1e6  # time in μs

        # Create figure with subplots for each frequency
        n_freqs = data.shape[1]
        fig, axes = plt.subplots(n_freqs, 2, figsize=(12, 2 * n_freqs), 
                                tight_layout=True, sharex=True)
        
        # Handle single frequency case
        if n_freqs == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_freqs):
            if show_iq:
                # I stream plot
                i_stream = np.real(data[:, i])
                axes[i, 0].plot(time_axis, i_stream)
                axes[i, 0].set_ylabel(f"I [a.u.]\n{freqs[i]/1e9:.3f} GHz")
                axes[i, 0].grid(True, alpha=0.3)

                # Q stream plot
                q_stream = np.imag(data[:, i])
                axes[i, 1].plot(time_axis, q_stream)
                axes[i, 1].set_ylabel(f"Q [a.u.]\n{freqs[i]/1e9:.3f} GHz")
                axes[i, 1].grid(True, alpha=0.3)
            else:
                # Amplitude plot
                amplitudes = np.abs(data[:, i])
                power_db = 20.0 * np.log10(amplitudes)
                axes[i, 0].plot(time_axis, power_db)
                axes[i, 0].set_ylabel(f"Power [dBFS]\n{freqs[i]/1e9:.3f} GHz")
                axes[i, 0].grid(True, alpha=0.3)

                # Phase plot
                phases = np.angle(data[:, i])
                axes[i, 1].plot(time_axis, phases)
                axes[i, 1].set_ylabel(f"Phase [rad]\n{freqs[i]/1e9:.3f} GHz")
                axes[i, 1].grid(True, alpha=0.3)

        # Set x-labels for bottom plots
        axes[-1, 0].set_xlabel("Time [μs]")
        axes[-1, 1].set_xlabel("Time [μs]")

        # Set title
        plot_type = "I/Q Streams" if show_iq else "Power/Phase"
        if title is not None:
            fig.suptitle(f"{title} - {sideband_label} ({plot_type})")
        else:
            fig.suptitle(f"TimeStream - {sideband_label} ({plot_type})")

        plt.show()

        return fig
