# -*- coding: utf-8 -*-
"""
Modified from https://github.com/intermod-pro/presto-measure/blob/master/sweep_power.py.
2D sweep of drive power and frequency in Lockin mode.
"""

from typing import List, Optional, Union

import h5py
import numpy as np
import numpy.typing as npt

from presto import lockin
from presto.utils import ProgressBar, asarray, recommended_dac_config

from .._base import Base
from ..calibrations import amp_to_power_dbm_hz
from ..config import get_presto_address, get_presto_port

FloatAny = Union[float, List[float], npt.NDArray[np.floating]]


class SweepPower(Base):
    def __init__(
        self,
        freq_center: float,
        freq_span: float,
        df: float,
        num_averages: int,
        amp_arr: FloatAny,
        output_port: int,
        input_port: int,
        dither: bool = True,
        num_skip: int = 0,
        device: Optional[str] = None,
        filter: Optional[str] = None,
        notes: Optional[str] = None,
        auto_fit: bool = True,
        attenuation_db: Optional[float] = None,
    ) -> None:
        self.freq_center = freq_center
        self.freq_span = freq_span
        self.df = df  # modified after tuning
        self.num_averages = num_averages
        self.amp_arr = asarray(amp_arr, np.float64)
        self.output_port = output_port
        self.input_port = input_port
        self.dither = dither
        self.num_skip = num_skip
        self.device = device
        self.filter = filter
        self.notes = notes
        self.auto_fit = auto_fit
        self.attenuation_db = attenuation_db

        self.freq_arr = None  # replaced by run
        self.resp_arr = None  # replaced by run
        self.fit_results = None  # list of per-amp fit dicts, replaced by run

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

        # Use the recommended DAC config for the center frequency
        dac_mode, dac_fsample = recommended_dac_config(self.freq_center)
        dc_params = dict(self.DC_PARAMS)
        dc_params["dac_mode"] = dac_mode
        dc_params["dac_fsample"] = dac_fsample

        with lockin.Lockin(
            address=presto_address,
            port=presto_port,
            ext_ref_clk=ext_ref_clk,
            **dc_params,
        ) as lck:
            lck.hardware.set_adc_attenuation(self.input_port, self.ADC_ATTENUATION)
            lck.hardware.set_dac_current(self.output_port, self.DAC_CURRENT)
            lck.hardware.set_inv_sinc(self.output_port, 0)

            nr_amps = len(self.amp_arr)

            # tune frequencies
            _, self.df = lck.tune(0.0, self.df)
            lck.set_df(self.df)

            f_start = self.freq_center - self.freq_span / 2
            f_stop = self.freq_center + self.freq_span / 2
            n_start = int(round(f_start / self.df))
            n_stop = int(round(f_stop / self.df))
            n_arr = np.arange(n_start, n_stop + 1)
            nr_freq = len(n_arr)
            self.freq_arr = self.df * n_arr
            self.resp_arr = np.zeros((nr_amps, nr_freq), np.complex128)

            lck.hardware.configure_mixer(
                freq=self.freq_arr[0], in_ports=self.input_port, out_ports=self.output_port
            )

            og = lck.add_output_group(self.output_port, 1)
            og.set_frequencies(0.0)
            og.set_amplitudes(self.amp_arr[0])
            og.set_phases(0.0, 0.0)

            lck.set_dither(self.dither, self.output_port)
            ig = lck.add_input_group(self.input_port, 1)
            ig.set_frequencies(0.0)

            lck.apply_settings()

            pb = ProgressBar(nr_amps * nr_freq)
            pb.start()
            for jj, amp in enumerate(self.amp_arr):
                og.set_amplitudes(amp)
                lck.apply_settings()

                for ii, freq in enumerate(self.freq_arr):
                    lck.hardware.configure_mixer(
                        freq=freq, in_ports=self.input_port, out_ports=self.output_port
                    )
                    lck.apply_settings()

                    _d = lck.get_pixels(self.num_skip + self.num_averages, quiet=True)
                    data_i = _d[self.input_port][1][:, 0]
                    data_q = _d[self.input_port][2][:, 0]
                    data = data_i.real + 1j * data_q.real  # using zero IF

                    self.resp_arr[jj, ii] = np.mean(data[-self.num_averages :])

                    pb.increment()

            pb.done()

            # Mute outputs at the end of the sweep
            og.set_amplitudes(0.0)
            lck.apply_settings()

        # Perform automatic per-amp fit analysis before saving (if enabled)
        if self.auto_fit:
            self._perform_fit()

        return self.save(save_filename=save_filename)

    def save(self, save_filename: Optional[str] = None) -> str:
        return super()._save(__file__, save_filename=save_filename)

    def _perform_fit(self) -> bool:
        """Fit the resonator silently at each drive amplitude and store the results.

        Populates :attr:`fit_results` with one entry per row of :attr:`amp_arr`
        (i.e. per drive power): the ``resonator_tools`` ``fitresults`` dict on
        success, or ``None`` if that amplitude's fit failed. The list is kept on
        the object only -- it is not written to HDF5 or MongoDB.

        :return: ``True`` if at least one amplitude was fit successfully.
        :rtype: bool
        """
        if self.freq_arr is None or self.resp_arr is None:
            return False

        try:
            from resonator_tools import circuit
        except ImportError:
            # resonator_tools not available, skip fitting
            return False

        fit_results = []
        any_success = False
        for jj in range(len(self.amp_arr)):
            resp = self.resp_arr[jj]
            try:
                # Center the fit on the amplitude minimum for this drive power
                f_ctr = self.freq_arr[np.argmin(np.abs(resp))]
                # Fit at most half of the sweep span
                f_min = max(f_ctr - self.freq_span / 4, self.freq_arr.min())
                f_max = min(f_ctr + self.freq_span / 4, self.freq_arr.max())

                port = circuit.notch_port(self.freq_arr, resp)
                port.autofit(fcrop=(f_min, f_max))

                fit_results.append(port.fitresults)
                any_success = True
            except Exception as e:
                # Fit failed for this amp; keep going so one bad row does not
                # prevent the measurement from being saved.
                print(f"WARN: Fit analysis failed at amp index {jj}: {e}")
                fit_results.append(None)

        self.fit_results = fit_results
        return any_success

    @classmethod
    def load(cls, load_filename: str) -> "SweepPower":
        with h5py.File(load_filename, "r") as h5f:
            freq_center = float(h5f.attrs["freq_center"])  # type: ignore
            freq_span = float(h5f.attrs["freq_span"])  # type: ignore
            df = float(h5f.attrs["df"])  # type: ignore
            num_averages = int(h5f.attrs["num_averages"])  # type: ignore
            output_port = int(h5f.attrs["output_port"])  # type: ignore
            input_port = int(h5f.attrs["input_port"])  # type: ignore
            dither = bool(h5f.attrs["dither"])  # type: ignore
            num_skip = int(h5f.attrs["num_skip"])  # type: ignore
            # Load optional parameters if they exist
            device = h5f.attrs.get("device", None)
            filter_param = h5f.attrs.get("filter", None)
            notes = h5f.attrs.get("notes", None)
            auto_fit = bool(h5f.attrs["auto_fit"]) if "auto_fit" in h5f.attrs else True
            attenuation_db = (
                float(h5f.attrs["attenuation_db"]) if "attenuation_db" in h5f.attrs else None
            )

            amp_arr: npt.NDArray[np.float64] = h5f["amp_arr"][()]  # type: ignore
            freq_arr: npt.NDArray[np.float64] = h5f["freq_arr"][()]  # type: ignore
            resp_arr: npt.NDArray[np.complex128] = h5f["resp_arr"][()]  # type: ignore

        self = cls(
            freq_center=freq_center,
            freq_span=freq_span,
            df=df,
            num_averages=num_averages,
            amp_arr=amp_arr,
            output_port=output_port,
            input_port=input_port,
            dither=dither,
            num_skip=num_skip,
            device=device,
            filter=filter_param,
            notes=notes,
            auto_fit=auto_fit,
            attenuation_db=attenuation_db,
        )
        self.freq_arr = freq_arr
        self.resp_arr = resp_arr

        return self

    def _drive_power(self) -> "tuple[npt.NDArray[np.float64], str]":
        """Return the drive-power axis and its label for plotting.

        Converts :attr:`amp_arr` to calibrated output power in dBm at
        :attr:`freq_center`. When :attr:`attenuation_db` is set, the power is
        referenced to the device input -- ``drive power - attenuation_db`` --
        and the label reflects that.

        :return: ``(power_dbm, label)`` where *power_dbm* has the same length
            as :attr:`amp_arr`.
        :rtype: tuple[numpy.ndarray, str]
        """
        power_dbm = np.asarray(
            amp_to_power_dbm_hz(self.freq_center, self.amp_arr), dtype=np.float64
        )
        if self.attenuation_db is not None:
            power_dbm = power_dbm - self.attenuation_db
            label = "Drive power at device [dBm]"
        else:
            label = "Drive power [dBm]"
        return power_dbm, label

    def _fit_vs_power(
        self,
    ) -> "tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]":
        """Extract per-amp fitted ``fr`` and ``Qi`` (diagonally corrected).

        Reads the ``resonator_tools`` ``fitresults`` dicts stored in
        :attr:`fit_results`, one per drive amplitude. Amplitudes with a failed
        (``None``) fit yield ``NaN`` so they are simply skipped when plotted.

        :return: ``(fr, fr_err, qi, qi_err)``, each an array of length
            ``len(amp_arr)``. ``fr``/``fr_err`` are in Hz.
        :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        """
        nr_amps = len(self.amp_arr)
        fr = np.full(nr_amps, np.nan)
        fr_err = np.full(nr_amps, np.nan)
        qi = np.full(nr_amps, np.nan)
        qi_err = np.full(nr_amps, np.nan)

        if self.fit_results is not None:
            for jj, res in enumerate(self.fit_results):
                if not res:
                    continue
                fr[jj] = res.get("fr", np.nan)
                fr_err[jj] = res.get("fr_err", np.nan)
                qi[jj] = res.get("Qi_dia_corr", np.nan)
                qi_err[jj] = res.get("Qi_dia_corr_err", np.nan)

        return fr, fr_err, qi, qi_err

    def analyze(self, norm: bool = True, portrait: bool = True):
        """Plot the 2D response map and the fitted resonator parameters.

        The left/top panel shows the response-amplitude map over frequency and
        drive power, overlaid with a scatter of the fitted resonant frequency
        at each drive power. The remaining two panels show the best-fit
        resonant frequency ``fr`` and internal quality factor ``Qi``
        (diagonally corrected) as a function of drive power, each with
        fit-error bars.

        Fits frequently fail at some -- or all -- drive powers; those points
        are simply omitted from every panel (and drawn without an error bar
        where the fit succeeded but reported no uncertainty), so a partial or
        empty fit never raises.

        When :attr:`attenuation_db` is set, the drive-power axis is referenced
        to the device input (drive power minus the attenuation).

        :param norm: Normalise the response map by the per-row drive amplitude.
        :type norm: bool
        :param portrait: Stack the panels vertically instead of side-by-side.
        :type portrait: bool
        :return: The created figure.
        :rtype: matplotlib.figure.Figure
        """
        if self.freq_arr is None:
            raise RuntimeError
        if self.resp_arr is None:
            raise RuntimeError

        import matplotlib.pyplot as plt

        # Ensure per-amp fits are available (e.g. after load(), which does not
        # persist fit_results); fit lazily so analyze() works standalone.
        if self.fit_results is None:
            self._perform_fit()

        if norm:
            resp_scaled = self.resp_arr / self.amp_arr[:, None]
        else:
            resp_scaled = self.resp_arr

        resp_dB = 20.0 * np.log10(np.abs(resp_scaled))
        power_dbm, power_label = self._drive_power()
        fr, fr_err, qi, qi_err = self._fit_vs_power()

        # Fits often fail for some (or all) drive powers -- those rows come back
        # as NaN. Drop them from the markers and draw no error bar where the fit
        # error is missing (NaN -> 0-length bar) so nothing raises or misleads.
        finite_fr = np.isfinite(fr)
        finite_qi = np.isfinite(qi)
        fr_err_safe = np.where(np.isfinite(fr_err), fr_err, 0.0)
        qi_err_safe = np.where(np.isfinite(qi_err), qi_err, 0.0)

        # choose limits for colorbar
        cutoff = 1.0  # %
        lowlim = np.percentile(resp_dB, cutoff)
        highlim = np.percentile(resp_dB, 100.0 - cutoff)

        # extent
        x_min = 1e-9 * self.freq_arr[0]
        x_max = 1e-9 * self.freq_arr[-1]
        dx = 1e-9 * (self.freq_arr[1] - self.freq_arr[0])
        y_min = power_dbm[0]
        y_max = power_dbm[-1]
        dy = power_dbm[1] - power_dbm[0] if len(power_dbm) > 1 else 1.0

        if portrait:
            fig1 = plt.figure(tight_layout=True, figsize=(6.4, 9.6))
            ax1 = fig1.add_subplot(2, 1, 1)
        else:
            fig1 = plt.figure(tight_layout=True, figsize=(12.8, 4.8))
            ax1 = fig1.add_subplot(1, 2, 1)
        im = ax1.imshow(
            resp_dB,
            origin="lower",
            aspect="auto",
            interpolation="none",
            extent=(x_min - dx / 2, x_max + dx / 2, y_min - dy / 2, y_max + dy / 2),
            vmin=lowlim,  # type: ignore
            vmax=highlim,  # type: ignore
        )
        # Mark the fitted resonant frequency on the response map (skip failed
        # fits). Constrain the axes to the map extent so stray points cannot
        # rescale it.
        if np.any(finite_fr):
            ax1.scatter(
                1e-9 * fr[finite_fr],
                power_dbm[finite_fr],
                s=16,
                c="r",
                marker="x",
                linewidths=1.0,
                label=r"fit $f_r$",
            )
            ax1.set_xlim(x_min - dx / 2, x_max + dx / 2)
            ax1.set_ylim(y_min - dy / 2, y_max + dy / 2)
            ax1.legend(loc="upper right", fontsize="small")
        ax1.set_xlabel("Frequency [GHz]")
        ax1.set_ylabel(power_label)
        cb = fig1.colorbar(im)
        if portrait:
            cb.set_label("Response amplitude [dB]")
        else:
            ax1.set_title("Response amplitude [dB]")

        if portrait:
            ax2 = fig1.add_subplot(4, 1, 3)
            ax3 = fig1.add_subplot(4, 1, 4, sharex=ax2)
        else:
            ax2 = fig1.add_subplot(2, 2, 2)
            ax3 = fig1.add_subplot(2, 2, 4, sharex=ax2)
            ax2.yaxis.set_label_position("right")
            ax2.yaxis.tick_right()
            ax3.yaxis.set_label_position("right")
            ax3.yaxis.tick_right()

        # Fitted resonant frequency vs. drive power (in GHz); failed fits (NaN)
        # are dropped so the line only connects successfully-fit points.
        if np.any(finite_fr):
            ax2.errorbar(
                power_dbm[finite_fr],
                1e-9 * fr[finite_fr],
                yerr=1e-9 * fr_err_safe[finite_fr],
                fmt=".-",
                capsize=3,
            )
        ax2.set_ylabel(r"Fit $f_r$ [GHz]")
        ax2.xaxis.set_tick_params(labelbottom=False)

        # Fitted internal quality factor (diag. corrected) vs. drive power
        if np.any(finite_qi):
            ax3.errorbar(
                power_dbm[finite_qi],
                qi[finite_qi],
                yerr=qi_err_safe[finite_qi],
                fmt=".-",
                capsize=3,
            )
        ax3.set_ylabel(r"Fit $Q_i$ (diag. corr.)")
        ax3.set_xlabel(power_label)

        # Use plt.show() (not fig1.show()) so that inside a notebook loop each
        # figure renders immediately and is closed afterwards -- otherwise the
        # inline backend queues every figure until the cell ends and shows them
        # all at once. Matches the other measurement classes.
        plt.show()
        return fig1
