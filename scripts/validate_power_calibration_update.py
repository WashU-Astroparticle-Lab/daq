"""Reproduce numerical diagnostics for the unfiltered calibration; no hardware access."""
import csv
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    spec=importlib.util.spec_from_file_location('builder',ROOT/'scripts/build_power_calibration.py')
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    src=builder.DEFAULT_INPUT_DIR
    with np.load(ROOT/'daq/calibrations/power_calibration.npz') as z:
        a={k:z[k].copy() for k in z.files}
    f,p=a['frequency_ghz'],a['power_dbm_at_amp_1'];ok=a['interpolation_allowed']
    report=dict(calibration_id=str(a['calibration_id']),reference_plane=str(a['reference_plane']),
        measured_frequency_nodes=len(f),selected_observations=len(builder.read_measurements(src)),
        amplitude_support={str(k):int(v) for k,v in zip(*np.unique(a['amplitude_support'],return_counts=True))},
        excluded_open_intervals_ghz=[[float(f[i]),float(f[i+1])] for i in range(len(ok)) if not ok[i]],
        hardware_validation={},leave_one_frequency_out={},
        meaning='Fit consistency and limited hardware spot checks, not a complete RF accuracy certification. Low-band single-anchor nodes use a model at other amplitudes and warn.')
    for name,col,scale,pcol in [('low_amplitude_checks.csv','frequency_mhz',1000,'median_peak_power_dbm'),
                                ('high_validation.csv','frequency_ghz',1,'measured_power_dbm')]:
        with (src/name).open(newline='') as stream:rows=list(csv.DictReader(stream))
        residual=[]
        for row in rows:
            idx=np.flatnonzero(np.isclose(f,float(row[col])/scale,rtol=0,atol=1e-12))
            assert len(idx)==1
            residual.append(float(row[pcol])-(p[idx[0]]+20*np.log10(float(row['amplitude']))))
        report['hardware_validation'][name]=dict(points=len(rows),max_abs_error_db=float(max(abs(np.array(residual)))),
            mean_signed_error_db=float(np.mean(residual)),residual_db=residual)
    for band,lo,hi in [('low',.1,2.1),('gap',2.1,2.5),('high',2.5,9.)]:
        errors=[]
        for i in range(1,len(f)-1):
            if lo<=f[i-1] and f[i+1]<=hi and ok[i-1] and ok[i]:
                predicted=np.interp(f[i],f[[i-1,i+1]],p[[i-1,i+1]])
                errors.append(abs(float(predicted-p[i])))
        report['leave_one_frequency_out'][band]=dict(points=len(errors),
            absolute_error_db_median_p95_max=np.quantile(errors,[.5,.95,1]).tolist() if errors else [],
            note='Withholding measured nodes tests a wider frequency interval than normal interpolation; not an accuracy bound.')
    path=ROOT/'docs/power_calibration_validation.json'
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
