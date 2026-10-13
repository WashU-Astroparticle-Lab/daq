# Historical October 2 source tables

These preserved tables contain 5,050 low-band and 13,100 high-band settings from the
October 2 acquisition, reviewed on October 11. All observations and review flags remain.

**Filter configuration clarified October 12:** low_power.csv (0.1–2.1 GHz) was acquired
through TWO cascaded Mini-Circuits VLFX-2500+ low-pass filters. It is excluded from the
current unfiltered calibration. high_power.csv (2.5–9 GHz) was acquired without filters
and is copied unchanged into ../2026-10-12/ for the current builder.

The policy in this folder and the original v1 package are historical. The current builder
reads ../2026-10-12/, which has direct unfiltered low-band anchors and gap measurements.
Do not combine this filtered low-band table with the unfiltered high band as one unfiltered
calibration, or describe their difference as pure filter loss.

Original reviewed source location: exports/Presentation_Review_2026-10-11/data/ in the
laboratory collection. Raw spectra are preserved in that collection, not duplicated here.
Git newline conversion is disabled for the historical source tables and policy.
