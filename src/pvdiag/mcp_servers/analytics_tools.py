"""Statistical and physical evidence; no fault labels or scenario truth are read."""

from __future__ import annotations

import pandas as pd
import pvlib

from pvdiag.mcp_servers.models import (
    AnomaliesResult,
    Anomaly,
    CompareRequest,
    DetectRequest,
    Evidence,
    ExpectedRequest,
    ExpectedResult,
    ExpectedSummary,
    Pattern,
    PatternsResult,
    PeerRequest,
    PeersResult,
    PeerSummary,
    PeriodsResult,
    PeriodSummary,
    Point,
)
from pvdiag.mcp_servers.store import ObservationStore, finite, segments


class AnalyticsTools:
    def __init__(self, store: ObservationStore):
        self.store = store

    def _signals(self, request: PeerRequest):
        self.store.check(request)
        if not set(request.device_ids) <= self.store.inverters:
            raise ValueError("peer comparison only accepts observed inverter AC channels")
        self.store.sensor(request.poa_device_id, "poa_wm2")

        def signals(start, end):
            power = pd.DataFrame(
                {d: self.store.series(d, "ac_power_kw", start, end) for d in request.device_ids}
            )
            daylight = self.store.series(request.poa_device_id, "poa_wm2", start, end) > 200
            raw = pd.DataFrame(index=power.index)
            for device in request.device_ids:
                peers = power.drop(columns=device)
                median = peers.median(axis=1).where(peers.notna().sum(axis=1) >= 2)
                raw[device] = power[device].div(median.where(median > 0)).where(daylight)
            return power, daylight, raw

        power, daylight, raw = signals(request.start, request.end)
        _, _, reference = signals(request.reference_start, request.reference_end)
        scales = reference.median()
        scales = scales.where((scales > 0) & (reference.notna().sum() >= 4))
        ratio = raw.div(scales, axis="columns")
        return power, daylight, ratio, reference, scales

    def peer_compare(self, request: PeerRequest) -> Evidence[PeersResult]:
        _, _, ratio, reference, scales = self._signals(request)
        devices, remaining, truncated = [], 64, False
        for device in request.device_ids:
            values = ratio[device].dropna()
            spans = []
            for mask in (ratio[device] < 0.9, ratio[device] > 1.15):
                found = segments(mask)
                spans.append(found[:remaining])
                truncated |= len(found) > remaining
                remaining -= len(spans[-1])
            devices.append(
                PeerSummary(
                    device_id=device,
                    reference_bins=int(reference[device].notna().sum()),
                    reference_ratio=finite(scales[device]),
                    comparable_daylight_bins=len(values),
                    median_ratio=finite(values.median()),
                    low_ratio_fraction=float((values < 0.9).mean()) if len(values) else None,
                    low_spans=spans[0],
                    high_spans=spans[1],
                )
            )
        return self.store.evidence(
            "peer_compare",
            request,
            PeersResult(devices=devices, spans_truncated=truncated),
            [
                "Reference must be representative; unknown or faulty reference yields unreliable normalization.",
                "At least two valid peers and four reference daylight bins are required.",
            ],
        )

    def detect_anomalies(self, request: DetectRequest) -> Evidence[AnomaliesResult]:
        power, daylight, ratio, _, _ = self._signals(request)
        anomalies = []
        for device in request.device_ids:
            values = ratio[device].dropna()
            history = values.shift(1).rolling(12, min_periods=4)
            z = (values - history.mean()).abs() / history.std().clip(lower=0.01)
            masks = [
                ("data_gap", power[device].isna(), None),
                ("zero_power", power[device].eq(0) & daylight, None),
                ("rolling_zscore", (z > 4).reindex(power.index, fill_value=False), z),
            ]
            for kind, mask, scores in masks:
                for span in segments(mask):
                    score = (
                        None
                        if scores is None
                        else finite(
                            scores[
                                (scores.index >= pd.Timestamp(span.start))
                                & (scores.index < pd.Timestamp(span.end))
                            ].max()
                        )
                    )
                    anomalies.append(Anomaly(device_id=device, kind=kind, span=span, score=score))
        return self.store.evidence(
            "detect_anomalies",
            request,
            AnomaliesResult(
                method=request.method, anomalies=anomalies[:64], truncated=len(anomalies) > 64
            ),
            [
                "Rolling z-score uses the preceding 12 valid daylight samples, min 4, std floor 0.01, threshold 4.",
                "A sustained loss may stop triggering z-score; inspect peer_compare for its full duration.",
            ],
        )

    def profile_pattern(self, request: PeerRequest) -> Evidence[PatternsResult]:
        power, daylight, ratio, _, _ = self._signals(request)
        timezone = self.store.metadata["System"]["timezone_code"]
        devices = []
        for device in request.device_ids:
            values = ratio[device].dropna()
            observed = power.loc[daylight, device].dropna()
            zero_fraction = float(observed.eq(0).mean()) if len(observed) else None
            median = finite(values.median())
            iqr = finite(values.quantile(0.75) - values.quantile(0.25))
            slope = None
            if len(values) >= 2:
                elapsed = pd.Series(
                    (values.index - values.index[0]).total_seconds() / 86400, index=values.index
                )
                slope = finite(values.cov(elapsed) / elapsed.var()) if elapsed.var() > 0 else None
            loss = values[values < 0.9]
            local = loss.index.tz_convert(timezone)
            anomalous_days = len(set(local.date))
            peak_hour = int(pd.Series(local.hour).mode().iloc[0]) if len(loss) else None
            if zero_fraction is not None and zero_fraction >= 0.5:
                shape = "zero_output"
            elif median is not None and median < 0.9 and iqr is not None and iqr < 0.08:
                shape = "level_shift"
            elif slope is not None and slope < -0.01 and len(values) >= 16:
                shape = "gradual"
            elif anomalous_days >= 3 and len(set(local.hour)) <= 3:
                shape = "time_localized"
            else:
                shape = "no_clear_pattern"
            devices.append(
                Pattern(
                    device_id=device,
                    daylight_bins=int(daylight.sum()),
                    observed_daylight_bins=len(observed),
                    zero_fraction=zero_fraction,
                    median_peer_ratio=median,
                    ratio_iqr=iqr,
                    slope_per_day=slope,
                    anomalous_days=anomalous_days,
                    peak_loss_local_hour=peak_hour,
                    shape=shape,
                )
            )
        return self.store.evidence(
            "profile_pattern",
            request,
            PatternsResult(local_timezone=timezone, devices=devices),
            [
                "Shape labels are descriptive heuristics, not root-cause verdicts; profile the anomaly interval."
            ],
        )

    def compare_periods(self, request: CompareRequest) -> Evidence[PeriodsResult]:
        power, _, ratio, _, _ = self._signals(request)
        devices = []
        for device in request.device_ids:
            before = self.store.series(
                device, "ac_power_kw", request.reference_start, request.reference_end
            )
            after = power[device]
            previous_energy, current_energy = float(before.sum() * 0.25), float(after.sum() * 0.25)
            devices.append(
                PeriodSummary(
                    device_id=device,
                    reference_known_energy_kwh=previous_energy,
                    current_known_energy_kwh=current_energy,
                    reference_coverage=float(before.notna().mean()),
                    current_coverage=float(after.notna().mean()),
                    energy_ratio=current_energy / previous_energy if previous_energy > 0 else None,
                    median_peer_ratio=finite(ratio[device].median()),
                )
            )
        return self.store.evidence(
            "compare_periods",
            request,
            PeriodsResult(devices=devices),
            [
                "Known energy excludes missing bins; raw energy ratios also reflect weather and period length."
            ],
        )

    def expected_power(self, request: ExpectedRequest) -> Evidence[ExpectedResult]:
        self.store.check(request)
        if not set(request.device_ids) <= self.store.inverters:
            raise ValueError("expected_power requires inverter devices")
        self.store.sensor(request.poa_device_id, "poa_wm2")
        self.store.sensor(request.ambient_device_id, "ambient_c")
        poa = self.store.series(request.poa_device_id, "poa_wm2", request.start, request.end)
        ambient = self.store.series(
            request.ambient_device_id, "ambient_c", request.start, request.end
        )
        temperature = pvlib.temperature.faiman(poa.clip(lower=0), ambient, wind_speed=1.0)
        devices = []
        for device in request.device_ids:
            params = self.store.parameters.get(device)
            if params is None:
                devices.append(
                    ExpectedSummary(
                        device_id=device,
                        status="unavailable",
                        reason="verified DC/AC ratings and gamma_pdc not provided",
                    )
                )
                continue
            dc = pvlib.pvsystem.pvwatts_dc(
                poa.clip(lower=0), temperature, params.rated_dc_kw, params.gamma_pdc
            )
            expected = pvlib.inverter.pvwatts(dc, pdc0=params.rated_ac_kw / 0.96, eta_inv_nom=0.96)
            actual = self.store.series(device, "ac_power_kw", request.start, request.end)
            comparable = (poa > 200) & expected.gt(0) & actual.notna() & expected.notna()
            if not comparable.any():
                devices.append(
                    ExpectedSummary(
                        device_id=device,
                        status="unavailable",
                        parameters=params,
                        reason="no comparable high-irradiance power/temperature bins",
                    )
                )
                continue
            performance = (actual / expected)[comparable]
            count = min(len(performance), request.max_points // len(request.device_ids))
            positions = (
                []
                if count == 0
                else (
                    [0]
                    if count == 1
                    else [i * (len(performance) - 1) // (count - 1) for i in range(count)]
                )
            )
            devices.append(
                ExpectedSummary(
                    device_id=device,
                    status="available",
                    parameters=params,
                    comparable_bins=int(comparable.sum()),
                    expected_known_energy_kwh=float(expected[comparable].sum() * 0.25),
                    median_performance_ratio=finite(
                        (actual[comparable] / expected[comparable]).median()
                    ),
                    mean_deviation_kw=finite((actual[comparable] - expected[comparable]).mean()),
                    performance_ratio_points=[
                        Point(
                            ts=performance.index[i].isoformat(), value=finite(performance.iloc[i])
                        )
                        for i in positions
                    ],
                    points_truncated=count < len(performance),
                )
            )
        return self.store.evidence(
            "expected_power",
            request,
            ExpectedResult(
                assumptions=[
                    "Explicit sourced per-device DC/AC ratings and gamma_pdc are required.",
                    "Faiman defaults u0=25, u1=6.84, assumed wind=1 m/s; inverter efficiency=0.96.",
                    "Expected energy is limited to comparable high-irradiance bins, not total generation.",
                ],
                devices=devices,
            ),
        )
