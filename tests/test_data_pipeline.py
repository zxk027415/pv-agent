import json

import pandas as pd
import pytest

from pvdiag.data.datacard import build_datacard
from pvdiag.data.datacard import main as card_main
from pvdiag.data.normalize import (
    SCHEMA,
    channels_from_metadata,
    normalize_files,
)
from pvdiag.data.normalize import (
    main as normalize_main,
)


@pytest.fixture
def metadata():
    return {
        "System": {"system_id": 2107, "timezone_code": "PST8PDT"},
        "Inverters": {
            "Inverter 0": {"inverter_id": 1, "name": "inv_fss_1", "model": "example"},
            "Inverter 1": {"inverter_id": 2, "name": "inv_fss_2", "model": "example"},
        },
        "Metrics": {
            "inv_01_ac_power_inv_1": {
                "metric_id": 1,
                "common_name": "AC power",
                "units": "kw",
                "source_type": "INVERTER",
                "source_id": None,
            },
            "inv_02_ac_power_inv_2": {
                "metric_id": 2,
                "common_name": "AC power",
                "units": "W",
                "source_type": "INVERTER",
                "source_id": None,
            },
            "poa": {
                "metric_id": 3,
                "common_name": "Irradiance POA",
                "units": "W/m^2",
                "source_type": "OTHER",
            },
            "temp": {
                "metric_id": 4,
                "common_name": "Temperature ambient",
                "units": "F",
                "source_type": "OTHER",
            },
        },
    }


def write_csv(tmp_path, **columns):
    path = tmp_path / "sample.csv"
    pd.DataFrame(columns).to_csv(path, index=False)
    return path


def test_local_timezone_units_schema_and_missing_values(tmp_path, metadata):
    path = write_csv(
        tmp_path,
        measured_on=["2019-01-01 12:00:00", "2019-07-01 12:00:00"],
        inv_01_ac_power_inv_1=[20.0, None],
        inv_02_ac_power_inv_2=[1000.0, 2000.0],
        temp=[32.0, 212.0],
        poa=[400.0, 900.0],
        ignored=[1, 2],
    )
    frame, audit = normalize_files([path], metadata, chunksize=1)
    assert list(frame) == SCHEMA
    assert str(frame.ts.dt.tz) == "UTC"
    assert set(frame.ts.dt.hour) == {19, 20}  # winter UTC-8, summer UTC-7
    assert frame[frame.device_id == "inv02"].value.tolist() == [1.0, 2.0]
    assert frame[frame.metric == "ambient_c"].value.tolist() == pytest.approx([0, 100])
    assert frame[frame.device_id == "inv01"].value.isna().sum() == 1
    assert audit["sources"][0]["input_rows"] == 2


def test_explicit_utc_column_and_half_open_bounds(tmp_path, metadata):
    path = write_csv(tmp_path, utc_measured_on=["2019-01-01 00:00", "2019-01-01 00:15"], poa=[0, 2])
    frame, audit = normalize_files([path], metadata, start="2019-01-01", end="2019-01-01 00:15")
    assert len(frame) == 1
    assert frame.ts.iloc[0] == pd.Timestamp("2019-01-01", tz="UTC")
    assert audit["sources"][0]["timezone"] == "UTC"


def test_dst_gaps_and_repeated_hours_are_excluded_and_counted(tmp_path, metadata):
    path = write_csv(
        tmp_path,
        measured_on=[
            "2019-03-10 02:15",
            "2019-11-03 01:15",
            "2019-11-03 02:15",
        ],
        poa=[1, 2, 3],
    )
    frame, audit = normalize_files([path], metadata)
    assert len(frame) == 1
    assert frame.ts.iloc[0] == pd.Timestamp("2019-11-03 10:15", tz="UTC")
    assert audit["sources"][0]["dst_rows_excluded"] == 2


def test_timezone_override(tmp_path, metadata):
    path = write_csv(tmp_path, measured_on=["2019-01-01 00:00"], poa=[1])
    frame, _ = normalize_files([path], metadata, source_timezone="UTC")
    assert frame.ts.iloc[0].hour == 0


def test_export_column_typo_resolves_by_metric_id_and_records_alias(tmp_path, metadata):
    path = write_csv(tmp_path, measured_on=["2019-01-01 12:00"], inv_01_ac_power_iinv_1=[20])
    frame, audit = normalize_files([path], metadata)
    assert frame.device_id.tolist() == ["inv01"]
    assert frame.value.tolist() == [20]
    assert audit["sources"][0]["column_aliases"] == {
        "inv_01_ac_power_iinv_1": "inv_01_ac_power_inv_1",
    }


@pytest.mark.parametrize("values", [[2, 3], [None, 3]])
def test_overlapping_sources_reject_conflicts(tmp_path, metadata, values):
    path = write_csv(tmp_path, measured_on=["2019-01-01 12:00"] * 2, poa=values)
    with pytest.raises(ValueError, match="conflicting observations"):
        normalize_files([path], metadata, chunksize=1)


def test_identical_duplicates_collapsed_across_chunks(tmp_path, metadata):
    path = write_csv(tmp_path, measured_on=["2019-01-01 12:00"] * 2, poa=[2, 2])
    frame, audit = normalize_files([path], metadata, chunksize=1)
    assert len(frame) == 1
    assert audit["duplicate_rows_collapsed"] == 1


@pytest.mark.parametrize("unit", ["kWh", "unknown"])
def test_energy_or_unknown_unit_is_not_guessed(metadata, unit):
    metadata["Metrics"]["inv_01_ac_power_inv_1"]["units"] = unit
    with pytest.raises(ValueError, match="unsupported unit"):
        channels_from_metadata(metadata)


@pytest.mark.parametrize("timestamps", [["bad"], [None]])
def test_invalid_timestamp_fails(tmp_path, metadata, timestamps):
    path = write_csv(tmp_path, measured_on=timestamps, poa=[2])
    with pytest.raises(ValueError):
        normalize_files([path], metadata)


def test_infinite_values_fail(tmp_path, metadata):
    path = write_csv(tmp_path, measured_on=["2019-01-01 12:00"], poa=[float("inf")])
    with pytest.raises(ValueError, match="infinite"):
        normalize_files([path], metadata)


def make_clean_frame(start, end, devices=("inv01", "inv02", "sensor_3")):
    ts = pd.date_range(start, end, freq="15min", tz="UTC", inclusive="left")
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "ts": ts,
                    "system_id": 2107,
                    "device_id": d,
                    "metric": "poa_wm2" if d == "sensor_3" else "ac_power_kw",
                    "value": 1.0,
                }
            )
            for d in devices
        ],
        ignore_index=True,
    )


def test_card_distinguishes_missing_bins_nan_and_zero(metadata):
    frame = make_clean_frame("2019-01-01", "2019-01-01 01:00")
    frame.loc[0, "value"] = None
    frame.loc[1, "value"] = 0.0
    frame = frame.drop(index=2)
    card = build_datacard(frame, metadata)
    inv = next(s for s in card["series"] if s["device_id"] == "inv01")
    assert inv["coverage"] == 0.5
    assert inv["missing_bins"] == 2  # one absent record and one NaN
    assert inv["missing_value_rows"] == 1
    assert inv["zero_value_rows"] == 1
    assert card["recommended_window"] is None
    assert card["inverters"][0]["rated_ac_kw"] is None


def test_card_requires_all_inverters_and_complete_months(metadata):
    frame = make_clean_frame("2019-01-01", "2019-07-01")
    card = build_datacard(frame, metadata)
    assert card["recommended_window"]["months"] == 6
    assert card["recommended_window"]["min_coverage"] == 1.0
    assert len(card["monthly"]) == 6
    partial = frame[frame.ts >= "2019-01-02"]
    assert build_datacard(partial, metadata)["recommended_window"] is None
    missing_device = frame[frame.device_id != "inv02"]
    missing = build_datacard(missing_device, metadata)
    assert missing["recommended_window"] is None
    assert "inv02" in missing["warnings"][0]


def test_card_caps_work_window_at_twelve_months(metadata):
    frame = make_clean_frame("2019-01-01", "2020-03-01")
    card = build_datacard(frame, metadata)
    assert card["recommended_window"]["months"] == 12
    assert card["recommended_window"]["end"] == "2020-01-01T00:00:00+00:00"


def test_card_counts_missing_window_edges_in_coverage(metadata):
    frame = make_clean_frame("2019-01-02", "2019-07-01")
    audit = {"system_id": 2107, "window_utc": ["2019-01-01", "2019-07-01"]}
    card = build_datacard(frame, metadata, normalization=audit)
    assert card["window"]["start"] == "2019-01-01T00:00:00+00:00"
    assert card["monthly"][0]["min_inverter_coverage"] == pytest.approx(30 / 31)
    assert card["recommended_window"]["months"] == 6
    inv = next(s for s in card["series"] if s["device_id"] == "inv01")
    assert inv["missing_bins"] == 96


def test_invalid_window_and_empty_selection_fail(tmp_path, metadata):
    path = write_csv(tmp_path, measured_on=["2019-01-01 12:00"], poa=[2])
    with pytest.raises(ValueError, match="start must precede end"):
        normalize_files([path], metadata, start="2019-02-01", end="2019-01-01")
    with pytest.raises(ValueError, match="no supported observations"):
        normalize_files([path], metadata, start="2020-01-01")


def test_pipeline_cli_parquet_and_json_roundtrip(tmp_path, metadata):
    raw = tmp_path / "raw" / "2107"
    (raw / "data").mkdir(parents=True)
    (raw / "metadata").mkdir()
    (raw / "metadata" / "2107_system_metadata.json").write_text(json.dumps(metadata))
    pd.DataFrame({"measured_on": ["2019-01-01 12:00"], "poa": [300]}).to_csv(
        raw / "data" / "2107_irradiance_data.csv", index=False
    )
    # The alternate revision must not be included by the default selection.
    (raw / "data" / "2107_irradiance_data_2025.csv").write_text("invalid")
    clean = tmp_path / "clean"
    assert normalize_main(["2107", "--raw-root", str(raw.parent), "--dest", str(clean)]) == 0
    assert card_main(["2107", "--raw-root", str(raw.parent), "--clean-root", str(clean)]) == 0
    card = json.loads((clean / "2107_card.json").read_text())
    assert card["rows"] == 1
    assert card["normalization"]["sources"][0]["timezone"] == "PST8PDT"
    assert pd.read_parquet(clean / "2107.parquet").value.tolist() == [300.0]
