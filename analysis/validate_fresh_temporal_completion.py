"""Read-only audit of the explicitly approved September 23 fresh temporal run.

Print the audit; do not recompute metrics, consolidate runs or modify outputs.
"""
import base64
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.consolidate_perturbation import read_yaml, settings_signature, validate_trial
from experiments.common import METRIC_KEYS
from experiments.perturbation_shared import load_data, period_windows

RUN = ROOT / 'output/perturbation/20260923_175125_279651'
LAUNCH = ROOT / 'output/perturbation/overnight_degrees_fresh_20260923_1752'
ANGLES = [0, .25, .5, 1, 2, 5, 10, 30, 45, 60, 90, 120, 135, 150,
          175, 180, 185, 210, 225, 240, 270, 300, 315, 330, 360]


def plot_payload(path):
    text = path.read_text(encoding='utf-8')
    text = text[text.rindex('Plotly.newPlot(') + len('Plotly.newPlot('):].lstrip()
    decoder = json.JSONDecoder()
    _, end = decoder.raw_decode(text)
    text = text[end:].lstrip().removeprefix(',').lstrip()
    data, end = decoder.raw_decode(text)
    layout, _ = decoder.raw_decode(text[end:].lstrip().removeprefix(',').lstrip())
    return data, layout


def array(value):
    if isinstance(value, dict) and 'bdata' in value:
        result = np.frombuffer(base64.b64decode(value['bdata']), dtype=value['dtype'])
        if 'shape' in value:
            result = result.reshape(tuple(map(int, value['shape'].split(','))))
        return result
    return np.asarray(value)


def check_curves(path, summary):
    data, layout = plot_payload(path)
    assert len(data) == len(summary.trial.unique()) * 5
    assert layout['height'] == 1150
    for trace in data:
        axis = trace.get('yaxis', 'y')
        metric = METRIC_KEYS[int(axis[1:] or '1') - 1]
        expected = summary[(summary.trial == trace['name']) & (summary.metric == metric)].sort_values('magnitude')
        np.testing.assert_allclose(array(trace['x']), expected.magnitude)
        np.testing.assert_allclose(array(trace['y']), expected.mean_distance)
        np.testing.assert_array_equal(array(trace['customdata']), expected[['n_valid', 'n_excluded']])
    assert 'degree' in layout['xaxis5']['title']['text'].lower()


def main():
    manifest = json.loads((LAUNCH / 'manifest.json').read_text())
    assert manifest['status'] == 'completed' and manifest['exit_code'] == 0
    assert read_yaml(RUN / 'run_state.yaml')['status'] == 'completed'
    for name, digest in manifest['source_sha256'].items():
        assert hashlib.sha256((LAUNCH / 'source' / name).read_bytes()).hexdigest() == digest, name
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    config = read_yaml(RUN / 'run_config.yaml')
    args = config['arguments']
    assert args['evaluations'] == ['temporal_offset']
    assert args['temporal_reuse_run'] is None and args['temporal_offset_grid'] == 'degrees'
    assert args['temporal_offset_sweep'] == ANGLES
    signature = settings_signature(config)
    assert signature == settings_signature(read_yaml(ROOT / 'output/perturbation/20260922_181016_100519/run_config.yaml'))
    assert signature['window_arguments'] == dict(temporal_window=5000, first_period=0, periods=1, seed=0)
    assert config['feature_scales'] == dict(x=1, y=1, t=42)
    launch_args = read_yaml(LAUNCH / 'source/experiments/temporal_degrees_cla.yaml')
    for key, value in launch_args.items():
        if key != 'output_dir':
            assert args[key] == value, key
    assert not list(RUN.rglob('*reuse*'))
    summaries, audits = [], []
    for number, expected_rows in enumerate([49000, 25000, 16500, 12375, 10000], 1):
        name = f'optical_chopper_data_f{number}'
        folder = RUN / name / 'temporal_offset'
        saved = read_yaml(folder / 'run_config.yaml')
        assert saved['arguments']['temporal_reuse_run'] is None
        assert saved['arguments']['temporal_offset_sweep'] == ANGLES
        summary, audit = validate_trial(RUN, name, 'temporal_offset', signature)
        assert audit['comparisons'] == expected_rows and audit['summary_rows'] == 125
        rows = pd.read_csv(folder / 'results.csv')
        windows = pd.read_csv(folder / 'windows.csv')
        assert not any('reuse' in col for col in rows.columns)
        assert not rows.duplicated(['trial', 'period_index', 'window_index', 'temporal_offset_us', 'metric']).any()
        np.testing.assert_array_equal(rows.phase_degrees, rows.magnitude)
        np.testing.assert_array_equal(rows.comparison_start_us, rows.window_start_us + rows.temporal_offset_us)
        np.testing.assert_array_equal(rows.comparison_end_us, rows.window_end_us + rows.temporal_offset_us)
        np.testing.assert_array_equal(rows.window_end_us - rows.window_start_us, 5000)
        np.testing.assert_array_equal(saved['arguments']['resolved_temporal_offsets_us'], np.rint(np.asarray(ANGLES) * audit['period_us'] / 360))
        trial, real, v2e = load_data(args['trials_dir'], name)
        try:
            assert trial.rotation_period_us == audit['period_us']
            assert trial.rotation_period_source == saved['trial']['rotation_period_source']
            pd.testing.assert_frame_equal(windows, period_windows(trial, real, 0, 5000), check_dtype=False)
            assert rows.comparison_end_us.max() <= real.t_end
            np.testing.assert_array_equal(rows.n_events_a, np.searchsorted(real.t, rows.window_end_us) - np.searchsorted(real.t, rows.window_start_us))
            np.testing.assert_array_equal(rows.n_events_b, np.searchsorted(real.t, rows.comparison_end_us) - np.searchsorted(real.t, rows.comparison_start_us))
        finally:
            real.close()
            v2e.close()
        signed = rows.loc[rows.metric.str.startswith('mmd'), 'distance_squared']
        audit.update(negative_signed_mmd=int((signed < 0).sum()), min_signed_mmd=float(signed.min()),
                     max_identity_distance=float(rows.loc[rows.magnitude == 0, 'distance'].abs().max()))
        saved_summary = pd.read_csv(folder / 'summary.csv')
        check_curves(folder / 'temporal_offset.html', saved_summary)
        clouds, _ = plot_payload(folder / 'point_cloud.html')
        assert len(clouds) == 2
        for cloud in clouds:
            for key in ['x', 'y', 'z']:
                assert 0 < array(cloud[key]).size <= 2000
                assert np.isfinite(array(cloud[key])).all()
        summaries.append(saved_summary)
        audits.append(audit)
        print(json.dumps(audit), flush=True)
    combined = pd.concat(summaries, ignore_index=True)
    keys = ['trial', 'perturbation', 'magnitude', 'metric']
    for filename in ['summary.csv', 'temporal_offset_summary.csv']:
        actual = pd.read_csv(RUN / filename)
        pd.testing.assert_frame_equal(actual.sort_values(keys).reset_index(drop=True), combined.sort_values(keys).reset_index(drop=True))
        assert len(actual) == 625
    check_curves(RUN / 'temporal_offset.html', combined)
    assert sum(a['comparisons'] for a in audits) == 112875
    assert sum(a['excluded'] for a in audits) == 0
    print(json.dumps(dict(status='PASSED', comparisons=112875, summary_rows=625, excluded=0,
                          verified_source_hashes=len(manifest['source_sha256']),
                          negative_signed_mmd=sum(a['negative_signed_mmd'] for a in audits),
                          plot_check='Saved HTML data/layout checked; no browser-rendered visual inspection',
                          started_at=manifest['started_at'], finished_at=manifest['finished_at'])), flush=True)


if __name__ == '__main__':
    main()
