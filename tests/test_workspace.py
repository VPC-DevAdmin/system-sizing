"""The parallel UI keeps Classic available and serves selection-scoped evidence."""
import json
import sqlite3

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from simulator.roofline import State, save_state, summarize
from simulator.service import create_app
from simulator.service.workspace import _read_evidence


def seed_evidence(base):
    directory = base / 'roofline' / 'candidate-1'
    directory.mkdir(parents=True)
    row = dict(model='test/model', engine='test-engine', max_num_seqs=64,
               output_tokens=128, out_tok_s=120, concurrency=64,
               confirmed=True, steady_state=True, success_rate=1,
               run_dir=str(directory), samples=100, errors=0)
    save_state(base / 'roofline.json', State(status='finished', results=[row],
               plan={'models': ['test/model'], 'engines': ['test-engine'],
                     'cells': [row]}))
    (directory / 'headline_sweep.json').write_text(json.dumps({
        'cohort_run_id': 'wanted', 'peak': row, 'rungs': [row],
        'model': 'test/model', 'engine': 'test-engine', 'status': 'ok',
    }))
    conn = sqlite3.connect(directory / 'run.db')
    conn.executescript('''
        CREATE TABLE cohort_measurements (
            measurement_id INTEGER PRIMARY KEY, cohort_run_id TEXT,
            target_pool_size INTEGER, step_index INTEGER);
        CREATE TABLE measurement_telemetry (
            measurement_id INTEGER, sampled_at_ms INTEGER, gpu_sm_util_pct REAL);
        INSERT INTO cohort_measurements VALUES (1, 'wanted', 64, 0);
        INSERT INTO cohort_measurements VALUES (2, 'wanted', 128, 1);
        INSERT INTO cohort_measurements VALUES (3, 'unrelated', 64, 2);
        INSERT INTO measurement_telemetry VALUES (1, 1000, 80);
        INSERT INTO measurement_telemetry VALUES (2, 2000, 99);
        INSERT INTO measurement_telemetry VALUES (3, 3000, 1);
    ''')
    conn.close()
    return row, directory


def test_pair_selection_uses_confirmation_and_completion_policy():
    common = dict(model='m', engine='e', steady_state=True)
    search = dict(common, out_tok_s=200, success_rate=1)
    confirmed = dict(common, out_tok_s=150, success_rate=1, confirmed=True)
    rejected = dict(common, out_tok_s=300, success_rate=.2, confirmed=True)
    other = dict(common, engine='other', out_tok_s=50, success_rate=1)
    summary = summarize([search, confirmed, rejected, other])
    assert summary['best_per_pair']['m']['e'] == confirmed
    assert summary['best_per_pair']['m']['other'] == other
    assert summary['fastest'] == confirmed


def test_evidence_is_scoped_to_selected_cohort_and_concurrency(tmp_path):
    seed_evidence(tmp_path)
    evidence = _read_evidence(tmp_path, 'test/model', 'test-engine', None)
    assert evidence['measurement']['measurement_id'] == 1
    assert [t['gpu_sm_util_pct'] for t in evidence['telemetry']] == [80]
    assert evidence['source'] == 'roofline/candidate-1'


def test_evidence_downsamples_entire_window_and_keeps_last_sample(tmp_path):
    _, directory = seed_evidence(tmp_path)
    conn = sqlite3.connect(directory / 'run.db')
    conn.executemany('INSERT INTO measurement_telemetry VALUES (1, ?, 80)',
                     [(i,) for i in range(1001, 2401)])
    conn.commit()
    conn.close()
    evidence = _read_evidence(tmp_path, 'test/model', 'test-engine', None)
    assert evidence['telemetry_total'] == 1401
    assert len(evidence['telemetry']) <= 601
    assert evidence['telemetry'][0]['sampled_at_ms'] == 1000
    assert evidence['telemetry'][-1]['sampled_at_ms'] == 2400


def test_evidence_legacy_database_is_explicitly_unavailable(tmp_path):
    _, directory = seed_evidence(tmp_path)
    conn = sqlite3.connect(directory / 'run.db')
    conn.execute('DROP TABLE measurement_telemetry')
    conn.commit()
    conn.close()
    evidence = _read_evidence(tmp_path, 'test/model', 'test-engine', None)
    assert evidence['telemetry'] == []
    assert 'unavailable' in evidence['telemetry_note']


def test_evidence_refuses_stored_paths_outside_runs(tmp_path):
    base = tmp_path / 'runs'
    row, _ = seed_evidence(base)
    row['run_dir'] = str(tmp_path)
    save_state(base / 'roofline.json', State(results=[row]))
    with pytest.raises(HTTPException) as exc:
        _read_evidence(base, 'test/model', 'test-engine', None)
    assert exc.value.status_code == 404


def test_evidence_refuses_summary_symlink_outside_runs(tmp_path):
    base = tmp_path / 'runs'
    _, directory = seed_evidence(base)
    external = tmp_path / 'external.json'
    external.write_text('{}')
    (directory / 'headline_sweep.json').unlink()
    (directory / 'headline_sweep.json').symlink_to(external)
    with pytest.raises(HTTPException) as exc:
        _read_evidence(base, 'test/model', 'test-engine', None)
    assert exc.value.status_code == 404


def test_both_interfaces_and_evidence_api(tmp_path):
    seed_evidence(tmp_path / 'runs')
    with TestClient(create_app(tmp_path / 'runs', catalog_dir=tmp_path / 'catalog')) as client:
        classic = client.get('/')
        assert classic.status_code == 200
        assert 'daylight.html#home' in classic.text
        assert 'daylight-tools.css' in classic.text
        assert 'src="app.js"' in classic.text
        workspace = client.get('/workspace.html')
        assert workspace.status_code == 200
        assert 'src="workspace.js"' in workspace.text
        assert 'must-revalidate' in workspace.headers['cache-control']
        for name in ['workspace.css', 'workspace.js', 'workspace-evidence.js',
                     'lib/workspace-data.js']:
            assert client.get('/' + name).status_code == 200
        endpoint = '/api/workspace/roofline-evidence'
        params = {'model': 'test/model', 'engine': 'test-engine'}
        assert client.get(endpoint, params=params).json()['selected']['out_tok_s'] == 120
        assert client.get(endpoint, params={**params, 'attempt': -1}).status_code == 422
        assert client.get(endpoint, params={**params, 'attempt': 99}).status_code == 404
        assert client.get(endpoint, params={**params, 'model': '../outside'}).status_code == 404


def test_simple_ui_is_served_beside_classic(tmp_path):
    client = TestClient(create_app(tmp_path / 'runs', catalog_dir=tmp_path / 'catalog'))
    page = client.get('/daylight.html')
    assert page.status_code == 200 and 'daylight.js' in page.text
    for asset in ('/daylight.js', '/daylight.css', '/lib/daylight-data.js'):
        assert client.get(asset).status_code == 200
    tools = client.get('/index.html').text
    assert 'daylight.html' in tools and client.get('/daylight-tools.css').status_code == 200
    # Every tool keeps its tab; opened with no tool named, capsim starts at Daylight.
    for view in ('prepare', 'optimizer', 'roofline', 'control', 'results', 'personas', 'infrastructure'):
        assert f'data-view="{view}"' in tools
    assert 'location.replace("daylight.html")' in tools
    for tool in ('control', 'roofline', 'optimizer', 'results', 'infrastructure', 'personas', 'prepare'):
        assert f'index.html#{tool}' in page.text


def test_headline_run_request_validates_its_shape(tmp_path):
    client = TestClient(create_app(tmp_path / 'runs', catalog_dir=tmp_path / 'catalog'))
    body = {'workload': {'kind': 'persona', 'id': 'headline_generation'},
            'profile': 'mock', 'new_run': True}
    bad = client.post('/api/runs', json={**body, 'shape': {'input_tokens': 'many'}})
    assert bad.status_code == 422 and 'shape' in bad.json()['detail']
    wrong = client.post('/api/runs', json={
        **body, 'workload': {'kind': 'persona', 'id': 'headline_ingest'},
        'shape': {'input_tokens': 128, 'output_tokens': 128}})
    assert wrong.status_code == 422
