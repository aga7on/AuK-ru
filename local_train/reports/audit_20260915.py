"""Read-only audit of retained benchmarks; writes only its own report files."""
import ast
import collections
import json
import pathlib
import sys
import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
report = {'suites': [], 'json_errors': [], 'audio_errors': [], 'syntax_errors': []}
for base in [ROOT/'local_tests', ROOT/'local_train/run_ru/evals', ROOT/'local_train/run_ru_s1/evals', ROOT/'local_train/user_reviews']:
    for d in [base] + sorted(p for p in base.rglob('*') if p.is_dir()):
        wavs = list(d.glob('*.wav'))
        js = list(d.glob('*.json'))
        if not wavs and not js:
            continue
        row = {'path': str(d.relative_to(ROOT)), 'wav_count': len(wavs), 'json': [], 'audio_seconds': 0, 'near_full_scale_files': []}
        for p in js:
            try:
                obj = json.loads(p.read_text(encoding='utf-8-sig'))
                records = obj if isinstance(obj, list) else []
                scores = [(r.get('judge') or {}).get('overall') for r in records if isinstance(r, dict)]
                scores = [s for s in scores if isinstance(s, (int, float))]
                row['json'].append({'name': p.name, 'records': len(obj), 'mean_judge_overall': sum(scores)/len(scores) if scores else None, 'status_counts': dict(collections.Counter(r.get('status') for r in records if isinstance(r,dict) and r.get('status')))})
            except Exception as e:
                report['json_errors'].append([str(p), str(e)])
        for p in wavs:
            try:
                x, sr = sf.read(p, dtype='float32', always_2d=True)
                row['audio_seconds'] += len(x)/sr
                if not len(x) or not np.isfinite(x).all():
                    report['audio_errors'].append([str(p), 'empty or nonfinite'])
                if len(x) and np.max(np.abs(x)) >= 0.999:
                    row['near_full_scale_files'].append(p.name)
            except Exception as e:
                report['audio_errors'].append([str(p), str(e)])
        report['suites'].append(row)
for base in ['src', 'local_train', 'local_tests', 'comfyui']:
    for p in (ROOT/base).rglob('*.py'):
        try:
            ast.parse(p.read_text(encoding='utf-8-sig'))
        except Exception as e:
            report['syntax_errors'].append([str(p), str(e)])
report['dataset'] = {}
paths = {}
for split in ['train','val']:
    records = [json.loads(s) for s in (ROOT/f'local_train/data/{split}.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    paths[split] = {c['audio_url'] for r in records for m in r['messages'] if m['role']=='assistant' for c in m['content'] if 'audio_url' in c}
    report['dataset'][split] = {'rows':len(records), 'hours':sum(r.get('duration',0) for r in records)/3600, 'user_audio_rows':sum(any(c.get('type')=='audio' for m in r['messages'] if m['role']=='user' for c in m['content']) for r in records), 'missing_targets':sum(not pathlib.Path(p).exists() for p in paths[split])}
report['dataset']['shared_target_paths'] = len(paths['train'] & paths['val'])
from auk.infer import quality
report['metric_probes'] = {
    'repeat_perfect': quality.recall('да да нет', 'да да нет'),
    'insertion': quality.recall('привет мир', 'привет мир лишние слова'),
    'wrong_order': quality.recall('мама любит папу', 'папу любит мама'),
    'english_wrong': quality.recall('hello world', 'completely wrong'),
    'silence_pause_excess': quality.pause_excess(np.zeros(24000, dtype=np.float32),24000),
}
try:
    quality.trim_silence(__import__('torch').zeros(1,100),24000)
    report['metric_probes']['short_audio'] = 'ok'
except Exception as e:
    report['metric_probes']['short_audio'] = type(e).__name__ + ': ' + str(e)
(OUT/'AUDIT_2026-09-15_inventory.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='suites'},ensure_ascii=False,indent=2))
for row in report['suites']:
    print(json.dumps(row,ensure_ascii=False))
