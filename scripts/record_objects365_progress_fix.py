#!/usr/bin/env python3
"""Record a bounded technical amendment before resuming the frozen experiment."""
import json
import shutil
from audit_objects365_confirmation_source import ROOT, BASE, now, sha256, write_json


def main():
    code = ROOT/'scripts/audit_objects365_confirmation_source.py'
    snapshot = BASE/'audit/audit_objects365_confirmation_source_before_progress_fix.py'
    ledger = BASE/'execution_code_freeze.sha256.json'
    original = json.loads(ledger.read_text())
    assert original[str(code)] == sha256(snapshot), 'Original source snapshot differs from freeze'
    changed = [path for path, digest in original.items() if sha256(path) != digest]
    assert changed == [str(code)], f'Unexpected changed frozen files: {changed}'
    assert not list((BASE/'benchmark').glob('**/candidate_outputs.npz'))
    assert not list((BASE/'annotations').glob('*/*/parsed/*.json'))
    previous = BASE/'audit/execution_code_freeze_before_progress_fix.sha256.json'
    if previous.exists():
        assert json.loads(previous.read_text()) == original
    else:
        shutil.copy2(ledger, previous)
    record = dict(recorded_at=now(), reason='Concurrent progress.tmp writers caused FileNotFoundError; use unique temporary files',
        affected_file=str(code), old_sha256=original[str(code)], new_sha256=sha256(code),
        previous_freeze=str(previous), original_source_snapshot=str(snapshot),
        prediction_read=False, annotation_requests_before_fix=0,
        unchanged=['data source','qualification','sample order','deduplication','budgets','models','weights','VLMs','metrics','tests'],
        restart='resume verified downloaded chunks and extracted images; retain failed logs')
    write_json(BASE/'audit/progress_write_technical_amendment.json', record)
    original[str(code)] = sha256(code)
    write_json(ledger, original)
    with (BASE/'research_log.md').open('a') as stream:
        stream.write(f"\n[{now()}] 技术修订：并发progress.tmp冲突，改用写入者独有临时文件。原失败日志、旧代码与旧冻结哈希保留于audit/。不修改抽样、标签或统计规则；新标注及模型预测前续跑。\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
