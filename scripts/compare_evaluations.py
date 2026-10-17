"""Compare exact answer AND citation outcomes; report regressions explicitly."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    before, after = [json.loads(p.read_text(encoding='utf-8')) for p in (args.baseline, args.candidate)]
    left = {row['query']: row for row in before['questions']}
    right = {row['query']: row for row in after['questions']}
    if len(left) != len(before['questions']) or set(left) != set(right):
        raise ValueError('Comparison requires identical unique questions')
    families = defaultdict(lambda: {'total': 0, 'baseline_pass': 0, 'candidate_pass': 0})
    improvements, regressions = [], []
    for query, row in right.items():
        old = left[query]
        if (row['expected_answer'], row['expected_citations']) != (old['expected_answer'], old['expected_citations']):
            raise ValueError('Gold labels changed between runs')
        family = families[row.get('family', 'sample')]
        family['total'] += 1
        family['baseline_pass'] += bool(old['passed'])
        family['candidate_pass'] += bool(row['passed'])
        if old['passed'] and not row['passed']:
            regressions.append({'query': query, 'baseline': old['actual'], 'candidate': row['actual']})
        if row['passed'] and not old['passed']:
            improvements.append({'query': query, 'baseline': old['actual'], 'candidate': row['actual']})
    result = {'kind': 'paired_development_comparison_not_official_score', 'total': len(right),
              'baseline_pass': sum(r['passed'] for r in left.values()),
              'candidate_pass': sum(r['passed'] for r in right.values()),
              'by_family': dict(sorted(families.items())), 'improvements': improvements, 'regressions': regressions}
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
