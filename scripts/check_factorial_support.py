"""Small rejection-sampling support smoke check; not a tail guarantee."""
import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path
from exposure.measure import load_rules
from exposure.render import sample_conditioned


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rules', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=82652)
    p.add_argument('--draws', type=int, default=10)
    p.add_argument('--max-attempts', type=int, default=10000)
    a = p.parse_args()
    if a.draws < 1 or a.max_attempts < 1: p.error('draws and max-attempts must be positive')
    raw = a.rules.read_bytes(); rules = load_rules(json.loads(raw)); rng = random.Random(a.seed)
    attempts, failures = [], []
    for name, predicates in rules.items():
        for draw in range(a.draws):
            try:
                _, count = sample_conditioned(rng, predicates, max_attempts=a.max_attempts)
                attempts.append(count)
            except ValueError:
                failures.append({'rule':name, 'draw':draw})
    result = {'rules_sha256':hashlib.sha256(raw).hexdigest(), 'seed':a.seed,
              'samples_per_condition':a.draws, 'conditions':len(rules),
              'max_attempts_per_draw':a.max_attempts, 'failed_draws':failures,
              'median_attempts':statistics.median(attempts) if attempts else None,
              'maximum_attempts':max(attempts) if attempts else None,
              'note':'Small smoke check; not a tail guarantee or evidence of benchmark validity.'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({k:result[k] for k in ('conditions','samples_per_condition','median_attempts','maximum_attempts','failed_draws')}))

if __name__ == '__main__':
    main()
