#!/usr/bin/env python3
"""Score next-layer forecasts against static cold usage and previous-token IDs.

All controls use the same number of cold candidates as the GPU forecast on
that visit. Previous-token candidates are filled from static usage as needed.
Transfer costs are lower bounds using the measured PCIe bandwidth; they do
not imply that staging meets a deadline or produces a throughput gain.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def analyze(path, usage_path, bandwidth):
    metadata, routes = {}, {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row.get('kind') == 'layer':
            metadata[row['layer']] = row
        else:
            key = (row['position'], row['layer'])
            if key in routes:
                raise ValueError(f'duplicate route row: {key}')
            routes[key] = row
    usage = defaultdict(lambda: defaultdict(int))
    for line in Path(usage_path).read_text().splitlines():
        words = line.split()
        if len(words) == 3 and all(word.isdigit() for word in words):
            layer, expert, count = map(int, words)
            usage[layer][expert] += count
    totals = {name: defaultdict(int) for name in ('proxy', 'static', 'previous')}
    staging = defaultdict(int)
    staged_layers = defaultdict(lambda: defaultdict(int))
    per_layer = defaultdict(lambda: {name: defaultdict(int) for name in totals})
    for (position, layer), source in sorted(routes.items()):
        target = source['prediction_layer']
        actual = routes.get((position, target))
        if actual is None or not source['prediction']:
            continue
        info = metadata[target]
        mask = info['cold_mask']
        def cold(values):
            result = []
            for expert in values:
                if not 0 <= expert < len(mask):
                    raise ValueError(f'invalid expert {expert} for layer {target}')
                if mask[expert] and expert not in result:
                    result.append(expert)
            return result
        truth = set(cold(actual['ids']))
        proxy = cold(source['prediction'])
        count = len(proxy)
        static = sorted((e for e, flag in enumerate(mask) if flag),
                        key=lambda e: (-usage[target][e], e))
        previous = cold(routes.get((position - 1, target), {'ids': []})['ids'])
        previous += [e for e in static if e not in previous]
        controls = {'proxy': proxy, 'static': static[:count], 'previous': previous[:count]}
        selected = actual.get('stage_ids', [])
        if selected:
            if len(selected) != len(actual['ids']) or any(slot < -1 for slot in selected):
                raise ValueError(f'invalid stage selection at position {position}, layer {target}')
            if source.get('epoch') != actual.get('epoch'):
                raise ValueError('forecast and consumer replay epochs differ')
            ready = {expert for expert, slot in zip(actual['ids'], selected) if slot >= 0}
            if not ready.issubset(truth.intersection(proxy)):
                raise ValueError('staged selection is outside the predicted cold routes')
            for accumulator in (staging, staged_layers[target]):
                accumulator['visits'] += 1
                accumulator['actual_cold'] += len(truth)
                accumulator['predicted_actual_cold'] += len(truth.intersection(proxy))
                accumulator['ready_actual_cold'] += len(ready)
                accumulator['mixed_visits'] += bool(truth)
                accumulator['fully_staged_mixed_visits'] += bool(truth) and ready == truth
        for name, candidates in controls.items():
            hit = len(truth.intersection(candidates))
            for accumulator in (totals[name], per_layer[target][name]):
                accumulator['visits'] += 1
                accumulator['mixed_visits'] += bool(truth)
                accumulator['actual_cold'] += len(truth)
                accumulator['candidates'] += len(candidates)
                accumulator['hits'] += hit
                accumulator['full_mixed_coverage'] += bool(truth) and hit == len(truth)
                accumulator['transfer_bytes'] += len(candidates) * info['expert_bytes']
                accumulator['wasted_bytes'] += (len(candidates) - hit) * info['expert_bytes']
    if not totals['proxy']['visits']:
        raise ValueError('trace has no scored next-layer predictions')
    def summarize(values):
        result = dict(values)
        result['cold_recall'] = values['hits'] / max(1, values['actual_cold'])
        result['precision'] = values['hits'] / max(1, values['candidates'])
        result['full_mixed_coverage_rate'] = values['full_mixed_coverage'] / max(1, values['mixed_visits'])
        result['pcie_lower_bound_us_per_visit'] = values['transfer_bytes'] / bandwidth * 1e6 / values['visits']
        return result
    def summarize_staging(values):
        result = dict(values)
        result['realized_cold_coverage'] = values['ready_actual_cold'] / max(1, values['actual_cold'])
        result['predicted_hits_ready_fraction'] = values['ready_actual_cold'] / max(1, values['predicted_actual_cold'])
        result['fully_staged_mixed_rate'] = values['fully_staged_mixed_visits'] / max(1, values['mixed_visits'])
        return result
    return {'trace': str(path), 'usage': str(usage_path), 'pcie_bytes_per_second': bandwidth,
            'summary': {name: summarize(values) for name, values in totals.items()},
            'staging': summarize_staging(staging) if staging['visits'] else None,
            'staging_per_layer': {layer: summarize_staging(values) for layer, values in staged_layers.items()},
            'per_layer': {layer: {name: summarize(values) for name, values in controls.items()}
                          for layer, controls in per_layer.items()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace', type=Path)
    p.add_argument('--usage', type=Path, default=Path('tests/moe_usage_survey/zeromoe_usage_survey_qwen4exp_usage_tap'))
    p.add_argument('--pcie-gbps', type=float, default=12.3)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.pcie_gbps <= 0:
        p.error('PCIe bandwidth must be positive')
    result = analyze(args.trace, args.usage, args.pcie_gbps * 1e9)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
