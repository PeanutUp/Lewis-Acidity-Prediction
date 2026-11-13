"""Strict, ID-aligned auditing of FIA prediction exports (stdlib only).

Usage: python metric_audit.py a.csv b.csv --output comparison.json
Each CSV must have mol_id,y_true,y_pred; all quantities must be in kJ/mol.
This tool does not repair labels, drop failures, or intersect incompatible tests.
"""
import argparse
import csv
import json
import math
from pathlib import Path


def load_predictions(path):
    rows = {}
    with open(path, newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not {'mol_id', 'y_true', 'y_pred'} <= set(reader.fieldnames or []):
            raise ValueError(f'{path}: require mol_id,y_true,y_pred')
        for line, row in enumerate(reader, 2):
            key = row['mol_id'].strip()
            if not key or key in rows:
                raise ValueError(f'{path}:{line}: empty or duplicated mol_id {key!r}')
            truth, pred = float(row['y_true']), float(row['y_pred'])
            if not math.isfinite(truth) or not math.isfinite(pred):
                raise ValueError(f'{path}:{line}: non-finite target or prediction')
            rows[key] = (truth, pred)
    if len(rows) < 2:
        raise ValueError(f'{path}: need at least two predictions')
    return rows


def metrics(rows):
    keys = sorted(rows)
    ys = [rows[k][0] for k in keys]
    errors = [rows[k][1] - rows[k][0] for k in keys]
    n = len(ys)
    mean = math.fsum(ys) / n
    sst = math.fsum((y - mean)**2 for y in ys)
    sse = math.fsum(e*e for e in errors)
    return dict(n=n, mae=math.fsum(abs(e) for e in errors)/n,
                rmse=math.sqrt(sse/n), mean_signed_error=math.fsum(errors)/n,
                r2=1-sse/sst if sst else None,
                target_population_variance=sst/n)


def compare(paths):
    exports = [load_predictions(p) for p in paths]
    ref = exports[0]
    for path, data in zip(paths[1:], exports[1:]):
        if set(data) != set(ref):
            raise ValueError(f'{path}: test IDs differ; '
                             f'missing={len(set(ref)-set(data))}, '
                             f'extra={len(set(data)-set(ref))}')
        for key in ref:
            if not math.isclose(ref[key][0], data[key][0], rel_tol=0, abs_tol=1e-6):
                raise ValueError(f'{path}: target mismatch for {key}')
    return {str(path): metrics(rows) for path, rows in zip(paths, exports)}


def implied_variance_interval(rmse, r2, rmse_digits=2, r2_digits=3):
    """Bounds from last-digit rounding, assuming conventional pooled R-squared."""
    dr, dq = 0.5*10**(-rmse_digits), 0.5*10**(-r2_digits)
    if not 0 <= r2-dq < r2+dq < 1:
        raise ValueError('This diagnostic requires a non-degenerate R2 below 1')
    return ((rmse-dr)**2/(1-r2+dq), (rmse+dr)**2/(1-r2-dq))


def original_table_audit():
    records = []
    for model, rmse, r2 in [('EGNN', 15.17, .966), ('DimeNet++', 15.99, .942)]:
        lo, hi = implied_variance_interval(rmse, r2)
        records.append(dict(model=model, rmse=rmse, r2=r2,
                            implied_variance=rmse**2/(1-r2),
                            rounding_interval=[lo, hi]))
    return dict(source='Annotated original PDF, Table 2, page 8, Total row',
                assumption='Unweighted pooled R2 = 1-SSE/SST; same observations and units',
                identity='Var_population(y) = RMSE^2/(1-R2)', records=records,
                rounding_intervals_overlap=max(x['rounding_interval'][0] for x in records)
                    <= min(x['rounding_interval'][1] for x in records),
                interpretation='Incompatible with the same test targets under the stated '
                    'assumption. The PDF alone cannot identify the cause. Alternative '
                    'definitions, fold averages, transcription, phase, filtering, or '
                    'different test sets require checking prediction files.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', nargs='*', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.predictions) if args.predictions else original_table_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
