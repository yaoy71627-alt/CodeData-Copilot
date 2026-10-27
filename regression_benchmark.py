"""Evaluate a reviewed truth subset, without calling unjudged findings false positives.

Truth IDs belong to annotations; Finding IDs belong to the detector. Matching requires
rule, exact source location, evidence, semantic status and an allowed severity.
This module evaluates reports only; it never changes findings or project scores.
"""
from collections import Counter
from fnmatch import fnmatchcase
from statistics import mean


def selected(finding, selector):
    return all(fnmatchcase(str(finding.get(key, '')), str(value))
               for key, value in selector.items())


def at_location(finding, truth):
    return finding.get('rule_id') == truth['rule_id'] and all(
        finding.get('location', {}).get(key) == value
        for key, value in truth['location'].items())


def evaluate_case(findings, annotation):
    ids = [item['id'] for item in findings]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate Finding IDs are not valid benchmark input')
    truths = annotation['positives']
    if len({item['truth_id'] for item in truths}) != len(truths):
        raise ValueError('Duplicate truth IDs')
    matched, failed, reviewed_ids, fp_ids = [], [], set(), set()
    for truth in truths:
        candidates = [f for f in findings if at_location(f, truth)]
        valid = [f for f in candidates
                 if f.get('verification_status') == truth['verification_status']
                 and f.get('severity') in truth['allowed_severities']
                 and (not truth.get('allowed_confidences') or f.get('confidence') in truth['allowed_confidences'])
                 and all(part in f.get('evidence', '') for part in truth['evidence_contains'])]
        if len(valid) == 1 and valid[0]['id'] not in reviewed_ids:
            finding = valid[0]
            matched.append({'truth_id': truth['truth_id'], 'finding_id': finding['id'],
                            'location': finding['location'], 'evidence': finding['evidence'],
                            'severity': finding['severity'], 'confidence': finding.get('confidence'), 'verification_status': finding['verification_status']})
            reviewed_ids.add(finding['id'])
        else:
            failed.append({'truth_id': truth['truth_id'], 'reason': 'missing' if not candidates else 'location_found_but_semantics_evidence_severity_or_uniqueness_failed',
                           'candidate_ids': [f['id'] for f in candidates]})
            reviewed_ids.update(f['id'] for f in candidates)
            fp_ids.update(f['id'] for f in candidates)
    negative_results = []
    for negative in annotation.get('negatives', []):
        violations = [f['id'] for f in findings if selected(f, negative['selector'])]
        fp_ids.update(violations)
        reviewed_ids.update(violations)
        negative_results.append({'truth_id': negative['truth_id'], 'passed': not violations,
                                 'violating_finding_ids': violations})
    tp, fp, fn = len(matched), len(fp_ids), len(failed)
    return {'scope': 'reviewed_subset_not_exhaustive_project_accuracy',
            'annotation_review': 'static_source_review_pending_user_adjudication',
            'tp': tp, 'fp': fp, 'fn': fn, 'reviewed_precision': tp / (tp + fp) if tp + fp else None,
            'supported_subset_recall': tp / (tp + fn) if tp + fn else None,
            'unadjudicated_finding_ids': sorted(set(ids) - reviewed_ids),
            'matched': matched, 'missed': failed, 'negative_assertions': negative_results,
            'negative_assertions_passed': sum(n['passed'] for n in negative_results),
            'reported_findings': len(findings),
            'reported_status_counts': dict(Counter(f['verification_status'] for f in findings))}


def summarize_cases(results):
    if not results:
        raise ValueError('No cases evaluated')
    metrics = ('reviewed_precision', 'supported_subset_recall')
    return {'scope': 'reviewed_subset_not_exhaustive_project_accuracy', 'cases': results,
            'totals': {key: sum(r[key] for r in results.values()) for key in ('tp', 'fp', 'fn')},
            'macro_average': {key: mean(r[key] for r in results.values() if r[key] is not None)
                              if any(r[key] is not None for r in results.values()) else None for key in metrics},
            'unadjudicated_count': sum(len(r['unadjudicated_finding_ids']) for r in results.values())}
