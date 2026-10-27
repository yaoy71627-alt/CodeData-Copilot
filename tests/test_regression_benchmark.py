from copy import deepcopy
import pytest
from regression_benchmark import evaluate_case, summarize_cases


def sample():
    finding = {'id': 'F-1', 'rule_id': 'CODE_RISK', 'location': {'file': 'model.py', 'line': 8},
               'evidence': 'scaler.fit_transform(X)', 'confidence': 'High', 'severity': 'High', 'verification_status': 'static_risk'}
    truth = {'positives': [{'truth_id': 'CASE-01', 'rule_id': 'CODE_RISK', 'location': {'file': 'model.py', 'line': 8},
              'verification_status': 'static_risk', 'allowed_severities': ['High'], 'allowed_confidences': ['High'], 'evidence_contains': ['fit_transform(X)']}],
             'negatives': [{'truth_id': 'CASE-N1', 'selector': {'rule_id': 'DQ_*', 'verification_status': 'confirmed_error'}}]}
    return finding, truth


def test_unknown_findings_are_not_automatically_false_positives():
    finding, truth = sample()
    other = dict(finding, id='F-2', rule_id='CODE_UNJUDGED')
    result = evaluate_case([finding, other], truth)
    assert (result['tp'], result['fp'], result['fn']) == (1, 0, 0)
    assert result['unadjudicated_finding_ids'] == ['F-2']


@pytest.mark.parametrize('field,value', [('severity', 'Low'), ('confidence', 'Low'), ('verification_status', 'confirmed_error'), ('evidence', 'invented')])
def test_inaccurate_semantics_or_evidence_cannot_score_as_tp(field, value):
    finding, truth = sample()
    finding[field] = value
    result = evaluate_case([finding], truth)
    assert (result['tp'], result['fp'], result['fn']) == (0, 1, 1)


def test_correct_rule_wrong_location_is_not_recall():
    finding, truth = sample()
    finding['location']['line'] = 9
    result = evaluate_case([finding], truth)
    assert result['fn'] == 1 and result['tp'] == 0
    assert result['unadjudicated_finding_ids'] == ['F-1']


def test_negative_assertion_is_audited_and_duplicate_ids_rejected():
    finding, truth = sample()
    bad = dict(finding, id='F-2', rule_id='DQ_STATISTIC', verification_status='confirmed_error')
    result = evaluate_case([finding, bad], truth)
    assert result['fp'] == 1 and not result['negative_assertions'][0]['passed']
    with pytest.raises(ValueError, match='Duplicate Finding'):
        evaluate_case([finding, deepcopy(finding)], truth)


def test_missing_case_has_zero_recall_and_macro_is_unweighted():
    finding, truth = sample()
    results = {'a': evaluate_case([finding], truth), 'b': evaluate_case([], truth)}
    summary = summarize_cases(results)
    assert summary['totals'] == {'tp': 1, 'fp': 0, 'fn': 1}
    assert summary['macro_average']['supported_subset_recall'] == .5
    assert results['b']['reviewed_precision'] is None
