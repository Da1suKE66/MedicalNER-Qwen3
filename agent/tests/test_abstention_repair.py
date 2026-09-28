from copy import deepcopy
import pytest
from data_repair.abstention_repair import make_case,validate_case,refresh,build


def anchor(symptom,disease,group=1,relation='is_associated_symptom_of'):
    return {'source':symptom,'target':disease,'group_id':group,'sample_id':1,
            'parent_id':'train:1','parent_sha256':'a'*64,'relation':relation,'evidence_span_ids':['SENT_001']}


def common_case():
    return make_case(anchor('Feature Alpha','Condition Kappa'),anchor('Feature Beta','Condition Kappa'),
                     missing=anchor('Feature Gamma','Condition Lambda'))


def test_common_context_negative_does_not_use_original_silver_absence():
    row=common_case();assert validate_case(row)
    assert len(row['pairs'])==12
    assert sum(x=='SUPPORTED_SILVER' for x in row['pair_supervision'].values())==2
    assert sum(x=='SYNTHETIC_NEGATIVE' for x in row['pair_supervision'].values())==8
    assert 'negative_proofs' not in row['prompt']
    assert row['eligible_for_rl'] is False


def test_cross_scope_negatives_leave_reverse_symptom_edges_unknown():
    row=make_case(anchor('Feature Alpha','Condition Kappa'),anchor('Feature Beta','Condition Lambda'))
    assert validate_case(row)
    assert sum(x=='SYNTHETIC_NEGATIVE' for x in row['pair_supervision'].values())==6
    nodes={e['id']:e for e in row['entities']}
    for p in row['pairs']:
        if nodes[p['source']]['label']=='Disease' and nodes[p['target']]['label']=='Symptom':
            assert row['pair_supervision'][p['pair_id']].startswith('UNKNOWN_')


def test_missing_entity_proof_rejects_present_name():
    with pytest.raises(ValueError):
        make_case(anchor('Feature Alpha','Condition Kappa'),anchor('Feature Beta','Condition Kappa'),
                  missing=anchor('Feature Alpha','Elsewhere'))


def test_no_cross_split_or_group_construction():
    with pytest.raises(ValueError):make_case(anchor('Alpha','Kappa'),anchor('Beta','Lambda',group=2))


def test_no_overlapping_concepts():
    with pytest.raises(ValueError):make_case(anchor('Feature','Kappa'),anchor('Feature Beta','Kappa'))


def test_modified_source_invalidates_proof():
    row=common_case();row['spans'][0]['text']='Different text'
    with pytest.raises(ValueError):validate_case(row)


def test_constructed_positive_cannot_be_marked_negative():
    row=common_case()
    pid=next(k for k,s in row['pair_supervision'].items() if s=='SUPPORTED_SILVER')
    row['pair_supervision'][pid]='SYNTHETIC_NEGATIVE'
    for d in row['target']['decisions']:
        if d['pair_id']==pid:d['relations']=[]
    row['abstention_provenance']['negative_proofs'][pid]={'kind':'no_explicit_relation_in_two_statement_context'}
    refresh(row)
    with pytest.raises(ValueError):validate_case(row)


def test_build_preserves_base_and_matched_inputs():
    base=[common_case(),make_case(anchor('Feature Gamma','Condition Lambda',group=2),
                                 anchor('Feature Delta','Condition Lambda',group=2),serial=1)]
    # A synthetic row can act as a simple supported fixture; only train anchors
    # are consumed, and original rows remain byte-for-byte identical objects.
    variants,report=build(base,[],max_cases=1)
    a,b=variants.values()
    assert a[:2]==b[:2]==base
    assert a[2]['prompt']==b[2]['prompt']
    assert a[2]['target']==b[2]['target']
    assert report['original_unknowns_unchanged']


def test_build_rejects_group_leakage():
    base=[common_case()];tune=deepcopy(base);tune[0]['split']='tune'
    with pytest.raises(ValueError):build(base,tune,max_cases=1)


def test_build_rejects_non_train_anchors():
    base=[common_case()];base[0]['split']='test'
    with pytest.raises(ValueError):build(base,[],max_cases=1)
