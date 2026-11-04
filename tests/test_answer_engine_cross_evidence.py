# Imports the canonical answer_engine module directly.
# (Previously pointed at a non-existent 'answer_engine_cross_evidence_fixed.py' -- C2 fix.)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.services import answer_engine as engine


def test_ocr_clock_normalization_and_location_time_link():
    chunks = [
        {
            'evidence_id': 'EVD-VIDEO', 'chunk_id': 'VID-C0002',
            'filename': 'Warehouse_CCTV_Test.mp4', 'modality': 'video',
            'text': 'OCR: 2026-01-31 21.14.27 CAM 02 WAREHOUSE ZONE B SIMULATED CCTV',
        },
        {
            'evidence_id': 'EVD-IMAGE', 'chunk_id': 'IMG-C0002',
            'filename': 'Warehouse_Asset_Checkout_Log.png', 'modality': 'image',
            'text': 'OCR: 2026-01-31 21:14:32 LOG-3187 LPT-8876 EMP-104 Warehouse Zone B Checkout',
        },
    ]
    claims = engine.build_claims(chunks)
    links = engine.correlate_claims(claims)
    assert any(link['match_type'] == 'location_and_temporal_proximity' for link in links)
    link = next(link for link in links if link['match_type'] == 'location_and_temporal_proximity')
    assert set(link['evidence_ids']) == {'EVD-VIDEO', 'EVD-IMAGE'}
    assert len(link['source_refs']) >= 2
    assert all(ref.get('chunk_id') for ref in link['source_refs'])


def test_repeated_camera_label_is_deduplicated_but_provenance_kept():
    chunks = [
        {'evidence_id': 'EVD-VIDEO', 'chunk_id': 'VID-C0002', 'filename': 'Warehouse_CCTV_Test.mp4',
         'modality': 'video', 'text': 'The camera is labeled CAM 02 WAREHOUSE ZONE B SIMULATED CCTV.'},
        {'evidence_id': 'EVD-VIDEO', 'chunk_id': 'VID-C0005', 'filename': 'Warehouse_CCTV_Test.mp4',
         'modality': 'video', 'text': 'The camera is labeled CAM 02 WAREHOUSE ZONE B SIMULATED CCTV.'},
    ]
    claims = engine.build_claims(chunks)
    # Label-only watermark observations are intentionally suppressed; the original
    # chunks remain untouched in the retrieval/index layer.
    assert claims == []


def test_png_with_cctv_in_filename_is_image_not_video():
    assert engine._modality({'filename': 'CCTV_Log.png', 'modality': 'image'}) == 'image'


def test_correlation_claim_ids_match_selected_best_pair():
    chunks = [
        {'evidence_id': 'A', 'chunk_id': 'A1', 'filename': 'a.txt', 'modality': 'document',
         'text': 'Unrelated introductory notes about the warehouse entrance.'},
        {'evidence_id': 'A', 'chunk_id': 'A2', 'filename': 'a.txt', 'modality': 'document',
         'text': 'Asset identifier LPT-8876 was recorded in Warehouse Zone B.'},
        {'evidence_id': 'B', 'chunk_id': 'B1', 'filename': 'b.mp4', 'modality': 'video',
         'text': 'Camera metadata identifies Warehouse Zone B.'},
        {'evidence_id': 'B', 'chunk_id': 'B2', 'filename': 'b.mp4', 'modality': 'video',
         'text': 'The video overlay displays asset identifier LPT-8876.'},
    ]
    claims = engine.build_claims(chunks)
    links = engine.correlate_claims(claims)
    identifier_link = next(link for link in links if link['match_type'] == 'identifier_match')
    by_id = {claim['claim_id']: claim for claim in claims}
    assert by_id[identifier_link['claim_ids'][0]]['text'].find('LPT-8876') >= 0
    assert by_id[identifier_link['claim_ids'][1]]['text'].find('LPT-8876') >= 0


def test_answer_includes_cross_exhibit_candidate_and_both_chunk_refs():
    chunks = [
        {'evidence_id': 'EVD-VIDEO', 'chunk_id': 'VID-C0002', 'filename': 'Warehouse_CCTV_Test.mp4',
         'modality': 'video', 'text': 'OCR: 2026-01-31 21.14.27 CAM 02 WAREHOUSE ZONE B SIMULATED CCTV'},
        {'evidence_id': 'EVD-IMAGE', 'chunk_id': 'IMG-C0002', 'filename': 'Warehouse_Asset_Checkout_Log.png',
         'modality': 'image', 'text': 'OCR: 2026-01-31 21:14:32 LOG-3187 LPT-8876 EMP-104 Warehouse Zone B Checkout'},
    ]
    result = engine.generate_answer('What connects the CCTV footage to the asset checkout log?', chunks)
    assert result['correlations']
    assert 'location_and_temporal_proximity' in result['answer']
    assert 'VID-C0002' in result['answer'] and 'IMG-C0002' in result['answer']


def test_literal_escaped_newlines_are_split_and_wrapper_removed():
    chunks = [{
        'evidence_id': 'EVD-VIDEO', 'chunk_id': 'VID-C0006',
        'filename': 'Warehouse_CCTV_Test.mp4', 'modality': 'video',
        'text': 'OCR: 2026-01-31 21.14.27 CAM 02 WAREHOUSE ZONE B SIMULATED CCTV\\n'
                'The camera is labeled "CAM 02 | WAREHOUSE - ZONE B | SIMULATED CCTV".\\n'
                'Interpretation rule: treat these as candidate observations, not ground truth.',
    }]
    claims = engine.build_claims(chunks)
    assert len(claims) == 1
    assert '21.14.27' in claims[0]['text']
    assert all('Interpretation rule' not in c['text'] for c in claims)


def test_cross_evidence_answer_prioritizes_linked_claims_and_compacts_repeated_labels():
    chunks = [
        {'evidence_id': 'EVD-VIDEO', 'chunk_id': 'VID-C0002', 'filename': 'Warehouse_CCTV_Test.mp4',
         'modality': 'video', 'text': 'OCR: 2026-01-31 21.14.27 CAM 02 WAREHOUSE ZONE B SIMULATED CCTV\\n'
         'The camera is labeled CAM 02 WAREHOUSE ZONE B SIMULATED CCTV.'},
        {'evidence_id': 'EVD-VIDEO', 'chunk_id': 'VID-C0005', 'filename': 'Warehouse_CCTV_Test.mp4',
         'modality': 'video', 'text': 'The camera is labeled CAM 02 WAREHOUSE ZONE B SIMULATED CCTV.'},
        {'evidence_id': 'EVD-IMAGE', 'chunk_id': 'IMG-C0003', 'filename': 'Warehouse_Asset_Checkout_Log.png',
         'modality': 'image', 'text': 'Inventory log records 2026-01-31 21:14:32 for asset LPT-8876 in Warehouse Zone B.'},
    ]
    result = engine.generate_answer('What connects the CCTV footage to the asset checkout log?', chunks)
    assert result['correlations']
    assert 'VID-C0002' in result['answer'] and 'IMG-C0003' in result['answer']
    assert result['answer'].count('camera is labeled') == 0
    assert '5s apart' in result['answer']


def test_closest_clock_pair_wins_when_multiple_video_times_match():
    chunks = [
        {'evidence_id': 'EVD-VIDEO', 'chunk_id': 'VID-27', 'filename': 'Warehouse_CCTV_Test.mp4',
         'modality': 'video', 'text': 'OCR: 2026-01-31 21.14.27 CAM 02 WAREHOUSE ZONE B SIMULATED CCTV'},
        {'evidence_id': 'EVD-VIDEO', 'chunk_id': 'VID-36', 'filename': 'Warehouse_CCTV_Test.mp4',
         'modality': 'video', 'text': 'OCR: 2026-01-31 21.14.36 CAM 02 WAREHOUSE ZONE B SIMULATED CCTV'},
        {'evidence_id': 'EVD-IMAGE', 'chunk_id': 'IMG-32', 'filename': 'Warehouse_Asset_Checkout_Log.png',
         'modality': 'image', 'text': 'Inventory log records 2026-01-31 21:14:32 checkout in Warehouse Zone B.'},
    ]
    links = engine.correlate_claims(engine.build_claims(chunks))
    link = next(x for x in links if x['match_type'] == 'location_and_temporal_proximity')
    assert any('4s apart' in anchor for anchor in link['shared_anchors'])
    assert 'VID-36' in {r['chunk_id'] for r in link['source_refs']}
