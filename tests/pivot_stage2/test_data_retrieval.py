import numpy as np
import pytest
import torch

from pivot_stage2.data import MIMODataset, valid_windows, causal_fill, loader, HorizonSampler
from pivot_stage2.encoding import build_encoder
from pivot_stage2.prepare import fit_scaler
from pivot_stage2.retrieval import Retriever


def test_every_history_and_target_edge_checked_and_empty_history_rejected():
    times = np.arange(9) * 3600 * 10**9
    mask = np.ones((9, 3), bool)
    segments = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])
    assert valid_windows(times, segments, mask, 3, 2, 3600 * 10**9).tolist() == [3, 4]
    mask[:3] = False
    starts = valid_windows(times, np.zeros(9), mask, 3, 2, 3600 * 10**9)
    assert 0 not in starts
    times[4:] += 3600 * 10**9
    assert not valid_windows(times[:5], np.zeros(5), np.ones((5, 3), bool), 3, 2, 3600 * 10**9).size


def test_causal_fill_does_not_read_future_and_masks_not_mutated():
    values = np.array([[np.nan, 2.], [1., np.nan], [np.nan, 9.]])
    mask = np.isfinite(values)
    before = mask.copy()
    result = causal_fill(values, mask)
    assert result.tolist() == [[0., 2.], [1., 2.], [1., 9.]]
    values[-1] = 999
    assert np.array_equal(result[:-1], causal_fill(values, mask)[:-1])
    assert np.array_equal(mask, before)


def test_scaler_train_only_with_structurally_unobserved_channel():
    v = np.array([[1., np.nan], [3., np.nan], [1e9, 10.]])
    mask = np.isfinite(v)
    a = fit_scaler(v, mask, np.array([True, True, False]), allow_empty=True)
    assert a["mean"] == [2., 0.] and a["scale"] == [1., 1.]
    assert a["fit_counts"] == [2, 0] and a["unidentified_channels"] == [1]
    v[-1] = -1e9
    assert fit_scaler(v, mask, np.array([True, True, False]), True) == a


@pytest.mark.parametrize("channels", [3, 20, 88])
def test_dataset_full_channel_contract_text_missing_keeps_identical_samples(bundle_factory, channels):
    directory = bundle_factory(channels=channels)
    numeric = MIMODataset(directory, "train", 3, strategy="none")
    text = MIMODataset(directory, "train", 3, strategy="latest")
    assert len(numeric) == len(text)
    a, b = numeric[0], text[0]
    assert a["sample_id"] == b["sample_id"]
    assert a["x_num"].shape == (8, channels) and a["y"].shape == (3, channels)
    assert not a["text_mask"].any() and b["text_mask"].any()
    assert torch.equal(a["y_mask"], b["y_mask"])
    batch = next(iter(loader(text, batch_size=1)))
    assert batch["x_num"].shape == (1, 8, channels)
    assert list(HorizonSampler(text, seed=3, shuffle=True)) == list(HorizonSampler(text, seed=3, shuffle=True))


def test_asof_filter_precedes_scoring_even_when_future_vector_would_dominate():
    docs = [{"document_id": str(i), "source": "fixture", "entity_id": "A", "issued_at_ns": t,
             "availability_policy": "fixture", "received_at_ns": None} for i, t in enumerate((0, 20 * 3600 * 10**9))]
    values = np.ones((2, 384), np.float32)

    class Guard:
        shape = values.shape

        def __getitem__(self, ids):
            assert np.all(np.asarray(ids) == 0), "Future embeddings accessed for scoring"
            return values[ids]

    r = Retriever(docs, Guard(), np.ones((1, 6, 384), np.float32), ["A"], lag_hours=1)
    ids, _, _ = r.retrieve(np.ones((3, 1)), np.ones((3, 1), bool), 3600 * 10**9)
    assert ids[0, 0] == 0 and (ids[0, 1:] == -1).all()


@pytest.mark.parametrize("strategy", ["latest", "semantic", "semantic_temporal", "random", "shuffled", "mean"])
def test_every_retrieval_strategy_preserves_legality_and_trace(bundle_factory, strategy):
    data = MIMODataset(bundle_factory(), "train", 1, strategy=strategy, lag_hours=3)
    item = data[0]
    assert item["text_mask"].any()
    for channel in item["text_metadata"]:
        for row in channel:
            assert row["available_at_ns"] <= row["origin_ns"] and row["asof_legal"]
            assert "3h" in row["policy"]


def test_tfidf_never_fits_future_documents(tmp_path):
    docs = [{"text": "train vocabulary finance market", "available_at_ns": 1},
            {"text": "futureuniquetoken secret", "available_at_ns": 100}]
    encoder = build_encoder(tmp_path / "encoder", docs, 10)
    assert encoder.meta["fit_document_count"] == 1
    assert "futureuniquetoken" not in encoder.vectorizer.vocabulary_
    assert not encoder.encode(["futureuniquetoken"]).any()


def test_negative_lag_and_unverified_receipt_cannot_backdate_candidates():
    values = np.zeros((0, 384), np.float32)
    queries = np.zeros((1, 6, 384), np.float32)
    with pytest.raises(ValueError, match="nonnegative"):
        Retriever([], values, queries, ["A"], lag_hours=-1)
    bad = {"document_id": "x", "issued_at_ns": 10, "received_at_ns": 5, "entity_id": "A"}
    with pytest.raises(ValueError, match="verified provenance"):
        Retriever([bad], np.zeros((1, 384), np.float32), queries, ["A"])


def test_legacy_stocknet_window_also_requires_observed_input():
    from scripts.pivot_sources.stocknet import price_task, parse_prices
    from scripts.pivot_sources.common import load_config
    spec = load_config()["sources"]["stocknet"].copy()
    spec.update(lookback=1, horizons=[1])
    calendar = [{"session": day} for day in ["2015-06-01", "2015-06-02", "2015-06-03"]]
    prices = {"A": parse_prices(b"Date,Close\n2015-06-01,100\n2015-06-02,110\n2015-06-03,121\n")}
    result = price_task(prices, ["A"], calendar, spec)
    assert result["per_split_windows"]["train"]["horizons"]["1"]["eligible_masked_windows"] == 1
