import argparse
import os

import numpy as np
import pandas as pd
import tqdm


def l2_normalize(arr, axis=1):
    norm = np.linalg.norm(arr, axis=axis, keepdims=True)
    return arr / np.clip(norm, 1e-12, None)


def embed_with_sentence_transformers(model_name, texts, batch_size, device):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name).to(device)
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb = model.encode(batch, batch_size=batch_size, show_progress_bar=False)
        embeddings.append(emb)
    return np.vstack(embeddings)


def embed_with_openai(model_name, texts):
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai package is required for GPT embeddings") from exc

    client = OpenAI()
    embeddings = []
    for i in range(0, len(texts), 512):
        batch = texts[i:i + 512]
        resp = client.embeddings.create(model=model_name, input=batch)
        embeddings.append([item.embedding for item in resp.data])
    return np.array([e for batch in embeddings for e in batch], dtype=np.float32)


def fit_or_load_pca(embeddings, pca_path):
    try:
        from sklearn.decomposition import PCA
        import joblib
    except Exception as exc:
        raise RuntimeError("scikit-learn and joblib are required for PCA") from exc

    if os.path.exists(pca_path):
        return joblib.load(pca_path)

    pca = PCA(n_components=384, random_state=0)
    pca.fit(embeddings)
    os.makedirs(os.path.dirname(pca_path), exist_ok=True)
    joblib.dump(pca, pca_path)
    return pca


def build_paths(dataset):
    base_textual = f"../../../dataset/Time-MMD/textual/{dataset}"
    base_numerical = f"../../../dataset/Time-MMD/numerical/{dataset}"
    numerical_parquet = f"{base_numerical}/{dataset}.parquet"
    query_parquet = f"{base_textual}/QueryTextPackage.parquet"
    return base_textual, numerical_parquet, query_parquet


def main():
    parser = argparse.ArgumentParser(description="Query text pre-embedding (multi model)")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--target_id", type=str, required=True)
    parser.add_argument("--ForecastingPoint", type=int, required=True)
    parser.add_argument("--backend", type=str, choices=["sentence_transformers", "openai_pca"], required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_tag", type=str, default=None)
    parser.add_argument("--pca_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    dataset = args.dataset
    target_id = args.target_id
    forecasting_point = args.ForecastingPoint
    backend = args.backend
    model_name = args.model_name
    model_tag = args.model_tag or model_name

    base_textual, numerical_parquet, query_parquet = build_paths(dataset)

    df = pd.read_parquet(numerical_parquet)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    time_span = df[["start_date", "end_date"]].values.tolist()

    des = pd.read_parquet(query_parquet)

    if forecasting_point:
        save_dir = f"{base_textual}/QueryText-embedding-{model_tag}-ForecastingPoint/{target_id}"
        os.makedirs(save_dir, exist_ok=True)
        qts = []
        qts_keys = []
        for row in df.itertuples(index=False):
            start_date = row.start_date
            end_date = row.end_date
            qt = f"From {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}."
            qts.append(qt)
            qts_keys.append(start_date)

        if backend == "sentence_transformers":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            pbar = tqdm.tqdm(total=len(qts), desc="Encoding")
            for i in range(0, len(qts), args.batch_size):
                batch = qts[i:i + args.batch_size]
                embeddings = embed_with_sentence_transformers(model_name, batch, args.batch_size, device)
                embeddings = l2_normalize(embeddings, axis=1)
                for j, emb in enumerate(embeddings):
                    key = qts_keys[i + j]
                    np.save(os.path.join(save_dir, str(key)[:10] + target_id + ".npy"), emb[np.newaxis, :])
                pbar.update(len(batch))
            pbar.close()
        else:
            embeddings = embed_with_openai(model_name, qts)
            pca_path = args.pca_path or os.path.join(save_dir, "pca_384.pkl")
            pca = fit_or_load_pca(embeddings, pca_path)
            reduced = pca.transform(embeddings)
            reduced = l2_normalize(reduced, axis=1)
            for i, emb in enumerate(reduced):
                key = qts_keys[i]
                np.save(os.path.join(save_dir, str(key)[:10] + target_id + ".npy"), emb[np.newaxis, :])
    else:
        save_dir = f"{base_textual}/QueryText-embedding-{model_tag}-Description/{target_id}"
        os.makedirs(save_dir, exist_ok=True)
        qt = f"{des[target_id].loc[0]}"

        if backend == "sentence_transformers":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            emb = embed_with_sentence_transformers(model_name, [qt], 1, device)
            emb = l2_normalize(emb, axis=1)
            np.save(os.path.join(save_dir, target_id + ".npy"), emb[0])
        else:
            embeddings = embed_with_openai(model_name, [qt])
            default_pca_path = (
                f"{base_textual}/QueryText-embedding-{model_tag}-ForecastingPoint/{target_id}/pca_384.pkl"
            )
            pca_path = args.pca_path or default_pca_path
            if not os.path.exists(pca_path):
                raise FileNotFoundError(
                    f"PCA file not found: {pca_path}. Run ForecastingPoint mode first or pass --pca_path."
                )
            pca = fit_or_load_pca(embeddings, pca_path)
            reduced = pca.transform(embeddings)
            reduced = l2_normalize(reduced, axis=1)
            np.save(os.path.join(save_dir, target_id + ".npy"), reduced[0])


if __name__ == "__main__":
    main()
