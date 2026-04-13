"""
Flexible Data Pipeline
Supports: CSV, API, Database
Auto-detects data source and runs model comparison
"""

import pandas as pd
import numpy as np
import requests
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# ══════════════════════════════════════════════════
# CONFIG — only change this part
# ══════════════════════════════════════════════════
CONFIG = {
    "source": "api",            # "csv" | "api" | "database"
    "path":   "/Users/mingyancai/Ontario Population/1710000501-eng.csv",
    # Ontario Data Catalogue — chronic disease (active datastore)
    "api_url": "https://data.ontario.ca/api/3/action/datastore_search?resource_id=84f51521-0393-404f-bdf3-d338bc9a66f9&limit=5000",
    "db_url":  "postgresql://user:password@localhost/health_db",
    "db_query": "SELECT * FROM patient_records LIMIT 10000",
    "models":  ["gaussian_copula", "ctgan"],
    "n_records": 5000,
    "output_dir": ".",
}

# ══════════════════════════════════════════════════
# STEP 1: Universal Data Loader
# ══════════════════════════════════════════════════
def load_data(config):
    source = config["source"]
    print(f"\n Loading data from: {source.upper()}")

    if source == "csv":
        path = config["path"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV not found: {path}")
        df = pd.read_csv(path)
        print(f"  CSV loaded: {df.shape[0]} rows x {df.shape[1]} cols")

    elif source == "api":
        url = config["api_url"]
        print(f"  Calling API: {url[:60]}...")
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            # Handle Ontario Data Catalogue API format
            if "result" in data and "records" in data["result"]:
                df = pd.DataFrame(data["result"]["records"])
            else:
                df = pd.DataFrame(data)
            print(f"  API loaded: {df.shape[0]} rows x {df.shape[1]} cols")
        except Exception as e:
            print(f"  API failed ({e}), falling back to synthetic data")
            df = generate_fallback_data(config["n_records"])

    elif source == "database":
        try:
            import sqlalchemy
            engine = sqlalchemy.create_engine(config["db_url"])
            df = pd.read_sql(config["db_query"], engine)
            print(f"  DB loaded: {df.shape[0]} rows x {df.shape[1]} cols")
        except Exception as e:
            print(f"  DB failed ({e}), falling back to synthetic data")
            df = generate_fallback_data(config["n_records"])

    else:
        raise ValueError(f"Unknown source: {source}. Use 'csv', 'api', or 'database'")

    return df


# ══════════════════════════════════════════════════
# STEP 2: Auto Data Cleaner
# ══════════════════════════════════════════════════
def auto_clean(df):
    print(f"\n Auto-cleaning data...")
    original_cols = df.shape[1]

    # Drop ID-like columns
    id_cols = [c for c in df.columns if any(x in c.lower() for x in ['id', '_id', 'index', 'unnamed'])]
    df = df.drop(columns=id_cols, errors='ignore')

    # Drop high-missing columns (>50%)
    missing_pct = df.isnull().mean()
    drop_cols = missing_pct[missing_pct > 0.5].index.tolist()
    df = df.drop(columns=drop_cols, errors='ignore')

    # Encode categoricals
    for col in df.select_dtypes(include='object').columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='raise')
        except:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Fill remaining nulls
    df = df.fillna(df.median(numeric_only=True))

    # Keep only numeric
    df = df.select_dtypes(include=[np.number])

    print(f"  Columns: {original_cols} → {df.shape[1]}")
    print(f"  Rows: {df.shape[0]}")
    return df


# ══════════════════════════════════════════════════
# STEP 3: Run Models (flexible model selection)
# ══════════════════════════════════════════════════
def run_models(df, config):
    results = {}
    n = config["n_records"]

    for model_name in config["models"]:
        print(f"\n Running {model_name.upper()}...")
        try:
            if model_name == "gaussian_copula":
                synthetic = run_gaussian_copula(df, n)
            elif model_name == "ctgan":
                synthetic = run_ctgan(df, n)
            elif model_name == "vine_copula":
                synthetic = run_vine_copula(df, n)
            else:
                print(f"  Unknown model: {model_name}, skipping")
                continue

            fidelity = calculate_fidelity(df, synthetic)
            results[model_name] = {
                "synthetic": synthetic,
                "fidelity": fidelity
            }
            print(f"  {model_name} fidelity: {fidelity:.1f}%")

        except Exception as e:
            print(f"  {model_name} failed: {e}")

    return results


def run_gaussian_copula(df, n):
    try:
        from sdv.single_table import GaussianCopulaSynthesizer
        from sdv.metadata import SingleTableMetadata
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        model = GaussianCopulaSynthesizer(metadata)
    except ImportError:
        from sdv.tabular import GaussianCopula
        model = GaussianCopula()
    model.fit(df)
    return model.sample(n)


def run_ctgan(df, n):
    try:
        from sdv.single_table import CTGANSynthesizer
        from sdv.metadata import SingleTableMetadata
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        model = CTGANSynthesizer(metadata, epochs=100)
    except ImportError:
        from sdv.tabular import CTGAN
        model = CTGAN(epochs=100)
    model.fit(df)
    return model.sample(n)


def run_vine_copula(df, n):
    """Vine Copula via pyvinecopulib"""
    try:
        import pyvinecopulib as pv
        from scipy.stats import norm

        data = df.values.astype(float)
        # Transform to uniform marginals
        u = np.zeros_like(data)
        for i in range(data.shape[1]):
            u[:, i] = (np.argsort(np.argsort(data[:, i])) + 1) / (len(data) + 1)

        cop = pv.Vinecop(data=u)
        u_sim = cop.simulate(n)

        # Transform back
        synthetic = np.zeros_like(u_sim)
        for i in range(data.shape[1]):
            synthetic[:, i] = np.quantile(data[:, i], u_sim[:, i])

        return pd.DataFrame(synthetic, columns=df.columns)
    except ImportError:
        print("  pyvinecopulib not installed, using Gaussian Copula instead")
        return run_gaussian_copula(df, n)


# ══════════════════════════════════════════════════
# STEP 4: Fidelity Score
# ══════════════════════════════════════════════════
def calculate_fidelity(real, synthetic):
    from scipy.stats import ks_2samp

    cols = [c for c in real.columns if c in synthetic.columns]
    ks_scores, corr_scores = [], []

    for col in cols:
        try:
            stat, _ = ks_2samp(real[col].dropna(), synthetic[col].dropna())
            ks_scores.append(1 - stat)
        except:
            pass

    try:
        corr_real = real[cols].corr().values
        corr_syn  = synthetic[cols].corr().values
        mask = ~(np.isnan(corr_real) | np.isnan(corr_syn))
        if mask.sum() > 0:
            corr_scores.append(1 - np.mean(np.abs(corr_real[mask] - corr_syn[mask])))
    except:
        pass

    all_scores = ks_scores + corr_scores
    return np.mean(all_scores) * 100 if all_scores else 0.0


# ══════════════════════════════════════════════════
# STEP 5: Charts
# ══════════════════════════════════════════════════
def generate_charts(real, results, output_dir):
    print("\n Generating charts...")
    model_names = list(results.keys())
    colors = ['#2e5b8e', '#e07060', '#60a060', '#9b59b6']

    # Chart 1: Fidelity comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    fidelities = [results[m]["fidelity"] for m in model_names]
    bars = ax.bar(model_names, fidelities,
                  color=colors[:len(model_names)], width=0.5, alpha=0.85)
    ax.set_ylim(0, 105)
    ax.axhline(90, color='gray', linestyle='--', linewidth=1, label='90% threshold')
    for bar, val in zip(bars, fidelities):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1,
                f'{val:.1f}%', ha='center', fontweight='bold')
    ax.set_ylabel('Fidelity Score (%)')
    ax.set_title('Model Fidelity Comparison', fontsize=13, fontweight='bold')
    ax.legend(); ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/flex_chart1_fidelity.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Chart 2: Distribution comparison (first 4 numeric cols)
    cols = real.select_dtypes(include=np.number).columns[:4]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        ax = axes[i]
        ax.hist(real[col].dropna(), bins=30, alpha=0.5, color='#333333', label='Real', density=True)
        for j, m in enumerate(model_names):
            syn = results[m]["synthetic"]
            if col in syn.columns:
                ax.hist(syn[col].dropna(), bins=30, alpha=0.5,
                        color=colors[j], label=m, density=True)
        ax.set_title(col, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Distribution Comparison: Real vs Synthetic', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/flex_chart2_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Chart 3: Correlation matrix — best model vs real
    best_model = max(results, key=lambda m: results[m]["fidelity"])
    best_syn = results[best_model]["synthetic"]
    shared_cols = [c for c in real.columns if c in best_syn.columns][:12]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, data, title in zip(axes,
                                [real[shared_cols], best_syn[shared_cols],
                                 (real[shared_cols].corr() - best_syn[shared_cols].corr()).abs()],
                                ['Real', f'Synthetic ({best_model})', 'Absolute Difference']):
        if title == 'Absolute Difference':
            corr = data
        else:
            corr = data.corr()
        im = ax.imshow(corr, cmap='RdBu_r' if title != 'Absolute Difference' else 'Reds',
                       vmin=-1 if title != 'Absolute Difference' else 0,
                       vmax=1 if title != 'Absolute Difference' else 0.5)
        ax.set_xticks(range(len(shared_cols)))
        ax.set_yticks(range(len(shared_cols)))
        ax.set_xticklabels(shared_cols, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(shared_cols, fontsize=7)
        ax.set_title(title, fontweight='bold')
        plt.colorbar(im, ax=ax)
    fig.suptitle('Correlation Matrix Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/flex_chart3_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Charts saved to {output_dir}/")


# ══════════════════════════════════════════════════
# Fallback: generate synthetic health data
# ══════════════════════════════════════════════════
def generate_fallback_data(n):
    print("  Generating fallback health dataset...")
    np.random.seed(42)
    age = np.random.normal(55, 18, n).clip(18, 95).astype(int)
    return pd.DataFrame({
        'age': age,
        'income': np.random.lognormal(10.5, 0.6, n).clip(15000, 200000),
        'num_rooms': np.random.randint(3, 9, n),
        'has_diabetes': (age > 50) * np.random.binomial(1, 0.3, n),
        'has_hypertension': (age > 45) * np.random.binomial(1, 0.35, n),
        'has_copd': np.random.binomial(1, 0.08, n),
        'has_heart_disease': (age > 55) * np.random.binomial(1, 0.2, n),
        'chronic_condition_count': np.random.poisson(1.5, n).clip(0, 6),
        'er_visits_12mo': np.random.poisson(0.8, n).clip(0, 10),
        'risk_score': (age / 100 + np.random.normal(0, 0.1, n)).clip(0, 1),
    })


# ══════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 50)
    print("  FLEXIBLE SYNTHETIC DATA PIPELINE")
    print("=" * 50)

    # 1. Load
    df_raw = load_data(CONFIG)

    # 2. Clean
    df_clean = auto_clean(df_raw)

    if df_clean.shape[1] < 2:
        print("  Not enough columns after cleaning, using fallback data")
        df_clean = generate_fallback_data(CONFIG["n_records"])

    # 3. Run models
    results = run_models(df_clean, CONFIG)

    # 4. Charts
    if results:
        generate_charts(df_clean, results, CONFIG["output_dir"])

    # 5. Summary
    print("\n" + "=" * 50)
    print("  RESULTS SUMMARY")
    print("=" * 50)
    for model, res in results.items():
        print(f"  {model:<20} Fidelity: {res['fidelity']:.1f}%")
    print(f"\n  Data source : {CONFIG['source'].upper()}")
    print(f"  Models run  : {', '.join(results.keys())}")
    print(f"  Records     : {CONFIG['n_records']}")
    print("=" * 50)
    print("\n To switch data source, change CONFIG at top of file:")
    print('  "source": "csv"      → local CSV file')
    print('  "source": "api"      → Ontario Data Catalogue API')
    print('  "source": "database" → PostgreSQL / any SQL database')
