"""
data_model.py — model Decision Tree + simpan/baca data UMKM via CSV
Label kategori dalam Bahasa Indonesia: Tumbuh, Stabil, Perlu Perhatian
"""
import os, numpy as np, pandas as pd, streamlit as st
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_umkm.csv")

KOLOM_CSV = [
    "timestamp", "nama_umkm",
    "pendapatan", "margin", "burn_rate", "transaksi", "rating", "digital_int", "digital_label",
    "hasil_label", "confidence"
]

# ── Label kategori (Bahasa Indonesia) ───────────────────────────────────────────
LABEL_TUMBUH         = "Berkembang"
LABEL_STABIL          = "Stabil"
LABEL_PERLU_PERHATIAN = "Mengalami Penurunan"

COLORS    = {LABEL_TUMBUH: "#28a745", LABEL_STABIL: "#007bff", LABEL_PERLU_PERHATIAN: "#dc3545"}
BG        = {LABEL_TUMBUH: "#d1fae5", LABEL_STABIL: "#dbeafe",  LABEL_PERLU_PERHATIAN: "#fee2e2"}
TEXT_COL  = {LABEL_TUMBUH: "#065f46", LABEL_STABIL: "#1e40af",  LABEL_PERLU_PERHATIAN: "#991b1b"}
EMOJI     = {LABEL_TUMBUH: "🌿",       LABEL_STABIL: "📊",        LABEL_PERLU_PERHATIAN: "⚠️"}

DIGITAL_MAP = {
    "Tidak / Jarang":               0,
    "Sebagian (satu platform)":     1,
    "Aktif (marketplace + medsos)": 2,
}


@st.cache_resource
def load_model():
    """Latih Decision Tree, return (clf, le, feature_names)."""
    rng = np.random.RandomState(42)
    n   = 600
    rev  = rng.randint(1_000_000, 15_000_000, n).astype(float)
    mar  = rng.uniform(-20, 40, n)
    brn  = rng.uniform(0.5, 1.3, n)
    tx   = rng.randint(20, 300, n).astype(float)
    rat  = rng.uniform(2.0, 5.0, n)
    dig  = rng.choice([0, 1, 2], n).astype(float)

    labels = []
    for i in range(n):
        s  = (35 if mar[i] >= 15 else 18 if mar[i] >= 0 else -10)
        s += (25 if brn[i] < 0.85 else 10 if brn[i] <= 1.0 else -15)
        s += (15 if rev[i] >= 7e6  else  8 if rev[i] >= 4e6  else  2)
        s += (12 if tx[i]  >= 130  else  6 if tx[i]  >= 80   else  1)
        s += ( 8 if rat[i] >= 4.5  else  4 if rat[i] >= 3.5  else  0)
        s += dig[i] * 2
        labels.append(LABEL_TUMBUH if s >= 70 else LABEL_STABIL if s >= 40 else LABEL_PERLU_PERHATIAN)

    X = pd.DataFrame({
        "pendapatan": rev, "margin": np.round(mar, 2),
        "burn_rate":  np.round(brn, 3), "transaksi": tx,
        "rating":     np.round(rat, 2), "digital_int": dig,
    })
    le  = LabelEncoder()
    clf = DecisionTreeClassifier(max_depth=5, min_samples_split=10,
                                  min_samples_leaf=5, random_state=42)
    clf.fit(X, le.fit_transform(labels))
    return clf, le, list(X.columns)


def hitung_net_profit_margin(pendapatan: float, laba_bersih: float) -> float:
    """
    Net Profit Margin (%) = (Laba Bersih / Pendapatan) x 100

    Laba Bersih = Pendapatan - Total Pengeluaran (semua biaya operasional,
    bahan baku, gaji, sewa, dll.)
    """
    if pendapatan == 0:
        return 0.0
    return (laba_bersih / pendapatan) * 100


def hitung_burn_rate_ratio(pendapatan: float, total_pengeluaran: float) -> float:
    """
    Burn Rate Ratio = Total Pengeluaran / Pendapatan

    Rasio < 1.0  -> pengeluaran lebih kecil dari pendapatan (sehat)
    Rasio = 1.0  -> pengeluaran sama dengan pendapatan (impas/BEP)
    Rasio > 1.0  -> pengeluaran melebihi pendapatan (defisit/boncos)
    """
    if pendapatan == 0:
        return 0.0
    return total_pengeluaran / pendapatan


def baca_data() -> pd.DataFrame:
    """Baca data_umkm.csv. Return DataFrame kosong bila belum ada."""
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame(columns=KOLOM_CSV)
    try:
        df = pd.read_csv(DATA_PATH)
        for k in KOLOM_CSV:
            if k not in df.columns:
                df[k] = ""
        return df
    except Exception:
        return pd.DataFrame(columns=KOLOM_CSV)


def simpan_data(nama_umkm, pendapatan, margin, burn_rate, transaksi,
                rating, digital_int, digital_label,
                hasil_label, confidence):
    """Tambahkan satu baris ke CSV."""
    baris = {
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "nama_umkm":    nama_umkm,
        "pendapatan":   pendapatan,
        "margin":       round(margin, 2),
        "burn_rate":    round(burn_rate, 3),
        "transaksi":    transaksi,
        "rating":       round(rating, 2),
        "digital_int":  digital_int,
        "digital_label": digital_label,
        "hasil_label":  hasil_label,
        "confidence":   round(confidence, 2),
    }
    df_lama = baca_data()
    df_baru = pd.concat([df_lama, pd.DataFrame([baris])], ignore_index=True)
    df_baru.to_csv(DATA_PATH, index=False)
