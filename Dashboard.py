"""
Dashboard.py — Dashboard utama, membaca data real dari submission UMKM
Jalankan: streamlit run Dashboard.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.tree import plot_tree
from data_model import (
    load_model, baca_data, COLORS, EMOJI,
    LABEL_TUMBUH, LABEL_STABIL, LABEL_PERLU_PERHATIAN,
)

st.set_page_config(page_title="Dashboard UMKM",
                   layout="wide", initial_sidebar_state="collapsed")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.hero-wrap{background:linear-gradient(135deg,#1B5E40 0%,#2E7D52 60%,#43A047 100%);
  border-radius:16px;padding:2.8rem 2.5rem 2.2rem;color:#fff;margin-bottom:1.8rem}
.hero-badge{display:inline-block;background:rgba(255,255,255,.18);
  border:1px solid rgba(255,255,255,.35);border-radius:20px;
  padding:4px 14px;font-size:.78rem;margin-bottom:1rem;letter-spacing:.05em}
.hero-title{font-size:2.4rem;font-weight:800;margin:0 0 .4rem;letter-spacing:-.5px}
.hero-sub{font-size:1.05rem;opacity:.88;max-width:640px;margin:0}

.kpi-card{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:1.1rem 1.2rem}
.kpi-label{font-size:.75rem;color:#6b7280;text-transform:uppercase;letter-spacing:.06em}
.kpi-value{font-size:2rem;font-weight:700;color:#111827;line-height:1.1}
.kpi-delta{font-size:.82rem}
.delta-pos{color:#16a34a}.delta-neg{color:#dc2626}.delta-neu{color:#6b7280}

.sec-head{font-size:1.15rem;font-weight:700;color:#111827;
  border-left:4px solid #1B5E40;padding-left:10px;margin:1.6rem 0 .9rem}

.pill{display:inline-flex;align-items:center;gap:6px;padding:5px 14px;
  border-radius:20px;font-size:.88rem;font-weight:600}
.pill-tumbuh{background:#d1fae5;color:#065f46}
.pill-stabil{background:#dbeafe;color:#1e40af}
.pill-perhatian{background:#fee2e2;color:#991b1b}

.summary-tbl{width:100%;border-collapse:collapse;font-size:.9rem}
.summary-tbl th{background:#f9fafb;color:#374151;font-weight:600;
  padding:8px 12px;text-align:left;border-bottom:2px solid #e5e7eb}
.summary-tbl td{padding:8px 12px;border-bottom:1px solid #f3f4f6;color:#111827}
.summary-tbl tr:last-child td{border-bottom:none}

.insight-card{background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;
  padding:1rem 1.2rem;font-size:.92rem;color:#14532d;line-height:1.65}
.empty-state{text-align:center;padding:3rem 1rem;color:#6b7280;font-size:1rem}

div[data-testid="stButton"] button{background:#1B5E40!important;color:#fff!important;
  border:none!important;border-radius:10px!important;font-weight:600!important}
hr.thin{border:none;border-top:1px solid #e5e7eb;margin:1.4rem 0}
</style>
""", unsafe_allow_html=True)

# ── Load ──────────────────────────────────────────────────────────────────────
clf, le, feature_names = load_model()
df = baca_data()   # data real dari UMKM yang sudah submit, kolom: hasil_label, pendapatan, dst.

# Bersihkan tipe data numerik (CSV kadang baca sebagai string)
for col_num in ["pendapatan","margin","burn_rate","transaksi","rating","confidence"]:
    if col_num in df.columns:
        df[col_num] = pd.to_numeric(df[col_num], errors="coerce")
df = df.dropna(subset=["hasil_label"]) if "hasil_label" in df.columns else df

has_data = len(df) > 0
KATEGORI = [LABEL_TUMBUH, LABEL_STABIL, LABEL_PERLU_PERHATIAN]

# ── HERO ──────────────────────────────────────────────────────────────────────
total_str = f"{len(df):,} UMKM" if has_data else "belum ada data"
st.markdown(f"""
<div class="hero-wrap">
  <div class="hero-badge">SISTEM KLASIFIKASI UMKM · DECISION TREE</div>
  <div class="hero-title">Dashboard Analitik UMKM</div>
  <div class="hero-sub">
    Pantau kondisi seluruh UMKM yang telah melakukan klasifikasi secara real-time.
    Setiap UMKM yang mengisi form akan otomatis muncul di dashboard ini.
    Saat ini tercatat <strong>{total_str}</strong> yang telah dianalisis.
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tombol navigasi ────────────────────────────────────────────────────────────
col_nav1, col_nav2, col_nav3 = st.columns([1,1,4])
with col_nav1:
    if st.button("Tambah Klasifikasi UMKM", use_container_width=True):
        st.switch_page("pages/Klasifikasi.py")
with col_nav2:
    if st.button("Refresh Data", use_container_width=True):
        st.rerun()

st.markdown("<hr class='thin'>", unsafe_allow_html=True)

# ── Jika belum ada data ────────────────────────────────────────────────────────
if not has_data:
    st.markdown("""
    <div class="empty-state">
      <div style="font-size:3rem">📭</div>
      <div style="font-size:1.3rem;font-weight:700;color:#374151;margin:.5rem 0">Belum ada data UMKM</div>
      <div>Dashboard akan terisi otomatis setelah UMKM melakukan klasifikasi pertama.</div>
    </div>
    """, unsafe_allow_html=True)
    col_e1, col_e2, col_e3 = st.columns([1,1,1])
    with col_e2:
        if st.button("Mulai Klasifikasi Sekarang →", use_container_width=True):
            st.switch_page("pages/Klasifikasi.py")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# Data sudah ada — tampilkan dashboard
# ═══════════════════════════════════════════════════════════════════════════════

dist          = df["hasil_label"].value_counts()
n_tumbuh      = dist.get(LABEL_TUMBUH, 0)
n_stabil      = dist.get(LABEL_STABIL, 0)
n_perhatian   = dist.get(LABEL_PERLU_PERHATIAN, 0)
total         = len(df)

# ── KPI ───────────────────────────────────────────────────────────────────────
k1,k2,k3,k4,k5 = st.columns(5, gap="small")

def kpi(col, label, value, delta, dtype="neu"):
    col.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-delta delta-{dtype}">{delta}</div>
    </div>""", unsafe_allow_html=True)

kpi(k1, "Total UMKM Terdaftar", f"{total:,}", "data submission nyata")
kpi(k2, "🌿 Berkembang",          f"{n_tumbuh:,}",    f"{n_tumbuh/total*100:.1f}% dari total",    "pos")
kpi(k3, "📊 Stabil",          f"{n_stabil:,}",    f"{n_stabil/total*100:.1f}% dari total",    "neu")
kpi(k4, "⚠️ Mengalami Penurunan", f"{n_perhatian:,}", f"{n_perhatian/total*100:.1f}% dari total", "neg")
avg_conf = df["confidence"].mean()
kpi(k5, "Rata-rata Keyakinan", f"{avg_conf:.1f}%", "confidence model", "pos")

st.markdown("<hr class='thin'>", unsafe_allow_html=True)

# ── ROW 1: Distribusi + Tren waktu ────────────────────────────────────────────
st.markdown('<div class="sec-head">Distribusi & Tren Klasifikasi</div>', unsafe_allow_html=True)

col_pie, col_trend = st.columns([1, 1.8], gap="medium")

with col_pie:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    wedge_c = [COLORS[k] for k in KATEGORI]
    nilai_pie = [n_tumbuh, n_stabil, n_perhatian]
    wedges, texts, autos = ax.pie(
        nilai_pie, labels=KATEGORI, colors=wedge_c, autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor":"white","linewidth":2}, textprops={"fontsize":11}
    )
    for a in autos: a.set_fontsize(10); a.set_fontweight("bold")
    ax.set_title("Proporsi Kategori UMKM", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout(); st.pyplot(fig); plt.close()

with col_trend:
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    trend = df.groupby(["date","hasil_label"]).size().unstack(fill_value=0).reset_index()
    for lbl in KATEGORI:
        if lbl not in trend.columns: trend[lbl] = 0

    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    if len(trend) >= 2:
        for lbl in KATEGORI:
            ax2.plot(trend["date"], trend[lbl], "o-", color=COLORS[lbl],
                     label=lbl, linewidth=2, markersize=6)
        ax2.set_title("Tren Klasifikasi per Hari", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Tanggal"); ax2.set_ylabel("Jumlah UMKM")
        ax2.legend(fontsize=9)
        ax2.spines[["top","right"]].set_visible(False)
        fig2.autofmt_xdate()
    else:
        vals_ = [dist.get(c,0) for c in KATEGORI]
        ax2.bar(KATEGORI, vals_, color=[COLORS[c] for c in KATEGORI], edgecolor="white", width=0.5)
        for i,(c,v) in enumerate(zip(KATEGORI,vals_)):
            ax2.text(i, v+0.05, str(v), ha="center", fontweight="bold")
        ax2.set_title("Distribusi per Kategori (hari ini)", fontsize=12, fontweight="bold")
        ax2.spines[["top","right"]].set_visible(False)

    plt.tight_layout(); st.pyplot(fig2); plt.close()

# ── ROW 2: Rata-rata metrik per kategori ──────────────────────────────────────
st.markdown('<div class="sec-head">Rata-rata Metrik per Kategori (Data Nyata UMKM)</div>',
            unsafe_allow_html=True)

col_tbl, col_bar = st.columns([1.3, 1], gap="medium")

with col_tbl:
    grp = df.groupby("hasil_label").agg(
        Jumlah         =("hasil_label","count"),
        Avg_Pendapatan =("pendapatan","mean"),
        Avg_Margin     =("margin","mean"),
        Avg_Burn       =("burn_rate","mean"),
        Avg_Rating     =("rating","mean"),
        Avg_Confidence =("confidence","mean"),
    ).reindex(KATEGORI).dropna(how="all")

    pills = {
        LABEL_TUMBUH:          '<span class="pill pill-tumbuh">🌿 Berkembang</span>',
        LABEL_STABIL:          '<span class="pill pill-stabil">📊 Stabil</span>',
        LABEL_PERLU_PERHATIAN: '<span class="pill pill-perhatian">⚠️ Mengalami Penurunan</span>',
    }
    rows_html = ""
    for lbl in KATEGORI:
        if lbl not in grp.index or pd.isna(grp.loc[lbl,"Jumlah"]): continue
        r = grp.loc[lbl]
        rows_html += f"""<tr>
            <td>{pills[lbl]}</td>
            <td>{int(r['Jumlah'])}</td>
            <td>Rp {r['Avg_Pendapatan']:,.0f}</td>
            <td>{r['Avg_Margin']:.1f}%</td>
            <td>{r['Avg_Burn']:.3f}</td>
            <td>{r['Avg_Rating']:.2f}</td>
            <td>{r['Avg_Confidence']:.1f}%</td>
        </tr>"""

    st.markdown(f"""
    <table class="summary-tbl">
      <thead><tr>
        <th>Kategori</th><th>Jumlah</th><th>Pendapatan</th>
        <th>Margin</th><th>Burn Rate</th><th>Rating</th><th>Confidence</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>""", unsafe_allow_html=True)

with col_bar:
    lbls_avail = [l for l in KATEGORI if l in grp.index and not pd.isna(grp.loc[l,"Jumlah"])]
    avg_revs   = [grp.loc[l,"Avg_Pendapatan"]/1_000_000 for l in lbls_avail]
    fig3, ax3  = plt.subplots(figsize=(5, 4))
    bars3 = ax3.bar(lbls_avail, avg_revs, color=[COLORS[l] for l in lbls_avail],
                     edgecolor="white", width=0.5)
    for bar, val in zip(bars3, avg_revs):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                 f"Rp {val:.1f}jt", ha="center", fontsize=9.5, fontweight="bold")
    ax3.set_ylabel("Rata-rata Pendapatan (Juta Rp)")
    ax3.set_title("Rata-rata Pendapatan\nper Kategori", fontsize=12, fontweight="bold")
    ax3.spines[["top","right"]].set_visible(False)
    plt.tight_layout(); st.pyplot(fig3); plt.close()

# ── ROW 3: Distribusi fitur ───────────────────────────────────────────────────
st.markdown('<div class="sec-head">Distribusi Fitur UMKM yang Terdaftar</div>',
            unsafe_allow_html=True)

feats = [
    ("pendapatan", "Pendapatan Bulanan", 1_000_000, "Juta Rp"),
    ("margin",     "Net Profit Margin",  1,          "%"),
    ("burn_rate",  "Burn Rate Ratio",    1,          ""),
    ("transaksi",  "Jumlah Transaksi",   1,          ""),
    ("rating",     "Rating Pelanggan",   1,          ""),
]
fig4, axes4 = plt.subplots(1, 5, figsize=(18, 4))
for ax, (col_, title_, div_, unit_) in zip(axes4, feats):
    for lbl in KATEGORI:
        sub = df[df["hasil_label"]==lbl][col_] / div_
        if len(sub): ax.hist(sub, bins=12, alpha=0.55, color=COLORS[lbl],
                              label=lbl, edgecolor="white")
    ax.set_title(title_, fontsize=9.5, fontweight="bold")
    ax.set_xlabel(unit_)
    ax.spines[["top","right"]].set_visible(False)
axes4[0].legend(fontsize=8)
plt.suptitle("Distribusi Setiap Fitur per Kategori UMKM (Data Nyata)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout(); st.pyplot(fig4); plt.close()

# ── ROW 4: Tabel data UMKM ────────────────────────────────────────────────────
st.markdown('<div class="sec-head">Daftar UMKM yang Telah Klasifikasi</div>',
            unsafe_allow_html=True)

df_show = df.copy()
df_show["Pendapatan"] = df_show["pendapatan"].apply(lambda x: f"Rp {x:,.0f}")
df_show["Margin"]     = df_show["margin"].apply(lambda x: f"{x:.2f}%")
df_show["Burn Rate"]  = df_show["burn_rate"].apply(lambda x: f"{x:.3f}")
df_show["Transaksi"]  = df_show["transaksi"].astype(int).astype(str)
df_show["Rating"]     = df_show["rating"].apply(lambda x: f"{x:.2f}")
df_show["Confidence"] = df_show["confidence"].apply(lambda x: f"{x:.1f}%")
df_show["Kategori"]   = df_show["hasil_label"].apply(lambda x: f"{EMOJI.get(x,'')} {x}")

cols_show = ["timestamp","nama_umkm","Kategori","Confidence",
             "Pendapatan","Margin","Burn Rate","Transaksi","Rating","digital_label"]
rename = {"timestamp":"Waktu","nama_umkm":"Nama UMKM","digital_label":"Adopsi Digital"}
df_tbl = df_show[cols_show].rename(columns=rename).iloc[::-1].reset_index(drop=True)

st.dataframe(df_tbl, use_container_width=True, height=320)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Data CSV", csv, "data_umkm.csv",
                   "text/csv", use_container_width=False)

# ── ROW 5: Radar perbandingan & Feature Importance ────────────────────────────
st.markdown('<div class="sec-head">Profil Dimensi & Faktor Penentu Model</div>',
            unsafe_allow_html=True)

col_radar, col_fi = st.columns([1,1], gap="medium")

with col_radar:
    grp_r = df.groupby("hasil_label").agg(
        Margin=("margin","mean"), Burn=("burn_rate","mean"),
        Rating=("rating","mean"), Tx=("transaksi","mean"),
    ).reindex(KATEGORI).dropna(how="all")

    def norm(s): return (s-s.min())/(s.max()-s.min()+1e-9) if s.notna().sum()>1 else s*0+0.5
    nd = pd.DataFrame({
        "Margin": norm(grp_r["Margin"]), "Burn": 1-norm(grp_r["Burn"]),
        "Rating": norm(grp_r["Rating"]), "Tx": norm(grp_r["Tx"]),
    }, index=grp_r.index).dropna()

    if len(nd) > 0:
        cats_r  = ["Margin\n(%)", "Burn Rate\n(inv)", "Rating", "Transaksi\n(norm)"]
        angles  = np.linspace(0, 2*np.pi, len(cats_r), endpoint=False).tolist()
        angles += angles[:1]

        fig5, ax5 = plt.subplots(figsize=(4.5,4.5), subplot_kw={"polar":True})
        for lbl in nd.index:
            vals = nd.loc[lbl].tolist() + [nd.loc[lbl].tolist()[0]]
            ax5.plot(angles, vals, "o-", linewidth=2, color=COLORS[lbl], label=lbl, alpha=0.9)
            ax5.fill(angles, vals, alpha=0.1, color=COLORS[lbl])
        ax5.set_xticks(angles[:-1]); ax5.set_xticklabels(cats_r, fontsize=10)
        ax5.set_yticklabels([]); ax5.set_ylim(0,1)
        ax5.set_title("Profil Dimensi per Kategori\n(data nyata, ternormalisasi)",
                      fontsize=11, fontweight="bold", pad=18)
        ax5.legend(loc="upper right", bbox_to_anchor=(1.35,1.1), fontsize=9)
        plt.tight_layout(); st.pyplot(fig5); plt.close()
    else:
        st.caption("Butuh data dari minimal 2 kategori berbeda untuk menampilkan radar chart.")

with col_fi:
    importances = clf.feature_importances_
    fi_df = pd.DataFrame({"Fitur":[f.replace("_"," ").title() for f in feature_names],
                           "Importance":importances}).sort_values("Importance",ascending=True)
    fi_cols = ["#28a745" if v>=0.20 else "#007bff" if v>=0.10 else "#6c757d"
               for v in fi_df["Importance"]]
    fig6, ax6 = plt.subplots(figsize=(5.5,4))
    bars6 = ax6.barh(fi_df["Fitur"], fi_df["Importance"], color=fi_cols,
                      edgecolor="white", height=0.55)
    for bar, val in zip(bars6, fi_df["Importance"]):
        ax6.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2,
                 f"{val:.3f}", va="center", fontsize=9.5)
    ax6.set_xlabel("Feature Importance Score")
    ax6.set_title("Faktor Penentu Klasifikasi\n(Decision Tree)", fontsize=12, fontweight="bold")
    ax6.spines[["top","right"]].set_visible(False)
    ax6.legend(handles=[
        mpatches.Patch(color="#28a745", label="Sangat penting (≥0.20)"),
        mpatches.Patch(color="#007bff", label="Penting (0.10–0.20)"),
        mpatches.Patch(color="#6c757d", label="Penunjang (<0.10)"),
    ], fontsize=8.5, loc="lower right")
    plt.tight_layout(); st.pyplot(fig6); plt.close()

# ── Insight otomatis ──────────────────────────────────────────────────────────
st.markdown('<div class="sec-head">Insight Otomatis dari Data UMKM</div>',
            unsafe_allow_html=True)

top_feat = feature_names[np.argmax(clf.feature_importances_)].replace("_"," ").title()
pct_t    = n_tumbuh/total*100
pct_p    = n_perhatian/total*100
avg_mt   = df[df["hasil_label"]==LABEL_TUMBUH]["margin"].mean()             if n_tumbuh>0 else 0
avg_mp   = df[df["hasil_label"]==LABEL_PERLU_PERHATIAN]["margin"].mean()    if n_perhatian>0 else 0
avg_bt   = df[df["hasil_label"]==LABEL_TUMBUH]["burn_rate"].mean()          if n_tumbuh>0 else 0
avg_bp   = df[df["hasil_label"]==LABEL_PERLU_PERHATIAN]["burn_rate"].mean() if n_perhatian>0 else 0
latest      = df.iloc[-1]["nama_umkm"]   if len(df) else "-"
latest_lbl  = df.iloc[-1]["hasil_label"] if len(df) else "-"

st.markdown(f"""
<div class="insight-card">
  Dari <strong>{total:,} UMKM</strong> yang telah melakukan klasifikasi, sebanyak
  <strong>{pct_t:.1f}%</strong> masuk kategori Berkembang dan
  <strong>{pct_p:.1f}%</strong> masuk kategori Mengalami Penurunan.<br><br>
  Faktor paling berpengaruh dalam menentukan kondisi usaha adalah
  <strong>{top_feat}</strong>
  (importance: <strong>{max(clf.feature_importances_):.3f}</strong>).<br><br>
  Rata-rata Net Profit Margin UMKM Berkembang: <strong>{avg_mt:.1f}%</strong>
  vs Mengalami Penurunan: <strong>{avg_mp:.1f}%</strong>.
  Burn Rate Berkembang: <strong>{avg_bt:.3f}</strong> vs Mengalami Penurunan: <strong>{avg_bp:.3f}</strong>
  — efisiensi pengeluaran adalah pembeda utama.<br><br>
  UMKM terakhir yang klasifikasi: <strong>{latest}</strong>
  dengan hasil <strong>{EMOJI.get(latest_lbl,'')} {latest_lbl}</strong>.
</div>
""", unsafe_allow_html=True)

# ── Pohon keputusan ───────────────────────────────────────────────────────────
st.markdown('<div class="sec-head">Pohon Keputusan yang Digunakan</div>', unsafe_allow_html=True)
with st.expander("Tampilkan / sembunyikan pohon keputusan", expanded=False):
    fig7, ax7 = plt.subplots(figsize=(22,10))
    plot_tree(clf, feature_names=feature_names, class_names=le.classes_,
              filled=True, rounded=True, fontsize=8, ax=ax7,
              impurity=False, proportion=True)
    plt.tight_layout(); st.pyplot(fig7); plt.close()

# ── CTA ───────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_c1, col_c2, col_c3 = st.columns([1,1.5,1])
with col_c2:
    st.markdown("### Tambahkan data UMKM baru?")
    if st.button("Mulai Klasifikasi UMKM →", use_container_width=True):
        st.switch_page("pages/Klasifikasi.py")

st.markdown("---")
st.markdown("<center><small>Dashboard UMKM · Decision Tree · Data real dari setiap submission UMKM</small></center>",
            unsafe_allow_html=True)
