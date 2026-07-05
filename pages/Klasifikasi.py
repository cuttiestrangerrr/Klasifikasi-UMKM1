"""
pages/Klasifikasi.py
Sidebar: input langsung (termasuk margin & burn rate)
Halaman: panduan kalkulator sebagai referensi + hasil klasifikasi
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.tree import export_text
from data_model import (
    load_model, simpan_data, COLORS, EMOJI, DIGITAL_MAP,
    LABEL_TUMBUH, LABEL_STABIL, LABEL_PERLU_PERHATIAN,
)

st.set_page_config(page_title="Klasifikasi UMKM",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
body{font-family:'Segoe UI',sans-serif}
.judul{font-size:1.85rem;font-weight:800;color:#1B5E40}
.subjudul{color:#555;font-size:.96rem;margin-bottom:1.2rem}
.kotak-hasil{padding:1.5rem;border-radius:14px;text-align:center;margin-bottom:1rem}
.berkembang-box{background:#d1fae5;border:2px solid #28a745}
.stabil-box{background:#dbeafe;border:2px solid #007bff}
.penurunan-box{background:#fee2e2;border:2px solid #dc3545}
.label-hasil{font-size:2.2rem;font-weight:900;letter-spacing:2px}
.berkembang-label{color:#065f46}.stabil-label{color:#1e40af}.penurunan-label{color:#991b1b}
.kotak-insight{padding:.9rem 1.2rem;border-radius:8px;border-left:4px solid;
               font-size:.92rem;line-height:1.65;margin-bottom:.8rem}
.berkembang-insight{background:#f0fdf4;border-color:#28a745;color:#14532d}
.stabil-insight{background:#eff6ff;border-color:#007bff;color:#1e3a8a}
.penurunan-insight{background:#fff5f5;border-color:#dc3545;color:#7f1d1d}
.baris-faktor{display:flex;justify-content:space-between;align-items:center;
              padding:6px 0;border-bottom:1px solid #f0f0f0;font-size:.9rem}
.bdg-pos{background:#d1fae5;color:#065f46;padding:2px 10px;border-radius:12px;font-size:.78rem;font-weight:600}
.bdg-neu{background:#fef9c3;color:#854d0e;padding:2px 10px;border-radius:12px;font-size:.78rem;font-weight:600}
.bdg-neg{background:#fee2e2;color:#991b1b;padding:2px 10px;border-radius:12px;font-size:.78rem;font-weight:600}
.panduan-wrap{background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:1.1rem 1.3rem}
.rumus-chip{display:inline-block;font-family:monospace;background:#eef2ff;color:#3730a3;
            padding:3px 10px;border-radius:6px;font-size:.88rem;margin:4px 0}
.contoh-box{background:#fff;border:1px solid #d1fae5;border-radius:8px;
            padding:.7rem 1rem;font-size:.85rem;color:#065f46;margin-top:.5rem}
.info-burn{background:#fffbeb;border:1px solid #fde68a;border-radius:8px;
           padding:.6rem 1rem;font-size:.85rem;color:#92400e;margin-top:.5rem}
</style>
""", unsafe_allow_html=True)

clf, le, fitur = load_model()

# ── Sidebar — persis seperti versi sebelumnya ───────────────────────────────────
with st.sidebar:
    st.markdown("## Data UMKM Kamu")
    st.divider()
    nama = st.text_input("Nama UMKM *", placeholder="Contoh: Warung Bu Sari")

    st.markdown("#### Finansial")
    pendapatan = st.number_input("Pendapatan Bulanan (Rp)", 0, 100_000_000, 6_000_000, 100_000)
    margin     = st.number_input("Net Profit Margin (%)", -50.0, 100.0, 10.0, 0.1,
                                  help="(Laba Bersih ÷ Pendapatan) × 100. Lihat panduan di halaman utama.")
    burn_rate  = st.number_input("Burn Rate Ratio", 0.0, 3.0, 0.90, 0.001, format="%.3f",
                                  help="Total Pengeluaran ÷ Pendapatan. Lihat panduan di halaman utama.")

    st.markdown("#### Operasional")
    transaksi  = st.number_input("Jumlah Transaksi / Bulan", 0, 2000, 100, 1)

    st.markdown("#### Pelanggan & Digital")
    rating     = st.number_input("Rata-rata Rating", 1.0, 5.0, 4.2, 0.01, format="%.2f")
    digital_lb = st.selectbox("Platform Digital", list(DIGITAL_MAP.keys()), index=1)

    st.divider()
    tombol = st.button("Analisis Sekarang", type="primary", use_container_width=True)
    if st.button("Dashboard", use_container_width=True):
        st.switch_page("Dashboard.py")

# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="judul">Klasifikasi Kondisi Usaha UMKM</div>', unsafe_allow_html=True)
st.markdown('<div class="subjudul">Masukkan data usahamu di sidebar, lalu klik <b>Analisis Sekarang</b>. '
            'Belum tahu cara menghitung Net Profit Margin & Burn Rate? Lihat panduan di bawah ini.</div>',
            unsafe_allow_html=True)

# ── Panduan Kalkulator (expander, tidak wajib dibuka) ───────────────────────────
with st.expander("Panduan: Cara Menghitung Net Profit Margin & Burn Rate Ratio", expanded=False):
    st.markdown('<div class="panduan-wrap">', unsafe_allow_html=True)

    col_p1, col_p2 = st.columns(2, gap="large")

    with col_p1:
        st.markdown("### Net Profit Margin (%)")
        st.markdown("""
Mengukur seberapa besar **keuntungan bersih** yang dihasilkan dari setiap rupiah pendapatan.
""")
        st.markdown('<div class="rumus-chip">Net Profit Margin (%) = (Laba Bersih ÷ Pendapatan) × 100</div>',
                    unsafe_allow_html=True)
        st.markdown("""
**Komponen:**
- **Pendapatan** = total omzet / pemasukan kotor sebulan
- **Laba Bersih** = Pendapatan − semua pengeluaran (bahan baku, gaji, sewa, listrik, transportasi, dll.)
- Hasilnya **boleh negatif** jika pengeluaran melebihi pendapatan (rugi)

**Interpretasi:**
| Nilai | Kondisi |
|-------|---------|
| ≥ 15% | 🌿 Sangat baik |
| 0% – 15% | 📊 Cukup |
| < 0% | ⚠️ Merugi |
""")
        st.markdown("""
<div class="contoh-box">
<b>Contoh:</b><br>
Pendapatan = Rp 6.000.000<br>
Total pengeluaran = Rp 4.500.000<br>
Laba Bersih = 6.000.000 − 4.500.000 = <b>Rp 1.500.000</b><br>
Net Profit Margin = (1.500.000 ÷ 6.000.000) × 100 = <b>25%</b>
</div>
""", unsafe_allow_html=True)

    with col_p2:
        st.markdown("### Burn Rate Ratio")
        st.markdown("""
Mengukur seberapa besar **proporsi pengeluaran** terhadap pendapatan — seberapa cepat uangmu "terbakar".
""")
        st.markdown('<div class="rumus-chip">Burn Rate Ratio = Total Pengeluaran ÷ Pendapatan</div>',
                    unsafe_allow_html=True)
        st.markdown("""
**Komponen:**
- **Total Pengeluaran** = semua biaya operasional bulan ini (bahan baku, gaji, sewa, listrik, dll.)
- **Pendapatan** = total omzet bulan ini
- Hasilnya adalah **angka desimal** (bukan persen)

**Interpretasi:**
| Nilai | Kondisi |
|-------|---------|
| < 0.85 | 🌿 Efisien — pengeluaran jauh di bawah pendapatan |
| 0.85 – 1.0 | 📊 Hati-hati — mendekati batas impas |
| > 1.0 | ⚠️ Defisit — pengeluaran melebihi pendapatan |
""")
        st.markdown("""
<div class="contoh-box">
<b>Contoh:</b><br>
Pendapatan = Rp 6.000.000<br>
Total pengeluaran = Rp 4.500.000<br>
Burn Rate Ratio = 4.500.000 ÷ 6.000.000 = <b>0.750</b> ✅ Efisien
</div>
<div class="info-burn" style="margin-top:.6rem">
💡 <b>Tips:</b> Burn Rate Ratio dan Net Profit Margin berasal dari data yang sama —
jika kamu sudah punya angka pengeluaran dan pendapatan, keduanya bisa dihitung sekaligus.
</div>
""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ── Tampilkan panduan kategori jika belum analisis ─────────────────────────────
if not tombol:
    st.info("Isi semua data di sidebar kiri, lalu klik **Analisis Sekarang**.")
    a, b, c = st.columns(3)
    a.success(f"**{EMOJI[LABEL_TUMBUH]} BERKEMBANG**\n\nProfit tinggi, biaya efisien, transaksi aktif. Saatnya ekspansi!")
    b.info(f"**{EMOJI[LABEL_STABIL]} STABIL**\n\nUsaha berjalan baik. Optimalkan efisiensi untuk naik ke kategori Berkembang.")
    c.error(f"**{EMOJI[LABEL_PERLU_PERHATIAN]} MENGALAMI PENURUNAN**\n\nAda tantangan finansial. Fokus perbaiki arus kas segera.")
    st.stop()

# ── Validasi ────────────────────────────────────────────────────────────────────
if not nama.strip():
    st.warning("⚠️ Mohon isi **Nama UMKM** di sidebar sebelum analisis.")
    st.stop()

# ── Prediksi ────────────────────────────────────────────────────────────────────
dig_int    = DIGITAL_MAP[digital_lb]
X_in       = np.array([[pendapatan, margin, burn_rate, transaksi, rating, dig_int]])
pred_enc   = clf.predict(X_in)[0]
pred_proba = clf.predict_proba(X_in)[0]
pred_label = le.inverse_transform([pred_enc])[0]
conf       = pred_proba[pred_enc] * 100
proba_dict = {le.classes_[i]: pred_proba[i] * 100 for i in range(len(le.classes_))}

# ── Simpan ke CSV ───────────────────────────────────────────────────────────────
simpan_data(nama.strip(), pendapatan, margin, burn_rate,
            transaksi, rating, dig_int, digital_lb, pred_label, conf)
st.success(f"Data **{nama.strip()}** berhasil dicatat — akan muncul di Dashboard!")

# ── Mapping style ────────────────────────────────────────────────────────────────
box_cls = {
    LABEL_TUMBUH:          "berkembang-box",
    LABEL_STABIL:          "stabil-box",
    LABEL_PERLU_PERHATIAN: "penurunan-box",
}
lbl_cls = {
    LABEL_TUMBUH:          "berkembang-label",
    LABEL_STABIL:          "stabil-label",
    LABEL_PERLU_PERHATIAN: "penurunan-label",
}
ins_cls = {
    LABEL_TUMBUH:          "berkembang-insight",
    LABEL_STABIL:          "stabil-insight",
    LABEL_PERLU_PERHATIAN: "penurunan-insight",
}

insights = {
    LABEL_TUMBUH:(
        "Usahamu menunjukkan performa yang sangat baik! Profit margin tinggi dan burn rate efisien "
        "adalah kombinasi ideal untuk terus berkembang. Pertahankan konsistensi dan pertimbangkan ekspansi."
    ),
    LABEL_STABIL:(
        "Kondisi usahamu stabil dan berjalan dengan baik, namun masih ada ruang untuk tumbuh. "
        "Tingkatkan efisiensi operasional dan perkuat kehadiran digital untuk mendorong lebih banyak transaksi."
    ),
    LABEL_PERLU_PERHATIAN:(
        "Usahamu sedang mengalami penurunan yang perlu segera ditangani. Prioritaskan perbaikan arus kas, "
        "kurangi pengeluaran tidak produktif, dan cari peluang pendapatan baru sesegera mungkin."
    ),
}
recs = {
    LABEL_TUMBUH:[
        "Ekspansi produk atau segmen pelanggan baru untuk menjaga momentum.",
        "Investasikan profit untuk otomasi & peningkatan kapasitas produksi.",
        "Bangun program loyalitas berbasis data transaksi pelanggan.",
        "Monitor KPI rutin agar performa terjaga saat skala usaha membesar.",
    ],
    LABEL_STABIL:[
        "Audit pengeluaran — temukan celah efisiensi yang bisa diperbaiki.",
        "Tingkatkan rating lewat peningkatan kualitas produk & pelayanan.",
        "Perluas jangkauan ke marketplace populer (Tokopedia, Shopee, dll.).",
        "Konsisten di media sosial untuk membangun brand awareness.",
    ],
    LABEL_PERLU_PERHATIAN:[
        "Segera kurangi biaya tidak esensial dan perbaiki arus kas.",
        "Reaktivasi pelanggan lama dengan promosi atau diskon khusus.",
        "Manfaatkan program UMKM pemerintah untuk akses modal & pendampingan.",
        "Diversifikasi produk berdasarkan kebutuhan dan feedback pelanggan.",
    ],
}

# ── Layout hasil ─────────────────────────────────────────────────────────────────
kiri, kanan = st.columns([1, 2], gap="large")

with kiri:
    st.markdown(f"""
    <div class="kotak-hasil {box_cls[pred_label]}">
        <div style="font-size:3rem">{EMOJI[pred_label]}</div>
        <div class="label-hasil {lbl_cls[pred_label]}">{pred_label.upper()}</div>
        <div style="font-size:.85rem;color:#666;margin-top:6px">Tingkat keyakinan model</div>
        <div style="font-size:2rem;font-weight:800;color:#111">{conf:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### Distribusi Probabilitas")
    for lb in [LABEL_TUMBUH, LABEL_STABIL, LABEL_PERLU_PERHATIAN]:
        pct = proba_dict.get(lb, 0)
        st.markdown(f"**{EMOJI[lb]} {lb}** `{pct:.1f}%`")
        st.progress(pct / 100)

    st.divider()
    st.markdown("##### Ringkasan Input")
    rows_in = {
        "Nama UMKM":     nama.strip(),
        "Pendapatan":    f"Rp {pendapatan:,.0f}",
        "Profit Margin": f"{margin:.2f}%",
        "Burn Rate":     f"{burn_rate:.3f}",
        "Transaksi":     f"{transaksi}/bln",
        "Rating":        f"{rating:.2f}/5",
        "Digital":       digital_lb,
    }
    for k, v in rows_in.items():
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;padding:4px 0;"
            f"border-bottom:1px solid #f0f0f0;font-size:.88rem'>"
            f"<span style='color:#6b7280'>{k}</span><b>{v}</b></div>",
            unsafe_allow_html=True)

with kanan:
    st.markdown("##### Analisis Faktor (diurutkan dari paling berpengaruh)")

    def bdg(impact):
        if impact=="pos": return "bdg-pos","▲ Baik"
        if impact=="neg": return "bdg-neg","▼ Perlu perbaikan"
        return "bdg-neu","→ Cukup"

    faktor_list = [
        {"nama":"Net Profit Margin","nilai":f"{margin:.2f}%",
         "impact":"pos" if margin>=15 else "neu" if margin>=0 else "neg",
         "ket":"Sangat baik (≥15%)" if margin>=15 else "Cukup (0–15%)" if margin>=0 else "Merugi (<0%)",
         "imp": clf.feature_importances_[fitur.index("margin")]},
        {"nama":"Burn Rate Ratio","nilai":f"{burn_rate:.3f}",
         "impact":"pos" if burn_rate<0.85 else "neu" if burn_rate<=1.0 else "neg",
         "ket":"Efisien (<0.85)" if burn_rate<0.85 else "Hati-hati (0.85–1.0)" if burn_rate<=1.0 else "Boros (>1.0)",
         "imp": clf.feature_importances_[fitur.index("burn_rate")]},
        {"nama":"Pendapatan Bulanan","nilai":f"Rp {pendapatan:,.0f}",
         "impact":"pos" if pendapatan>=7e6 else "neu" if pendapatan>=4e6 else "neg",
         "ket":"Tinggi (≥7jt)" if pendapatan>=7e6 else "Menengah (4–7jt)" if pendapatan>=4e6 else "Rendah (<4jt)",
         "imp": clf.feature_importances_[fitur.index("pendapatan")]},
        {"nama":"Volume Transaksi","nilai":f"{transaksi}/bln",
         "impact":"pos" if transaksi>=130 else "neu" if transaksi>=80 else "neg",
         "ket":"Aktif (≥130)" if transaksi>=130 else "Sedang (80–130)" if transaksi>=80 else "Rendah (<80)",
         "imp": clf.feature_importances_[fitur.index("transaksi")]},
        {"nama":"Rating Pelanggan","nilai":f"{rating:.2f}/5",
         "impact":"pos" if rating>=4.5 else "neu" if rating>=3.5 else "neg",
         "ket":"Sangat baik (≥4.5)" if rating>=4.5 else "Baik (3.5–4.5)" if rating>=3.5 else "Perlu ditingkatkan",
         "imp": clf.feature_importances_[fitur.index("rating")]},
        {"nama":"Adopsi Digital","nilai":digital_lb,
         "impact":"pos" if dig_int==2 else "neu" if dig_int==1 else "neg",
         "ket":"Multi-platform" if dig_int==2 else "Satu platform" if dig_int==1 else "Belum digital",
         "imp": clf.feature_importances_[fitur.index("digital_int")]},
    ]
    faktor_list.sort(key=lambda x: x["imp"], reverse=True)

    for i, fa in enumerate(faktor_list, 1):
        bc, bt = bdg(fa["impact"])
        st.markdown(f"""
        <div class="baris-faktor">
          <span><b>{i}. {fa['nama']}</b>
            <span style="color:#9ca3af;font-size:.78rem"> imp:{fa['imp']:.3f}</span><br>
            <small style="color:#888">{fa['nilai']} — {fa['ket']}</small>
          </span>
          <span class="{bc}">{bt}</span>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("##### Insight")
    st.markdown(f'<div class="kotak-insight {ins_cls[pred_label]}">{insights[pred_label]}</div>',
                unsafe_allow_html=True)
    st.markdown("##### Rekomendasi")
    for r in recs[pred_label]: st.markdown(f"- {r}")

# ── Visualisasi ──────────────────────────────────────────────────────────────────
st.divider()
st.markdown("### Visualisasi Hasil")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

cats = [LABEL_TUMBUH, LABEL_STABIL, LABEL_PERLU_PERHATIAN]
vals = [proba_dict[c] for c in cats]
bars = ax1.bar(cats, vals, color=[COLORS[c] for c in cats], edgecolor="white", width=.5)
for bar, val, lb in zip(bars, vals, cats):
    bar.set_alpha(1.0 if lb==pred_label else .3)
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.8,
             f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
ax1.set_ylim(0, 115); ax1.set_ylabel("Probabilitas (%)")
ax1.set_title("Distribusi Probabilitas Kelas", fontsize=12, fontweight="bold")
ax1.spines[["top","right"]].set_visible(False)

fi_names = [f["nama"] for f in faktor_list]
fi_vals  = [f["imp"]  for f in faktor_list]
fi_cols  = [COLORS[LABEL_TUMBUH] if f["impact"]=="pos"
            else COLORS[LABEL_PERLU_PERHATIAN] if f["impact"]=="neg"
            else "#ffc107" for f in faktor_list]
ax2.barh(fi_names[::-1], fi_vals[::-1], color=fi_cols[::-1], edgecolor="white", height=.55)
for i, v in enumerate(fi_vals[::-1]):
    ax2.text(v+.003, i, f"{v:.3f}", va="center", fontsize=9.5)
ax2.set_xlabel("Feature Importance")
ax2.set_title("Pengaruh Faktor", fontsize=12, fontweight="bold")
ax2.spines[["top","right"]].set_visible(False)
ax2.legend(handles=[
    mpatches.Patch(color=COLORS[LABEL_TUMBUH],          label="Kondisi Baik"),
    mpatches.Patch(color="#ffc107",                      label="Kondisi Cukup"),
    mpatches.Patch(color=COLORS[LABEL_PERLU_PERHATIAN], label="Perlu Perbaikan"),
], fontsize=8.5, loc="lower right")

plt.tight_layout(); st.pyplot(fig); plt.close()

with st.expander("Aturan Pohon Keputusan"):
    st.code(export_text(clf, feature_names=fitur, max_depth=5))

st.divider()
ca, cb = st.columns(2)
with ca:
    if st.button("Analisis UMKM Lain", use_container_width=True): st.rerun()
with cb:
    if st.button("Lihat Dashboard", use_container_width=True):
        st.switch_page("Dashboard.py")

st.markdown("---")
st.markdown("<center><small>Klasifikasi UMKM · Decision Tree · Data tersimpan otomatis ke dashboard</small></center>",
            unsafe_allow_html=True)
