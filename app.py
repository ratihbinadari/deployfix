"""
================================================================================
 SISTEM PREDIKSI TINGKAT KEPARAHAN CEDERA KECELAKAAN LALU LINTAS
 Deployment Model Machine Learning - Tema Jasa Raharja
================================================================================
Cara menjalankan:
    streamlit run app.py

Pastikan file model "injury_severity_model.pkl" berada satu folder dengan
file ini (atau ubah MODEL_PATH di bawah).
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# ==============================================================================
# KONFIGURASI
# ==============================================================================
MODEL_PATH = "injury_severity_model.pkl"

FEATURE_COLS = [
    "usia",
    "usia_kendaraan_tahun",
    "jumlah_kendaraan_terlibat",
    "provinsi",
    "gender",
    "jenis_kecelakaan",
    "jenis_kendaraan",
    "jenis_klaim",
]

# Daftar kategori di bawah ini diambil langsung dari kategori yang dipelajari
# OneHotEncoder pada model (injury_severity_model.pkl) agar konsisten dengan
# data yang digunakan saat training.
PROVINSI_LIST = [
    "Aceh", "Bali", "Bangka Belitung", "Banten", "Bengkulu", "DI Yogyakarta",
    "DKI Jakarta", "Gorontalo", "Jambi", "Jawa Barat", "Jawa Tengah",
    "Jawa Timur", "Kalimantan Barat", "Kalimantan Selatan",
    "Kalimantan Tengah", "Kalimantan Timur", "Kalimantan Utara",
    "Kepulauan Riau", "Lampung", "Maluku", "Maluku Utara",
    "Nusa Tenggara Barat", "Nusa Tenggara Timur", "Papua", "Papua Barat",
    "Riau", "Sulawesi Barat", "Sulawesi Selatan", "Sulawesi Tengah",
    "Sulawesi Tenggara", "Sulawesi Utara", "Sumatera Barat",
    "Sumatera Selatan", "Sumatera Utara", "Yogyakarta",
]

GENDER_LIST = ["Laki-laki", "Perempuan"]

JENIS_KECELAKAAN_LIST = [
    "Lalu Lintas Jalan",
    "Penumpang Angkutan Umum",
    "Other",
]

JENIS_KENDARAAN_LIST = [
    "Sepeda Motor",
    "Mobil Penumpang",
    "Angkutan Umum/Bus",
    "Truk/Angkutan Barang",
    "Lainnya",
    "Other",
]

JENIS_KLAIM_LIST = [
    "Lalu Lintas Jalan",
    "Penumpang Angkutan Umum",
    "Lainnya",
]

# Nilai default numerik (median dari data training, sesuai imputer model)
DEFAULT_USIA = 40
DEFAULT_USIA_KENDARAAN = 12
DEFAULT_JUMLAH_KENDARAAN = 2

# ==============================================================================
# KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    page_title="Prediksi Keparahan Cedera | Jasa Raharja",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# CUSTOM CSS - TEMA JASA RAHARJA (Navy & Oranye)
# ==============================================================================
st.markdown(
    """
    <style>
        :root {
            --jr-navy: #00234B;
            --jr-navy-light: #0B3D75;
            --jr-orange: #F26E22;
            --jr-orange-light: #FF8C42;
            --jr-grey: #F4F6F9;
        }

        html, body, [class*="css"] {
            font-family: 'Poppins', 'Helvetica Neue', Poppins, sans-serif;
        }

        .stApp {
            background-color: var(--jr-grey);
        }

        /* Header banner */
        .jr-header {
            background: linear-gradient(90deg, var(--jr-navy) 0%, var(--jr-navy-light) 100%);
            padding: 28px 32px;
            border-radius: 14px;
            margin-bottom: 24px;
            border-left: 8px solid var(--jr-orange);
            box-shadow: 0 4px 14px rgba(0,0,0,0.15);
        }
        .jr-header h1 {
            color: #FFFFFF;
            font-size: 30px;
            font-weight: 800;
            margin: 0;
        }
        .jr-header p {
            color: #D8E3F0;
            font-size: 15px;
            margin-top: 6px;
            margin-bottom: 0;
        }
        .jr-badge {
            display: inline-block;
            background-color: var(--jr-orange);
            color: white;
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
        }

        /* Section card */
        .jr-card {
            background-color: #FFFFFF;
            border-radius: 14px;
            padding: 24px 26px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.06);
            border: 1px solid #E4E9F0;
            margin-bottom: 20px;
        }
        .jr-card h3 {
            color: var(--jr-navy);
            font-weight: 700;
            border-bottom: 3px solid var(--jr-orange);
            display: inline-block;
            padding-bottom: 4px;
            margin-bottom: 18px;
        }

        /* Buttons */
        div.stButton > button, div.stDownloadButton > button {
            background-color: var(--jr-orange);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 26px;
            font-weight: 700;
            font-size: 15px;
            transition: 0.2s;
            width: 100%;
        }
        div.stButton > button:hover, div.stDownloadButton > button:hover {
            background-color: var(--jr-orange-light);
            color: white;
            transform: translateY(-1px);
        }

        /* Result boxes */
        .result-berat {
            background-color: #FDEDEC;
            border: 2px solid #E74C3C;
            border-radius: 14px;
            padding: 26px;
            text-align: center;
        }
        .result-ringan {
            background-color: #EAFAF1;
            border: 2px solid #27AE60;
            border-radius: 14px;
            padding: 26px;
            text-align: center;
        }
        .result-title {
            font-size: 14px;
            letter-spacing: 1px;
            color: #555;
            font-weight: 600;
            text-transform: uppercase;
        }
        .result-value {
            font-size: 34px;
            font-weight: 800;
            margin: 8px 0;
        }
        .result-berat .result-value { color: #C0392B; }
        .result-ringan .result-value { color: #1E8449; }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: var(--jr-navy);
        }
        section[data-testid="stSidebar"] * {
            color: #F1F1F1 !important;
        }
        section[data-testid="stSidebar"] hr {
            border-color: rgba(255,255,255,0.2);
        }

        footer {visibility: hidden;}
        .jr-footer {
            text-align: center;
            color: #8A94A6;
            font-size: 12px;
            margin-top: 30px;
            padding-top: 14px;
            border-top: 1px solid #E4E9F0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# LOAD MODEL
# ==============================================================================
@st.cache_resource(show_spinner="Memuat model prediksi...")
def load_model(model_path: str):
    if not os.path.exists(model_path):
        return None, None

    # Beberapa versi scikit-learn menyimpan atribut internal ColumnTransformer
    # secara berbeda (mis. `_RemainderColsList`). Jika versi scikit-learn di
    # environment deployment berbeda dari versi saat model dilatih, tambahkan
    # kelas kompatibilitas ini agar file pickle tetap bisa dimuat.
    try:
        import sklearn.compose._column_transformer as _ct
        if not hasattr(_ct, "_RemainderColsList"):
            class _RemainderColsList(list):
                pass
            _ct._RemainderColsList = _RemainderColsList
    except Exception:
        pass

    try:
        model = joblib.load(model_path)
        return model, None
    except Exception as e:
        return None, str(e)


def predict_severity(new_data: pd.DataFrame, pipe) -> pd.DataFrame:
    """
    Fungsi deployment: menerima data laporan awal kecelakaan
    (kolom sesuai `FEATURE_COLS`) dan mengembalikan prediksi tingkat
    keparahan beserta probabilitasnya.
    """
    proba_berat = pipe.predict_proba(new_data[FEATURE_COLS])[:, 1]
    pred = pipe.predict(new_data[FEATURE_COLS])

    result = new_data.copy()
    result["prob_luka_berat"] = proba_berat
    result["prediksi_tingkat_cedera"] = np.where(pred == 1, "Luka Berat", "Luka Ringan")
    return result


model, load_error = load_model(MODEL_PATH)

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("### 🚦 Jasa Raharja")
    st.caption("Sistem Pendukung Keputusan Klaim")
    st.markdown("---")
    st.markdown("#### Menu")
    menu = st.radio(
        label="Pilih mode",
        options=["🧍 Prediksi Individu", "📂 Prediksi Massal (CSV)", "ℹ️ Tentang Model"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    if model is not None:
        st.success("Model berhasil dimuat ✔")
    elif load_error:
        st.error("Model gagal dimuat")
        st.caption("Kemungkinan versi scikit-learn berbeda dengan saat training (scikit-learn 1.6.1).")
    else:
        st.error("Model belum ditemukan")
        st.caption(f"Letakkan file **{MODEL_PATH}** satu folder dengan app.py")
    st.markdown("---")
    st.caption(f"© {datetime.now().year} — Aplikasi internal bantu keputusan awal.")
    st.caption("Bukan pengganti keputusan medis/asesmen resmi.")

# ==============================================================================
# HEADER
# ==============================================================================
st.markdown(
    """
    <div class="jr-header">
        <div class="jr-badge">PT JASA RAHARJA (PERSERO)</div>
        <h1>🚦 Prediksi Tingkat Keparahan Cedera Kecelakaan</h1>
        <p>Estimasi awal tingkat keparahan cedera korban berdasarkan data laporan kecelakaan,
        untuk mendukung proses triase dan penanganan klaim yang lebih cepat.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None:
    if load_error:
        st.error(
            f"⚠️ Gagal memuat model **{MODEL_PATH}**.\n\n"
            f"Detail error: `{load_error}`\n\n"
            "Ini sering terjadi karena versi **scikit-learn** di environment deployment berbeda "
            "dari versi saat model dilatih (**scikit-learn 1.6.1**). "
            "Coba jalankan: `pip install scikit-learn==1.6.1` lalu ulangi."
        )
    else:
        st.warning(
            f"⚠️ File model **{MODEL_PATH}** tidak ditemukan di direktori aplikasi. "
            "Silakan letakkan file pickle model di folder yang sama dengan `app.py`, lalu jalankan ulang."
        )
    st.stop()

# ==============================================================================
# MODE 1: PREDIKSI INDIVIDU
# ==============================================================================
if menu == "🧍 Prediksi Individu":
    st.markdown('<div class="jr-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Data Laporan Awal Kecelakaan")

    col1, col2, col3 = st.columns(3)

    with col1:
        usia = st.number_input(
            "Usia Korban (tahun)", min_value=0, max_value=120, value=DEFAULT_USIA, step=1
        )
        provinsi = st.selectbox("Provinsi Kejadian", PROVINSI_LIST, index=PROVINSI_LIST.index("Jawa Timur"))
        jenis_kecelakaan = st.selectbox("Jenis Kecelakaan", JENIS_KECELAKAAN_LIST)

    with col2:
        usia_kendaraan_tahun = st.number_input(
            "Usia Kendaraan (tahun)", min_value=0, max_value=60, value=DEFAULT_USIA_KENDARAAN, step=1
        )
        gender = st.selectbox("Jenis Kelamin", GENDER_LIST)
        jenis_kendaraan = st.selectbox("Jenis Kendaraan", JENIS_KENDARAAN_LIST)

    with col3:
        jumlah_kendaraan_terlibat = st.number_input(
            "Jumlah Kendaraan Terlibat", min_value=1, max_value=20, value=DEFAULT_JUMLAH_KENDARAAN, step=1
        )
        jenis_klaim = st.selectbox("Jenis Klaim", JENIS_KLAIM_LIST)

    st.markdown("</div>", unsafe_allow_html=True)

    predict_clicked = st.button("🔍 Prediksi Tingkat Keparahan", use_container_width=True)

    if predict_clicked:
        input_df = pd.DataFrame(
            [{
                "usia": usia,
                "usia_kendaraan_tahun": usia_kendaraan_tahun,
                "jumlah_kendaraan_terlibat": jumlah_kendaraan_terlibat,
                "provinsi": provinsi,
                "gender": gender,
                "jenis_kecelakaan": jenis_kecelakaan,
                "jenis_kendaraan": jenis_kendaraan,
                "jenis_klaim": jenis_klaim,
            }]
        )

        try:
            hasil = predict_severity(input_df, model)
            pred_label = hasil.loc[0, "prediksi_tingkat_cedera"]
            prob_berat = hasil.loc[0, "prob_luka_berat"]

            st.markdown("### 🎯 Hasil Prediksi")
            colA, colB = st.columns([1, 1.4])

            with colA:
                box_class = "result-berat" if pred_label == "Luka Berat" else "result-ringan"
                icon = "🔴" if pred_label == "Luka Berat" else "🟢"
                st.markdown(
                    f"""
                    <div class="{box_class}">
                        <div class="result-title">Prediksi Tingkat Cedera</div>
                        <div class="result-value">{icon} {pred_label}</div>
                        <div class="result-title">Probabilitas Luka Berat</div>
                        <div class="result-value" style="font-size:24px;">{prob_berat*100:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with colB:
                st.markdown("**Tingkat probabilitas luka berat**")
                st.progress(min(max(float(prob_berat), 0.0), 1.0))
                if pred_label == "Luka Berat":
                    st.error(
                        "Model mengindikasikan risiko cedera **berat**. "
                        "Disarankan prioritas penanganan dan verifikasi lanjutan oleh tim asesmen."
                    )
                else:
                    st.success(
                        "Model mengindikasikan risiko cedera **ringan**. "
                        "Tetap lakukan verifikasi standar sesuai prosedur klaim."
                    )
                st.caption(
                    "Catatan: hasil ini merupakan estimasi awal berbasis data historis dan "
                    "tidak menggantikan asesmen medis maupun keputusan klaim resmi."
                )

            with st.expander("Lihat detail data input & output model"):
                st.dataframe(hasil, use_container_width=True)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

# ==============================================================================
# MODE 2: PREDIKSI MASSAL (CSV)
# ==============================================================================
elif menu == "📂 Prediksi Massal (CSV)":
    st.markdown('<div class="jr-card">', unsafe_allow_html=True)
    st.markdown("### 📂 Prediksi Massal dari File CSV")
    st.write(
        "Unggah file CSV berisi data laporan kecelakaan dengan kolom berikut (nama harus sama persis):"
    )
    st.code(", ".join(FEATURE_COLS), language="text")

    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            missing_cols = [c for c in FEATURE_COLS if c not in df_input.columns]

            if missing_cols:
                st.error(f"Kolom berikut tidak ditemukan pada file: {', '.join(missing_cols)}")
            else:
                with st.spinner("Menjalankan prediksi..."):
                    hasil_batch = predict_severity(df_input, model)

                st.markdown('<div class="jr-card">', unsafe_allow_html=True)
                st.markdown("### ✅ Hasil Prediksi Massal")

                total = len(hasil_batch)
                jumlah_berat = int((hasil_batch["prediksi_tingkat_cedera"] == "Luka Berat").sum())
                jumlah_ringan = total - jumlah_berat

                m1, m2, m3 = st.columns(3)
                m1.metric("Total Data", f"{total:,}")
                m2.metric("Prediksi Luka Berat", f"{jumlah_berat:,}")
                m3.metric("Prediksi Luka Ringan", f"{jumlah_ringan:,}")

                st.dataframe(hasil_batch, use_container_width=True)

                csv_out = hasil_batch.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Unduh Hasil Prediksi (CSV)",
                    data=csv_out,
                    file_name="hasil_prediksi_keparahan_cedera.csv",
                    mime="text/csv",
                )
                st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Gagal memproses file: {e}")

# ==============================================================================
# MODE 3: TENTANG MODEL
# ==============================================================================
else:
    st.markdown('<div class="jr-card">', unsafe_allow_html=True)
    st.markdown("### ℹ️ Tentang Model")
    st.write(
        """
        Model ini merupakan model klasifikasi biner (**AdaBoostClassifier**) yang memprediksi
        **tingkat keparahan cedera** korban kecelakaan lalu lintas berdasarkan data laporan awal,
        dengan dua kelas keluaran:
        - **Luka Berat**
        - **Luka Ringan**
        """
    )

    st.markdown("#### Fitur yang digunakan model")
    fitur_info = pd.DataFrame(
        {
            "Fitur": FEATURE_COLS,
            "Tipe": [
                "Numerik", "Numerik", "Numerik",
                "Kategorikal", "Kategorikal", "Kategorikal", "Kategorikal", "Kategorikal",
            ],
            "Keterangan": [
                "Usia korban (tahun)",
                "Usia kendaraan yang terlibat (tahun)",
                "Jumlah kendaraan yang terlibat dalam kecelakaan",
                "Provinsi lokasi kejadian",
                "Jenis kelamin korban",
                "Jenis kecelakaan (mis. lalu lintas jalan)",
                "Jenis kendaraan yang terlibat",
                "Jenis klaim yang diajukan",
            ],
        }
    )
    st.dataframe(fitur_info, use_container_width=True, hide_index=True)

    st.markdown("#### Pipeline Pra-pemrosesan (ColumnTransformer)")
    colx, coly = st.columns(2)
    with colx:
        st.markdown(
            """
            **Fitur Numerik**
            1. `SimpleImputer` — mengisi nilai kosong
            2. `RobustScaler` — menyamakan skala, tahan outlier
            """
        )
    with coly:
        st.markdown(
            """
            **Fitur Kategorikal**
            1. `SimpleImputer` — mengisi nilai kosong
            2. `OneHotEncoder` — mengubah kategori menjadi biner
            """
        )

    st.info(
        "Model dimuat dari file pickle **injury_severity_model.pkl** berupa scikit-learn "
        "`Pipeline` (ColumnTransformer + model klasifikasi) yang dilatih sebelumnya."
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown(
    """
    <div class="jr-footer">
        Aplikasi Internal Prediksi Tingkat Keparahan Cedera &bull; PT Jasa Raharja (Persero) &bull;
        Dibuat untuk mendukung proses asesmen klaim.
    </div>
    """,
    unsafe_allow_html=True,
)
