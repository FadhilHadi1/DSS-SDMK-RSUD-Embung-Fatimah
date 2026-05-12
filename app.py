import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import math
import holidays
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 1. SETUP DASHBOARD
st.set_page_config(page_title="DSS Optimasi Perawat RSUD ARIMA", layout="wide")

# Library hari libur Indonesia
id_holidays = holidays.Indonesia()

# Data Perawat Eksisting RSUD
data_perawat_eksisting = {
    "Penyakit Dalam": 3,
    "Rehabilitasi Medik": 6,
    "Jantung dan Pembuluh Darah": 2,
    "Paru": 2,
    "Umum MCU": 3
}

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stMetric { border: 1px solid #e6e9ef; padding: 15px; border-radius: 10px; background: #ffffff; }
    .main { background-color: #f0f2f6; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏥 Smart DSS: Manajemen SDMK Poliklinik")

# 2. DETEKSI MODEL ARIMA
files = [f for f in os.listdir('.') if f.startswith('model_arima_') and f.endswith('.pkl')]
list_poli_display = [f.replace('model_arima_', '').replace('.pkl', '').replace('_', ' ') for f in files]

if not list_poli_display:
    st.error("Model ARIMA .pkl tidak ditemukan! Pastikan file model ada di folder yang sama dengan app.py")
else:
    # --- SIDEBAR ---
    st.sidebar.header("🕹️ Control Panel")
    pilihan_poli = st.sidebar.selectbox("Pilih Poliklinik", list_poli_display)
    tgl_analisis = st.sidebar.date_input("Tanggal Analisis", value=datetime.now())
    
    st.sidebar.divider()
    st.sidebar.write("**Standar Waktu Aktivitas (Menit)**")
    m1 = st.sidebar.number_input("Pengkajian Awal", value=10)
    m2 = st.sidebar.number_input("Mengisi Asuhan Keperawatan", value=15)
    m3 = st.sidebar.number_input("Mendampingi Dokter", value=10)
    total_waktu = m1 + m2 + m3
    jam_efektif = st.sidebar.slider("Menit Kerja Efektif (1 Shift)", 300, 480, 420)

    # --- PROSES MODEL ARIMA ---
    nama_file_pkl = f"model_arima_{pilihan_poli.replace(' ', '_')}.pkl"
    model = joblib.load(nama_file_pkl)
    
    # Konversi tgl_analisis ke datetime murni
    target_date = datetime.combine(tgl_analisis, datetime.min.time())
    
    is_weekend = target_date.weekday() >= 5  
    is_holiday = target_date in id_holidays
    
    # Hitung Steps untuk ARIMA
    hari_ini = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    selisih_hari = (target_date - hari_ini).days
    steps = max(1, selisih_hari + 1)

    if is_weekend or is_holiday:
        pred_pasien = 0
        perawat_ideal = 0
        is_tutup = True
    else:
        is_tutup = False
        forecast = model.forecast(steps=steps)
        
        if isinstance(forecast, (pd.Series, pd.DataFrame)):
            val_final = forecast.iloc[-1]
        else:
            val_final = forecast[-1]
        
        pred_pasien = max(0, round(float(val_final)))
        
        # Hitung Kebutuhan SDM
        kebutuhan_hitung = (pred_pasien * total_waktu) / jam_efektif
        perawat_ideal = math.ceil(kebutuhan_hitung) if kebutuhan_hitung > 0 else 1
    
    perawat_sekarang = data_perawat_eksisting.get(pilihan_poli, 3) 
    selisih = perawat_sekarang - perawat_ideal 

    # --- TAMPILAN DASHBOARD ---
    tabs = st.tabs(["📊 Analisis Harian", "📈 Tren 30 Hari", "🔄 Realokasi SDMK"])

    with tabs[0]:
        st.subheader(f"Hasil Analisis: {pilihan_poli}")
        st.info(f"Target Tanggal: {tgl_analisis}")
        
        if is_tutup:
            st.warning("⚠️ **STATUS: UNIT TUTUP.** (Weekend atau Tanggal Merah)")
            
        # Membuat kolom metrik
        col1, col2, col3 = st.columns(3)
        
        col1.metric("Prediksi Pasien", f"{pred_pasien} Orang")
        col2.metric("Kebutuhan Ideal", f"{perawat_ideal} Perawat")
        
        if is_tutup:
            col3.metric("Status Saat Ini", f"{perawat_sekarang} Perawat", delta="OFF")
        elif selisih < 0:
            # delta_color="normal" -> Negatif otomatis MERAH
            col3.metric("Status Saat Ini", f"{perawat_sekarang} Perawat", delta=f"{selisih} (Kurang)", delta_color="normal")
            st.error(f"⚠️ **PERINGATAN:** Unit sedang kekurangan {abs(selisih)} tenaga perawat.")
        elif selisih > 0:
            # delta_color="normal" -> Positif otomatis HIJAU
            col3.metric("Status Saat Ini", f"{perawat_sekarang} Perawat", delta=f"+{selisih} (Lebih)", delta_color="normal")
            st.success(f"✅ **KETERANGAN:** Unit memiliki surplus {selisih} tenaga perawat.")
        else:
            col3.metric("Status Saat Ini", f"{perawat_sekarang} Perawat", delta="Optimal")

        st.write("---")
        st.write("### 📝 Rumus Perhitungan Kebutuhan")
        st.latex(rf""" \text{{Kebutuhan SDMK}} = \left\lceil \frac{{\text{{Prediksi Pasien}} \times \text{{Total Waktu}}}}{{\text{{Jam Efektif}}}} \right\rceil """)
        st.latex(rf""" \text{{Kebutuhan SDMK}} = \left\lceil \frac{{{pred_pasien} \times {total_waktu}}}{{{jam_efektif}}} \right\rceil = {perawat_ideal} \text{{ perawat}} """)

    with tabs[1]:
        st.subheader(f"Proyeksi Tren 30 Hari ke Depan")
        forecast_30 = model.forecast(steps=30)
        
        y_vals = forecast_30.values if hasattr(forecast_30, 'values') else forecast_30
        
        dates = [target_date + timedelta(days=x) for x in range(30)]
        df_30 = pd.DataFrame({'Tanggal': dates, 'Prediksi': y_vals})
        df_30['Tanggal'] = pd.to_datetime(df_30['Tanggal'])
        
        df_30['Prediksi'] = df_30['Prediksi'].apply(lambda x: max(0, round(float(x))))
        df_30['Kebutuhan'] = df_30['Prediksi'].apply(lambda x: math.ceil((x * total_waktu)/jam_efektif))
        
        df_30['is_holiday'] = df_30['Tanggal'].apply(lambda x: x in id_holidays)
        df_plot = df_30[(df_30['Tanggal'].dt.dayofweek < 5) & (df_30['is_holiday'] == False)].copy()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot['Tanggal'], y=df_plot['Kebutuhan'], mode='lines+markers', name='Kebutuhan Ideal'))
        fig.add_trace(go.Scatter(x=df_plot['Tanggal'], y=[perawat_sekarang]*len(df_plot), name='Kapasitas Eksisting', line=dict(dash='dash', color='red')))
        fig.update_layout(xaxis_title="Tanggal", yaxis_title="Jumlah Perawat", template="plotly_white")
        
        st.plotly_chart(fig, width="stretch")
        st.dataframe(df_plot[['Tanggal', 'Prediksi', 'Kebutuhan']], width="stretch", hide_index=True)

    with tabs[2]:
        st.subheader("💡 Saran Optimalisasi Antar Unit")
        if is_tutup:
            st.info("Unit sedang tutup. Saran realokasi tersedia pada hari kerja.")
        else:
            status_semua = []
            for p_name in list_poli_display:
                m_temp = joblib.load(f"model_arima_{p_name.replace(' ', '_')}.pkl")
                f_temp = m_temp.forecast(steps=steps)
                
                v_temp = f_temp.iloc[-1] if hasattr(f_temp, 'iloc') else f_temp[-1]
                val_temp = max(0, round(float(v_temp)))
                
                keb_temp = math.ceil((val_temp * total_waktu) / jam_efektif) if val_temp > 0 else 1
                eks_temp = data_perawat_eksisting.get(p_name, 3)
                sel_temp = eks_temp - keb_temp
                status_semua.append({"Poliklinik": p_name, "Kebutuhan": keb_temp, "Eksisting": eks_temp, "Status": sel_temp})
            
            df_status = pd.DataFrame(status_semua)
            
            if selisih < 0:
                st.error(f"⚠️ **{pilihan_poli}** saat ini kekurangan {abs(selisih)} perawat.")
                surplus = df_status[df_status['Status'] > 0]
                if not surplus.empty:
                    st.info("🚀 **Rekomendasi Penarikan Personel:**")
                    for _, r in surplus.iterrows():
                        st.success(f"🟢 **{r['Poliklinik']}** (Surplus {r['Status']} orang)")
                else:
                    st.warning("📢 Seluruh unit padat. Tidak ada unit dengan surplus perawat.")
            else:
                st.success(f"✅ **{pilihan_poli}** saat ini Aman.")

            st.write("---")
            st.write("**Tabel Monitoring Seluruh Unit:**")
            
            def style_status(val):
                color = 'red' if val < 0 else 'green' if val > 0 else 'black'
                return f'color: {color}'
            
            st.dataframe(df_status.style.map(style_status, subset=['Status']), width="stretch")