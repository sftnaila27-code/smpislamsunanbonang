import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import time
from io import BytesIO

# ====================== KONFIGURASI TAMPILAN ======================
st.set_page_config(
    page_title="Sistem Clustering Prestasi Siswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== CSS GLOBAL ======================
st.markdown("""
<style>
    /* HILANGKAN GARIS PUTIH DI ATAS */
    header { display: none !important; }
    .main .block-container { padding-top: 0rem !important; padding-bottom: 0rem !important; }
    .stApp header { background-color: transparent !important; }
    [data-testid="stSidebar"] { padding-top: 0rem !important; }
    
    /* Warna Dasar */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        color: #e2e8f0;
        font-family: 'Segoe UI', 'Poppins', sans-serif;
    }
    
    /* Header & Judul */
    h1, h2, h3, h4, h5 {
        color: #2dd4bf !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    /* Kartu / Panel */
    .card {
        background: rgba(15, 25, 35, 0.75);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(45, 212, 191, 0.25);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        border-color: rgba(45, 212, 191, 0.6);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
    }
    
    /* Tombol */
    .stButton>button {
        background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%);
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(20, 184, 166, 0.3);
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(20, 184, 166, 0.4);
        background: linear-gradient(135deg, #0d9488 0%, #2dd4bf 100%);
    }
    
    /* Upload Box */
    .stFileUploader {
        background-color: rgba(15, 25, 35, 0.7);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px dashed #2dd4bf;
    }
    
    /* Tabel */
    .dataframe {
        background-color: rgba(15, 25, 35, 0.6) !important;
        border-radius: 12px !important;
        border: 1px solid #334155 !important;
        color: #e2e8f0 !important;
    }
    .dataframe th {
        background-color: #0f766e !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 10px !important;
    }
    .dataframe td {
        padding: 8px !important;
    }
    
    /* Info Box */
    .st-info {
        background-color: rgba(45, 212, 191, 0.1) !important;
        border: 1px solid #2dd4bf !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
    }
    
    .st-success {
        background-color: rgba(20, 184, 166, 0.15) !important;
        border: 1px solid #14b8a6 !important;
        border-radius: 12px !important;
    }
    
    .st-warning {
        background-color: rgba(245, 158, 11, 0.15) !important;
        border: 1px solid #f59e0b !important;
        border-radius: 12px !important;
    }
    
    hr { border-color: #2dd4bf40 !important; margin: 1.5rem 0 !important; }
    
    /* Metric Card */
    .metric-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(59, 130, 246, 0.05));
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid rgba(16, 185, 129, 0.2);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(16, 185, 129, 0.5);
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.15);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #34d399;
        margin: 0.3rem 0;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ====================== TAMPILAN SELAMAT DATANG  ======================
st.markdown("""
<style>
.welcome-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 85vh;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border-radius: 24px;
    margin: 0.5rem 0;
    padding: 2rem;
    position: relative;
    overflow: hidden;
}

.welcome-container::before {
    content: '';
    position: absolute;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(16, 185, 129, 0.1), transparent);
    top: -100px;
    right: -100px;
    border-radius: 50%;
}

.welcome-container::after {
    content: '';
    position: absolute;
    width: 250px;
    height: 250px;
    background: radial-gradient(circle, rgba(59, 130, 246, 0.08), transparent);
    bottom: -80px;
    left: -80px;
    border-radius: 50%;
}

.welcome-card {
    position: relative;
    z-index: 2;
    background: rgba(15, 25, 35, 0.85);
    backdrop-filter: blur(20px);
    border-radius: 32px;
    padding: 3rem 3.5rem;
    border: 1px solid rgba(45, 212, 191, 0.25);
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
    max-width: 550px;
    width: 100%;
    text-align: center;
}

.welcome-icon {
    font-size: 4.5rem;
    margin-bottom: 0.5rem;
    display: block;
}

.welcome-title {
    font-size: 2.5rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 0.3rem;
}

.welcome-title span {
    background: linear-gradient(135deg, #2dd4bf, #10b981);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.welcome-divider {
    width: 60px;
    height: 3px;
    background: linear-gradient(90deg, #2dd4bf, #10b981);
    margin: 0.8rem auto;
    border-radius: 3px;
}

.welcome-subtitle {
    font-size: 1.1rem;
    color: #cbd5e1;
    line-height: 1.8;
}

.welcome-subtitle .highlight {
    color: #2dd4bf;
    font-weight: 600;
}

.welcome-school {
    display: block;
    margin-top: 0.3rem;
    font-weight: 500;
    color: #94a3b8;
}

.welcome-badge {
    display: inline-block;
    margin-top: 1.2rem;
    padding: 0.4rem 1.5rem;
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(45, 212, 191, 0.3);
    border-radius: 50px;
    color: #34d399;
    font-size: 0.75rem;
    letter-spacing: 1px;
}

.welcome-features {
    display: flex;
    justify-content: center;
    gap: 1.5rem;
    margin-top: 1.5rem;
    flex-wrap: wrap;
}

.welcome-feature {
    color: #94a3b8;
    font-size: 0.8rem;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(255, 255, 255, 0.03);
}
</style>

<div class="welcome-container">
    <div class="welcome-card">
        <span class="welcome-icon">🎓</span>
        <h1 class="welcome-title">
            SELAMAT <span>DATANG!</span>
        </h1>
        <div class="welcome-divider"></div>
        <p class="welcome-subtitle">
            di <span class="highlight">Sistem Pengelompokan Siswa Berprestasi</span>
            <span class="welcome-school">SMP Islam Sunan Bonang</span>
        </p>
        <div class="welcome-badge">✨ K-Means Clustering ✨</div>
        <div class="welcome-features">
            <span class="welcome-feature">3 Cluster</span>
            <span class="welcome-feature">Akurat</span>
            <span class="welcome-feature">Cepat</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-8px); }
    }
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 5px rgba(16, 185, 129, 0.3), 0 0 10px rgba(16, 185, 129, 0.2); }
        50% { box-shadow: 0 0 20px rgba(16, 185, 129, 0.6), 0 0 30px rgba(59, 130, 246, 0.3); }
    }
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0f1e 0%, #0d1425 50%, #0a0f1e 100%) !important;
        border-right: 1px solid rgba(16, 185, 129, 0.2);
    }
    [data-testid="stSidebar"] > div:first-child { background: transparent !important; }
    
    .premium-header {
        text-align: center;
        padding: 2rem 0.5rem 1.5rem 0.5rem;
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
    }
    .premium-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(16, 185, 129, 0.05), transparent);
        animation: shimmer 8s infinite linear;
    }
    
    .premium-logo {
        background: linear-gradient(135deg, #10b981, #3b82f6, #10b981);
        background-size: 200% 200%;
        width: 85px;
        height: 85px;
        border-radius: 28px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1.2rem auto;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.4), 0 0 0 3px rgba(59, 130, 246, 0.2);
        animation: float 3s ease-in-out infinite, glow 2s ease-in-out infinite;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        cursor: pointer;
    }
    .premium-logo:hover {
        transform: scale(1.08) rotate(8deg);
        box-shadow: 0 15px 40px rgba(16, 185, 129, 0.6), 0 0 0 5px rgba(59, 130, 246, 0.3);
    }
    .premium-logo span { font-size: 44px; filter: drop-shadow(0 4px 8px rgba(0,0,0,0.3)); }
    
    .premium-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #10b981, #3b82f6, #06b6d4, #10b981);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
        letter-spacing: 1.5px;
        animation: shimmer 4s infinite linear;
    }
    .premium-subtitle {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #64748b;
        font-size: 0.7rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-weight: 500;
    }
    .premium-badge {
        display: inline-block;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 20px;
        padding: 2px 8px;
        font-size: 0.6rem;
        color: #34d399;
        margin-top: 6px;
    }
    
    /* Menu - TEKS PUTIH */
    .stRadio > div { gap: 8px; padding: 0.75rem 0; }
    .stRadio label {
        background: rgba(15, 25, 45, 0.6) !important;
        backdrop-filter: blur(10px);
        border-radius: 14px !important;
        padding: 12px 18px !important;
        margin: 5px 0 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: 1px solid rgba(16, 185, 129, 0.15) !important;
        font-weight: 500 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        letter-spacing: 0.3px !important;
    }
    .stRadio label div p { color: #ffffff !important; }
    .stRadio label:hover div p { color: #2dd4bf !important; }
    .stRadio [data-baseweb="radio"]:checked + div p {
        color: #fbbf24 !important;
        font-weight: bold !important;
    }
    .stRadio label:hover {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(59, 130, 246, 0.1)) !important;
        border-color: rgba(16, 185, 129, 0.5) !important;
        transform: translateX(8px) scale(1.02) !important;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2) !important;
    }
    .stRadio [role="radiogroup"] [data-baseweb="radio"]:checked + div {
        background: linear-gradient(135deg, #10b981, #059669) !important;
        border-color: #34d399 !important;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4) !important;
    }
    
    .premium-divider {
        margin: 1.8rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #10b981, #3b82f6, #10b981, transparent);
        position: relative;
    }
    .premium-divider::before {
        content: '✦';
        position: absolute;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        background: #0a0f1e;
        color: #10b981;
        padding: 0 10px;
        font-size: 12px;
    }
    
    .premium-info-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(59, 130, 246, 0.05));
        border-radius: 20px;
        padding: 1.2rem;
        margin-top: 1rem;
        border: 1px solid rgba(16, 185, 129, 0.2);
        backdrop-filter: blur(10px);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    .premium-info-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(16, 185, 129, 0.1), transparent);
        transition: left 0.5s ease;
    }
    .premium-info-card:hover::before { left: 100%; }
    .premium-info-card:hover {
        transform: translateY(-5px);
        border-color: rgba(16, 185, 129, 0.5);
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.15);
    }
    
    .info-card-title {
        font-weight: 700;
        margin: 0 0 0.9rem 0;
        color: #34d399;
        font-size: 0.85rem;
        text-align: center;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }
    
    .premium-cluster-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 0;
        font-size: 0.8rem;
        border-bottom: 1px dashed rgba(16, 185, 129, 0.1);
        transition: all 0.3s ease;
    }
    .premium-cluster-item:hover {
        padding-left: 8px;
        border-bottom-color: rgba(16, 185, 129, 0.3);
    }
    
    .cluster-premium-badge {
        width: 30px;
        height: 30px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 0.7rem;
        transition: all 0.3s ease;
    }
    .cluster-badge-1 { background: linear-gradient(135deg, #10b981, #34d399); box-shadow: 0 2px 8px rgba(16, 185, 129, 0.4); }
    .cluster-badge-2 { background: linear-gradient(135deg, #f59e0b, #fbbf24); box-shadow: 0 2px 8px rgba(245, 158, 11, 0.4); }
    .cluster-badge-3 { background: linear-gradient(135deg, #ef4444, #f87171); box-shadow: 0 2px 8px rgba(239, 68, 68, 0.4); }
    .cluster-premium-badge:hover { transform: scale(1.1) rotate(5deg); }
    
    .premium-stat-card {
        background: rgba(16, 185, 129, 0.05);
        border-radius: 16px;
        padding: 0.8rem;
        margin-top: 0.5rem;
    }
    .stat-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        font-size: 0.75rem;
    }
    .stat-value { font-weight: 800; font-size: 1.1rem; }
    .stat-value-1 { color: #34d399; }
    .stat-value-2 { color: #fbbf24; }
    .stat-value-3 { color: #f87171; }
    
    .premium-footer {
        text-align: center;
        padding: 1.2rem 0.5rem;
        margin-top: 1.5rem;
        border-top: 1px solid rgba(16, 185, 129, 0.15);
        font-size: 0.7rem;
    }
    .premium-footer-text { color: #475569; line-height: 1.5; }
    .premium-footer-text strong {
        background: linear-gradient(135deg, #10b981, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # HEADER
    st.markdown("""
    <div class='premium-header'>
        <div class='premium-logo'><span>🏆</span></div>
        <h2 class='premium-title'>PRESTASI HUB</h2>
        <p class='premium-subtitle'>Powered Clustering System</p>
        <div class='premium-badge'>✨ K-Means Algorithm ✨</div>
    </div>
    """, unsafe_allow_html=True)
    
    # MENU
    menu = st.radio(
        "",
        ["🏠 Dashboard", "📤 Upload Data", "📊 Seleksi & Normalisasi", "🎯 Hasil Clustering", "📈 Analisis Visual", "⚙️ Tentang Sistem"],
        index=0
    )
    
    st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)
    
    # INFO CLUSTER
    st.markdown("""
    <div class='premium-info-card'>
        <div class='info-card-title'>✦ KATEGORI PRESTASI ✦</div>
        <div class='premium-cluster-item'>
            <div class='cluster-premium-badge cluster-badge-1'>C1</div>
            <div><span style='color:#34d399; font-weight:700;'>🌟 Elite</span> <span style='color:#94a3b8;'>(Berprestasi Tinggi)</span></div>
        </div>
        <div class='premium-cluster-item'>
            <div class='cluster-premium-badge cluster-badge-2'>C2</div>
            <div><span style='color:#fbbf24; font-weight:700;'>📘 Progresif</span> <span style='color:#94a3b8;'>(Kemampuan Sedang)</span></div>
        </div>
        <div class='premium-cluster-item'>
            <div class='cluster-premium-badge cluster-badge-3'>C3</div>
            <div><span style='color:#f87171; font-weight:700;'>🌱 Binaan</span> <span style='color:#94a3b8;'>(Perlu Pembinaan)</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # STATISTIK DINAMIS
    if 'hasil_akhir' in st.session_state and st.session_state['hasil_akhir'] is not None:
        df_sidebar = st.session_state['hasil_akhir']
        jml_siswa = len(df_sidebar)
        jml_cluster1 = len(df_sidebar[df_sidebar['Cluster'] == 1]) if 'Cluster' in df_sidebar.columns else 0
        jml_cluster2 = len(df_sidebar[df_sidebar['Cluster'] == 2]) if 'Cluster' in df_sidebar.columns else 0
        jml_cluster3 = len(df_sidebar[df_sidebar['Cluster'] == 3]) if 'Cluster' in df_sidebar.columns else 0
        
        pct1 = (jml_cluster1/jml_siswa*100) if jml_siswa > 0 else 0
        pct2 = (jml_cluster2/jml_siswa*100) if jml_siswa > 0 else 0
        pct3 = (jml_cluster3/jml_siswa*100) if jml_siswa > 0 else 0
        
        st.markdown(f"""
        <div class='premium-info-card' style='margin-top: 1rem;'>
            <div class='info-card-title'>📊 STATISTIK REAL-TIME</div>
            <div class='premium-stat-card'>
                <div class='stat-item'><span>👨‍🎓 Total Siswa</span><span class='stat-value' style='color:#34d399;'>{jml_siswa}</span></div>
                <div class='stat-item'><span>🌟 Elite (C1)</span><span><span class='stat-value stat-value-1'>{jml_cluster1}</span> <span style='color:#475569;'>({pct1:.1f}%)</span></span></div>
                <div class='stat-item'><span>📘 Progresif (C2)</span><span><span class='stat-value stat-value-2'>{jml_cluster2}</span> <span style='color:#475569;'>({pct2:.1f}%)</span></span></div>
                <div class='stat-item'><span>🌱 Binaan (C3)</span><span><span class='stat-value stat-value-3'>{jml_cluster3}</span> <span style='color:#475569;'>({pct3:.1f}%)</span></span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style='margin-top: 0.8rem;'>
            <div style='background: #1e293b; border-radius: 10px; height: 6px; overflow: hidden;'>
                <div style='width: {pct1}%; background: linear-gradient(90deg, #10b981, #34d399); height: 6px; float: left;'></div>
                <div style='width: {pct2}%; background: linear-gradient(90deg, #f59e0b, #fbbf24); height: 6px; float: left;'></div>
                <div style='width: {pct3}%; background: linear-gradient(90deg, #ef4444, #f87171); height: 6px; float: left;'></div>
            </div>
            <div style='display: flex; justify-content: space-between; margin-top: 6px;'>
                <span style='font-size: 0.6rem; color:#34d399;'>Elite</span>
                <span style='font-size: 0.6rem; color:#fbbf24;'>Progresif</span>
                <span style='font-size: 0.6rem; color:#f87171;'>Binaan</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='premium-info-card' style='margin-top: 1rem; text-align: center;'>
            <div class='info-card-title'>⚡ QUICK TIP</div>
            <p style='font-size: 0.7rem; color:#94a3b8; margin:0;'>Upload data nilai siswa untuk melihat statistik real-time</p>
            <div style='margin-top: 8px;'>📁 → 📤 Upload Data</div>
        </div>
        """, unsafe_allow_html=True)
    
    # FOOTER
    st.markdown("""
    <div class='premium-footer'>
        <div class='premium-footer-text'>
            <strong>PRESTASI HUB v2.0</strong><br>
            Powered by K-Means Clustering<br>
            <span style='color:#334155;'>© 2026 · SMP Islam Sunan Bonang</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ====================== FUNGSI PREPROCESSING ======================
def preprocessing_data(df):
    """Seleksi Atribut & Normalisasi Min-Max"""
    try:
        kolom_identitas = ['NO', 'NAMA', 'NIS', 'KELAS', 'NISN']
        kolom_id_tersedia = [col for col in kolom_identitas if col.upper() in [c.upper() for c in df.columns]]
        kolom_nilai = [col for col in df.columns if col.upper() not in [c.upper() for c in kolom_identitas]]
        
        if not kolom_nilai:
            raise ValueError("Kolom nilai mata pelajaran tidak ditemukan.")
        
        data_nilai = df[kolom_nilai].apply(pd.to_numeric, errors='coerce')
        data_nilai = data_nilai.fillna(data_nilai.mean())
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_normalisasi = scaler.fit_transform(data_nilai)
        data_normalisasi = pd.DataFrame(data_normalisasi, columns=kolom_nilai)
        
        return data_normalisasi, data_nilai, kolom_id_tersedia, kolom_nilai
    except Exception as e:
        st.error(f"❌ Kesalahan Preprocessing: {str(e)}")
        return None, None, None, None

# ====================== FUNGSI CLUSTERING ======================
def algoritma_kmeans(data, n_cluster=3):
    """Algoritma K-Means Clustering"""
    try:
        kmeans = KMeans(n_clusters=n_cluster, init='k-means++', max_iter=300, n_init=10, random_state=42)
        label_cluster = kmeans.fit_predict(data)
        centroid = kmeans.cluster_centers_
        silhouette = silhouette_score(data, label_cluster)
        return label_cluster, centroid, silhouette
    except Exception as e:
        st.error(f"❌ Kesalahan Clustering: {str(e)}")
        return None, None, None

# ====================== FUNGSI LABEL ======================
def beri_label_cluster(rata_rata_nilai):
    """Pemberian Label Cluster"""
    try:
        urutan = rata_rata_nilai.sort_values(ascending=False).index
        label_map = {urutan[0]: 1, urutan[1]: 2, urutan[2]: 3}
        nama_label = {
            1: "✨ Cluster 1 - Elite (Berprestasi Tinggi)",
            2: "📘 Cluster 2 - Progresif (Kemampuan Sedang)",
            3: "🌱 Cluster 3 - Binaan (Perlu Pembinaan)"
        }
        warna_label = {1: "#0f766e", 2: "#b45309", 3: "#9b2c2c"}
        return label_map, nama_label, warna_label
    except Exception as e:
        st.error(f"❌ Kesalahan Label: {str(e)}")
        return {}, {}, {}

# ====================== FUNGSI PERANGKINGAN ======================
def hitung_perangkingan(data_awal, data_nilai):
    """Penentuan Peringkat Siswa"""
    try:
        data_hasil = data_awal.copy()
        data_hasil['RATA_RATA_NILAI'] = data_nilai.mean(axis=1).round(2)
        data_hasil = data_hasil.sort_values(by='RATA_RATA_NILAI', ascending=False).reset_index(drop=True)
        data_hasil['PERINGKAT'] = data_hasil.index + 1
        return data_hasil
    except Exception as e:
        st.error(f"❌ Kesalahan Perangkingan: {str(e)}")
        return data_awal

# ====================== FUNGSI ELBOW ======================
def hitung_elbow_method(data):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    return wcss

# ====================== FUNGSI SILHOUETTE ======================
def hitung_silhouette_per_cluster(data):
    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(data)
    silhouette_avg = silhouette_score(data, cluster_labels)
    sample_silhouette_values = silhouette_samples(data, cluster_labels)
    return silhouette_avg, sample_silhouette_values, cluster_labels

# ====================== FUNGSI DOWNLOAD EXCEL ======================
def convert_df_to_excel(df):
    """Mengkonversi DataFrame ke file Excel murni (.xlsx) dengan multiple sheets"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Data utama hasil clustering
        df.to_excel(writer, index=False, sheet_name='Hasil Clustering')
        
        # Sheet 2: Informasi
        info_df = pd.DataFrame({
            'Informasi': ['Total Siswa', 'Jumlah Cluster', 'Tanggal Proses', 'Metode', 'Jumlah Atribut'],
            'Nilai': [
                len(df), 
                len(df['Cluster'].unique()) if 'Cluster' in df.columns else 0,
                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'K-Means Clustering',
                len([col for col in df.columns if col not in ['NO', 'NAMA', 'NIS', 'KELAS', 'NISN', 'Cluster', 'Keterangan', 'PERINGKAT', 'RATA_RATA_NILAI', 'Cluster_Asli']])
            ]
        })
        info_df.to_excel(writer, index=False, sheet_name='Informasi')
        
        # Sheet 3: Statistik per cluster
        if 'Cluster' in df.columns and 'RATA_RATA_NILAI' in df.columns:
            stat_cluster = df.groupby('Cluster').agg({
                'RATA_RATA_NILAI': ['count', 'mean', 'min', 'max', 'std']
            }).round(2)
            stat_cluster.columns = ['Jumlah', 'Rata-rata', 'Minimum', 'Maksimum', 'Std Dev']
            
            # Tambahkan label cluster
            if 'Keterangan' in df.columns:
                label_cluster = df.groupby('Cluster')['Keterangan'].first()
                stat_cluster['Keterangan'] = label_cluster
            
            stat_cluster.to_excel(writer, sheet_name='Statistik Cluster')
        
        # Sheet 4: Top 10 Siswa Berprestasi
        if 'PERINGKAT' in df.columns and 'RATA_RATA_NILAI' in df.columns:
            top10 = df.nsmallest(10, 'PERINGKAT')
            top10.to_excel(writer, index=False, sheet_name='Top 10 Siswa')
    
    processed_data = output.getvalue()
    return processed_data

# ================================================================
# ====================== HALAMAN DASHBOARD =======================
# ================================================================
if menu == "🏠 Dashboard":
    st.markdown("""
    <style>
    .dashboard-header { text-align: center; padding: 1rem 0 2rem 0; }
    .dashboard-title { font-size: 2.8rem; background: linear-gradient(135deg, #10b981, #3b82f6, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; }
    .dashboard-subtitle { color: #94a3b8; font-size: 1.1rem; }
    </style>
    <div class='dashboard-header'>
        <h1 class='dashboard-title'>🏠 DASHBOARD PRESTASI</h1>
        <p class='dashboard-subtitle'>Sistem Pengelompokan Siswa Berbasis K-Means Clustering</p>
        <div style='width: 80px; height: 3px; background: linear-gradient(90deg, #10b981, #3b82f6); margin: 1rem auto; border-radius: 3px;'></div>
    </div>
    """, unsafe_allow_html=True)
    
    if 'hasil_akhir' in st.session_state and st.session_state['hasil_akhir'] is not None:
        df = st.session_state['hasil_akhir']
        jml_siswa = len(df)
        jml_c1 = len(df[df['Cluster'] == 1]) if 'Cluster' in df.columns else 0
        jml_c2 = len(df[df['Cluster'] == 2]) if 'Cluster' in df.columns else 0
        jml_c3 = len(df[df['Cluster'] == 3]) if 'Cluster' in df.columns else 0
        silhouette = st.session_state.get('silhouette', 0)
        rata = df['RATA_RATA_NILAI'].mean() if 'RATA_RATA_NILAI' in df.columns else 0
        tertinggi = df['RATA_RATA_NILAI'].max() if 'RATA_RATA_NILAI' in df.columns else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"<div class='metric-card'><div style='font-size:2rem;'>👨‍🎓</div><div class='metric-value'>{jml_siswa}</div><div class='metric-label'>Total Siswa</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><div style='font-size:2rem;'>📊</div><div class='metric-value'>{rata:.1f}</div><div class='metric-label'>Rata-rata Nilai</div></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'><div style='font-size:2rem;'>🏆</div><div class='metric-value'>{tertinggi:.1f}</div><div class='metric-label'>Nilai Tertinggi</div></div>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<div class='metric-card'><div style='font-size:2rem;'>📈</div><div class='metric-value'>{silhouette:.3f}</div><div class='metric-label'>Silhouette Score</div></div>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("<h3 style='color:#34d399; text-align:center;'>📊 DISTRIBUSI SISWA PER CLUSTER</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div class='metric-card' style='border-left:4px solid #10b981;'><div style='font-size:1.5rem;'>🌟 ELITE</div><div class='metric-value' style='color:#34d399;'>{jml_c1}</div><div class='metric-label'>Siswa</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card' style='border-left:4px solid #f59e0b;'><div style='font-size:1.5rem;'>📘 PROGRESIF</div><div class='metric-value' style='color:#fbbf24;'>{jml_c2}</div><div class='metric-label'>Siswa</div></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card' style='border-left:4px solid #ef4444;'><div style='font-size:1.5rem;'>🌱 BINAAN</div><div class='metric-value' style='color:#f87171;'>{jml_c3}</div><div class='metric-label'>Siswa</div></div>", unsafe_allow_html=True)
        
        # Top 5
        st.markdown("---")
        st.markdown("<h3 style='color:#34d399; text-align:center;'>🏆 TOP 5 SISWA BERPRESTASI</h3>", unsafe_allow_html=True)
        kolom_nama = None
        for nama in ['NAMA', 'Nama', 'nama', 'NAMA SISWA']:
            if nama in df.columns:
                kolom_nama = nama
                break
        if kolom_nama and 'RATA_RATA_NILAI' in df.columns:
            top5 = df.nlargest(5, 'RATA_RATA_NILAI')[[kolom_nama, 'RATA_RATA_NILAI', 'Cluster', 'Keterangan']].reset_index(drop=True)
            top5.index = top5.index + 1
            st.dataframe(top5, use_container_width=True)
    else:
        st.markdown("""
        <div style='text-align:center; padding:3rem; background:rgba(16,185,129,0.05); border-radius:24px; border:1px dashed rgba(16,185,129,0.3);'>
            <div style='font-size:4rem;'>📂</div>
            <h3 style='color:#34d399;'>Belum Ada Data</h3>
            <p style='color:#94a3b8;'>Silakan upload data nilai siswa terlebih dahulu</p>
            <div style='margin-top:1rem;'>📁 → 📤 Upload Data</div>
        </div>
        """, unsafe_allow_html=True)

# ================================================================
# ====================== HALAMAN UPLOAD DATA =====================
# ================================================================
elif menu == "📤 Upload Data":
    st.markdown("<h1 style='color:#2dd4bf;'>📤 Upload & Proses Data Siswa</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#cbd5e1; margin-bottom:2rem;'>Unggah file data nilai siswa untuk memulai proses analisis</p>", unsafe_allow_html=True)
    
    # INFORMASI FORMAT FILE YANG DIDUKUNG
    st.info("📁 **Format yang didukung:** .xlsx, .xls, .xlsm, .xlsb, .csv")
    
    uploaded_file = st.file_uploader(
        "Pilih File Data Nilai", 
        type=['xlsx', 'xls', 'xlsm', 'xlsb', 'csv'],
        help="Support format: .xlsx, .xls, .xlsm, .xlsb, .csv"
    )
    
    if uploaded_file is not None:
        try:
            # BACA FILE SESUAI EKSTENSI
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            st.success(f"✅ File berhasil diunggah dan dibaca! ({file_extension.upper()})")
            st.markdown("<h3 style='color:#2dd4bf;'>📋 Data Siswa Lengkap</h3>", unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, height=300)
            st.session_state['df_mentah'] = df
            
            st.markdown("---")
            proses_btn = st.button("🚀 MULAI PROSES CLUSTERING", type="primary", use_container_width=True)
            
            if proses_btn:
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                # Tahap 1: Preprocessing
                progress_text.text("🔄 Tahap 1/3: Preprocessing Data (Seleksi Atribut & Normalisasi)...")
                progress_bar.progress(20)
                time.sleep(0.5)
                
                data_normalisasi, data_nilai, kolom_id, kolom_nilai = preprocessing_data(df)
                if data_normalisasi is None:
                    st.stop()
                
                st.session_state['data_normalisasi'] = data_normalisasi
                st.session_state['data_nilai'] = data_nilai
                st.session_state['kolom_nilai'] = kolom_nilai
                
                progress_text.text("✅ Tahap 1/3: Preprocessing Selesai!")
                progress_bar.progress(40)
                time.sleep(0.5)
                
                # Tahap 2: Clustering
                progress_text.text("🔄 Tahap 2/3: Proses Clustering K-Means...")
                progress_bar.progress(50)
                time.sleep(0.5)
                
                label_cluster, centroid, nilai_silhouette = algoritma_kmeans(data_normalisasi, n_cluster=3)
                if label_cluster is None:
                    st.stop()
                
                progress_text.text(f"✅ Tahap 2/3: Clustering Selesai! (Silhouette Score: {nilai_silhouette:.4f})")
                progress_bar.progress(70)
                time.sleep(0.5)
                
                # Tahap 3: Label & Perangkingan
                progress_text.text("🔄 Tahap 3/3: Pemberian Label & Perangkingan...")
                progress_bar.progress(80)
                time.sleep(0.5)
                
                df_hasil = df.copy()
                df_hasil['Cluster_Asli'] = label_cluster
                df_hasil['RATA_RATA_NILAI'] = data_nilai.mean(axis=1).round(2)
                rata_per_cluster = df_hasil.groupby('Cluster_Asli')['RATA_RATA_NILAI'].mean()
                peta_label, nama_cluster, warna_cluster = beri_label_cluster(rata_per_cluster)
                df_hasil['Cluster'] = df_hasil['Cluster_Asli'].map(peta_label)
                df_hasil['Keterangan'] = df_hasil['Cluster'].map(nama_cluster)
                df_hasil = hitung_perangkingan(df_hasil, data_nilai)
                
                progress_text.text("✅ Tahap 3/3: Selesai!")
                progress_bar.progress(100)
                time.sleep(0.5)
                
                # Simpan ke session state
                st.session_state['hasil_akhir'] = df_hasil
                st.session_state['centroid'] = centroid
                st.session_state['silhouette'] = nilai_silhouette
                st.session_state['warna'] = warna_cluster
                
                progress_text.empty()
                progress_bar.empty()
                
                # 🎈 BALON TERBANG!
                st.balloons()
                
                st.success("🎉 PROSES CLUSTERING BERHASIL! Silakan lihat hasil di menu berikut:")
                st.info("📊 Lihat hasil di menu **Hasil Clustering** atau **Dashboard**")
                
        except Exception as e:
            st.error(f"❌ Gagal memproses data: {str(e)}")

# ================================================================
# ================== HALAMAN SELEKSI & NORMALISASI ===============
# ================================================================
elif menu == "📊 Seleksi & Normalisasi":
    st.markdown("<h1 style='color:#2dd4bf;'>📊 Seleksi Atribut & Normalisasi Data</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#cbd5e1; margin-bottom:2rem;'>Proses preprocessing data sebelum clustering</p>", unsafe_allow_html=True)
    
    if 'hasil_akhir' not in st.session_state or st.session_state['hasil_akhir'] is None:
        st.warning("⚠️ Silakan upload data dan proses clustering terlebih dahulu!")
    else:
        df_original = st.session_state.get('df_mentah', None)
        data_normalisasi = st.session_state.get('data_normalisasi', None)
        data_nilai = st.session_state.get('data_nilai', None)
        kolom_nilai = st.session_state.get('kolom_nilai', None)
        
        if df_original is not None and data_normalisasi is not None:
            st.success("✅ Data preprocessing tersedia!")
            
            # Ringkasan
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(data_nilai)}</div><div class='metric-label'>📊 Jumlah Data</div></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(kolom_nilai) if kolom_nilai else 0}</div><div class='metric-label'>📚 Jumlah Atribut Nilai</div></div>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<div class='metric-card'><div class='metric-value'>0 - 1</div><div class='metric-label'>🎯 Rentang Normalisasi</div></div>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Step 1: Seleksi Atribut
            st.markdown("<h3 style='color:#2dd4bf;'>📋 STEP 1: SELEKSI ATRIBUT</h3>", unsafe_allow_html=True)
            st.markdown("<p style='color:#94a3b8;'>Kolom identitas dihapus, kolom nilai digunakan untuk clustering</p>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                <div style='background: rgba(239, 68, 68, 0.1); border-radius: 12px; padding: 1rem; border-left: 4px solid #ef4444;'>
                    <h4 style='color:#f87171;'>❌ KOLOM IDENTITAS (DIHAPUS)</h4>
                </div>
                """, unsafe_allow_html=True)
                for kol in ['NO', 'NAMA', 'NIS', 'KELAS', 'NISN']:
                    if kol in df_original.columns:
                        st.markdown(f"<div style='background:rgba(239,68,68,0.05); padding:0.5rem; border-radius:8px; margin:0.25rem 0;'><span style='color:#f87171;'>✖</span> {kol}</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style='background: rgba(16, 185, 129, 0.1); border-radius: 12px; padding: 1rem; border-left: 4px solid #10b981;'>
                    <h4 style='color:#34d399;'>✅ KOLOM NILAI (DIPAKAI)</h4>
                </div>
                """, unsafe_allow_html=True)
                for kol in kolom_nilai:
                    st.markdown(f"<div style='background:rgba(16,185,129,0.05); padding:0.5rem; border-radius:8px; margin:0.25rem 0;'><span style='color:#34d399;'>✔</span> {kol}</div>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Tampilkan Data Nilai
            st.markdown("<h3 style='color:#2dd4bf;'>📋 DATA NILAI (SETELAH SELEKSI ATRIBUT)</h3>", unsafe_allow_html=True)
            st.dataframe(data_nilai.head(10), use_container_width=True)
            
            st.markdown("---")
            
            # Step 2: Normalisasi
            st.markdown("<h3 style='color:#2dd4bf;'>📊 STEP 2: NORMALISASI MIN-MAX</h3>", unsafe_allow_html=True)
            st.markdown("""
            <div style='background: rgba(16, 185, 129, 0.1); border-radius: 12px; padding: 1rem; border-left: 4px solid #10b981; margin-bottom:1rem;'>
                <b style='color:#2dd4bf;'>📐 RUMUS:</b><br>
                <span style='color:#cbd5e1;'>X' = (X - Xmin) / (Xmax - Xmin)</span><br>
                <span style='color:#94a3b8; font-size:0.8rem;'>Mengubah data ke rentang 0-1</span>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<p style='color:#f87171; font-weight:bold;'>SEBELUM NORMALISASI</p>", unsafe_allow_html=True)
                st.dataframe(data_nilai.head(5), use_container_width=True)
            with col2:
                st.markdown("<p style='color:#34d399; font-weight:bold;'>SESUDAH NORMALISASI</p>", unsafe_allow_html=True)
                st.dataframe(data_normalisasi.head(5).round(4), use_container_width=True)
            
            # Grafik perbandingan
            st.markdown("---")
            st.markdown("<h3 style='color:#2dd4bf; text-align:center;'>📈 PERBANDINGAN SEBELUM & SESUDAH NORMALISASI</h3>", unsafe_allow_html=True)
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=100)
            sample_orig = data_nilai.iloc[0, :].values
            axes[0].bar(range(len(sample_orig)), sample_orig, color='#ef4444', alpha=0.7)
            axes[0].set_title('Sebelum Normalisasi', color='white', fontsize=12)
            axes[0].set_ylabel('Nilai Asli', color='white')
            axes[0].set_xlabel('Kolom Nilai', color='white')
            axes[0].set_facecolor('#1e293b')
            axes[0].tick_params(colors='white')
            axes[0].grid(True, alpha=0.2)
            
            sample_norm = data_normalisasi.iloc[0, :].values
            axes[1].bar(range(len(sample_norm)), sample_norm, color='#34d399', alpha=0.7)
            axes[1].set_title('Sesudah Normalisasi', color='white', fontsize=12)
            axes[1].set_ylabel('Nilai Normalisasi (0-1)', color='white')
            axes[1].set_xlabel('Kolom Nilai', color='white')
            axes[1].set_facecolor('#1e293b')
            axes[1].tick_params(colors='white')
            axes[1].grid(True, alpha=0.2)
            axes[1].set_ylim(0, 1)
            
            fig.patch.set_facecolor('#0f172a')
            st.pyplot(fig, use_container_width=True)
        else:
            st.warning("⚠️ Data preprocessing belum lengkap. Silakan proses clustering terlebih dahulu!")

# ================================================================
# =================== HALAMAN HASIL CLUSTERING ===================
# ================================================================
elif menu == "🎯 Hasil Clustering":
    st.markdown("<h1 style='color:#2dd4bf; margin-bottom:1.5rem;'>🎯 Hasil Pengelompokan Siswa</h1>", unsafe_allow_html=True)
    
    if 'hasil_akhir' not in st.session_state or st.session_state['hasil_akhir'] is None:
        st.warning("⚠️ Silakan upload file dan proses data terlebih dahulu")
    else:
        df = st.session_state['hasil_akhir'].copy()
        
        jml_c1 = len(df[df['Cluster'] == 1]) if 'Cluster' in df.columns else 0
        jml_c2 = len(df[df['Cluster'] == 2]) if 'Cluster' in df.columns else 0
        jml_c3 = len(df[df['Cluster'] == 3]) if 'Cluster' in df.columns else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div style='background:linear-gradient(135deg,#0f766e,#14b8a6); padding:1rem; border-radius:16px; text-align:center;'><p style='margin:0; color:white;'>🌟 Elite</p><h2 style='margin:0; color:white;'>{jml_c1}</h2></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div style='background:linear-gradient(135deg,#78350f,#b45309); padding:1rem; border-radius:16px; text-align:center;'><p style='margin:0; color:white;'>📘 Progresif</p><h2 style='margin:0; color:#fcd34d;'>{jml_c2}</h2></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div style='background:linear-gradient(135deg,#7f1d1d,#9b2c2c); padding:1rem; border-radius:16px; text-align:center;'><p style='margin:0; color:white;'>🌱 Binaan</p><h2 style='margin:0; color:#f87171;'>{jml_c3}</h2></div>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("<h3 style='color:#2dd4bf;'>📋 Data Lengkap Hasil Clustering & Peringkat</h3>", unsafe_allow_html=True)
        
        kolom_nama = None
        for nama in ['NAMA', 'Nama', 'nama', 'NAMA SISWA']:
            if nama in df.columns:
                kolom_nama = nama
                break
        
        urutan = []
        if kolom_nama: urutan.append(kolom_nama)
        if 'PERINGKAT' in df.columns: urutan.append('PERINGKAT')
        if 'NO' in df.columns: urutan.append('NO')
        if 'NIS' in df.columns: urutan.append('NIS')
        if 'KELAS' in df.columns: urutan.append('KELAS')
        if 'RATA_RATA_NILAI' in df.columns: urutan.append('RATA_RATA_NILAI')
        if 'Cluster' in df.columns: urutan.append('Cluster')
        if 'Keterangan' in df.columns: urutan.append('Keterangan')
        
        if urutan:
            st.dataframe(df[urutan], use_container_width=True, height=400)
        
        st.markdown("---")
        st.markdown("<h3 style='color:#2dd4bf;'>🏆 Daftar Siswa Elite (Cluster 1)</h3>", unsafe_allow_html=True)
        
        if 'Cluster' in df.columns and kolom_nama is not None:
            elite = df[df['Cluster'] == 1].sort_values(by='PERINGKAT', ascending=True).reset_index(drop=True)
            if len(elite) > 0:
                st.dataframe(elite[urutan], use_container_width=True)
            else:
                st.info("ℹ️ Tidak ada siswa di kategori Elite")
        
        # ===== TOMBOL DOWNLOAD EXCEL =====
        st.markdown("---")
        st.markdown("<h3 style='color:#2dd4bf;'>📥 Download Hasil Clustering</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("""
            <div style='background: rgba(16, 185, 129, 0.05); border-radius: 12px; padding: 0.8rem;'>
                <p style='color:#94a3b8; margin:0; font-size:0.9rem;'>
                    📄 Download file Excel murni (.xlsx) berisi:
                    <br>• Data lengkap hasil clustering (Sheet 1)
                    <br>• Informasi proses (Sheet 2)
                    <br>• Statistik per cluster (Sheet 3)
                    <br>• Top 10 siswa berprestasi (Sheet 4)
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Tombol download
            excel_data = convert_df_to_excel(df)
            st.download_button(
                label="📥 DOWNLOAD EXCEL",
                data=excel_data,
                file_name=f"Hasil_Clustering_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                type="primary"
            )

# ================================================================
# =================== HALAMAN ANALISIS VISUAL ====================
# ================================================================
elif menu == "📈 Analisis Visual":
    st.markdown("<h1 style='color:#2dd4bf;'>📈 Analisis Visual Hasil Clustering</h1>", unsafe_allow_html=True)
    
    if 'hasil_akhir' not in st.session_state:
        st.warning("⚠️ Silakan upload dan proses data terlebih dahulu")
    else:
        df = st.session_state['hasil_akhir']
        data_norm = st.session_state['data_normalisasi']
        
        # Elbow Method
        st.markdown("<h3 style='color:#2dd4bf;'>📉 Grafik Elbow Method</h3>", unsafe_allow_html=True)
        wcss = hitung_elbow_method(data_norm)
        fig1, ax1 = plt.subplots(figsize=(10,5), dpi=100)
        ax1.plot(range(1, 11), wcss, marker='o', color='#2dd4bf', linewidth=2, markersize=8)
        ax1.set_title('Elbow Method - Penentuan Jumlah Cluster Optimal', color='white', fontsize=14)
        ax1.set_xlabel('Jumlah Cluster (k)', color='white')
        ax1.set_ylabel('WCSS', color='white')
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#1e293b')
        fig1.patch.set_facecolor('#0f172a')
        ax1.tick_params(colors='white')
        ax1.axvline(x=3, color='#f87171', linestyle='--', linewidth=2, label='k = 3 (Terpilih)')
        ax1.legend(facecolor='#1e293b', labelcolor='white')
        st.pyplot(fig1, use_container_width=True)
        
        # Silhouette
        st.markdown("<h3 style='color:#2dd4bf;'>📊 Grafik Silhouette Score</h3>", unsafe_allow_html=True)
        silhouette_avg, sample_values, cluster_labels = hitung_silhouette_per_cluster(data_norm)
        fig2, ax2 = plt.subplots(figsize=(10,5), dpi=100)
        y_lower = 10
        colors_sil = ['#14b8a6', '#f59e0b', '#ef4444']
        for i in range(3):
            ith = sample_values[cluster_labels == i]
            ith.sort()
            size = ith.shape[0]
            y_upper = y_lower + size
            ax2.fill_betweenx(np.arange(y_lower, y_upper), 0, ith, color=colors_sil[i], alpha=0.7)
            ax2.text(-0.05, y_lower + 0.5 * size, f'Cluster {i+1}', color='white')
            y_lower = y_upper + 10
        ax2.axvline(x=silhouette_avg, color='#2dd4bf', linestyle='--', linewidth=2, label=f'Rata-rata: {silhouette_avg:.4f}')
        ax2.set_title('Analisis Silhouette untuk 3 Cluster', color='white', fontsize=14)
        ax2.set_xlabel('Nilai Koefisien Silhouette', color='white')
        ax2.set_ylabel('Sampel Data', color='white')
        ax2.set_yticks([])
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('#1e293b')
        fig2.patch.set_facecolor('#0f172a')
        ax2.tick_params(colors='white')
        ax2.legend(facecolor='#1e293b', labelcolor='white')
        st.pyplot(fig2, use_container_width=True)
        
        # Jumlah per Cluster
        st.markdown("<h3 style='color:#2dd4bf;'>📊 Jumlah Siswa per Cluster</h3>", unsafe_allow_html=True)
        jumlah = df['Cluster'].value_counts().sort_index()
        label = ['Cluster 1\nElite', 'Cluster 2\nProgresif', 'Cluster 3\nBinaan']
        warna = ['#14b8a6', '#f59e0b', '#ef4444']
        fig3, ax3 = plt.subplots(figsize=(8,4), dpi=100)
        bars = ax3.bar(label, jumlah, color=warna, alpha=0.85, edgecolor='white', linewidth=1.5)
        ax3.set_ylabel("Jumlah Siswa", color='white')
        ax3.set_xlabel("Kategori Prestasi", color='white')
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_facecolor('#1e293b')
        fig3.patch.set_facecolor('#0f172a')
        ax3.tick_params(colors='white')
        for bar, count in zip(bars, jumlah):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(count), ha='center', va='bottom', color='white', fontweight='bold')
        st.pyplot(fig3, use_container_width=True)
        
        # Sebaran Nilai
        st.markdown("<h3 style='color:#2dd4bf;'>📌 Sebaran Nilai Rata-Rata Siswa</h3>", unsafe_allow_html=True)
        fig4, ax4 = plt.subplots(figsize=(10,5), dpi=100)
        warna_map = {1:'#14b8a6', 2:'#f59e0b', 3:'#ef4444'}
        label_map = {1:'Elite ✨', 2:'Progresif 📘', 3:'Binaan 🌱'}
        for cluster_num in [1,2,3]:
            data_plot = df[df['Cluster']==cluster_num]
            ax4.scatter(data_plot['PERINGKAT'], data_plot['RATA_RATA_NILAI'], 
                       c=warna_map[cluster_num], label=label_map[cluster_num], 
                       alpha=0.7, s=80, edgecolors='white', linewidth=1)
        ax4.legend(facecolor='#1e293b', labelcolor='white')
        ax4.set_ylabel("Rata-Rata Nilai", color='white')
        ax4.set_xlabel("Peringkat Siswa", color='white')
        ax4.grid(True, alpha=0.2)
        ax4.set_facecolor('#1e293b')
        fig4.patch.set_facecolor('#0f172a')
        ax4.tick_params(colors='white')
        st.pyplot(fig4, use_container_width=True)

# ================================================================
# =================== HALAMAN TENTANG SISTEM =====================
# ================================================================
elif menu == "⚙️ Tentang Sistem":
    st.markdown("""
    <div class='card'>
        <h1 style='color:#2dd4bf;'>⚙️ Tentang Sistem</h1>
        <hr>
        <p style='font-size:1.1rem; line-height:1.8; color:#cbd5e1;'>
            <b style='color:#2dd4bf;'>Metode Pengolahan:</b>
        </p>
        <ul style='color:#cbd5e1; font-size:1rem; line-height:1.8;'>
            <li>✅ Seleksi Atribut</li>
            <li>✅ Normalisasi Min-Max</li>
            <li>✅ K-Means Clustering (3 Cluster)</li>
            <li>✅ Perhitungan Rata-Rata Nilai</li>
            <li>✅ Penentuan Peringkat (1 s.d dst)</li>
            <li>✅ Evaluasi: Elbow Method & Silhouette Score</li>
        </ul>
        <hr>
        <p style='text-align:center; color:#94a3b8; font-size:0.9rem; margin-top:1rem;'>
            © 2026 Sistem Pengelompokan Prestasi Siswa - SMP Islam Sunan Bonang
        </p>
    </div>
    """, unsafe_allow_html=True)