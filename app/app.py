"""Dashboard ulasan e-wallet (DANA, GoPay, OVO).

Jalankan dari folder repo:  streamlit run app/app.py
Semua angka dibaca dari app/data/ (hasil 05_insight.ipynb 7.10-7.12). Dashboard tidak melatih model; contoh ulasan
yang ditampilkan sudah disaring dan disamarkan.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from grafik import (ASPEK, GARIS, KELAS, NAMA, NAMA_ASPEK, NAMA_SENTIMEN, URUTAN_APLIKASI, WARNA, WARNA_SENTIMEN,
                    angka, label_bulan, persen, posisi_label, rapikan, rgba)
from praproses import teks_penuh

FOLDER_APP = Path(__file__).parent
FOLDER_DATA = FOLDER_APP / 'data'
FOLDER_MODEL = FOLDER_APP / 'models'
DATA_WAJIB = ['kubus', 'harian', 'versi', 'kata_keluhan', 'contoh_ulasan', 'matriks_prioritas', 'sentimen_vs_rating',
              'lonjakan_versi', 'insiden', 'respons_developer', 'respons_per_aspek', 'model_dev_sentimen',
              'model_dev_aspek']
FILE_WAJIB = [FOLDER_DATA / f'{n}.parquet' for n in DATA_WAJIB] + [
    FOLDER_DATA / 'info.json', FOLDER_MODEL / 'tfidf_sentimen_gold.joblib', FOLDER_MODEL / 'tfidf_aspek.joblib',
    FOLDER_APP / 'kamus' / 'slang_final.csv', FOLDER_APP / 'kamus' / 'stopword.txt',
]
MERAH_INSIDEN = '#e34948'

st.set_page_config(page_title='Ulasan E-Wallet', layout='wide')


@st.cache_data
def muat(nama):
    return pd.read_parquet(FOLDER_DATA / f'{nama}.parquet')


@st.cache_data
def muat_info():
    return json.loads((FOLDER_DATA / 'info.json').read_text(encoding='utf-8'))


@st.cache_resource
def muat_model(nama):
    return joblib.load(FOLDER_MODEL / f'{nama}.joblib')


def tampilkan(fig):
    st.plotly_chart(fig, use_container_width=True, theme='streamlit')


def judul_aplikasi(nama):
    st.markdown(f"<h4 style='color:{WARNA[nama]};margin:0 0 .25rem 0'>{NAMA[nama]}</h4>", unsafe_allow_html=True)


def ringkas_kubus(k):
    """Angka kunci dari potongan kubus: jumlah ulasan, rating rata-rata, % isi keluhan, % bintang 4-5 yang mengeluh."""
    n = k['jumlah'].sum()
    b45 = k[k['rating'] >= 4]
    n45 = b45['jumlah'].sum()
    return {
        'n': n,
        'rating': (k['rating'] * k['jumlah']).sum() / n if n else np.nan,
        'keluhan': k.loc[k['sentimen'] == 'neg', 'jumlah'].sum() / n * 100 if n else np.nan,
        'tersembunyi': b45.loc[b45['sentimen'] == 'neg', 'jumlah'].sum() / n45 * 100 if n45 else np.nan,
    }


def label_periode(info):
    awal, akhir = info['periode'].split(' s.d. ')
    return f'{label_bulan(awal)} - {label_bulan(akhir)}'


# ----------------------------------------------------------------------------------------- Sekilas 3 aplikasi
def halaman_sekilas():
    info, kubus = muat_info(), muat('kubus')
    respons = muat('respons_developer').set_index('aplikasi')
    matriks = muat('matriks_prioritas')

    st.title('Ulasan e-wallet: DANA, GoPay, dan OVO')
    st.caption(f"{angka(info['jumlah_ulasan'])} ulasan Google Play · {label_periode(info)} · "
               'sentimen dan aspek dibaca oleh model IndoBERT')

    for kolom, nama in zip(st.columns(3), URUTAN_APLIKASI):
        r = ringkas_kubus(kubus[kubus['aplikasi'] == nama])
        with kolom, st.container(border=True):
            judul_aplikasi(nama)
            a, b = st.columns(2)
            a.metric('Ulasan', angka(r['n']))
            b.metric('Rating rata-rata', angka(r['rating'], 2))
            a.metric('Isi berupa keluhan', persen(r['keluhan']))
            b.metric('Dibalas developer', persen(respons.loc[nama, 'persen_dibalas_semua']))

    kiri, kanan = st.columns(2)
    with kiri:
        st.subheader('Keluhan per bulan')
        bulanan = (kubus.assign(neg=np.where(kubus['sentimen'] == 'neg', kubus['jumlah'], 0))
                   .groupby(['aplikasi', 'bulan'])[['jumlah', 'neg']].sum().reset_index())
        bulanan['persen'] = bulanan['neg'] / bulanan['jumlah'] * 100
        fig = go.Figure()
        for nama in URUTAN_APLIKASI:
            b = bulanan[bulanan['aplikasi'] == nama].sort_values('bulan')
            fig.add_trace(go.Scatter(x=b['bulan'].map(label_bulan), y=b['persen'], name=NAMA[nama],
                                     mode='lines+markers', line=dict(color=WARNA[nama], width=2), marker=dict(size=6),
                                     hovertemplate=f'{NAMA[nama]}: ' + '%{y:.1f}%<extra></extra>'))
        fig.update_layout(yaxis=dict(title='% ulasan', rangemode='tozero'), hovermode='x unified')
        tampilkan(rapikan(fig, tinggi=360))
        st.caption('Persen ulasan tiap bulan yang isinya keluhan.')
    with kanan:
        st.subheader('Bintang vs isi ulasan')
        data = muat('sentimen_vs_rating')
        data['label'] = data['aplikasi'].map(NAMA) + ' · ' + data['kelompok_rating'].astype(str) + ' ★'
        fig = go.Figure()
        for kelas in KELAS:
            fig.add_trace(go.Bar(
                y=data['label'], x=data[kelas], name=f'Isi {NAMA_SENTIMEN[kelas].lower()}', orientation='h',
                marker_color=WARNA_SENTIMEN[kelas], texttemplate='%{x:.0f}%', textposition='inside',
                insidetextanchor='middle', hovertemplate='%{y}: %{x:.1f}%<extra></extra>',
            ))
        fig.update_layout(barmode='stack', bargap=0.3, uniformtext=dict(minsize=10, mode='hide'),
                          xaxis=dict(title='% ulasan', range=[0, 100]), yaxis=dict(autorange='reversed'))
        tampilkan(rapikan(fig, tinggi=360))
        st.caption('Banyak ulasan berbintang 4-5 yang isinya tetap keluhan.')

    st.subheader('Aspek yang paling sering dibicarakan')
    posisi = np.arange(len(ASPEK))[::-1]
    fig = go.Figure()
    for geser, nama in zip([0.22, 0, -0.22], URUTAN_APLIKASI):
        a = matriks[matriks['aplikasi'] == nama].set_index('aspek').loc[ASPEK]
        fig.add_trace(go.Scatter(
            x=a['persen_koreksi'], y=posisi + geser, mode='markers', name=NAMA[nama],
            marker=dict(size=11, color=WARNA[nama], line=dict(width=1.5, color='rgba(255,255,255,0.9)')),
            error_x=dict(type='data', symmetric=False, array=a['persen_atas'] - a['persen_koreksi'],
                         arrayminus=a['persen_koreksi'] - a['persen_bawah'], color=WARNA[nama], thickness=1, width=0),
            customdata=[NAMA_ASPEK[x] for x in ASPEK],
            hovertemplate=f'{NAMA[nama]} · ' + '%{customdata}: %{x:.1f}%<extra></extra>',
        ))
    fig.update_layout(xaxis=dict(title='% ulasan yang menyebut aspek', rangemode='tozero'),
                      yaxis=dict(tickvals=posisi, ticktext=[NAMA_ASPEK[a] for a in ASPEK], showgrid=False))
    tampilkan(rapikan(fig, tinggi=460, grid_x=True))
    st.caption('Persen ulasan yang menyebut tiap aspek, sudah dikoreksi dari kesalahan model (garis = rentang 95%).')

    st.subheader('Matriks prioritas')
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=[NAMA[n] for n in URUTAN_APLIKASI])
    for i, nama in enumerate(URUTAN_APLIKASI, start=1):
        m = matriks[matriks['aplikasi'] == nama].reset_index(drop=True)
        fig.add_trace(go.Scatter(
            x=m['persen_koreksi'], y=m['dampak'], mode='markers+text', showlegend=False,
            text=m['aspek'].map(NAMA_ASPEK), textposition=posisi_label(m['persen_koreksi'], m['dampak']),
            textfont=dict(size=11),
            marker=dict(size=12, color=WARNA[nama], line=dict(width=1.5, color='rgba(255,255,255,0.9)')),
            customdata=m['kontribusi'],
            hovertemplate='<b>%{text}</b><br>Frekuensi %{x:.1f}% · dampak %{y:.2f} bintang'
                          '<br>Rating rata-rata turun %{customdata:.2f} bintang<extra></extra>',
        ), row=1, col=i)
        fig.add_vline(x=m['persen_koreksi'].mean(), line_width=1, line_color=GARIS, row=1, col=i)
        fig.add_hline(y=m['dampak'].mean(), line_width=1, line_color=GARIS, row=1, col=i)
    fig.update_xaxes(title_text='% ulasan', rangemode='tozero')
    fig.update_yaxes(title_text='Dampak (bintang lebih rendah)', row=1, col=1)
    tampilkan(rapikan(fig, tinggi=440, legenda=False, grid_x=True))
    st.caption('Kanan atas = sering muncul dan paling menurunkan rating. Di bawah nol = berasosiasi dengan rating '
               'lebih tinggi.')

    kiri, kanan = st.columns([3, 2])
    with kiri:
        st.subheader('Seberapa sering developer membalas')
        kategori = {
            'persen_dibalas_bintang_1_2_isi_negatif': '1-2 ★, isi keluhan',
            'persen_dibalas_bintang_4_5_isi_negatif': '4-5 ★, isi keluhan',
            'persen_dibalas_bintang_4_5_isi_positif': '4-5 ★, isi pujian',
        }
        fig = go.Figure()
        for nama in URUTAN_APLIKASI:
            fig.add_trace(go.Bar(
                x=list(kategori.values()), y=[respons.loc[nama, k] for k in kategori], name=NAMA[nama],
                marker_color=WARNA[nama], texttemplate='%{y:.0f}%', textposition='outside',
                hovertemplate=f'{NAMA[nama]} · ' + '%{x}: %{y:.1f}% dibalas<extra></extra>',
            ))
        fig.update_layout(barmode='group', bargap=0.25, bargroupgap=0.08,
                          yaxis=dict(title='% ulasan dibalas', range=[0, 105]))
        tampilkan(rapikan(fig, tinggi=360))
        st.caption('Keluhan berbintang 4-5 mudah terlewat kalau ulasan dipilah menurut bintang.')
    with kanan:
        st.subheader('Waktu sampai dibalas')
        median = respons['median_jam_balasan'].reindex(URUTAN_APLIKASI)
        fig = go.Figure(go.Bar(
            y=[NAMA[n] for n in URUTAN_APLIKASI], x=median.fillna(0), orientation='h',
            marker_color=[WARNA[n] for n in URUTAN_APLIKASI],
            text=[f'{angka(v, 1)} jam' if pd.notna(v) else 'tidak ada balasan' for v in median],
            textposition='outside', hovertemplate='%{y}: %{text}<extra></extra>',
        ))
        fig.update_layout(xaxis=dict(title='Median jam', range=[0, max(median.max() * 1.6, 1)]),
                          yaxis=dict(autorange='reversed'))
        tampilkan(rapikan(fig, tinggi=360, legenda=False))
        st.caption('Median jam dari ulasan ditulis sampai dibalas.')

    with st.expander('Lihat angka'):
        st.dataframe(matriks[['aplikasi', 'aspek', 'persen_koreksi', 'persen_bawah', 'persen_atas', 'dampak',
                              'kontribusi']].round(2), hide_index=True, use_container_width=True)
        st.dataframe(respons.round(1), use_container_width=True)


# ------------------------------------------------------------------------------------------ Jelajah per aplikasi
def halaman_jelajah():
    kubus, harian = muat('kubus'), muat('harian')
    st.title('Per aplikasi')

    a, b, c = st.columns([1.2, 2, 1.6])
    nama = a.radio('Aplikasi', URUTAN_APLIKASI, format_func=NAMA.get, horizontal=True)
    daftar_bulan = sorted(kubus['bulan'].unique())
    mulai, akhir = b.select_slider('Periode', options=daftar_bulan, value=(daftar_bulan[0], daftar_bulan[-1]),
                                   format_func=label_bulan)
    bintang = c.multiselect('Bintang', [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5], format_func=lambda r: f'{r} ★')
    if not bintang:
        st.info('Pilih minimal satu bintang.')
        return

    k = kubus[(kubus['aplikasi'] == nama) & kubus['bulan'].between(mulai, akhir) & kubus['rating'].isin(bintang)]
    r = ringkas_kubus(k)
    if not r['n']:
        st.info('Tidak ada ulasan untuk filter ini.')
        return
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Ulasan', angka(r['n']))
    m2.metric('Rating rata-rata', angka(r['rating'], 2))
    m3.metric('Isi berupa keluhan', persen(r['keluhan']))
    m4.metric('Bintang 4-5 tapi mengeluh', persen(r['tersembunyi']) if pd.notna(r['tersembunyi']) else '-')

    kiri, kanan = st.columns(2)
    with kiri:
        st.subheader('Isi ulasan per bintang')
        jumlah = k.groupby(['rating', 'sentimen'])['jumlah'].sum().unstack(fill_value=0).reindex(columns=KELAS,
                                                                                                fill_value=0)
        bagian = jumlah.div(jumlah.sum(axis=1), axis=0) * 100
        fig = go.Figure()
        for kelas in KELAS:
            fig.add_trace(go.Bar(
                x=[f'{i} ★' for i in bagian.index], y=bagian[kelas], name=f'Isi {NAMA_SENTIMEN[kelas].lower()}',
                marker_color=WARNA_SENTIMEN[kelas], customdata=jumlah[kelas],
                hovertemplate='%{x}: %{y:.1f}% (%{customdata:,} ulasan)<extra></extra>',
            ))
        fig.update_layout(barmode='stack', bargap=0.35, yaxis=dict(title='% ulasan', range=[0, 100]))
        tampilkan(rapikan(fig, tinggi=340))
        st.caption('Tiap batang = satu kelompok bintang; warna = isi ulasan.')
    with kanan:
        st.subheader('Ulasan per hari')
        h = harian[(harian['aplikasi'] == nama) & harian['rating'].isin(bintang)
                   & harian['hari'].dt.strftime('%Y-%m').between(mulai, akhir)]
        h = h.groupby('hari').agg(jumlah=('jumlah', 'sum'), negatif=('negatif', 'sum'),
                                  lonjakan=('hari_lonjakan', 'max')).reset_index()
        h['persen'] = h['negatif'] / h['jumlah'] * 100
        fig = go.Figure(go.Scatter(
            x=h['hari'], y=h['jumlah'], mode='lines', line=dict(color=WARNA[nama], width=1.5), name='Ulasan',
            customdata=h['persen'], hovertemplate='%{x|%d %b %Y}: %{y:,} ulasan · %{customdata:.0f}% keluhan'
                                                  '<extra></extra>',
        ))
        puncak = h[h['lonjakan']]
        fig.add_trace(go.Scatter(x=puncak['hari'], y=puncak['jumlah'], mode='markers', name='Hari gangguan',
                                 marker=dict(size=10, color=MERAH_INSIDEN), hoverinfo='skip'))
        fig.update_layout(yaxis_title='Ulasan per hari')
        tampilkan(rapikan(fig, tinggi=340))
        st.caption('Titik merah = hari gangguan (volume > 3x hari biasa).')

    st.subheader('Apa yang dibicarakan')
    n = r['n']
    per_aspek = pd.DataFrame({s: [k.loc[k['sentimen'] == s, a].sum() / n * 100 for a in ASPEK] for s in KELAS},
                             index=ASPEK)
    per_aspek = per_aspek.loc[per_aspek.sum(axis=1).sort_values().index]
    fig = go.Figure()
    for kelas, label in [('neg', 'Keluhan'), ('net', 'Netral'), ('pos', 'Pujian')]:
        fig.add_trace(go.Bar(
            y=[NAMA_ASPEK[a] for a in per_aspek.index], x=per_aspek[kelas], name=label, orientation='h',
            marker_color=WARNA_SENTIMEN[kelas], hovertemplate='%{y} · ' + label + ': %{x:.1f}% ulasan<extra></extra>',
        ))
    fig.update_layout(barmode='stack', bargap=0.3, xaxis_title='% ulasan yang menyebut aspek')
    tampilkan(rapikan(fig, tinggi=380, grid_x=True))
    st.caption('Panjang batang = seberapa sering aspek disebut; warna = apakah disebut sebagai keluhan atau pujian.')

    st.subheader('Aspek dari bulan ke bulan')
    per_bulan = k.groupby('bulan')[['jumlah'] + ASPEK].sum()
    persen_bulan = per_bulan[ASPEK].div(per_bulan['jumlah'], axis=0) * 100
    fig = go.Figure(go.Heatmap(
        z=persen_bulan.T.to_numpy(), x=[label_bulan(b) for b in persen_bulan.index], y=[NAMA_ASPEK[a] for a in ASPEK],
        colorscale=[[0, rgba(WARNA[nama], 0.04)], [1, WARNA[nama]]], texttemplate='%{z:.0f}', textfont=dict(size=11),
        hovertemplate='%{y} · %{x}: %{z:.1f}% ulasan<extra></extra>', colorbar=dict(title='%', thickness=12),
    ))
    fig.update_layout(yaxis=dict(autorange='reversed'))
    tampilkan(rapikan(fig, tinggi=380, legenda=False))
    st.caption('Angka = % ulasan tiap bulan yang menyebut aspek; makin gelap makin sering.')

    st.subheader('Lihat lebih dekat satu aspek')
    keluhan_terbanyak = max(ASPEK, key=lambda a: k.loc[k['sentimen'] == 'neg', a].sum())
    aspek = st.selectbox('Aspek', ASPEK, index=ASPEK.index(keluhan_terbanyak), format_func=NAMA_ASPEK.get)
    kiri, kanan = st.columns([2, 3])
    with kiri:
        kata = muat('kata_keluhan')
        kata = kata[(kata['aplikasi'] == nama) & (kata['aspek'] == aspek)].head(10).iloc[::-1]
        if kata.empty:
            st.caption('Keluhan untuk aspek ini terlalu sedikit.')
        else:
            fig = go.Figure(go.Bar(y=kata['ngram'], x=kata['frekuensi'], orientation='h',
                                   marker_color=WARNA_SENTIMEN['neg'],
                                   hovertemplate='%{y}: %{x:,} kali<extra></extra>'))
            fig.update_layout(xaxis_title='Muncul (kali)', title=dict(text='Kata yang khas di keluhan', x=0))
            tampilkan(rapikan(fig, tinggi=360, legenda=False, grid_x=True))
            st.caption('Kata/frasa yang paling membedakan keluhan aspek ini dari keluhan lain (seluruh periode).')
    with kanan:
        jenis = st.radio('Contoh ulasan', ['neg', 'pos'], format_func={'neg': 'Keluhan', 'pos': 'Pujian'}.get,
                         horizontal=True)
        contoh = muat('contoh_ulasan')
        contoh = contoh[(contoh['aplikasi'] == nama) & (contoh['aspek'] == aspek) & (contoh['sentimen'] == jenis)]
        if contoh.empty:
            st.caption('Belum ada contoh yang lolos penyaringan untuk pilihan ini.')
        for c in contoh.head(3).itertuples():
            with st.container(border=True):
                st.markdown(f'"{c.teks}"')
                st.caption(f"{'★' * c.rating}{'☆' * (5 - c.rating)} · {label_bulan(c.bulan)}")
        st.caption('Contoh ulasan asli yang sudah disaring (tanpa angka panjang, tautan, dan kata kasar).')


# ----------------------------------------------------------------------------------------------- Versi & insiden
def halaman_versi():
    st.title('Update & gangguan')
    nama = st.radio('Aplikasi', URUTAN_APLIKASI, format_func=NAMA.get, horizontal=True)

    v = muat('versi')
    v = v[v['aplikasi'] == nama].copy()
    lonjakan = muat('lonjakan_versi')
    lonjakan = lonjakan[(lonjakan['aplikasi'] == nama) & lonjakan['lonjakan']]
    aspek_naik = lonjakan.groupby('versi')['aspek'].apply(lambda s: ', '.join(NAMA_ASPEK[a] for a in s))
    v['aspek_naik'] = v['versi'].map(aspek_naik)
    v['ukuran'] = 9 + 26 * np.sqrt(v['jumlah'] / v['jumlah'].max())
    insiden = muat('insiden')
    insiden = insiden[insiden['aplikasi'] == nama].sort_values('hari')

    fig = go.Figure()
    for ada, label, simbol, opasitas in [(False, 'Versi', 'circle', 0.55), (True, 'Versi dengan lonjakan keluhan',
                                                                            'diamond', 1.0)]:
        bagian = v[v['aspek_naik'].notna() == ada]
        fig.add_trace(go.Scatter(
            x=bagian['pertama_muncul'], y=bagian['persen_keluhan'], mode='markers', name=label,
            marker=dict(size=bagian['ukuran'], color=WARNA[nama], symbol=simbol, opacity=opasitas,
                        line=dict(width=1, color='rgba(255,255,255,0.9)')),
            customdata=np.column_stack([bagian['versi'], bagian['jumlah'].map(angka), bagian['aspek_naik'].fillna('-')]),
            hovertemplate='Versi %{customdata[0]} · mulai %{x|%d %b %Y}<br>%{y:.1f}% keluhan · %{customdata[1]} '
                          'ulasan<br>Aspek yang melonjak: %{customdata[2]}<extra></extra>',
        ))
    for hari in insiden['hari']:
        fig.add_vline(x=pd.Timestamp(hari).strftime('%Y-%m-%d'), line_width=1, line_color=MERAH_INSIDEN)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Hari gangguan',
                             line=dict(color=MERAH_INSIDEN, width=1)))
    fig.update_layout(yaxis=dict(title='% ulasan berisi keluhan', rangemode='tozero'), xaxis_title='Versi mulai dipakai')
    tampilkan(rapikan(fig, tinggi=440))
    st.caption('Tiap titik = satu versi (makin besar = makin banyak ulasan). Belah ketupat = versi yang keluhannya '
               'melonjak; garis merah = hari gangguan.')

    if insiden.empty:
        return
    st.subheader('Apa yang dikeluhkan di hari gangguan')
    hari_pilih = st.selectbox('Hari gangguan', list(insiden['hari']), format_func=lambda h: h.strftime('%d %b %Y'))
    baris = insiden[insiden['hari'] == hari_pilih].iloc[0]
    kubus = muat('kubus')
    normal = kubus[(kubus['aplikasi'] == nama) & ~kubus['hari_lonjakan']]
    persen_normal = normal[ASPEK].sum() / normal['jumlah'].sum() * 100
    fig = go.Figure([
        go.Bar(x=[NAMA_ASPEK[a] for a in ASPEK], y=[baris[a] for a in ASPEK], name=hari_pilih.strftime('%d %b %Y'),
               marker_color=MERAH_INSIDEN, hovertemplate='%{x}: %{y:.1f}% ulasan<extra></extra>'),
        go.Bar(x=[NAMA_ASPEK[a] for a in ASPEK], y=persen_normal.to_numpy(), name='Hari biasa', marker_color=GARIS,
               hovertemplate='%{x}: %{y:.1f}% ulasan<extra></extra>'),
    ])
    fig.update_layout(barmode='group', bargap=0.25, bargroupgap=0.08, yaxis_title='% ulasan yang menyebut aspek')
    tampilkan(rapikan(fig, tinggi=380))
    st.caption(f"{angka(baris['jumlah'])} ulasan di hari itu, {persen(baris['persen_negatif'], 0)} berisi keluhan "
               f"(hari biasa {persen(baris['persen_negatif_normal'], 0)}).")

    with st.expander('Lihat angka lonjakan per versi'):
        tabel = muat('lonjakan_versi')
        tabel = tabel[tabel['aplikasi'] == nama].assign(aspek=lambda t: t['aspek'].map(NAMA_ASPEK))
        st.dataframe(tabel[['versi', 'aspek', 'n_versi', 'persen_versi', 'persen_lain', 'selisih_pp',
                            'selisih_pp_tanpa_lonjakan']].round(1), hide_index=True, use_container_width=True)


# -------------------------------------------------------------------------------------------------- Coba model
CONTOH = {
    'Top up belum masuk': 'top up dari bank sudah terpotong tapi saldo belum masuk, cs cuma suruh nunggu 2x24 jam',
    'Pujian': 'aplikasinya gampang dipakai dan transfernya cepet, mantap',
    'Masalah login': 'gak bisa login katanya perangkat dimodifikasi padahal hp original',
    'Paylater belum muncul': 'kenapa fitur paylater di akun saya belum muncul juga',
    'Biaya admin': 'tiap top up kena biaya admin terus, mending pindah aplikasi lain',
}


def kata_berpengaruh(pipeline, teks, kelas, n=8):
    """Kontribusi tiap kata (fitur TF-IDF kata) ke kelas terprediksi: nilai TF-IDF x koefisien (dipusatkan)."""
    vektor, model = pipeline.named_steps['vektor'], pipeline.named_steps['model']
    nama_fitur = vektor.transformer_list[0][1].get_feature_names_out()
    baris = vektor.transform([teks]).tocsr()
    koef = model.coef_ - model.coef_.mean(axis=0)
    k = list(model.classes_).index(kelas)
    hasil = [(nama_fitur[j], v * koef[k, j]) for j, v in zip(baris.indices, baris.data) if j < len(nama_fitur)]
    return pd.DataFrame(hasil, columns=['kata', 'kontribusi']).sort_values('kontribusi', ascending=False).head(n)


def halaman_coba():
    st.title('Coba model')
    st.caption('Ketik ulasan, model menebak sentimen dan aspeknya. Model demo: TF-IDF + Logistic Regression '
               '(versi ringan dari model analisis). Teks tidak disimpan.')
    contoh = st.selectbox('Contoh', ['(tulis sendiri)'] + list(CONTOH))
    teks = st.text_area('Ulasan', value=CONTOH.get(contoh, ''), height=100, max_chars=1000,
                        placeholder='Tulis ulasan aplikasi e-wallet')
    if not teks.strip():
        return
    masukan = teks_penuh(teks)
    if not masukan:
        st.warning('Tidak ada kata yang tersisa setelah praproses. Coba tulis ulasan yang lebih lengkap.')
        return

    model_s, model_a = muat_model('tfidf_sentimen_gold'), muat_model('tfidf_aspek')
    prob_s = pd.Series(model_s.predict_proba([masukan])[0], index=model_s.classes_)
    kelas = prob_s.idxmax()
    prob_a = pd.Series(model_a.predict_proba([masukan])[0], index=ASPEK)

    kiri, kanan = st.columns(2)
    with kiri, st.container(border=True):
        st.markdown(f'**Sentimen: {NAMA_SENTIMEN[kelas]}** ({persen(prob_s[kelas] * 100, 0)})')
        fig = go.Figure(go.Bar(
            x=[prob_s[k] * 100 for k in KELAS], y=[NAMA_SENTIMEN[k] for k in KELAS], orientation='h',
            marker_color=[WARNA_SENTIMEN[k] for k in KELAS], texttemplate='%{x:.0f}%', textposition='outside',
            hovertemplate='%{y}: %{x:.1f}%<extra></extra>',
        ))
        fig.update_layout(xaxis=dict(range=[0, 110], title='Probabilitas (%)'), yaxis=dict(autorange='reversed'))
        tampilkan(rapikan(fig, tinggi=210, legenda=False))
        kata = kata_berpengaruh(model_s, masukan, kelas)
        kata = kata[kata['kontribusi'] > 0].iloc[::-1]
        if not kata.empty:
            fig = go.Figure(go.Bar(y=kata['kata'], x=kata['kontribusi'], orientation='h',
                                   marker_color=WARNA_SENTIMEN[kelas], hovertemplate='%{y}<extra></extra>'))
            fig.update_layout(xaxis=dict(title='Pengaruh', showticklabels=False),
                              title=dict(text=f'Kata yang mendorong "{NAMA_SENTIMEN[kelas].lower()}"', x=0))
            tampilkan(rapikan(fig, tinggi=260, legenda=False))
    with kanan, st.container(border=True):
        terdeteksi = [NAMA_ASPEK[a] for a in ASPEK if prob_a[a] >= 0.5]
        st.markdown(f"**Aspek: {', '.join(terdeteksi) if terdeteksi else 'tidak ada aspek spesifik'}**")
        urut = prob_a.sort_values()
        fig = go.Figure(go.Bar(
            x=urut * 100, y=[NAMA_ASPEK[a] for a in urut.index], orientation='h',
            marker_color=['#2a78d6' if p >= 0.5 else GARIS for p in urut],
            hovertemplate='%{y}: %{x:.1f}%<extra></extra>',
        ))
        fig.add_vline(x=50, line_width=1, line_color='#898781')
        fig.update_layout(xaxis=dict(range=[0, 100], title='Probabilitas (%) · garis = ambang 50%'))
        tampilkan(rapikan(fig, tinggi=480, legenda=False))


# ----------------------------------------------------------------------------------------------------- Tentang
def halaman_tentang():
    info = muat_info()
    st.title('Data & metode')
    st.markdown(
        f"- **Data**: {angka(info['jumlah_ulasan'])} ulasan Google Play ({label_periode(info)}) setelah cleaning. "
        'Nama dan foto pengguna tidak pernah diambil.\n'
        '- **Label manual**: +-1.300 ulasan dilabel sentimen dan 8 aspek, dengan bintang dan nama aplikasi '
        'disembunyikan saat pelabelan.\n'
        f"- **Model analisis**: {info['sentimen']} (sentimen) dan {info['aspek']} (aspek)."
    )

    st.subheader('Akurasi model sentimen')
    sentimen = muat('model_dev_sentimen').copy()
    batas = sentimen['ci_95'].str.split('-', expand=True).astype(float)
    sentimen['bawah'], sentimen['atas'] = batas[0], batas[1]
    sentimen = sentimen.sort_values('macro_f1')
    fig = go.Figure(go.Scatter(
        x=sentimen['macro_f1'], y=sentimen['model'], mode='markers',
        marker=dict(size=11, color=np.where(sentimen['kelompok'] == 'IndoBERT', '#2a78d6', '#898781')),
        error_x=dict(type='data', symmetric=False, array=sentimen['atas'] - sentimen['macro_f1'],
                     arrayminus=sentimen['macro_f1'] - sentimen['bawah'], color=GARIS, thickness=1.5, width=0),
        hovertemplate='%{y}: macro-F1 %{x:.3f}<extra></extra>',
    ))
    fig.update_layout(xaxis=dict(title='Macro-F1 (garis = rentang 95%)', range=[0.4, 0.95]),
                      yaxis=dict(title='', showgrid=False))
    tampilkan(rapikan(fig, tinggi=360, legenda=False, grid_x=True))
    st.caption('Biru = IndoBERT, abu-abu = pembanding. Model berlabel rating bintang kalah jauh dari model berlabel '
               'manual. Dinilai di 855 ulasan berlabel manual (cross-validation).')

    st.subheader('Akurasi model aspek')
    aspek = muat('model_dev_aspek')
    fig = go.Figure()
    for model, warna in zip(aspek['model'], ['#898781', '#c3c2b7', '#2a78d6']):
        baris = aspek[aspek['model'] == model].iloc[0]
        fig.add_trace(go.Bar(x=[NAMA_ASPEK[a] for a in ASPEK], y=[baris[f'f1_{a}'] for a in ASPEK], name=model,
                             marker_color=warna, hovertemplate=f'{model} · ' + '%{x}: F1 %{y:.2f}<extra></extra>'))
    fig.update_layout(barmode='group', bargap=0.25, bargroupgap=0.05, yaxis=dict(title='F1', range=[0, 1]))
    tampilkan(rapikan(fig, tinggi=360))

    path_test = FOLDER_DATA / 'model_test_sentimen.parquet'
    if path_test.exists():
        st.subheader('Skor di test set')
        st.dataframe(pd.read_parquet(path_test), hide_index=True, use_container_width=True)

    with st.expander('Keterbatasan'):
        st.markdown(
            '- Ulasan bertulisan tidak mewakili semua pengguna (skor Play Store jauh lebih tinggi).\n'
            '- Aspek dideteksi tanpa sentimen per aspek; hubungan aspek dan rating bersifat korelasional.\n'
            '- Versi aplikasi berimpit dengan periode waktu; tanggal ulasan adalah waktu terakhir diedit.\n'
            '- Skor model berasal dari cross-validation di 855 ulasan berlabel manual. Model juga dipilih dengan '
            'data ini, jadi skornya sedikit optimistis.\n'
            '- Label manual dibuat oleh satu orang.'
        )


# -------------------------------------------------------------------------------------------------- Navigasi
HALAMAN = {
    'Ringkasan': halaman_sekilas,
    'Per aplikasi': halaman_jelajah,
    'Update & gangguan': halaman_versi,
    'Coba model': halaman_coba,
    'Data & metode': halaman_tentang,
}

hilang = [str(p.relative_to(FOLDER_APP)) for p in FILE_WAJIB if not p.exists()]
if hilang:
    st.error('File berikut belum ada di folder app/. Jalankan sel 7.11 dan 7.12 di 05_insight.ipynb, lalu unduh '
             'folder `app/` dari Google Drive ke repo:\n\n' + '\n'.join(f'- `{h}`' for h in hilang))
    st.stop()

pilihan = st.sidebar.radio('Menu', list(HALAMAN))
HALAMAN[pilihan]()
