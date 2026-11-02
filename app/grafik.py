"""Konstanta tampilan dan pembantu grafik Plotly untuk dashboard."""
URUTAN_APLIKASI = ['dana', 'gopay', 'ovo']
NAMA = {'dana': 'DANA', 'gopay': 'GoPay', 'ovo': 'OVO'}
# Warna mengikuti aplikasi dan tidak pernah berubah (palet tervalidasi, 3 slot pertama aman untuk buta warna)
WARNA = {'dana': '#2a78d6', 'gopay': '#eb6834', 'ovo': '#1baf7a'}

ASPEK = ['transaksi', 'keamanan', 'akun', 'error_performa', 'cs', 'biaya_promo', 'paylater', 'kemudahan']
NAMA_ASPEK = {
    'transaksi': 'Transaksi', 'keamanan': 'Keamanan', 'akun': 'Akun', 'error_performa': 'Error/performa',
    'cs': 'Layanan CS', 'biaya_promo': 'Biaya/promo', 'paylater': 'Paylater', 'kemudahan': 'Kemudahan',
}

KELAS = ['neg', 'net', 'pos']
NAMA_SENTIMEN = {'neg': 'Negatif', 'net': 'Netral', 'pos': 'Positif'}
# Polaritas: dua kutub berlawanan (merah vs ungu) + abu-abu netral, sengaja berbeda dari warna aplikasi
WARNA_SENTIMEN = {'neg': '#e34948', 'net': '#a8a69e', 'pos': '#6250d6'}

GARIS = '#c3c2b7'
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def rapikan(fig, tinggi=420, legenda=True, grid_x=False):
    """Gaya bersama: font sistem, pemisah desimal Indonesia, legenda di atas, grid sumbu x hanya kalau diminta."""
    fig.update_layout(
        height=tinggi,
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(family=FONT),
        separators=',.',
        showlegend=legenda,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0, title_text=''),
        hoverlabel=dict(font_size=13),
    )
    fig.update_xaxes(showgrid=grid_x, zeroline=False)
    fig.update_yaxes(zeroline=False)
    return fig


def posisi_label(x, y):
    """Pilih posisi label titik agar label yang berdekatan tidak bertumpuk (atas, bawah, kanan, kiri)."""
    x, y = list(x), list(y)
    jarak_x = 0.12 * ((max(x) - min(x)) or 1)
    jarak_y = 0.12 * ((max(y) - min(y)) or 1)
    pilihan = ['top center', 'bottom center', 'middle right', 'middle left']
    posisi = [None] * len(x)
    for i in sorted(range(len(x)), key=lambda k: -y[k]):
        dipakai = {posisi[j] for j in range(len(x))
                   if posisi[j] and abs(x[j] - x[i]) < jarak_x and abs(y[j] - y[i]) < jarak_y}
        posisi[i] = next((p for p in pilihan if p not in dipakai), 'top center')
    return posisi


BULAN_ID = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des']


def label_bulan(bulan):
    """'2026-07' -> 'Jul 2026'."""
    tahun, nomor = str(bulan).split('-')
    return f'{BULAN_ID[int(nomor) - 1]} {tahun}'


def rgba(warna_hex, alpha):
    r, g, b = (int(warna_hex[i:i + 2], 16) for i in (1, 3, 5))
    return f'rgba({r},{g},{b},{alpha})'


def angka(x, desimal=0):
    """Format angka gaya Indonesia: titik pemisah ribuan, koma desimal."""
    teks = f'{x:,.{desimal}f}'
    return teks.replace(',', '_').replace('.', ',').replace('_', '.')


def persen(x, desimal=1):
    return f'{angka(x, desimal)}%'
