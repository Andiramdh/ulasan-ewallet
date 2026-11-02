"""Praproses teks untuk dashboard.

HARUS sama persis dengan notebooks/01_data.ipynb sel 2.8, karena model TF-IDF dilatih dengan `teks_penuh` hasil
fungsi itu. Kamus (`slang_final.csv`) dan stopword (`stopword.txt`) disalin dari Drive oleh 05_insight.ipynb (7.11).
"""
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

FOLDER_KAMUS = Path(__file__).parent / 'kamus'

# Frasa yang disatukan sebelum pemetaan per kata
ATURAN_FRASA = [
    (r'\bgo\s*-?\s*pay\b', 'gopay'),
    (r'\btop\s*-?\s*up\b', 'topup'),
    (r'\bpay\s*-?\s*later\b', 'paylater'),
    (r'\bcustomer\s+service\b', 'cs'),
    (r'\be\s*-?\s*wallet\b', 'ewallet'),
]
POLA_URL = re.compile(r'https?://\S+|www\.\S+')
POLA_EMAIL = re.compile(r'\S+@\S+')
POLA_ANGKA_PANJANG = re.compile(r'\d{6,}')


@lru_cache(maxsize=1)
def kamus_slang():
    tabel = pd.read_csv(FOLDER_KAMUS / 'slang_final.csv', dtype=str, keep_default_na=False)
    return dict(zip(tabel['slang'], tabel['formal']))


@lru_cache(maxsize=1)
def stopword():
    return frozenset((FOLDER_KAMUS / 'stopword.txt').read_text(encoding='utf-8').split())


def normalisasi(teks):
    """Teks -> daftar token ternormalisasi (huruf kecil, frasa disatukan, slang dipetakan)."""
    kamus = kamus_slang()
    teks = teks.lower()
    teks = POLA_URL.sub(' ', teks)
    teks = POLA_EMAIL.sub(' email ', teks)
    teks = POLA_ANGKA_PANJANG.sub(' nomor ', teks)
    for pola, ganti in ATURAN_FRASA:
        teks = re.sub(pola, ganti, teks)
    teks = re.sub(r'([a-z])\1{2,}', r'\1', teks)  # mantappp -> mantap
    token = []
    for t in re.findall(r'[a-z0-9]+', teks):
        token.extend(kamus.get(t, t).split())
    return token


def teks_penuh(teks):
    """Input model TF-IDF: token ternormalisasi tanpa stopword (kata negasi dipertahankan)."""
    daftar_stopword = stopword()
    return ' '.join(w for w in normalisasi(teks) if w not in daftar_stopword)
