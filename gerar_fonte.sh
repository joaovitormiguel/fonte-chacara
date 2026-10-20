#!/bin/sh
# Regera a fonte a partir da foto (src/placa.jpg). Requer: python3, numpy, opencv, scipy, scikit-image, fonttools.
set -e
cd "$(dirname "$0")"
python3 work/seg2.py        # separa letras da madeira
python3 work/extract.py     # recorta cada letra em alta resolução
python3 work/clean.py       # limpa e normaliza altura
python3 work/build_masks.py # desenha todos os glifos (originais + novos)
python3 work/build_font.py  # vetoriza e gera build/Chacara-Regular.otf/.ttf
