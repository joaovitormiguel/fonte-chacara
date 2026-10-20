# Fonte Chacara

Fonte feita a partir das letras de madeira recortadas à mão da placa da Chácara São Francisco (`src/placa.jpg`).

**Página e download:** https://joaovitormiguel.github.io/fonte-chacara/

## Arquivos

- `build/Chacara-Regular.otf` — Mac, Adobe, Figma
- `build/Chacara-Regular.ttf` — Windows, Office, Google Docs/Canva (upload)
- `build/Chacara.zip` — os dois arquivos da fonte + instruções (é o download da página)
- `index.html` — página da fonte, com testador de texto
- `build/amostra.png` — prévia rápida de todos os caracteres

Para instalar, dê dois cliques no `.otf` (Mac) ou no `.ttf` (Windows) e clique em "Instalar".

## O que tem na fonte

- Só maiúsculas. Digitar em minúsculo produz a **2ª versão** de cada letra, então dá para variar alternando a caixa (`ArArA`) em qualquer programa.
- **Letras da placa:** A, C, F, H, I, N, O, R, S. Estas foram vetorizadas da foto.
- **Letras desenhadas no mesmo estilo:** B D E G J K L M P Q T U V W X Y Z, números de 0 a 9, pontuação e símbolos (`R$ % & # ° º ª « »` etc.).
- Acentos do português e do espanhol: Á À Â Ã Ä É Ê Í Ó Ô Õ Ú Ü Ç Ñ.
- **Variação automática:** a fonte usa 3 dos 5 "A" da placa, 3 "C", 2 "R" e 2 "S" como versões da mesma letra. Quando uma letra se repete, a fonte alterna entre as versões (recurso OpenType `calt`, ligado por padrão na maioria dos programas). O, H e I ganharam uma versão espelhada, e as letras novas têm duas versões com o corte um pouco diferente.
- **Acabamento:** os contornos da foto passam por um filtro que tira as ondulações dos veios da madeira sem mudar a forma da letra, e os furos internos viram elipses regulares. A força da suavização de cada letra fica em `FAIR`, em `work/glyphs.py`.
- Kerning automático para pares como `TA`, `AV`, `LT` e `F.`.

## Regerar

```sh
./gerar_fonte.sh
```

Requer `python3` com `numpy`, `opencv-python`, `scipy`, `scikit-image` e `fonttools`. Os desenhos das letras novas estão em `work/glyphs.py`, e o nome, o espaçamento e o kerning ficam em `work/build_font.py`.
