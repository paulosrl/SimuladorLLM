"""
Muiraquitã - Simulador LLM 2D
MPPA - CIIA | Escritório de Inovação e Inteligência Artificial
"""

import numpy as np
import plotly.graph_objects as go

from simulador_streamlit import (
    GRUPOS,
    CORES_GRUPOS,
    COMMON_WORD,
    normalizar_texto,
    identificar_grupo,
    executar_interface
)

INFO_GRAFICO_2D = """
**Como usar:**
- 🔍 Zoom: Role o mouse
- ✋ Pan: Clique e arraste
- 🎚️ Filtrar: Clique na legenda para ocultar/mostrar grupos

**Legenda:**
- Círculos grandes: centros dos grupos
- Círculos pequenos: palavras de cada grupo
- Círculos brancos: palavras compartilhadas entre grupos
- Seta branca: indica o grupo sugerido para o texto analisado
- ❓ amarelo: texto sem contexto claro
"""


def criar_grafico_2d_plotly(texto_busca=""):
    nomes_grupos = list(GRUPOS.keys())
    n_grupos = len(nomes_grupos)
    angulos_grupos = np.linspace(0, 2 * np.pi, n_grupos, endpoint=False)
    raio_grupos = 6.0

    coords_grupos = {}
    for i, nome_grupo in enumerate(nomes_grupos):
        x = raio_grupos * np.cos(angulos_grupos[i])
        y = raio_grupos * np.sin(angulos_grupos[i])
        coords_grupos[nome_grupo] = (x, y)

    fig = go.Figure()
    fig.add_shape(
        type="circle",
        x0=-raio_grupos * 1.05,
        y0=-raio_grupos * 1.05,
        x1=raio_grupos * 1.05,
        y1=raio_grupos * 1.05,
        line=dict(color="rgba(255,255,255,0.1)", dash="dot")
    )

    # Centros dos grupos
    for nome_grupo, (x, y) in coords_grupos.items():
        cor = CORES_GRUPOS[nome_grupo]
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='text',
            text=[nome_grupo],
            textposition='top center',
            textfont=dict(size=24, color='white', family="Montserrat, sans-serif"),
            name=nome_grupo,
            hoverinfo='text'
        ))

    # Palavras dos grupos
    for nome_grupo, palavras_grupo in GRUPOS.items():
        centro_x, centro_y = coords_grupos[nome_grupo]
        palavras_lista = sorted(palavras_grupo)
        n_palavras = len(palavras_lista)
        raio_interno = 2.2
        cor_base = CORES_GRUPOS[nome_grupo]

        for j, palavra in enumerate(palavras_lista):
            angulo = 2 * np.pi * j / max(n_palavras, 1)
            x = centro_x + raio_interno * np.cos(angulo)
            y = centro_y + raio_interno * np.sin(angulo)

            grupos_da_palavra = [ng for ng, pg in GRUPOS.items() if palavra in pg]
            palavra_compartilhada = len(grupos_da_palavra) > 1
            palavra_eh_comum = (
                normalizar_texto(palavra) == normalizar_texto(COMMON_WORD)
            )

            cor_marker = "#FFFFFF" if palavra_compartilhada else cor_base
            cor_texto = "#FFFFFF" if palavra_compartilhada else cor_base
            tamanho = 18 if palavra_eh_comum else 14

            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers+text',
                marker=dict(
                    size=tamanho,
                    color=cor_marker,
                    opacity=0.9,
                    symbol='circle',
                    line=dict(width=1.5, color='black')
                ),
                text=[palavra],
                textposition='top center',
                textfont=dict(size=11, color=cor_texto),
                name=f"{nome_grupo} · {palavra}",
                hoverinfo='text',
                showlegend=False
            ))

    # Destaque da busca
    if texto_busca:
        grupo_identificado, _ = identificar_grupo(texto_busca)
        if grupo_identificado:
            centro_x, centro_y = coords_grupos[grupo_identificado]
            vetor = np.array([centro_x, centro_y], dtype=float)
            norma = np.linalg.norm(vetor)
            if norma == 0:
                vetor = np.array([0.0, 1.0])
                norma = 1.0
            direcao = vetor / norma
            texto_pos = vetor + direcao * 2.5

            fig.add_annotation(
                x=centro_x,
                y=centro_y,
                ax=texto_pos[0],
                ay=texto_pos[1],
                text=f"★ {texto_busca[:30]} ★",
                showarrow=True,
                arrowcolor="#FFFFFF",
                arrowwidth=2,
                font=dict(size=14, color='white'),
                bgcolor="rgba(0,0,0,0.6)",
                bordercolor="#FFFFFF",
                borderwidth=1,
                align="center"
            )
        else:
            fig.add_annotation(
                x=0,
                y=0,
                text="❓",
                showarrow=False,
                font=dict(size=32, color='#FFD700')
            )
            fig.add_annotation(
                x=0,
                y=-1.2,
                text=f"{texto_busca[:32]}<br>(SEM CONTEXTO)",
                showarrow=False,
                font=dict(size=14, color='#FFD700'),
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="#FFD700",
                borderwidth=1,
                align="center"
            )

    fig.update_layout(
        height=700,
        width=900,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#000000',
        showlegend=True,
        font=dict(color='white'),
    )

    fig.update_xaxes(
        title="",
        visible=False,
        showgrid=False
    )
    fig.update_yaxes(
        title="",
        visible=False,
        showgrid=False,
        scaleanchor='x',
        scaleratio=1
    )

    return fig


def main():
    executar_interface(
        criar_grafico_func=criar_grafico_2d_plotly,
        info_grafico_texto=INFO_GRAFICO_2D,
        titulo_pagina="Muiraquitã - Simulador LLM 2D",
        titulo_cabecalho="🧠 Muiraquitã - Simulador de LLM (2D)"
    )


if __name__ == "__main__":
    main()
