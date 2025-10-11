"""
Muiraquitã - Simulador LLM 3D
MPPA - CIIA | Escritório de Inovação e Inteligência Artificial
"""

import streamlit as st
import numpy as np
import unicodedata
import json
from datetime import datetime
import plotly.graph_objects as go

# ==================== DADOS ====================

GRUPOS = {
    "Transportes": {"carro", "avião", "ônibus", "bicicleta", "motocicleta", "trem", "veículo"},
    "Móveis": {"cadeira", "mesa", "banco", "armário", "sofá", "cômoda", "cama"},
    "Animais": {"cachorro", "gato", "focinho", "rato", "leão", "tigre", "baleia"},
    "Financeiro": {"banco", "moeda", "cédula", "caixa", "dinheiro", "investimento", "juros"}
}

CONTEXTO_GRUPOS = {
    "Transportes": {
        "viajar", "dirigir", "pilotar", "velocidade", "motor", "combustível", 
        "passageiro", "estrada", "rua", "aeroporto", "garagem", "estacionar",
        "viagem", "roda", "acelerar", "freio", "transporte", "locomover", "tráfego",
        "partida", "chegada", "embarque", "desembarque"
    },
    "Móveis": {
        "sentar", "sentei", "sentou", "sentado", "sentada", "sentem", "casa", 
        "sala", "quarto", "madeira", "decoração", "móvel", "conforto", "decorar", 
        "residência", "apartamento", "escritório", "design", "estofado", "montagem", 
        "ergonomia", "praça", "jardim", "parque", "acomodar", "descansar", "repousar",
        "apoiar", "encostar"
    },
    "Animais": {
        "pet", "animal", "bicho", "selvagem", "doméstico", "natureza", "zoológico",
        "veterinário", "pelo", "pata", "cauda", "mamífero", "espécie",
        "fauna", "ração", "alimentar", "latir", "miar", "rugir"
    },
    "Financeiro": {
        "dinheiro", "pagar", "receber", "conta", "depósito", "depositar", "depositei",
        "saque", "sacar", "transferência", "transferir", "agência", "gerente", 
        "aplicar", "investir", "economizar", "cartão", "cheque", "empréstimo", 
        "financiamento", "juros", "taxa", "saldo", "crédito", "débito", "poupança", 
        "correntista", "cofre", "financeiro", "bancário", "caixa eletrônico"
    }
}

PALAVRAS_INFERENCIA = {
    "Animais": ["tartaruga", "cobra", "pássaro", "peixe", "elefante", "girafa", 
                "macaco", "urso", "lobo", "raposa", "coelho", "hamster", "papagaio"],
    "Transportes": ["moto", "barco", "navio", "helicóptero", "metrô", "taxi",
                   "caminhão", "van", "scooter", "patinete", "skate"],
    "Móveis": ["estante", "escrivaninha", "poltrona", "banqueta", "criado-mudo",
              "guarda-roupa", "buffet", "aparador", "rack", "prateleira"],
    "Financeiro": ["pix", "boleto", "nota", "real", "dólar", "euro", "bitcoin",
                  "ação", "fundo", "renda", "lucro"]
}

CORES_GRUPOS = {
    "Transportes": "#FF4444",
    "Móveis": "#4488FF", 
    "Animais": "#44FF44",
    "Financeiro": "#FFAA00"
}

PESOS = {
    "contexto": 3.0,
    "inferencia": 2.5,
    "principal": 0.2
}

# ==================== FUNÇÕES ====================

# Teste

def normalizar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto_nfd = unicodedata.normalize('NFD', texto)
    texto_sem_acento = ''.join(char for char in texto_nfd if unicodedata.category(char) != 'Mn')
    return texto_sem_acento.lower().strip()

def calcular_similaridade_levenshtein(palavra1, palavra2):
    if palavra1 == palavra2:
        return 1.0
    len1, len2 = len(palavra1), len(palavra2)
    if len1 == 0 or len2 == 0:
        return 0.0

    matriz = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        matriz[i][0] = i
    for j in range(len2 + 1):
        matriz[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            custo = 0 if palavra1[i-1] == palavra2[j-1] else 1
            matriz[i][j] = min(
                matriz[i-1][j] + 1,
                matriz[i][j-1] + 1,
                matriz[i-1][j-1] + custo
            )

    distancia = matriz[len1][len2]
    similaridade = 1 - (distancia / max(len1, len2))
    return max(0.0, similaridade)

def calcular_similaridade_caracteres(palavra1, palavra2):
    if palavra1 == palavra2:
        return 1.0
    set1, set2 = set(palavra1.lower()), set(palavra2.lower())
    if not set1 or not set2:
        return 0.0
    intersecao = len(set1.intersection(set2))
    uniao = len(set1.union(set2))
    return intersecao / uniao if uniao > 0 else 0.0

def calcular_similaridade_composta(palavra_busca, palavra_comparacao):
    palavra_busca = normalizar_texto(palavra_busca)
    palavra_comparacao = normalizar_texto(palavra_comparacao)

    if palavra_busca == palavra_comparacao:
        return 100.0

    sim_levenshtein = calcular_similaridade_levenshtein(palavra_busca, palavra_comparacao)
    sim_caracteres = calcular_similaridade_caracteres(palavra_busca, palavra_comparacao)
    bonus_substring = 0.2 if palavra_busca in palavra_comparacao or palavra_comparacao in palavra_busca else 0.0

    similaridade_final = (sim_levenshtein * 0.6 + sim_caracteres * 0.4 + bonus_substring) * 100
    return min(100.0, similaridade_final)

def calcular_similaridades_palavra(palavra_busca):
    if not palavra_busca or len(palavra_busca.strip()) < 2:
        return {}

    palavra_busca = palavra_busca.strip()
    similaridades = {}

    for nome_grupo, palavras_grupo in GRUPOS.items():
        similaridades[nome_grupo] = []
        for palavra in sorted(palavras_grupo):
            similaridade = calcular_similaridade_composta(palavra_busca, palavra)
            similaridades[nome_grupo].append({'palavra': palavra, 'similaridade': similaridade})
        similaridades[nome_grupo].sort(key=lambda x: x['similaridade'], reverse=True)

    return similaridades

def detectar_palavras_ambiguas(texto):
    texto_normalizado = normalizar_texto(texto)
    palavras = texto_normalizado.split()
    ambiguas = []

    for palavra in palavras:
        palavra_limpa = palavra.strip('.,!?;:')
        grupos_encontrados = []
        for nome_grupo, palavras_grupo in GRUPOS.items():
            palavras_normalizadas = {normalizar_texto(p) for p in palavras_grupo}
            if palavra_limpa in palavras_normalizadas:
                grupos_encontrados.append(nome_grupo)
        if len(grupos_encontrados) > 1:
            ambiguas.append((palavra_limpa, grupos_encontrados))

    return ambiguas

def analisar_contexto(texto_completo, pesos=None):
    if not isinstance(texto_completo, str) or not texto_completo.strip():
        return {nome: 0.0 for nome in GRUPOS.keys()}

    if pesos is None:
        pesos = PESOS

    texto_normalizado = normalizar_texto(texto_completo)
    scores_contexto = {nome: 0.0 for nome in GRUPOS.keys()}

    for nome_grupo, palavras_contexto in CONTEXTO_GRUPOS.items():
        for palavra_ctx in palavras_contexto:
            palavra_norm = normalizar_texto(palavra_ctx)
            if palavra_norm in texto_normalizado:
                scores_contexto[nome_grupo] += pesos["contexto"]

    for nome_grupo, palavras_grupo in GRUPOS.items():
        for palavra_principal in palavras_grupo:
            palavra_norm = normalizar_texto(palavra_principal)
            if f" {palavra_norm} " in f" {texto_normalizado} " or                texto_normalizado.startswith(palavra_norm + " ") or                texto_normalizado.endswith(" " + palavra_norm):
                scores_contexto[nome_grupo] += pesos["principal"]

    for nome_grupo, palavras_relacionadas in PALAVRAS_INFERENCIA.items():
        for palavra_rel in palavras_relacionadas:
            palavra_norm = normalizar_texto(palavra_rel)
            if palavra_norm in texto_normalizado:
                scores_contexto[nome_grupo] += pesos["inferencia"]

    total = sum(scores_contexto.values())
    if total > 0:
        scores_contexto = {k: v/total for k, v in scores_contexto.items()}

    return scores_contexto

def identificar_grupo(texto):
    scores = analisar_contexto(texto)
    if not scores or all(v == 0 for v in scores.values()):
        return None, scores
    grupo_principal = max(scores.items(), key=lambda x: x[1])
    if grupo_principal[1] > 0.1:
        return grupo_principal[0], scores
    return None, scores

def detectar_palavras_desconhecidas(texto):
    texto_normalizado = normalizar_texto(texto)
    palavras = texto_normalizado.split()
    todas_conhecidas = set()

    for palavras_grupo in GRUPOS.values():
        todas_conhecidas.update(normalizar_texto(p) for p in palavras_grupo)
    for palavras_ctx in CONTEXTO_GRUPOS.values():
        todas_conhecidas.update(normalizar_texto(p) for p in palavras_ctx)
    for palavras_inf in PALAVRAS_INFERENCIA.values():
        todas_conhecidas.update(normalizar_texto(p) for p in palavras_inf)

    desconhecidas = []
    for palavra in palavras:
        palavra_limpa = palavra.strip('.,!?;:')
        if len(palavra_limpa) > 2 and palavra_limpa not in todas_conhecidas:
            desconhecidas.append(palavra_limpa)

    return desconhecidas

def criar_grafico_3d_plotly(texto_busca=""):
    nomes_grupos = list(GRUPOS.keys())
    n_grupos = len(nomes_grupos)
    angulos_grupos = np.linspace(0, 2*np.pi, n_grupos, endpoint=False)
    raio_grupos = 5.0

    coords_grupos = {}
    for i, nome_grupo in enumerate(nomes_grupos):
        x = raio_grupos * np.cos(angulos_grupos[i])
        y = raio_grupos * np.sin(angulos_grupos[i])
        coords_grupos[nome_grupo] = (x, y, 0)

    fig = go.Figure()

    for nome_grupo, (x, y, z) in coords_grupos.items():
        cor = CORES_GRUPOS[nome_grupo]
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers+text',
            marker=dict(size=25, color=cor, opacity=0.7, line=dict(width=3, color='black')),
            text=nome_grupo,
            textposition='bottom center',
            textfont=dict(size=18, color='white'),  # AUMENTADO 30%
            name=nome_grupo,
            showlegend=True
        ))

    for nome_grupo, palavras_grupo in GRUPOS.items():
        centro_x, centro_y, centro_z = coords_grupos[nome_grupo]
        palavras_lista = sorted(palavras_grupo)
        n_palavras = len(palavras_lista)
        raio_interno = 1.2
        cor = CORES_GRUPOS[nome_grupo]

        for j, palavra in enumerate(palavras_lista):
            angulo = 2 * np.pi * j / n_palavras
            x = centro_x + raio_interno * np.cos(angulo)
            y = centro_y + raio_interno * np.sin(angulo)
            z = np.random.randn() * 0.3

            grupos_da_palavra = [ng for ng, pg in GRUPOS.items() if palavra in pg]

            if len(grupos_da_palavra) > 1:
                cor_marker = "#9B59B6"
                simbolo = 'diamond'
                tamanho = 14
                cor_borda = 'red'
            else:
                cor_marker = cor
                simbolo = 'circle'
                tamanho = 9
                cor_borda = 'black'

            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='markers+text',
                marker=dict(size=tamanho, color=cor_marker, opacity=0.8,
                           symbol=simbolo, line=dict(width=1, color=cor_borda)),
                text=palavra,
                textposition='top center',
                textfont=dict(size=8, color=cor),
                name=palavra,
                showlegend=False
            ))

    if texto_busca:
        grupo_identificado, scores = identificar_grupo(texto_busca)

        if grupo_identificado:
            centro_x, centro_y, centro_z = coords_grupos[grupo_identificado]
            coord_busca = (centro_x, centro_y, 1.2)
            cor_destaque = CORES_GRUPOS[grupo_identificado]

            fig.add_trace(go.Scatter3d(
                x=[coord_busca[0]], y=[coord_busca[1]], z=[coord_busca[2]],
                mode='markers+text',
                marker=dict(size=35, color=cor_destaque, symbol='diamond',
                           opacity=1.0, line=dict(width=5, color='black')),
                text=f'★ {texto_busca[:30]} ★',
                textposition='top center',
                textfont=dict(size=12, color='red'),
                name='Texto Analisado',
                showlegend=True
            ))
        else:
            coord_busca = (0, 0, 2.2)
            fig.add_trace(go.Scatter3d(
                x=[coord_busca[0]], y=[coord_busca[1]], z=[coord_busca[2]],
                mode='markers+text',
                marker=dict(size=35, color='gray', symbol='diamond',
                           opacity=1.0, line=dict(width=5, color='black')),
                text=f'★ {texto_busca[:30]} ★<br>(SEM CONTEXTO)',
                textposition='top center',
                textfont=dict(size=11, color='white'),
                name='Sem Contexto',
                showlegend=True
            ))

    grupos_lista = ', '.join(GRUPOS.keys())
    titulo = f'Grupos Semânticos: {grupos_lista}'
    if texto_busca:
        titulo += f'\nEntrada: "{texto_busca[:50]}"'

    fig.update_layout(
        title=dict(text=titulo, font=dict(color='white', size=16)),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y', 
            zaxis_title='Z',
            camera=dict(eye=dict(x=1.2, y=1.2, z=0.8)),
            bgcolor='black',
            xaxis=dict(gridcolor='gray', color='white', range=[-8, 8]),
            yaxis=dict(gridcolor='gray', color='white', range=[-8, 8]),
            zaxis=dict(gridcolor='gray', color='white', range=[-2, 4])
        ),
        width=900,
        height=700,
        margin=dict(l=0, r=0, b=0, t=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='black',
        font=dict(color='white')
    )

    return fig

# ==================== INTERFACE ====================

def main():
    st.set_page_config(
        page_title="Muiraquitã - Simulador LLM 3D",
        page_icon="🧠",
        layout="wide"
    )

    st.title("🧠 Muiraquitã - Simulador de LLM")
    st.markdown("**MPPA - CIIA | Escritório de Inovação e Inteligência Artificial**")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("🎛️ Painel de Controle")

        st.subheader("Digite uma frase ou palavra:")
        texto_entrada = st.text_input(
            "Insira o texto para análise semântica:",
            placeholder="Digite uma palavra ou uma pequena frase...",
            help="Digite uma frase com contexto"
        )

        col_btn1, col_btn2, col_btn3 = st.columns(3)
        with col_btn1:
            analisar = st.button("🔍 Analisar", type="primary", use_container_width=True)
        with col_btn2:
            limpar = st.button("🔄 Limpar", use_container_width=True)
        with col_btn3:
            inicial = st.button("🏠 Inicial", use_container_width=True)

        # BOTÃO INICIAL AGORA LIMPA TUDO
        if limpar or inicial:
            st.session_state.resultado_analise = ""
            st.session_state.palavra_atual = ""
            st.session_state.scores = {}
            st.rerun()

        if 'resultado_analise' not in st.session_state:
            st.session_state.resultado_analise = ""
        if 'palavra_atual' not in st.session_state:
            st.session_state.palavra_atual = ""
        if 'scores' not in st.session_state:
            st.session_state.scores = {}

        if analisar and texto_entrada:
            grupo_identificado, scores = identificar_grupo(texto_entrada)
            ambiguas = detectar_palavras_ambiguas(texto_entrada)
            desconhecidas = detectar_palavras_desconhecidas(texto_entrada)

            st.session_state.palavra_atual = texto_entrada.split()[0].strip('.,!?;:') if texto_entrada else ""

            resultado = ""
            if ambiguas:
                resultado += "🔀 **PALAVRAS AMBÍGUAS:**\n"
                for palavra_amb, grupos_amb in ambiguas:
                    resultado += f"  • '{palavra_amb}' pertence a: {', '.join(grupos_amb)}\n"
                resultado += "\n"

            if desconhecidas:
                resultado += "❓ **PALAVRAS DESCONHECIDAS:**\n"
                resultado += f"  {', '.join(desconhecidas)}\n\n"

            if grupo_identificado:
                resultado += f"✅ **GRUPO:** {grupo_identificado}\n"
                resultado += f"   Confiança: {scores[grupo_identificado]*100:.1f}%\n\n"
            else:
                resultado += "⚠️ **SEM CONTEXTO**\n\n"

            resultado += "🎯 **Pertinência por grupo:**\n"

            st.session_state.resultado_analise = resultado
            st.session_state.scores = scores

        if st.session_state.resultado_analise:
            st.subheader("📊 Análise Detalhada:")
            st.markdown(st.session_state.resultado_analise)

            # GRÁFICO DE BARRAS
            if 'scores' in st.session_state and st.session_state.scores:
                scores_ordenados = sorted(st.session_state.scores.items(), 
                                        key=lambda x: x[1], reverse=True)
                grupos_names = [g[0] for g in scores_ordenados if g[1] > 0.001]
                scores_values = [g[1]*100 for g in scores_ordenados if g[1] > 0.001]
                cores_barras = [CORES_GRUPOS[g] for g in grupos_names]

                fig_barras = go.Figure(data=[
                    go.Bar(
                        x=grupos_names,
                        y=scores_values,
                        marker_color=cores_barras,
                        text=[f"{v:.1f}%" for v in scores_values],
                        textposition='outside'
                    )
                ])

                fig_barras.update_layout(
                    height=300,
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_title="Grupos",
                    yaxis_title="Pertinência (%)",
                    yaxis_range=[0, max(scores_values)*1.2 if scores_values else 100],
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )

                st.plotly_chart(fig_barras, use_container_width=True)

        elif not st.session_state.resultado_analise:
            st.subheader("📊 Análise Detalhada:")
            st.markdown("""
            🎨 **Estado inicial carregado!**

            Mostrando estrutura dos grupos semânticos.

            💡 **Digite uma palavra e clique em Analisar!**

            **Exemplos:**
            • 'banco' - palavra ambígua
            • 'carro' - palavra específica  
            • 'sentar no banco' - frase contextual

            🎯 **MPPA - GIIA**
            """)

        if st.session_state.palavra_atual:
            st.subheader("🔍 Similaridades")
            similaridades = calcular_similaridades_palavra(st.session_state.palavra_atual)

            resultado_sim = f"🔍 **Similaridades: '{st.session_state.palavra_atual}'**\n"
            resultado_sim += "─" * 40 + "\n\n"

            for nome_grupo in GRUPOS.keys():
                if nome_grupo in similaridades:
                    resultado_sim += f"**• {nome_grupo}:**\n"
                    top_similares = similaridades[nome_grupo][:5]
                    for item in top_similares:
                        palavra = item['palavra']
                        sim = item['similaridade']
                        icone = "🟢" if sim >= 80 else "🟡" if sim >= 50 else "⚪"
                        resultado_sim += f"  {icone} {palavra:<11} {sim:5.1f}%\n"
                    resultado_sim += "\n"

            st.markdown(resultado_sim)
        else:
            st.subheader("🔍 Similaridades")
            st.markdown("Digite uma palavra e analise para ver similaridades.")

        st.subheader("📚 Grupos Semânticos")
        for nome, palavras in GRUPOS.items():
            with st.expander(f"{nome} ({len(palavras)} palavras)"):
                st.write(", ".join(sorted(palavras)))

        st.subheader("💾 Exportar")
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            if st.button("📊 Gráfico", use_container_width=True):
                st.info("Clique direito no gráfico > Save image")
        with col_exp2:
            if st.button("📄 Dados", use_container_width=True):
                if st.session_state.palavra_atual:
                    similaridades = calcular_similaridades_palavra(st.session_state.palavra_atual)
                    dados_export = {
                        'palavra': st.session_state.palavra_atual,
                        'timestamp': datetime.now().isoformat(),
                        'similaridades': similaridades
                    }
                    json_str = json.dumps(dados_export, indent=2, ensure_ascii=False)
                    st.download_button(
                        "📥 Download JSON",
                        json_str,
                        f"similaridades_{st.session_state.palavra_atual}.json",
                        "application/json",
                        use_container_width=True
                    )

    with col2:
        st.header("🎨 Visualização 3D")
        fig = criar_grafico_3d_plotly(texto_entrada if analisar and texto_entrada else "")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("ℹ️ Sobre o Gráfico"):
            st.markdown("""
            **Como usar:**
            - 🔍 Zoom: Role o mouse
            - 🖱️ Rotacionar: Clique e arraste
            - 📏 Pan: Shift + arraste
            - 🏠 Reset: Clique duplo

            **Legenda:**
            - Círculos grandes: Centros dos grupos
            - Círculos pequenos: Palavras (mesma cor do grupo)
            - Diamantes roxos: Palavras ambíguas
            - Diamante destaque: Texto analisado
            """)

if __name__ == "__main__":
    main()
