"""
Muiraquitã - Simulador LLM 3D
MPPA - CIIA | Escritório de Inovação e Inteligência Artificial
"""

import streamlit as st
import numpy as np
import unicodedata
import json
from datetime import datetime
from typing import Dict, Set, List, Tuple
import plotly.graph_objects as go

# ==================== DADOS ====================

GRUPOS = {
    "Transportes": {
        "carro", "avião", "ônibus", "bicicleta", "motocicleta", "trem", "veículo",
        "barco", "navio", "metrô", "estrada", "aeroporto", "garagem", "rota",
        "lavagem","combustível","posto de gasolina","gasolina","diesel","etanol","flex",
        "transporte","locomover","tráfego","acidente","engarrafamento","congestionamento"
    },
    "Móveis": {
        "cadeira", "mesa", "banco", "armário", "sofá", "cômoda", "cama",
        "poltrona", "estante","lavagem","geladeira","fogão","micro-ondas","forno",
        "máquina de lavar","ventilador","ar condicionado","televisão","TV",
        "decoração", "residência", "apartamento", "escritório", "design",
        "limpeza",
    },
    "Animais": {
        "cachorro", "gato", "focinho", "rato", "leão", "tigre", "baleia",
        "rabo", "crocodilo", "cavalo", "ferradura","mucura","onça","arara",
        "jacaré","réptil","selva","pantanal","floresta","aquático","marinho",
        "mamífero", "espécie", "fauna", "veterinário", "alimentar", "selvagem",    
    },
    "Financeiro": {
        "banco", "moeda", "cédula", "caixa", "dinheiro", "investimento", "juros",
        "pix", "boleto", "cartão", "cheque", "saldo", "crédito", "débito",
        "depósito", "deposito", "depósito bancário", "transferência", "poupança",
        "saque", "extrato", "cofre", "aplicação","assalto","financiamento","empréstimo",
        "lavanderia","lavagem","dinheiro sujo","corrupção","suborno","propina"
    }
}

COMMON_WORD = "banco"
COMMON_WORD_GROUPS = ["Móveis", "Financeiro", "Transportes"]


CONTEXTO_GRUPOS = {
    "Transportes": {
        "viajar", "dirigir", "pilotar", "velocidade", "motor", "combustível", 
        "passageiro", "estrada", "rua", "aeroporto", "garagem", "estacionar",
        "viagem", "roda", "acelerar", "freio", "transporte", "locomover", "tráfego",
        "partida", "chegada", "embarque", "desembarque", "trajeto", "rodovia",
        "ponto", "embarcar", "porto", "metrô"
    },
    "Móveis": {
        "sentar", "sentei", "sentou", "sentado", "sentada", "sentem", "casa", 
        "sala", "quarto", "madeira", "decoração", "móvel", "conforto", "decorar", 
        "residência", "apartamento", "escritório", "design", "estofado", "montagem", 
        "ergonomia", "praça", "jardim", "parque", "acomodar", "descansar", "repousar",
        "apoiar", "encostar", "interior", "decoração"
    },
    "Animais": {
        "pet", "animal", "bicho", "selvagem", "doméstico", "natureza", "zoológico",
        "veterinário", "pelo", "pata", "cauda", "mamífero", "espécie",
        "fauna", "ração", "alimentar", "latir", "miar", "rugir", "jacaré",
        "réptil", "selva", "pantanal", "floresta", "crocodilo", "aquático"
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
    "Animais": [
        "tartaruga", "cobra", "pássaro", "peixe", "elefante", "girafa",
        "macaco", "urso", "lobo", "raposa", "coelho", "hamster", "papagaio",
        "jacaré", "crocodilo", "cavalo", "lagarto", "onça", "sapo",
        "hipopótamo", "panda", "golfinho", "falcão", "abutre","jabuti","jaboti",
        "arara","mucura","réptil","anfíbio","aquático","marinho","selvagem",
        "perema","catita","jiboia","serpente","surucucu","anaconda"
    ],
    "Transportes": [
        "moto", "barco", "navio", "helicóptero", "metrô", "taxi",
        "caminhão", "van", "scooter", "patinete", "skate", "uber",
        "barca", "bicicletário", "teleférico", "bondinho", "aeronave"
    ],
    "Móveis": [
        "estante", "escrivaninha", "poltrona", "banqueta", "criado-mudo",
        "guarda-roupa", "buffet", "aparador", "rack", "prateleira",
        "puff", "cômoda", "sapateira", "balcão", "cabideiro", "divã"
    ],
    "Financeiro": [
        "pix", "boleto", "nota", "real", "dólar", "euro", "bitcoin",
        "ação", "fundo", "renda", "lucro", "poupança", "cartão",
        "investidor", "fintech", "depósito", "transferência", "remessa",
        "depósito bancário", "tesouro", "derivativo", "swap", "portfólio",
        "cidade", "prefeitura", "governo", "imposto", "tributo", "corrupção", "suborno", 
        "Brasil",
    ]
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

SESSION_STATE_DEFAULTS = {
    "texto_entrada": "",
    "texto_analisado": "",
    "resultado_analise": "",
    "palavra_atual": "",
    "scores": {},
    "grupo_identificado": None,
    "_reset_requested": False,
    "force_rerun": False
}

INFO_GRAFICO_3D = """
**Como usar:**
- 🔍 Zoom: Role o mouse
- 🖱️ Rotacionar: Clique e arraste
- 📏 Pan: Shift + arraste
- 🏠 Reset: Clique duplo

**Legenda:**
- Círculos grandes: Centros dos grupos
- Círculos pequenos: Palavras (mesma cor do grupo)
- Marcadores brancos: Palavras compartilhadas entre grupos
- Seta branca: indica o grupo apontado pelo texto analisado
- ❓ amarelo: texto sem contexto claro
"""

# ==================== FUNÇÕES ====================

def normalizar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto_nfd = unicodedata.normalize('NFD', texto)
    texto_sem_acento = ''.join(char for char in texto_nfd if unicodedata.category(char) != 'Mn')
    return texto_sem_acento.lower().strip()

def sincronizar_palavra_comum():
    palavra_norm = normalizar_texto(COMMON_WORD)
    for nome, palavras in list(GRUPOS.items()):
        if not isinstance(palavras, set):
            GRUPOS[nome] = set(palavras)

    for nome in COMMON_WORD_GROUPS:
        if nome in GRUPOS:
            if COMMON_WORD not in GRUPOS[nome]:
                GRUPOS[nome].add(COMMON_WORD)

    for nome, palavras in list(GRUPOS.items()):
        if nome not in COMMON_WORD_GROUPS:
            GRUPOS[nome] = {p for p in palavras if normalizar_texto(p) != palavra_norm}

sincronizar_palavra_comum()

def obter_palavras_compartilhadas(grupos: Dict[str, Set[str]]) -> List[Tuple[str, List[str]]]:
    ocorrencias: Dict[str, Set[str]] = {}
    representantes: Dict[str, str] = {}

    for nome_grupo, palavras in grupos.items():
        for palavra in palavras:
            chave = normalizar_texto(palavra)
            if not chave:
                continue
            ocorrencias.setdefault(chave, set()).add(nome_grupo)
            representantes.setdefault(chave, palavra)

    compartilhadas: List[Tuple[str, List[str]]] = []
    for chave, nomes in ocorrencias.items():
        if len(nomes) > 1:
            compartilhadas.append((representantes.get(chave, chave), sorted(nomes)))

    compartilhadas.sort(key=lambda item: (item[0].lower(), item[1]))
    return compartilhadas

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

def _buscar_grupos_por_palavra(palavra):
    palavra_normalizada = normalizar_texto(palavra)
    if not palavra_normalizada:
        return set()

    grupos_encontrados = set()
    for nome_grupo, palavras_grupo in GRUPOS.items():
        palavras_normalizadas = {normalizar_texto(p) for p in palavras_grupo}
        if palavra_normalizada in palavras_normalizadas:
            grupos_encontrados.add(nome_grupo)

    return grupos_encontrados

def calcular_similaridades_palavra(palavra_busca, grupo_contexto=None):
    if not palavra_busca or len(palavra_busca.strip()) < 2:
        return {}

    palavra_busca = palavra_busca.strip()
    palavra_busca_norm = normalizar_texto(palavra_busca)
    grupos_palavra = _buscar_grupos_por_palavra(palavra_busca_norm)
    similaridades = {}

    for nome_grupo, palavras_grupo in GRUPOS.items():
        similaridades[nome_grupo] = []
        for palavra in sorted(palavras_grupo):
            similaridade_base = calcular_similaridade_composta(palavra_busca, palavra)
            palavra_norm = normalizar_texto(palavra)

            if palavra_norm == palavra_busca_norm:
                similaridade_final = 100.0
            else:
                bonus = 0.0

                if grupos_palavra:
                    if nome_grupo in grupos_palavra:
                        bonus += 35.0
                    else:
                        bonus -= 5.0

                if grupo_contexto:
                    if nome_grupo == grupo_contexto:
                        bonus += 15.0
                    else:
                        bonus -= 5.0

                similaridade_final = max(0.0, min(100.0, similaridade_base + bonus))

            similaridades[nome_grupo].append({
                'palavra': palavra,
                'similaridade': similaridade_final
            })
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
    max_score = grupo_principal[1]
    grupos_top = []
    for nome, valor in scores.items():
        if valor > 0 and abs(valor - max_score) < 1e-6:
            grupos_top.append(nome)
    if len(grupos_top) == 1 and max_score > 0.1:
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
        fig.add_trace(go.Scatter3d(
            x=[x],
            y=[y],
            z=[z],
            mode='text',
            text=nome_grupo,
            textposition='middle center',
            textfont=dict(size=18, color='white'),
            name=nome_grupo,
            showlegend=False
        ))

    for nome_grupo, palavras_grupo in GRUPOS.items():
        centro_x, centro_y, centro_z = coords_grupos[nome_grupo]
        palavras_lista = sorted(palavras_grupo)
        n_palavras = len(palavras_lista)
        raio_interno = 0.8
        cor = CORES_GRUPOS[nome_grupo]

        for j, palavra in enumerate(palavras_lista):
            angulo = 2 * np.pi * j / n_palavras
            x = centro_x + raio_interno * np.cos(angulo)
            y = centro_y + raio_interno * np.sin(angulo)
            z = np.random.randn() * 0.3

            grupos_da_palavra = [ng for ng, pg in GRUPOS.items() if palavra in pg]

            if len(grupos_da_palavra) > 1:
                cor_marker = "#FFFFFF"
                simbolo = 'circle'
                tamanho = 10
                cor_borda = 'black'
                cor_texto = "#FFFFFF"
            else:
                cor_marker = cor
                simbolo = 'circle'
                tamanho = 9
                cor_borda = 'black'
                cor_texto = cor

            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='markers+text',
                marker=dict(size=tamanho, color=cor_marker, opacity=0.8,
                           symbol=simbolo, line=dict(width=1, color=cor_borda)),
                text=palavra,
                textposition='top center',
                textfont=dict(size=10, color=cor_texto),
                name=palavra,
                showlegend=False
            ))

    if texto_busca:
        grupo_identificado, scores = identificar_grupo(texto_busca)

        if grupo_identificado:
            centro_x, centro_y, centro_z = coords_grupos[grupo_identificado]
            coord_busca = (centro_x, centro_y, 1.5)
            cor_destaque = CORES_GRUPOS[grupo_identificado]

            vetor = np.array([
                centro_x - coord_busca[0],
                centro_y - coord_busca[1],
                centro_z - coord_busca[2]
            ])
            if np.linalg.norm(vetor) < 1e-6:
                vetor = np.array([0.0, 0.0, -0.6])

            fig.add_trace(go.Cone(
                x=[coord_busca[0]],
                y=[coord_busca[1]],
                z=[coord_busca[2]],
                u=[vetor[0]],
                v=[vetor[1]],
                w=[vetor[2]],
                sizemode="absolute",
                sizeref=0.5,
                anchor="tail",
                showscale=False,
                colorscale=[[0, "#FFFFFF"], [1, "#FFFFFF"]],
                name='Indicador de Grupo',
                showlegend=False
            ))

            fig.add_trace(go.Scatter3d(
                x=[coord_busca[0]],
                y=[coord_busca[1]],
                z=[coord_busca[2]],
                mode='text',
                text=f'★ {texto_busca[:30]} ★',
                textposition='bottom center',
                textfont=dict(size=12, color='white'),
                name='Texto Analisado',
                showlegend=False
            ))
        else:
            coord_busca = (0, 0, 2.2)
            fig.add_trace(go.Scatter3d(
                x=[coord_busca[0]],
                y=[coord_busca[1]],
                z=[coord_busca[2] + 0.4],
                mode='text',
                text="❓",
                textposition='middle center',
                textfont=dict(size=24, color='#FFD700'),
                name='Sem Contexto (Indicação)',
                showlegend=False
            ))
            fig.add_trace(go.Scatter3d(
                x=[coord_busca[0]],
                y=[coord_busca[1]],
                z=[coord_busca[2]],
                mode='text',
                text=f"{texto_busca[:30]}<br>(SEM CONTEXTO)",
                textposition='top center',
                textfont=dict(size=14, color='#FFD700'),
                name='Sem Contexto',
                showlegend=False
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
        height=700,
        autosize=True,
        margin=dict(l=0, r=0, b=0, t=25),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='black',
        font=dict(color='white')
    )

    return fig

# ==================== INTERFACE ====================

def inicializar_estado():
    for chave, valor in SESSION_STATE_DEFAULTS.items():
        if chave not in st.session_state:
            st.session_state[chave] = valor.copy() if isinstance(valor, dict) else valor


def resetar_estado():
    for chave, valor in SESSION_STATE_DEFAULTS.items():
        st.session_state[chave] = valor.copy() if isinstance(valor, dict) else valor


def executar_interface(
    criar_grafico_func,
    info_grafico_texto,
    titulo_pagina,
    titulo_cabecalho="🧠 Muiraquitã - Simulador de LLM",
    icone_pagina="🧠"
):
    st.set_page_config(
        page_title=titulo_pagina,
        page_icon=icone_pagina,
        layout="wide"
    )

    inicializar_estado()
    if st.session_state.force_rerun:
        st.session_state.force_rerun = False
    if st.session_state._reset_requested:
        resetar_estado()

    st.title(titulo_cabecalho)
    st.markdown("**MPPA - CIIA | Escritório de Inovação e Inteligência Artificial**")

    st.subheader("Digite uma frase ou palavra:")
    st.text_input(
        "Insira o texto para análise semântica:",
        placeholder="Digite uma palavra ou uma pequena frase...",
        help="Digite uma frase com contexto",
        key="texto_entrada"
    )

    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        analisar = st.button("🔍 Analisar", type="primary", use_container_width=True)
    with col_btn2:
        limpar = st.button("🔄 Limpar", use_container_width=True)
    with col_btn3:
        inicial = st.button("🏠 Inicial", use_container_width=True)

    if limpar or inicial:
        st.session_state._reset_requested = True
        st.rerun()

    texto_analisar = st.session_state.texto_entrada.strip()

    if analisar:
        if not texto_analisar:
            st.warning("Digite uma palavra ou frase para análise.")
        else:
            grupo_identificado, scores = identificar_grupo(texto_analisar)
            ambiguas = detectar_palavras_ambiguas(texto_analisar)
            desconhecidas = sorted(set(detectar_palavras_desconhecidas(texto_analisar)))

            st.session_state.palavra_atual = (
                texto_analisar.split()[0].strip('.,!?;:')
                if texto_analisar else ""
            )

            resultado = []
            if ambiguas:
                resultado.append("🔀 **PALAVRAS AMBÍGUAS:**")
                for palavra_amb, grupos_amb in ambiguas:
                    resultado.append(f"  • '{palavra_amb}' pertence a: {', '.join(grupos_amb)}")
                resultado.append("")

            if desconhecidas:
                resultado.append("❓ **PALAVRAS DESCONHECIDAS:**")
                resultado.append(f"  {', '.join(desconhecidas)}")
                resultado.append("")

            if grupo_identificado:
                resultado.append(f"✅ **GRUPO:** {grupo_identificado}")
                resultado.append(f"   Confiança: {scores[grupo_identificado]*100:.1f}%")
                resultado.append("")
            else:
                resultado.append("⚠️ **SEM CONTEXTO**")
                resultado.append("")

            resultado.append("🎯 **Pertinência por grupo:**")
            for nome, valor in sorted(scores.items(), key=lambda item: item[1], reverse=True):
                resultado.append(f"   - {nome}: {valor*100:.1f}%")

            st.session_state.resultado_analise = "\n".join(resultado).strip()
            st.session_state.scores = dict(scores)
            st.session_state.texto_analisado = texto_analisar
            st.session_state.grupo_identificado = grupo_identificado
            st.session_state.force_rerun = True
            st.rerun()

    st.subheader("🌐 Visualização 3D")
    fig = criar_grafico_func(st.session_state.texto_analisado)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ Sobre o Gráfico"):
        st.markdown(info_grafico_texto.strip())

    st.subheader("📊 Análise Detalhada:")
    if st.session_state.resultado_analise:
        st.markdown(st.session_state.resultado_analise)
    else:
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

    st.subheader("📈 Gráfico de Pertinência")
    if 'scores' in st.session_state and st.session_state.scores:
        scores_ordenados = sorted(
            st.session_state.scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        grupos_names = [g[0] for g in scores_ordenados]
        scores_values = [g[1] * 100 for g in scores_ordenados]
        cores_barras = [CORES_GRUPOS[g] for g in grupos_names]
        yaxis_max = max(scores_values) * 1.2 if scores_values else 100
        yaxis_max = max(yaxis_max, 10)

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
            yaxis_range=[0, yaxis_max],
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )

        st.plotly_chart(fig_barras, use_container_width=True)
    else:
        st.info("Nenhuma análise realizada ainda.")

    st.subheader("🔍 Similaridades")
    if st.session_state.palavra_atual:
        grupo_foco = st.session_state.get("grupo_identificado")
        similaridades_por_grupo = calcular_similaridades_palavra(
            st.session_state.palavra_atual,
            grupo_contexto=grupo_foco
        )

        todas_similaridades = []
        for nome_grupo, itens in similaridades_por_grupo.items():
            for item in itens:
                todas_similaridades.append({
                    "grupo": nome_grupo,
                    "palavra": item["palavra"],
                    "similaridade": item["similaridade"],
                    "cor": CORES_GRUPOS.get(nome_grupo, "#FFFFFF")
                })

        if not todas_similaridades:
            st.markdown("Nenhum resultado de similaridade para esta palavra.")
        else:
            limite = 35.0
            similares_filtrados = [
                item for item in todas_similaridades
                if item["similaridade"] >= limite
            ]

            if grupo_foco:
                similares_filtrados = [
                    item for item in similares_filtrados
                    if item["grupo"] == grupo_foco
                ]
                if len(similares_filtrados) <= 1:
                    candidatos_grupo = sorted(
                        similaridades_por_grupo.get(grupo_foco, []),
                        key=lambda x: x["similaridade"],
                        reverse=True
                    )[:5]
                    similares_filtrados = [
                        {
                            "grupo": grupo_foco,
                            "palavra": item["palavra"],
                            "similaridade": item["similaridade"],
                            "cor": CORES_GRUPOS.get(grupo_foco, "#FFFFFF")
                        }
                        for item in candidatos_grupo
                    ]

            if not similares_filtrados:
                st.markdown(
                    f"Sem similaridades acima de {limite:.0f}% "
                    f"para o grupo detectado ({grupo_foco if grupo_foco else 'n/d'}). "
                    "Listando os resultados mais próximos, independentemente do grupo."
                )
                similares_filtrados = sorted(
                    todas_similaridades,
                    key=lambda x: x["similaridade"],
                    reverse=True
                )[:20]

            similares_filtrados.sort(key=lambda x: x["similaridade"], reverse=True)
            st.markdown(f"🔍 **Similaridades globais para '{st.session_state.palavra_atual}'**")
            for item in similares_filtrados[:20]:
                st.markdown(
                    f"<span style='display:inline-flex; align-items:center;'>"
                    f"<span style='width:10px; height:10px; border-radius:50%; "
                    f"background:{item['cor']}; display:inline-block; margin-right:8px;'></span>"
                    f"{item['palavra']} &mdash; <em>{item['grupo']}</em> "
                    f"({item['similaridade']:.1f}%)"
                    f"</span>",
                    unsafe_allow_html=True
                )
    else:
        st.markdown("Digite uma palavra e analise para ver similaridades.")

    st.subheader("📚 Grupos Semânticos")
    for nome, palavras in GRUPOS.items():
        with st.expander(f"{nome} ({len(palavras)} palavras)"):
            st.write(", ".join(sorted(palavras)))

    palavras_compartilhadas = obter_palavras_compartilhadas(GRUPOS)
    st.subheader("🔁 Palavras em mais de um grupo")
    if palavras_compartilhadas:
        for palavra, grupos_relacionados in palavras_compartilhadas:
            grupos_fmt = ", ".join(grupos_relacionados)
            st.markdown(f"- `{palavra}` → {grupos_fmt}")
    else:
        st.markdown("Nenhuma palavra se repete entre os grupos atuais.")


def main():
    executar_interface(
        criar_grafico_func=criar_grafico_3d_plotly,
        info_grafico_texto=INFO_GRAFICO_3D,
        titulo_pagina="Muiraquitã - Simulador LLM 3D"
    )


if __name__ == "__main__":
    main()
