"""
Simulador LLM - Análise Contextual Semântica 3D
Versão com Logo MPPA/GIIA e Análise de Similaridade
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D
import unicodedata
import json
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageTk  # Para carregar e redimensionar a logo

# ==================== CONFIGURAÇÕES E DADOS ====================

# Definição dos grupos semânticos
GRUPOS = {
    "Transportes": {"carro", "avião", "ônibus", "bicicleta", "motocicleta", "trem", "veículo"},
    "Móveis": {"cadeira", "mesa", "banco", "armário", "sofá", "cômoda", "cama"},
    "Animais": {"cachorro", "gato", "focinho", "rato", "leão", "tigre", "baleia"},
    "Financeiro": {"banco", "moeda", "cédula", "caixa", "dinheiro", "investimento", "juros"}
}

# Palavras contextuais
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

# Palavras de inferência
PALAVRAS_INFERENCIA = {
    "Animais": ["tartaruga", "cobra", "pássaro", "peixe", "elefante", "girafa", 
                "macaco", "urso", "lobo", "raposa", "coelho", "hamster", "papagaio",
                "jacaré", "crocodilo", "tubarão", "golfinho", "pato", "galinha",
                "vaca", "cavalo", "porco", "ovelha", "cabra", "camelo", "avestruz"],
    "Transportes": ["moto", "barco", "navio", "helicóptero", "metrô", "taxi",
                   "caminhão", "van", "ônibus", "scooter", "patinete", "skate"],
    "Móveis": ["estante", "escrivaninha", "poltrona", "banqueta", "criado-mudo",
              "guarda-roupa", "buffet", "aparador", "rack", "prateleira"],
    "Financeiro": ["pix", "boleto", "nota", "real", "dólar", "euro", "bitcoin",
                  "ação", "fundo", "renda", "lucro", "débito", "crédito"]
}

# Cores dos grupos
CORES_GRUPOS = {
    "Transportes": "#FF4444",
    "Móveis": "#4488FF",
    "Animais": "#44FF44",
    "Financeiro": "#FFAA00"
}

# Pesos para análise contextual
PESOS = {
    "contexto": 3.0,
    "inferencia": 2.5,
    "principal": 0.2
}

# ==================== FUNÇÕES UTILITÁRIAS ====================

def normalizar_texto(texto):
    """
    Remove acentuação e converte texto para minúsculas.
    """
    if not isinstance(texto, str):
        return ""
    
    texto_nfd = unicodedata.normalize('NFD', texto)
    texto_sem_acento = ''.join(
        char for char in texto_nfd 
        if unicodedata.category(char) != 'Mn'
    )
    
    return texto_sem_acento.lower().strip()


def carregar_logo(caminho_arquivo, tamanho=(80, 80)):
    """
    Carrega e redimensiona a logo.
    """
    try:
        # Carregar imagem
        imagem = Image.open(caminho_arquivo)
        
        # Redimensionar mantendo proporção
        imagem.thumbnail(tamanho, Image.Resampling.LANCZOS)
        
        # Converter para PhotoImage do tkinter
        return ImageTk.PhotoImage(imagem)
    except Exception as e:
        print(f"Erro ao carregar logo: {e}")
        return None


def calcular_similaridade_levenshtein(palavra1, palavra2):
    """
    Calcula a similaridade usando distância de Levenshtein normalizada.
    """
    if palavra1 == palavra2:
        return 1.0
    
    len1, len2 = len(palavra1), len(palavra2)
    if len1 == 0 or len2 == 0:
        return 0.0
    
    # Matriz de distâncias
    matriz = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    # Inicializar primeira linha e coluna
    for i in range(len1 + 1):
        matriz[i][0] = i
    for j in range(len2 + 1):
        matriz[0][j] = j
    
    # Calcular distâncias
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            custo = 0 if palavra1[i-1] == palavra2[j-1] else 1
            matriz[i][j] = min(
                matriz[i-1][j] + 1,      # deleção
                matriz[i][j-1] + 1,      # inserção
                matriz[i-1][j-1] + custo # substituição
            )
    
    distancia = matriz[len1][len2]
    max_len = max(len1, len2)
    similaridade = 1 - (distancia / max_len)
    
    return max(0.0, similaridade)


def calcular_similaridade_caracteres(palavra1, palavra2):
    """
    Calcula similaridade baseada em caracteres comuns.
    """
    if palavra1 == palavra2:
        return 1.0
    
    set1 = set(palavra1.lower())
    set2 = set(palavra2.lower())
    
    if not set1 or not set2:
        return 0.0
    
    intersecao = len(set1.intersection(set2))
    uniao = len(set1.union(set2))
    
    return intersecao / uniao if uniao > 0 else 0.0


def calcular_similaridade_composta(palavra_busca, palavra_comparacao):
    """
    Calcula similaridade composta usando múltiplas métricas.
    """
    palavra_busca = normalizar_texto(palavra_busca)
    palavra_comparacao = normalizar_texto(palavra_comparacao)
    
    if palavra_busca == palavra_comparacao:
        return 100.0
    
    # Similaridade de Levenshtein (peso 60%)
    sim_levenshtein = calcular_similaridade_levenshtein(palavra_busca, palavra_comparacao)
    
    # Similaridade de caracteres (peso 40%)
    sim_caracteres = calcular_similaridade_caracteres(palavra_busca, palavra_comparacao)
    
    # Bonus para substring
    bonus_substring = 0.0
    if palavra_busca in palavra_comparacao or palavra_comparacao in palavra_busca:
        bonus_substring = 0.2
    
    # Similaridade composta
    similaridade_final = (sim_levenshtein * 0.6 + sim_caracteres * 0.4 + bonus_substring) * 100
    
    return min(100.0, similaridade_final)


def obter_todas_palavras():
    """
    Retorna todas as palavras dos grupos organizadas por categoria.
    """
    todas_palavras = {}
    
    for nome_grupo, palavras_grupo in GRUPOS.items():
        todas_palavras[nome_grupo] = list(palavras_grupo)
    
    return todas_palavras


def calcular_similaridades_palavra(palavra_busca):
    """
    Calcula similaridades de uma palavra com todas as outras palavras dos grupos.
    """
    if not palavra_busca or len(palavra_busca.strip()) < 2:
        return {}
    
    palavra_busca = palavra_busca.strip()
    todas_palavras = obter_todas_palavras()
    similaridades = {}
    
    for nome_grupo, palavras_grupo in todas_palavras.items():
        similaridades[nome_grupo] = []
        
        for palavra in sorted(palavras_grupo):
            similaridade = calcular_similaridade_composta(palavra_busca, palavra)
            similaridades[nome_grupo].append({
                'palavra': palavra,
                'similaridade': similaridade
            })
        
        # Ordenar por similaridade decrescente
        similaridades[nome_grupo].sort(key=lambda x: x['similaridade'], reverse=True)
    
    return similaridades


def detectar_palavras_ambiguas(texto):
    """
    Identifica palavras que pertencem a múltiplos grupos semânticos.
    """
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
    """
    Analisa o contexto completo do texto e calcula scores de pertinência.
    """
    if not isinstance(texto_completo, str) or not texto_completo.strip():
        return {nome: 0.0 for nome in GRUPOS.keys()}
    
    if pesos is None:
        pesos = PESOS
    
    texto_normalizado = normalizar_texto(texto_completo)
    scores_contexto = {nome: 0.0 for nome in GRUPOS.keys()}
    
    # Analisar palavras contextuais
    for nome_grupo, palavras_contexto in CONTEXTO_GRUPOS.items():
        for palavra_ctx in palavras_contexto:
            palavra_norm = normalizar_texto(palavra_ctx)
            if palavra_norm in texto_normalizado:
                scores_contexto[nome_grupo] += pesos["contexto"]
    
    # Analisar palavras principais
    for nome_grupo, palavras_grupo in GRUPOS.items():
        for palavra_principal in palavras_grupo:
            palavra_norm = normalizar_texto(palavra_principal)
            if f" {palavra_norm} " in f" {texto_normalizado} " or \
               texto_normalizado.startswith(palavra_norm + " ") or \
               texto_normalizado.endswith(" " + palavra_norm):
                scores_contexto[nome_grupo] += pesos["principal"]
    
    # Analisar inferências
    for nome_grupo, palavras_relacionadas in PALAVRAS_INFERENCIA.items():
        for palavra_rel in palavras_relacionadas:
            palavra_norm = normalizar_texto(palavra_rel)
            if palavra_norm in texto_normalizado:
                scores_contexto[nome_grupo] += pesos["inferencia"]
    
    # Normalizar scores
    total = sum(scores_contexto.values())
    if total > 0:
        scores_contexto = {k: v/total for k, v in scores_contexto.items()}
    
    return scores_contexto


def identificar_grupo(texto):
    """
    Identifica o grupo principal baseado na análise de contexto.
    """
    scores = analisar_contexto(texto)
    
    if not scores or all(v == 0 for v in scores.values()):
        return None, scores
    
    grupo_principal = max(scores.items(), key=lambda x: x[1])
    
    if grupo_principal[1] > 0.1:
        return grupo_principal[0], scores
    
    return None, scores


def detectar_palavras_desconhecidas(texto):
    """
    Identifica palavras que não pertencem a nenhum grupo ou contexto.
    """
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


# ==================== VISUALIZAÇÃO 3D ====================

def criar_grafico_3d(ax, texto_busca=None):
    """
    Cria o gráfico 3D do espaço semântico.
    """
    ax.clear()
    
    # Criar coordenadas para grupos
    nomes_grupos = list(GRUPOS.keys())
    n_grupos = len(nomes_grupos)
    angulos_grupos = np.linspace(0, 2*np.pi, n_grupos, endpoint=False)
    raio_grupos = 5.0
    
    coords_grupos = {}
    for i, nome_grupo in enumerate(nomes_grupos):
        x = raio_grupos * np.cos(angulos_grupos[i])
        y = raio_grupos * np.sin(angulos_grupos[i])
        coords_grupos[nome_grupo] = (x, y, 0)
    
    # Posicionar palavras
    coords_palavras = {}
    
    for nome_grupo, palavras_grupo in GRUPOS.items():
        centro_x, centro_y, centro_z = coords_grupos[nome_grupo]
        palavras_lista = sorted(palavras_grupo)
        n_palavras = len(palavras_lista)
        raio_interno = 1.2
        
        for j, palavra in enumerate(palavras_lista):
            angulo = 2 * np.pi * j / n_palavras
            x = centro_x + raio_interno * np.cos(angulo)
            y = centro_y + raio_interno * np.sin(angulo)
            z = np.random.randn() * 0.3
            
            # Criar chave única para cada palavra em cada grupo
            chave = f"{palavra}_{nome_grupo}"
            coords_palavras[chave] = (x, y, z, palavra, nome_grupo)
    
    # Plotar centros dos grupos
    for nome_grupo, (x, y, z) in coords_grupos.items():
        cor = CORES_GRUPOS[nome_grupo]
        ax.scatter(x, y, z, c=cor, s=500, alpha=0.4, edgecolors='black', 
                  linewidths=3, marker='o', zorder=1)
        ax.text(x, y, z-0.6, nome_grupo, fontsize=11, weight='bold', 
               ha='center', color='black', zorder=2)
    
    # Plotar palavras
    for chave, (x, y, z, palavra, nome_grupo) in coords_palavras.items():
        # Verificar se a palavra existe em múltiplos grupos
        grupos_da_palavra = []
        for ng, pg in GRUPOS.items():
            if palavra in pg:
                grupos_da_palavra.append(ng)
        
        if len(grupos_da_palavra) > 1:
            # Palavra ambígua - diamante roxo
            cor = "#9B59B6"
            ax.scatter(x, y, z, c=cor, s=140, alpha=0.9, edgecolors='red', 
                      linewidths=2.5, zorder=3, marker='D')
            ax.text(x, y, z, f' {palavra}', fontsize=8, weight='bold', 
                   color='purple', alpha=0.95, zorder=3)
        else:
            # Palavra normal
            cor = CORES_GRUPOS[nome_grupo]
            ax.scatter(x, y, z, c=cor, s=90, alpha=0.75, edgecolors='black', 
                      linewidths=0.8, zorder=3)
            ax.text(x, y, z, f' {palavra}', fontsize=8, alpha=0.8, zorder=3)
    
    # Adicionar texto de busca se fornecido
    if texto_busca:
        grupo_identificado, scores = identificar_grupo(texto_busca)
        
        if grupo_identificado:
            centro_x, centro_y, centro_z = coords_grupos[grupo_identificado]
            coord_busca = (centro_x, centro_y, 1.2)
            cor_destaque = CORES_GRUPOS[grupo_identificado]
            
            ax.scatter(coord_busca[0], coord_busca[1], coord_busca[2], 
                      c=cor_destaque, s=900, marker='*', edgecolors='black', 
                      linewidths=5, zorder=10)
            
            texto_display = f'★ {texto_busca[:30]} ★'
            ax.text(coord_busca[0], coord_busca[1], coord_busca[2] + 0.9, 
                   texto_display, fontsize=12, weight='bold', color='darkred',
                   ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.6', facecolor=cor_destaque, 
                            edgecolor='black', linewidth=3, alpha=0.95),
                   zorder=11)
        else:
            coord_busca = (0, 0, 2.2)
            ax.scatter(coord_busca[0], coord_busca[1], coord_busca[2], 
                      c='gray', s=900, marker='*', edgecolors='black', 
                      linewidths=5, zorder=10)
            
            texto_display = f'★ {texto_busca[:30]} ★\n(SEM CONTEXTO)'
            ax.text(coord_busca[0], coord_busca[1], coord_busca[2] + 0.9, 
                   texto_display, fontsize=11, weight='bold', color='black',
                   ha='center', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.6', facecolor='lightgray', 
                            edgecolor='black', linewidth=3, alpha=0.95),
                   zorder=11)
    
    # Configurar gráfico
    ax.set_xlabel('X', fontsize=11, weight='bold')
    ax.set_ylabel('Y', fontsize=11, weight='bold')
    ax.set_zlabel('Z', fontsize=11, weight='bold')
    
    grupos = ', '.join(GRUPOS.keys())
    titulo = f'Grupos Semânticos: {grupos}'
    if texto_busca:
        titulo += f'\nEntrada: "{texto_busca[:50]}"'
    
    ax.set_title(titulo, fontsize=12, weight='bold', pad=20)
    
    limite = 8
    ax.set_xlim([-limite, limite])
    ax.set_ylim([-limite, limite])
    ax.set_zlim([-2, 4])
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.view_init(elev=20, azim=45)


# ==================== TOOLTIP ====================

class ToolTip:
    """
    Classe para criar tooltips customizados em widgets Tkinter.
    """
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        
        widget.bind("<Enter>", self.mostrar)
        widget.bind("<Leave>", self.esconder)
    
    def mostrar(self, event=None):
        if self.tooltip_window or not self.text:
            return
        
        x = self.widget.winfo_rootx() + 25
        y = self.widget.winfo_rooty() + 25
        
        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                        background="#FFFFDD", relief=tk.SOLID, borderwidth=1,
                        font=("Arial", 9, "normal"), padx=5, pady=3)
        label.pack()
    
    def esconder(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


# ==================== APLICAÇÃO PRINCIPAL ====================

class AplicacaoLLM:
    """
    Aplicação principal do Simulador LLM com análise contextual 3D e logo MPPA.
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Simulador LLM 3D v1.0 - Escritório de Inovação e IA - MPPA")
        self.root.geometry("1600x850")
        
        self.zoom_level = 1.0
        self.default_limits = 8
        self.palavra_atual = ""
        self.logo_image = None
        
        self._configurar_estilo()
        self._carregar_logo()
        self._criar_interface()
        self.mostrar_inicial()
    
    def _configurar_estilo(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Accent.TButton', 
                       font=('Arial', 10, 'bold'),
                       padding=8)
    
    def _carregar_logo(self):
        """Carrega a logo do MPPA/GIIA."""
        try:
            # Tentar carregar a logo do arquivo
            self.logo_image = carregar_logo("Mui.png", (40, 80))
        except Exception as e:
            print(f"Logo não encontrada: {e}")
            self.logo_image = None
    
    def _criar_interface(self):
        # Frame superior para logo
        top_frame = ttk.Frame(self.root)
        top_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=10, pady=(10, 0))
        self.root.columnconfigure(0, weight=1)
        
        # TOOLBAR DO MATPLOTLIB NO TOPO (EM CIMA!)
        toolbar_top_frame = ttk.Frame(top_frame)
        toolbar_top_frame.pack(side=tk.LEFT, padx=(15, 0), pady=5)

        # Criar toolbar temporária para mover depois
        self.temp_toolbar_frame = toolbar_top_frame

        # Logo no canto superior direito
        if self.logo_image:
            logo_label = tk.Label(top_frame, image=self.logo_image, 
                                 background='white', borderwidth=2, relief=tk.RAISED)
            logo_label.pack(side=tk.RIGHT, padx=(0, 10), pady=5)
            ToolTip(logo_label, "Escritório de Inovação e\nInteligência Artificial\nMPPA - Pará")
        
        # Frame principal abaixo da logo
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.rowconfigure(1, weight=1)
        main_frame.columnconfigure(1, weight=3)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        self._criar_painel_controle(main_frame)
        self._criar_painel_grafico(main_frame)
    
    def _criar_painel_controle(self, parent):
        control_frame = ttk.LabelFrame(parent, text="🎛️ Painel de Controle", 
                                       padding="15")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), 
                          padx=(10, 5), pady=(5, 10))
        # Impedir expansão horizontal do painel de controle
        control_frame.columnconfigure(0, weight=1)
        
        title_label = ttk.Label(control_frame, 
                               text="Muiraquitã - Simulador de LLM\n MPPA - CIIA", 
                               font=('Arial', 12, 'bold'),
                               justify=tk.CENTER,
                               foreground="#1e3a8a")
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 15))
        
        ttk.Label(control_frame, text="Digite uma frase ou palavra:", 
                 font=('Arial', 10, 'bold')).grid(row=1, column=0, columnspan=2, 
                                                  sticky=tk.W, pady=(5, 3))
        
        self.entrada = ttk.Entry(control_frame, width=30, font=('Arial', 11))
        self.entrada.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), 
                         pady=(0, 15))
        self.entrada.bind('<Return>', lambda e: self.analisar_texto())
        
        ToolTip(self.entrada, 
               "Digite uma frase com contexto.\nEx: 'Vou sentar no banco da praça'")
        
        self._criar_botoes_acao(control_frame)
        self._criar_area_resultados(control_frame)
        self._criar_area_similaridades(control_frame)
        self._criar_area_grupos(control_frame)
    
    def _criar_botoes_acao(self, parent):
        btn_frame = ttk.Frame(parent)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=(0, 15))
        
        self.btn_analisar = ttk.Button(btn_frame, text="🔍 Analisar", 
                                       command=self.analisar_texto,
                                       style='Accent.TButton')
        self.btn_analisar.grid(row=0, column=0, padx=3)
        ToolTip(self.btn_analisar, "Analisa o texto e atualiza o gráfico")
        
        self.btn_limpar = ttk.Button(btn_frame, text="🔄 Limpar", 
                                     command=self.limpar)
        self.btn_limpar.grid(row=0, column=1, padx=3)
        ToolTip(self.btn_limpar, "Limpa o campo de entrada e reseta")
        
        self.btn_inicial = ttk.Button(btn_frame, text="🏠 Inicial", 
                                      command=self.mostrar_inicial)
        self.btn_inicial.grid(row=0, column=2, padx=3)
        ToolTip(self.btn_inicial, "Mostra o estado inicial do gráfico")
        
        btn_frame2 = ttk.Frame(parent)
        btn_frame2.grid(row=4, column=0, columnspan=2, pady=(0, 15))
        
        self.btn_exportar_img = ttk.Button(btn_frame2, text="💾 Exportar Gráfico", 
                                          command=self.exportar_grafico)
        self.btn_exportar_img.grid(row=0, column=0, padx=3)
        ToolTip(self.btn_exportar_img, "Salva o gráfico como imagem PNG")
        
        self.btn_exportar_dados = ttk.Button(btn_frame2, text="📊 Exportar Dados", 
                                            command=self.exportar_dados)
        self.btn_exportar_dados.grid(row=0, column=1, padx=3)
        ToolTip(self.btn_exportar_dados, "Salva os dados de análise em JSON")
    
    def _criar_area_resultados(self, parent):
        ttk.Label(parent, text="📊 Análise Detalhada:", 
                 font=('Arial', 10, 'bold')).grid(row=5, column=0, columnspan=2, 
                                                  sticky=tk.W, pady=(10, 5))
        
        self.resultado_texto = scrolledtext.ScrolledText(
            parent, width=48, height=9, 
            font=('Courier', 9), wrap=tk.WORD,
            borderwidth=2, relief=tk.SUNKEN
        )
        self.resultado_texto.grid(row=6, column=0, columnspan=2, 
                                 sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.resultado_texto.tag_config('success', foreground='#008800', 
                                       font=('Courier', 9, 'bold'))
        self.resultado_texto.tag_config('warning', foreground='#FF6600', 
                                       font=('Courier', 9, 'bold'))
        self.resultado_texto.tag_config('error', foreground='#CC0000', 
                                       font=('Courier', 9, 'bold'))
        self.resultado_texto.tag_config('info', foreground='#0066CC', 
                                       font=('Courier', 9, 'bold'))
    
    def _criar_area_similaridades(self, parent):
        """Cria a área de similaridades."""
        sim_frame = ttk.LabelFrame(parent, text="🔍 Similaridades", padding="5")
        sim_frame.grid(row=7, column=0, columnspan=2, 
                       sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.similaridade_texto = scrolledtext.ScrolledText(
            sim_frame, width=48, height=7, 
            font=('Courier', 8), wrap=tk.WORD,
            borderwidth=2, relief=tk.SUNKEN
        )
        self.similaridade_texto.pack(fill=tk.BOTH, expand=True)
        
        # Configurar tags de cores para grupos
        for nome_grupo, cor in CORES_GRUPOS.items():
            self.similaridade_texto.tag_config(f'grupo_{nome_grupo}', 
                                              foreground=cor,
                                              font=('Courier', 8, 'bold'))
        
        self.similaridade_texto.tag_config('alta_sim', 
                                          background='#E8F5E8',
                                          font=('Courier', 8, 'bold'))
        self.similaridade_texto.tag_config('media_sim', 
                                          background='#FFF8DC')
        
        ToolTip(self.similaridade_texto, 
               "Mostra similaridade entre palavra pesquisada\ne todas palavras dos grupos")
    
    def _criar_area_grupos(self, parent):
        ttk.Label(parent, text="📚 Grupos Semânticos:", 
                 font=('Arial', 10, 'bold')).grid(row=8, column=0, columnspan=2, 
                                                  sticky=tk.W, pady=(10, 5))
        
        self.grupos_texto = scrolledtext.ScrolledText(
            parent, width=48, height=5, 
            font=('Courier', 8), wrap=tk.WORD,
            borderwidth=2, relief=tk.SUNKEN
        )
        self.grupos_texto.grid(row=9, column=0, columnspan=2, 
                              sticky=(tk.W, tk.E))
        
        for nome, palavras in GRUPOS.items():
            cor = CORES_GRUPOS[nome]
            self.grupos_texto.insert(tk.END, f"• {nome}:\n", nome)
            self.grupos_texto.tag_config(nome, foreground=cor, 
                                        font=('Courier', 8, 'bold'))
            self.grupos_texto.insert(tk.END, 
                                    f"  {', '.join(sorted(palavras))}\n\n")
        
        self.grupos_texto.config(state=tk.DISABLED)
    
    def _criar_painel_grafico(self, parent):
        graph_frame = ttk.LabelFrame(parent, text="🎨 Visualização 3D", 
                                     padding="10")
        graph_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), 
                        padx=(5, 10), pady=(5, 10))
        # Permitir que o gráfico expanda
        graph_frame.columnconfigure(0, weight=1)
        graph_frame.rowconfigure(0, weight=1)
        
        self.fig = plt.Figure(figsize=(11, 8.5), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar_frame = ttk.Frame(graph_frame)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
        
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()
        
        self._criar_controles_zoom(graph_frame)
    
    def _criar_controles_zoom(self, parent):
        zoom_frame = ttk.Frame(parent)
        zoom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        ttk.Label(zoom_frame, text="🔍 Zoom:", 
                 font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=5)
        
        btn_zoom_in = ttk.Button(zoom_frame, text="➕ Ampliar", 
                                command=self.zoom_in, width=12)
        btn_zoom_in.pack(side=tk.LEFT, padx=2)
        ToolTip(btn_zoom_in, "Aumenta o zoom do gráfico")
        
        btn_zoom_out = ttk.Button(zoom_frame, text="➖ Reduzir", 
                                 command=self.zoom_out, width=12)
        btn_zoom_out.pack(side=tk.LEFT, padx=2)
        ToolTip(btn_zoom_out, "Diminui o zoom do gráfico")
        
        btn_zoom_reset = ttk.Button(zoom_frame, text="🔄 Resetar", 
                                   command=self.zoom_reset, width=12)
        btn_zoom_reset.pack(side=tk.LEFT, padx=2)
        ToolTip(btn_zoom_reset, "Restaura o zoom padrão")
    
    def mostrar_inicial(self):
        try:
            criar_grafico_3d(self.ax)
            self.canvas.draw()
            
            self.resultado_texto.config(state=tk.NORMAL)
            self.resultado_texto.delete(1.0, tk.END)
            self.resultado_texto.insert(tk.END, "🎨 Estado inicial carregado!\n\n")
            self.resultado_texto.insert(tk.END, 
                "Mostrando estrutura dos grupos semânticos.\n\n")
            self.resultado_texto.insert(tk.END, 
                "💡 Digite uma palavra e clique em Analisar!\n\n", 'info')
            self.resultado_texto.insert(tk.END, 
                "Exemplos de teste:\n")
            self.resultado_texto.insert(tk.END, 
                "• 'banco' - palavra ambígua\n")
            self.resultado_texto.insert(tk.END, 
                "• 'carro' - palavra específica\n")
            self.resultado_texto.insert(tk.END, 
                "• 'sentar no banco' - frase contextual\n")
            self.resultado_texto.config(state=tk.DISABLED)
            
            # Limpar similaridades
            self._limpar_similaridades()
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao mostrar estado inicial:\n{str(e)}")
    
    def _limpar_similaridades(self):
        """Limpa a área de similaridades."""
        self.similaridade_texto.config(state=tk.NORMAL)
        self.similaridade_texto.delete(1.0, tk.END)
        self.similaridade_texto.insert(tk.END, 
            "🔍 Digite uma palavra e analise\npara ver similaridades\n\n")
        self.similaridade_texto.insert(tk.END, 
            "A similaridade será calculada entre\nsua palavra e todas as palavras\ndos grupos semânticos.\n\n")
        self.similaridade_texto.insert(tk.END,
            "🎯 Desenvolvido pelo GIIA\nEscritório de Inovação - MPPA", 'info')
        self.similaridade_texto.config(state=tk.DISABLED)
    
    def _exibir_similaridades(self, palavra_busca):
        """Exibe as similaridades da palavra buscada."""
        similaridades = calcular_similaridades_palavra(palavra_busca)
        
        self.similaridade_texto.config(state=tk.NORMAL)
        self.similaridade_texto.delete(1.0, tk.END)
        
        self.similaridade_texto.insert(tk.END, 
            f"🔍 Similaridades: '{palavra_busca}'\n")
        self.similaridade_texto.insert(tk.END, "─" * 40 + "\n\n")
        
        for nome_grupo in GRUPOS.keys():
            if nome_grupo in similaridades:
                self.similaridade_texto.insert(tk.END, 
                    f"• {nome_grupo}:\n", f'grupo_{nome_grupo}')
                
                # Pegar top 5 mais similares do grupo
                top_similares = similaridades[nome_grupo][:5]
                
                for item in top_similares:
                    palavra = item['palavra']
                    sim = item['similaridade']
                    
                    # Determinar tag baseada na similaridade
                    if sim >= 80:
                        tag = 'alta_sim'
                        icone = "🟢"
                    elif sim >= 50:
                        tag = 'media_sim'  
                        icone = "🟡"
                    else:
                        tag = ''
                        icone = "⚪"
                    
                    linha = f"  {icone} {palavra:<11} {sim:5.1f}%\n"
                    self.similaridade_texto.insert(tk.END, linha, tag)
                
                self.similaridade_texto.insert(tk.END, "\n")
        
        self.similaridade_texto.config(state=tk.DISABLED)
    
    def analisar_texto(self):
        texto = self.entrada.get().strip()
        
        if not texto:
            messagebox.showwarning("Atenção", "Por favor, digite algo!")
            return
        
        if len(texto) < 2:
            messagebox.showwarning("Atenção", 
                "Texto muito curto! Digite pelo menos 2 caracteres.")
            return
        
        try:
            criar_grafico_3d(self.ax, texto)
            self.canvas.draw()
            
            grupo_identificado, scores = identificar_grupo(texto)
            ambiguas = detectar_palavras_ambiguas(texto)
            desconhecidas = detectar_palavras_desconhecidas(texto)
            
            self._exibir_resultados(texto, grupo_identificado, scores, 
                                   ambiguas, desconhecidas)
            
            # Extrair primeira palavra para análise de similaridade
            primeira_palavra = texto.split()[0].strip('.,!?;:')
            self.palavra_atual = primeira_palavra
            self._exibir_similaridades(primeira_palavra)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao analisar texto:\n{str(e)}")
    
    def _exibir_resultados(self, texto, grupo, scores, ambiguas, desconhecidas):
        self.resultado_texto.config(state=tk.NORMAL)
        self.resultado_texto.delete(1.0, tk.END)
        
        # Palavras ambíguas
        if ambiguas:
            self.resultado_texto.insert(tk.END, "🔀 PALAVRAS AMBÍGUAS:\n", 
                                       'warning')
            for palavra_amb, grupos_amb in ambiguas:
                self.resultado_texto.insert(tk.END, 
                    f"  • '{palavra_amb}' pertence a:\n")
                for g in grupos_amb:
                    self.resultado_texto.insert(tk.END, f"    → {g}\n")
            self.resultado_texto.insert(tk.END, "\n")
        
        # Palavras desconhecidas
        if desconhecidas:
            self.resultado_texto.insert(tk.END, 
                "❓ PALAVRAS DESCONHECIDAS:\n", 'info')
            self.resultado_texto.insert(tk.END, 
                f"  {', '.join(desconhecidas)}\n\n")
        
        # Grupo identificado
        if grupo:
            self.resultado_texto.insert(tk.END, 
                f"✅ GRUPO: {grupo}\n", 'success')
            self.resultado_texto.insert(tk.END, 
                f"   Confiança: {scores[grupo]*100:.1f}%\n\n")
        else:
            self.resultado_texto.insert(tk.END, 
                "⚠️ SEM CONTEXTO\n", 'warning')
            self.resultado_texto.insert(tk.END, "\n")
        
        # Scores detalhados
        self.resultado_texto.insert(tk.END, "🎯 Pertinência por grupo:\n")
        self.resultado_texto.insert(tk.END, "─" * 45 + "\n")
        
        scores_ordenados = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for nome_grupo, score in scores_ordenados:
            if score > 0.001:
                num_blocos = int(score * 18)
                barra = "██" * num_blocos
                espacos = "  " * (18 - num_blocos)
                
                self.resultado_texto.insert(tk.END, 
                    f"{nome_grupo:12}:  {barra}{espacos} {score*100:4.1f}%\n")
        
        self.resultado_texto.config(state=tk.DISABLED)
    
    def limpar(self):
        self.entrada.delete(0, tk.END)
        self.palavra_atual = ""
        self.mostrar_inicial()
    
    def exportar_grafico(self):
        try:
            filepath = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                initialfile=f"grafico_semantico_MPPA_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            
            if filepath:
                self.fig.savefig(filepath, dpi=300, bbox_inches='tight', 
                                facecolor='white', edgecolor='none')
                messagebox.showinfo("Sucesso", 
                    f"Gráfico exportado!\n{filepath}")
        
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao exportar gráfico:\n{str(e)}")
    
    def exportar_dados(self):
        try:
            if not self.palavra_atual:
                messagebox.showwarning("Atenção", 
                    "Não há dados para exportar!")
                return
            
            filepath = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialfile=f"similaridades_MPPA_{self.palavra_atual}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            if filepath:
                similaridades = calcular_similaridades_palavra(self.palavra_atual)
                
                dados_exportar = {
                    'palavra_analisada': self.palavra_atual,
                    'timestamp_exportacao': datetime.now().isoformat(),
                    'instituicao': 'MPPA - GIIA',
                    'similaridades': similaridades
                }
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(dados_exportar, f, ensure_ascii=False, indent=2)
                
                messagebox.showinfo("Sucesso", 
                    f"Dados exportados!\n{filepath}")
        
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao exportar dados:\n{str(e)}")
    
    def zoom_in(self):
        self.zoom_level *= 0.8
        self.aplicar_zoom()
    
    def zoom_out(self):
        self.zoom_level *= 1.25
        self.aplicar_zoom()
    
    def zoom_reset(self):
        self.zoom_level = 1.0
        self.aplicar_zoom()
    
    def aplicar_zoom(self):
        limite = self.default_limits * self.zoom_level
        self.ax.set_xlim([-limite, limite])
        self.ax.set_ylim([-limite, limite])
        self.ax.set_zlim([-limite/4, limite/2])
        self.canvas.draw()


# ==================== EXECUÇÃO ====================

def main():
    """Função principal de execução."""
    root = tk.Tk()
    app = AplicacaoLLM(root)
    
    # Configurar ícone da janela se possível
    try:
        if app.logo_image:
            root.iconphoto(True, app.logo_image)
    except:
        pass
    
    # Centralizar janela
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()


if __name__ == "__main__":
    main()

