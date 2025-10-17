## Muiraquitã – Simulador LLM

Aplicação Streamlit que demonstra, de forma visual, como uma LLM poderia agrupar textos por proximidade semântica. O usuário informa frases curtas, a aplicação calcula o quanto elas se aproximam de grupos pré-definidos e atualiza gráficos interativos (3D e 2D). O objetivo é permitir que qualquer pessoa entenda, de maneira acessível, como um modelo de linguagem poderia tomar decisões de classificação.

### Como o simulador funciona

1. **Vocabulário controlado** – Em `simulador_streamlit.py` há dicionários com listas de palavras associadas a cada domínio (`GRUPOS`, `CONTEXTO_GRUPOS`, `PALAVRAS_INFERENCIA`). Eles representam o “conhecimento” da LLM fictícia.
2. **Palavra comum configurável** – As constantes `COMMON_WORD` e `COMMON_WORD_GROUPS` determinam um termo que aparece em vários domínios (ex.: `banco`). A função `sincronizar_palavra_comum` garante que essa palavra esteja apenas nos grupos corretos.
3. **Normalização e comparação** – O texto digitado é normalizado (acentos removidos, caixa baixa) e comparado com os vocabulários usando distância de Levenshtein e similaridade de caracteres. Esses cálculos simulam “proximidade semântica”.
4. **Pontuação por contexto** – A função `analisar_contexto` soma pesos quando encontra termos do usuário nos conjuntos de contexto/inferência, aproximando o comportamento de um modelo que entenda pistas indiretas.
5. **Identificação do grupo** – `identificar_grupo` escolhe o domínio com maior pontuação; se nenhuma pontuação for relevante, o resultado fica “SEM CONTEXTO”.
6. **Visualização** – A camada de interface (`executar_interface`) exibe o texto analisado, barras de pontuação, similaridades detalhadas e gráficos Plotly. Depois de cada análise o app executa `st.rerun()` para garantir que a tela reflita o estado mais recente.

### Visualização 3D (`simulador_streamlit.py`)

- Usa um layout circular 3D com cones e marcadores coloridos para destacar o grupo vencedor.
- Palavras compartilhadas ganham marcador branco; a palavra comum (`banco`) é listada no rodapé junto com qualquer outra repetição detectada por `obter_palavras_compartilhadas`.
- As barras de pontuação mostram a confiança relativa em cada domínio.

### Visualização 2D (`simulador_streamlit_2d.py`)

- Reaproveita o mesmo vocabulário e lógica de identificação.
- Removeu o marcador circular interno para o nome dos grupos; apenas o texto aparece com fonte maior.
- Palavras compartilhadas continuam brancas e a palavra comum recebe um marcador maior.

### Geração de vocabulário (opcional)

- O simulador opera 100% offline com os dicionários definidos em código. Entretanto, o projeto inclui `temp.py`, que demonstra como consultar o Gemini (`google-generativeai`) para gerar novos vocabulários.
- `requirements.txt` e `pyproject.toml` listam `python-dotenv` para ler `GEMINI_API_KEY` do `.env`. Se preferir, basta ignorar o script e manter o vocabulário estático.

### Estrutura dos principais arquivos

| Arquivo | Função |
| --- | --- |
| `simulador_streamlit.py` | Core do simulador 3D: dados, análise, UI e gráficos Plotly. |
| `simulador_streamlit_2d.py` | Visualização alternativa em 2D usando o mesmo núcleo lógico. |
| `temp.py` | Teste simples para verificar a chave do Gemini e gerar conteúdo de validação. |
| `requirements.txt` / `pyproject.toml` | Dependências para instalar com `pip`. |

### Objetivo educacional

O projeto não é uma LLM real; ele apenas aplica regras transparentes para mostrar como diferentes pistas (palavra exata, contexto, inferência) podem influenciar a classificação de um texto. A partir dele é possível:

- Explicar visualmente o conceito de espaço semântico.
- Demonstrar o impacto de vocabulário compartilhado e ambiguidades.
- Simular a “conversão” de uma frase em pontuações por domínio, semelhante a como um modelo de linguagem poderia operar.

Sinta-se à vontade para adaptar os dicionários, cores e pesos para que a simulação reflita domínios distintos ou casos de uso específicos.

### Pré-requisitos

- Python 3.12 ou superior
- Dependências listadas em `requirements.txt`
- Opcional: chave da API Gemini (`GEMINI_API_KEY`) para geração dinâmica de vocabulário (`temp.py`).

### Configuração

1. Crie e ative um ambiente virtual.
2. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

3. (Opcional) Crie um arquivo `.env` com a chave:

   ```
   GEMINI_API_KEY="sua_chave"
   ```

### Execução

- Visualização 3D:

  ```bash
  streamlit run simulador_streamlit.py
  ```

- Visualização 2D:

  ```bash
  streamlit run simulador_streamlit_2d.py
  ```

### Teste rápido da API Gemini

Use o script auxiliar:

```bash
python temp.py
```

Ele valida o carregamento do `.env`, inicializa o modelo `gemini-pro-latest` e envia um prompt simples, confirmando se a resposta é adequada.
