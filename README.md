## Muiraquitã – Simulador LLM

Aplicação Streamlit que demonstra agrupamento semântico em duas visualizações (3D e 2D). O simulador aceita textos curtos, analisa a proximidade com grupos temáticos e exibe resultados interativos.

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
