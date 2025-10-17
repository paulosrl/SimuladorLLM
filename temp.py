import google.generativeai as genai
import os
import sys
from dotenv import load_dotenv

def main_test():
    """
    Script final para testar a geração de conteúdo (pergunta e resposta)
    com um modelo válido da sua lista.
    """
    
    print("Iniciando o teste da API Gemini...")

    # --- 1. Carregar o arquivo .env ---
    try:
        load_dotenv()
        print("Arquivo .env carregado.")
    except Exception as e:
        print(f"Erro ao tentar carregar o dotenv: {e}")
        sys.exit(1)

    # --- 2. Obter a Chave da API do Ambiente ---
    API_KEY = os.environ.get("GEMINI_API_KEY")

    if not API_KEY:
        print("ERRO CRÍTICO: 'GEMINI_API_KEY' não foi encontrada.")
        sys.exit(1)
    
    # --- 3. Configurar a API ---
    try:
        genai.configure(api_key=API_KEY)
        print("Biblioteca google-generativeai configurada com sucesso.")
    except Exception as e:
        print(f"Erro ao configurar o genai: {e}")
        sys.exit(1)


    # --- 4. Inicializar o Modelo (USANDO O MODELO DA SUA LISTA) ---
    # Usando 'gemini-pro-latest' que apareceu na sua lista de modelos
    model_name = "models/gemini-pro-latest" 
    print(f"Inicializando o modelo '{model_name}'...")
    
    try:
        model = genai.GenerativeModel(model_name)
        print("Modelo inicializado.")
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        sys.exit(1)


    # --- 5. Enviar o Prompt de Teste (PERGUNTA) ---
    prompt = "Olá! Faça um teste simples: qual é a capital da França?"
    print(f"\nEnviando prompt: '{prompt}'")
    
    try:
        # Gera o conteúdo
        response = model.generate_content(prompt)

        # --- 6. Imprimir a Resposta (RESPOSTA GERADA) ---
        print("\n" + "="*50)
        print("RESPOSTA RECEBIDA DA API:")
        print(response.text)
        print("="*50)

        if "PARIS" in response.text.upper():
            print("\nStatus: [TESTE BEM-SUCEDIDO]")
            print("A API do Gemini respondeu corretamente.")
        else:
            print(f"\nStatus: [TESTE FALHOU (Resposta Inesperada)]")
            print(f"Recebido: {response.text}")

    except Exception as e:
        print("\n" + "!"*50)
        print(f"ERRO AO CHAMAR A API DO GEMINI:")
        print(f"Detalhes: {e}")
        print("!"*50)
        print("\nStatus: [TESTE FALHOU (Erro na Execução)]")


# Ponto de entrada padrão do Python
if __name__ == "__main__":
    main_test()