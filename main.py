# --- 1. IMPORTAÇÕES E CONFIGURAÇÕES GERAIS ---
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt

# --- Configurações dos Caminhos ---
CAMINHO_DADOS_BRUTOS = "data/01_raw"
CAMINHO_DADOS_PROCESSADOS = "data/03_processed"
CAMINHO_MODELOS = "models"

# --- Parâmetros do Pipeline ---
PERCENTUAL_TREINO = 0.8
JANELA_DE_TEMPO = 60
EPOCHS = 100
BATCH_SIZE = 32

# Garante que os diretórios de saída existam
os.makedirs(CAMINHO_DADOS_PROCESSADOS, exist_ok=True)
os.makedirs(CAMINHO_MODELOS, exist_ok=True)

print("Ambiente configurado. Iniciando o pipeline...")

# --- 2. ENCONTRAR TODOS OS ARQUIVOS DE AÇÕES ---
arquivos_csv = [f for f in os.listdir(CAMINHO_DADOS_BRUTOS) if f.endswith('.csv')]
print(f"Encontrados {len(arquivos_csv)} arquivos de ações para processar: {arquivos_csv}")

# --- 3. LOOP PRINCIPAL PARA PROCESSAR CADA AÇÃO ---
for nome_arquivo in arquivos_csv:
    ticker = nome_arquivo.replace(".csv", "")
    print(f"\n{'=' * 60}")
    print(f"INICIANDO PROCESSAMENTO PARA O ATIVO: {ticker}")
    print(f"{'=' * 60}")

    try:
        # --- ETAPA DE PRÉ-PROCESSAMENTO ---
        print(f"[{ticker}] - Carregando e processando dados...")
        caminho_arquivo_completo = os.path.join(CAMINHO_DADOS_BRUTOS, nome_arquivo)

        df = pd.read_csv(caminho_arquivo_completo, header=None, skiprows=3, index_col=0, parse_dates=[0])
        df.dropna(inplace=True)

        dados_fechamento = df.iloc[:, 3].values.reshape(-1, 1)  # Coluna 4 (posição 3) = Close

        ponto_divisao = int(len(dados_fechamento) * PERCENTUAL_TREINO)
        dados_treino = dados_fechamento[:ponto_divisao]
        dados_teste = dados_fechamento[ponto_divisao:]

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(dados_treino)
        dados_treino_normalizados = scaler.transform(dados_treino)
        dados_teste_normalizados = scaler.transform(dados_teste)


        def criar_sequencias(dados, janela):
            X, y = [], []
            for i in range(janela, len(dados)):
                X.append(dados[i - janela:i, 0])
                y.append(dados[i, 0])
            return np.array(X), np.array(y)


        X_train, y_train = criar_sequencias(dados_treino_normalizados, JANELA_DE_TEMPO)
        X_test, y_test = criar_sequencias(dados_teste_normalizados, JANELA_DE_TEMPO)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        print(f"[{ticker}] - Dados processados. Shape de X_train: {X_train.shape}")

        # --- ETAPA DE CONSTRUÇÃO DO MODELO ---
        print(f"[{ticker}] - Construindo o modelo LSTM...")
        model = Sequential([
            Input(shape=(X_train.shape[1], 1)),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # --- ETAPA DE TREINAMENTO ---
        print(f"[{ticker}] - Iniciando treinamento...")
        model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0  # Usamos verbose=0 para não poluir a saída do loop
        )
        print(f"[{ticker}] - Treinamento concluído.")

        # --- ETAPA DE SALVAMENTO ---
        caminho_scaler = os.path.join(CAMINHO_MODELOS, f"{ticker}_scaler.pkl")
        joblib.dump(scaler, caminho_scaler)

        caminho_modelo = os.path.join(CAMINHO_MODELOS, f"{ticker}_lstm_model.h5")
        model.save(caminho_modelo)
        print(f"[{ticker}] - Modelo e scaler salvos com sucesso!")

    except Exception as e:
        print(f"!!!!!! ERRO AO PROCESSAR O ATIVO {ticker} !!!!!!")
        print(f"Detalhe do erro: {e}")
        # 'continue' faz o loop pular para o próximo arquivo em caso de erro
        continue

print(f"\n{'=' * 60}")
print("PIPELINE FINALIZADO PARA TODOS OS ATIVOS.")
print(f"{'=' * 60}")