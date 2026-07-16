# ROADMAP — Defesa Espacial & Consciência Situacional
## Daniel Palma · UNIFEI · OPD/COAFA

---

## 🎯 Visão Geral

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SISTEMA DE DEFESA ESPACIAL                       │
│                                                                     │
│   DETECÇÃO ──→ TRIAGEM ──→ CLASSIFICAÇÃO ──→ RESPOSTA ──→ DEFESA   │
│                                                                     │
│   ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌────────┐  ┌────────┐  │
│   │ Sensores│  │ Filtros │  │   IA +   │  │ Análise│  │ Ação   │  │
│   │ Radar   │→ │ Automat.│→ │ Física   │→ │ Risco  │→ │ Protet.│  │
│   │ Óptico  │  │ Sinais  │  │ Matemát. │  │ Humana │  │        │  │
│   │ EM      │  │         │  │          │  │        │  │        │  │
│   └─────────┘  └─────────┘  └──────────┘  └────────┘  └────────┘  │
│                                                                     │
│   FASE 1       FASE 2       FASE 3        FASE 4     FASE 5        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📊 FASE 1 — Análise Exploratória de Dados (AED)

### 1.1 Dados Disponíveis no OPD

| Tipo de Dado | Fonte | Volume | Formato |
|--------------|-------|--------|---------|
| Imagens noturnas | Telescópio OPD | TB/noite | FITS, TIFF |
| Catálogo de objetos | LNA/NORAD | Milhares | CSV, JSON |
| Efemérides | JPL/NASA | Contínuo | SPICE, TLE |
| Dados climáticos | INPE/Estação local | Horário | JSON |
| Dados eletromagnéticos | Sensores EM | Contínuo | Binário |

### 1.2 Técnicas de AED Aplicadas

```python
# Pipeline de Análise Exploratória para Dados Astronômicos
# Daniel Palma — UNIFEI

import pandas as pd
import numpy as np
from astropy.io import fits
from scipy import stats

class AnaliseEspacial:
    """Pipeline de AED para detecção e classificação de objetos."""
    
    def __init__(self):
        self.dados = None
        self.outliers = None
        self.padrao = None
    
    def carregar_dados(self, caminho):
        """Carrega dados do telescópio (FITS) ou catálogo (CSV)."""
        if caminho.endswith('.fits'):
            with fits.open(caminho) as hdul:
                self.dados = hdul[1].data
        else:
            self.dados = pd.read_csv(caminho)
        return self
    
    def detectar_outliers(self, metodo='iqr', threshold=1.5):
        """Detecta anomalias usando IQR, Z-Score ou Isolation Forest."""
        if metodo == 'iqr':
            Q1 = self.dados.quantile(0.25)
            Q3 = self.dados.quantile(0.75)
            IQR = Q3 - Q1
            mask = (self.dados < Q1 - threshold*IQR) | (self.dados > Q3 + threshold*IQR)
            self.outliers = self.dados[mask.any(axis=1)]
        elif metodo == 'zscore':
            z = np.abs(stats.zscore(self.dados.select_dtypes(include=[np.number])))
            self.outliers = self.dados[(z > threshold).any(axis=1)]
        return self
    
    def analisar_trajetoria(self, posicoes, tempos):
        """Calcula velocidade, aceleração e classifica órbita."""
        velocidades = np.diff(posicoes, axis=0) / np.diff(tempos)[:, None]
        aceleracoes = np.diff(velocidades, axis=0) / np.diff(tempos[1:])[:, None]
        
        # Classificação por energia específica
        energia = 0.5 * np.linalg.norm(velocidades, axis=1)**2
        
        return {
            'velocidade_media': np.mean(np.linalg.norm(velocidades, axis=1)),
            'aceleracao_media': np.mean(np.linalg.norm(aceleracoes, axis=1)),
            'energia_especifica': np.mean(energia),
            'tipo_orbita': self._classificar_orbita(energia)
        }
    
    def _classificar_orbita(self, energia):
        """Classifica tipo de órbita pela energia específica."""
        e_med = np.mean(energia)
        if e_med < 0:
            return 'ELÍPTICA (ligada)'
        elif e_med == 0:
            return 'PARABÓLICA (escape)'
        else:
            return 'HIPERBÓLICA (passagem)'
    
    def correlacao_variaveis(self):
        """Matriz de correlação para identificar variáveis relevantes."""
        numeric_cols = self.dados.select_dtypes(include=[np.number]).columns
        corr = self.dados[numeric_cols].corr()
        
        # Pares mais correlacionados
        pares = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                if abs(corr.iloc[i, j]) > 0.7:
                    pares.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
        
        return corr, sorted(pares, key=lambda x: abs(x[2]), reverse=True)
```

### 1.3 Métricas de Consciência Situacional

| Métrica | Descrição | Fórmula |
|---------|-----------|---------|
| **Taxa de Detecção** | Objetos detectados / objetos reais | TP / (TP + FN) |
| **Precisão** | Detecções corretas / total de detecções | TP / (TP + FP) |
| **Latência de Detecção** | Tempo entre aparecimento e detecção | Δt = t_det - t_apar |
| **Cobertura Espacial** | % do céu monitorado | Ω_monitorado / Ω_total |
| **Taxa de Falsos Positivos** | Falsos alarmes / hora | FP / tempo |

---

## 🔬 FASE 2 — Física Matemática Aplicada

### 2.1 Mecânica Orbital

```python
import numpy as np
from scipy.integrate import odeint

class MecanicaOrbital:
    """Simulação de mecânica orbital para predição de trajetórias."""
    
    MU_TERRA = 3.986004418e14  # m³/s²
    R_TERRA = 6.371e6          # m
    
    @staticmethod
    def equacoes_movimento(y, t, mu=MU_TERRA):
        """Equações de movimento kepleriano perturbado."""
        r = y[:3]
        v = y[3:]
        r_norm = np.linalg.norm(r)
        
        # Aceleração gravitacional (2 corpo)
        a_grav = -mu * r / r_norm**3
        
        # Perturbação J2 (achatamento terrestre)
        J2 = 1.08263e-3
        z2 = r[2]**2
        r2 = r_norm**2
        fator_J2 = 1.5 * J2 * (mu / r_norm**2) * (R_TERRA / r_norm)**2
        a_J2 = np.array([
            fator_J2 * r[0] / r_norm * (5 * z2 / r2 - 1),
            fator_J2 * r[1] / r_norm * (5 * z2 / r2 - 1),
            fator_J2 * r[2] / r_norm * (5 * z2 / r2 - 3)
        ])
        
        # Perturbação atmosférica (simplificada)
        rho = 1e-12  # densidade atmosférica a 400km (kg/m³)
        Cd = 2.2     # coeficiente de arrasto
        A = 1.0      # área efetiva (m²)
        m = 100.0    # massa (kg)
        v_rel = v    # velocidade relativa ao ar
        a_drag = -0.5 * rho * Cd * A / m * np.linalg.norm(v_rel) * v_rel
        
        return np.concatenate([v, a_grav + a_J2 + a_drag])
    
    def propagar_orbita(self, estado_inicial, tempo, dt=60):
        """Propaga órbita no tempo usando odeint."""
        t = np.arange(0, tempo, dt)
        solucao = odeint(self.equacoes_movimento, estado_inicial, t)
        return t, solucao
    
    def calcular_risco_colisao(self, obj1, obj2, margem=1000):
        """Calcula probabilidade de colisão entre dois objetos."""
        # Propagar ambas as órbitas
        t1, pos1 = self.propagar_orbita(obj1, 86400)  # 24h
        t2, pos2 = self.propagar_orbita(obj2, 86400)
        
        # Distância mínima
        distancias = np.linalg.norm(pos1[:, :3] - pos2[:, :3], axis=1)
        dist_min = np.min(distancias)
        t_min = t[np.argmin(distancias)]
        
        # Probabilidade (modelo Poisson)
        raio_conjunto = 10  # metros (soma dos raios)
        volume_perigo = np.pi * raio_conjunto**2 * np.linalg.norm(pos1[0, 3:]) * 86400
        probabilidade = 1 - np.exp(-dist_min / margem)
        
        return {
            'distancia_minima': dist_min,
            'tempo_minimo': t_min,
            'risco': 'ALTO' if dist_min < margem else 'MÉDIO' if dist_min < 2*margem else 'BAIXO',
            'probabilidade': probabilidade
        }
    
    def manobra_evasiva(self, estado_atual, obstaculo, delta_v_max=10):
        """Calcula delta-v ótimo para evitar colisão."""
        r = estado_atual[:3] - obstaculo[:3]
        v = estado_atual[3:] - obstaculo[3:]
        
        # Direção perpendicular ao vetor relativo
        direcao = np.cross(r, np.cross(v, r))
        direcao = direcao / np.linalg.norm(direcao)
        
        # Magnitude mínima para escapar
        dist = np.linalg.norm(r)
        tempo_colisao = dist / np.linalg.norm(v)
        delta_v = direcao * min(delta_v_max, dist / tempo_colisao)
        
        return {
            'delta_v': delta_v,
            'magnitude': np.linalg.norm(delta_v),
            'direcao': direcao,
            'nova_velocidade': estado_atual[3:] + delta_v
        }
```

### 2.2 Propagação de Erros e Incerteza

```python
class PropagacaoIncerteza:
    """Propagação de incerteza para medições astronômicas."""
    
    @staticmethod
    def monte_carlo(estado_inicial, incerteza, n_simulacoes=1000, tempo=3600):
        """Simulação Monte Carlo para propagação de incerteza."""
        resultados = []
        orbital = MecanicaOrbital()
        
        for _ in range(n_simulacoes):
            # Perturbar estado inicial
            estado_perturbado = estado_inicial + np.random.normal(0, incerteza)
            
            # Propagar
            t, sol = orbital.propagar_orbita(estado_perturbado, tempo)
            resultados.append(sol[-1])
        
        resultados = np.array(resultados)
        
        return {
            'media': np.mean(resultados, axis=0),
            'desvio': np.std(resultados, axis=0),
            'percentil_95': np.percentile(resultados, 95, axis=0),
            'percentil_5': np.percentile(resultados, 5, axis=0),
            'covariancia': np.cov(resultados.T)
        }
    
    @staticmethod
    def filtro_kalman(observacoes, tempos, estado_inicial, P_inicial):
        """Filtro de Kalman para rastreamento de objetos."""
        n = len(estado_inicial)
        x = estado_inicial.copy()
        P = P_inicial.copy()
        
        # Matrizes do modelo (simplificado)
        F = np.eye(n)  # Transição de estado
        H = np.eye(n)  # Observação
        Q = np.eye(n) * 0.01  # Ruído do processo
        R = np.eye(n) * 1.0   # Ruído da medição
        
        estados_estimados = []
        covariancias = []
        
        for z in observacoes:
            # Predição
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q
            
            # Atualização
            K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
            x = x_pred + K @ (z - H @ x_pred)
            P = (np.eye(n) - K @ H) @ P_pred
            
            estados_estimados.append(x.copy())
            covariancias.append(P.copy())
        
        return np.array(estados_estimados), np.array(covariancias)
```

---

## 🤖 FASE 3 — Inteligência Artificial

### 3.1 Classificação de Objetos

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler

class ClassificadorEspacial:
    """IA para classificação de objetos espaciais."""
    
    CATEGORIAS = {
        0: 'satélite_ativo',
        1: 'debris_pequeno',      # < 10cm
        2: 'debris_medio',         # 10cm - 1m
        3: 'debris_grande',        # > 1m
        4: 'meteoroide',
        5: 'objeto_anomalo',       # Comportamento não padrão
        6: 'falso_positivo'
    }
    
    def __init__(self):
        self.modelo_cnn = None
        self.modelo_rf = None
        self.scaler = StandardScaler()
    
    def criar_cnn_imagens(self, input_shape=(128, 128, 1)):
        """CNN para classificação de imagens do telescópio."""
        self.modelo_cnn = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dense(len(self.CATEGORIAS), activation='softmax')
        ])
        
        self.modelo_cnn.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return self.modelo_cnn
    
    def criar_modelo_tabular(self):
        """Random Forest para dados tabulares (trajetória, magnitude, etc.)."""
        self.modelo_rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42
        )
        return self.modelo_rf
    
    def detectar_anomalias(self, dados, contaminacao=0.05):
        """Isolation Forest para detecção de objetos anômalos."""
        modelo = IsolationForest(
            n_estimators=200,
            contamination=contaminacao,
            random_state=42
        )
        
        dados_scaled = self.scaler.fit_transform(dados)
        predicoes = modelo.fit_predict(dados_scaled)
        scores = modelo.decision_function(dados_scaled)
        
        anomalias = dados[predicoes == -1]
        
        return {
            'anomalias': anomalias,
            'scores': scores,
            'n_anomalias': len(anomalias),
            'percentual': len(anomalias) / len(dados) * 100
        }
    
    def extrair_features(self, dados_brutos):
        """Extrai features relevantes para classificação."""
        features = {}
        
        # Features de trajetória
        if 'posicao' in dados_brutos and 'tempo' in dados_brutos:
            pos = dados_brutos['posicao']
            t = dados_brutos['tempo']
            
            velocidades = np.diff(pos, axis=0) / np.diff(t)[:, None]
            aceleracoes = np.diff(velocidades, axis=0) / np.diff(t[1:])[:, None]
            
            features['velocidade_media'] = np.mean(np.linalg.norm(velocidades, axis=1))
            features['velocidade_max'] = np.max(np.linalg.norm(velocidades, axis=1))
            features['aceleracao_media'] = np.mean(np.linalg.norm(aceleracoes, axis=1))
            features['trajetoria_retilinea'] = self._retilineidade(pos)
            features['variacao_aceleracao'] = np.std(np.linalg.norm(aceleracoes, axis=1))
        
        # Features de magnitude (brilho)
        if 'magnitude' in dados_brutos:
            mag = dados_brutos['magnitude']
            features['magnitude_media'] = np.mean(mag)
            features['variacao_magnitude'] = np.std(mag)
            features['periodicidade'] = self._detectar_periodicidade(mag)
        
        # Features de formato (se imagem)
        if 'imagem' in dados_brutos:
            img = dados_brutos['imagem']
            features['elongacao'] = self._calcular_elongacao(img)
            features['brilho_central'] = np.max(img)
            features['fwhm'] = self._calcular_fwhm(img)
        
        return features
    
    def _retilineidade(self, posicoes):
        """Calcula quão retilínea é a trajetória (0=circular, 1=retilínea)."""
        if len(posicoes) < 3:
            return 1.0
        
        # Ajuste linear
        t = np.arange(len(posicoes))
        coef = np.polyfit(t, posicoes[:, 0], 1)
        pred = np.polyval(coef, t)
        residuos = np.sum((posicoes[:, 0] - pred)**2)
        total = np.sum((posicoes[:, 0] - np.mean(posicoes[:, 0]))**2)
        
        return 1 - (residuos / total) if total > 0 else 1.0
    
    def _detectar_periodicidade(self, sinal):
        """Detecta periodicidade usando FFT."""
        fft = np.fft.fft(sinal - np.mean(sinal))
        frequencias = np.fft.fftfreq(len(sinal))
        magnitudes = np.abs(fft)
        
        # Pico dominante (excluindo DC)
        idx_pico = np.argmax(magnitudes[1:]) + 1
        periodo = 1 / frequencias[idx_pico] if frequencias[idx_pico] != 0 else 0
        
        return abs(periodo)
    
    def _calcular_elongacao(self, imagem):
        """Calcula elongação da mancha (indicador de movimento)."""
        from skimage.measure import regionprops
        from skimage.filters import threshold_otsu
        
        thresh = threshold_otsu(imagem)
        binary = imagem > thresh
        props = regionprops(binary.astype(int))
        
        if props:
            return props[0].major_axis_length / props[0].minor_axis_length
        return 1.0
    
    def _calcular_fwhm(self, imagem):
        """Calcula Full Width at Half Maximum (resolução angular)."""
        centro = np.array(imagem.shape) // 2
        perfil = imagem[centro[0], :]
        half_max = np.max(perfil) / 2
        
        acima = perfil > half_max
        if np.any(acima):
            return np.sum(acima)
        return 0
```

### 3.2 Pipeline de Consciência Situacional

```python
class PipelineConsciencia:
    """Pipeline completo de consciência situacional espacial."""
    
    def __init__(self):
        self.analise = AnaliseEspacial()
        self.orbital = MecanicaOrbital()
        self.ia = ClassificadorEspacial()
        self.incerteza = PropagacaoIncerteza()
        self.historico = []
        self.alertas = []
    
    def processar_noite(self, dados_brutos):
        """Processa uma noite completa de observação."""
        resultado = {
            'timestamp': pd.Timestamp.now(),
            'objetos_detectados': 0,
            'objetos_classificados': {},
            'anomalias': [],
            'alertas': [],
            'metricas': {}
        }
        
        # 1. Detecção
        objetos = self.analise.carregar_dados(dados_brutos).detectar_outliers()
        resultado['objetos_detectados'] = len(objetos.outliers)
        
        # 2. Classificação
        for obj in objetos.outliers:
            features = self.ia.extrair_features(obj)
            classe = self.ia.modelo_rf.predict([list(features.values())])[0]
            confianca = self.ia.modelo_rf.predict_proba([list(features.values())])[0].max()
            
            categoria = ClassificadorEspacial.CATEGORIAS[classe]
            
            if categoria not in resultado['objetos_classificados']:
                resultado['objetos_classificados'][categoria] = []
            
            resultado['objetos_classificados'][categoria].append({
                'features': features,
                'confianca': confianca
            })
            
            # 3. Análise de risco para debris
            if 'debris' in categoria:
                risco = self.orbital.calcular_risco_colisao(obj, estado_estacao_iss)
                if risco['risco'] in ['ALTO', 'MÉDIO']:
                    resultado['alertas'].append({
                        'tipo': 'RISCO_COLISAO',
                        'objeto': obj,
                        'risco': risco,
                        'prioridade': 'ALTA' if risco['risco'] == 'ALTO' else 'MÉDIA'
                    })
            
            # 4. Detecção de anomalias
            if categoria == 'objeto_anomalo':
                resultado['anomalias'].append({
                    'objeto': obj,
                    'features': features,
                    'confianca': confianca
                })
        
        # 5. Métricas
        resultado['metricas'] = {
            'taxa_deteccao': resultado['objetos_detectados'] / self._objetos_esperados(),
            'precisao': self._calcular_precisao(resultado),
            'cobertura': self._calcular_cobertura(),
            'latencia_media': self._calcular_latencia()
        }
        
        self.historico.append(resultado)
        return resultado
    
    def gerar_relatorio(self, periodo='7d'):
        """Gera relatório de consciência situacional."""
        dados = [h for h in self.historico if h['timestamp'] > pd.Timestamp.now() - pd.Timedelta(periodo)]
        
        relatorio = {
            'periodo': periodo,
            'total_observacoes': len(dados),
            'total_objetos': sum(d['objetos_detectados'] for d in dados),
            'distribuicao': {},
            'alertas_criticos': [],
            'tendencias': {}
        }
        
        # Distribuição por categoria
        for d in dados:
            for cat, objs in d['objetos_classificados'].items():
                if cat not in relatorio['distribuicao']:
                    relatorio['distribuicao'][cat] = 0
                relatorio['distribuicao'][cat] += len(objs)
        
        # Alertas críticos
        for d in dados:
            for alerta in d['alertas']:
                if alerta['prioridade'] == 'ALTA':
                    relatorio['alertas_criticos'].append(alerta)
        
        return relatorio
    
    def _objetos_esperados(self):
        """Retorna número esperado de objetos (baseline)."""
        return 100  # Placeholder - ajustar com dados reais
    
    def _calcular_precisao(self, resultado):
        """Calcula precisão das classificações."""
        return 0.85  # Placeholder - ajustar com validação
    
    def _calcular_cobertura(self):
        """Calcula cobertura espacial."""
        return 0.75  # Placeholder - ajustar com dados reais
    
    def _calcular_latencia(self):
        """Calcula latência média de detecção."""
        return 2.5  # segundos - placeholder
```

---

## 🛡️ FASE 4 — Defesa e Resposta

### 4.1 Classificação de Ameaças

```
┌─────────────────────────────────────────────────────────────────┐
│                    TAXONOMIA DE AMEAÇAS                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  NATURAIS                                                       │
│  ├── Meteoroides (< 1m) → Monitoramento                        │
│  ├── Asteroides (1m - 10km) → Alerta + Rastreamento            │
│  └── Cometas → Detecção precoce + Predição                      │
│                                                                 │
│  ARTIFICIAIS                                                    │
│  ├── Debris operacional → Monitoramento passivo                 │
│  ├── Debris de colisão → Alerta + Manobra                       │
│  ├── Satélites inativos → Catálogo + Predição                   │
│  └── Fragmentação → Detecção rápida + Alerta                    │
│                                                                 │
│  ANÔMALOS                                                       │
│  ├── Trajetória não-kepleriana → Investigação                   │
│  ├── Mudança de velocidade → Análise profunda                   │
│  ├── Emissão EM incomum → Classificação                         │
│  └── Comportamento desconhecido → Alerta + Documentação         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Protocolo de Resposta

```python
class ProtocoloResposta:
    """Protocolos de resposta para diferentes tipos de ameaça."""
    
    PROTOCOLOS = {
        'METEOROIDE_PEQUENO': {
            'prioridade': 'BAIXA',
            'acao': 'Monitoramento contínuo',
            'tempo_resposta': '24h',
            'notificar': ['Observatório', 'Pesquisadores']
        },
        'DEBRIS_PERIGOSO': {
            'prioridade': 'ALTA',
            'acao': 'Alerta + Cálculo de manobra evasiva',
            'tempo_resposta': '1h',
            'notificar': ['LNA', 'AEB', 'Operadores de satélites']
        },
        'OBJETO_ANOMALO': {
            'prioridade': 'MÉDIA',
            'acao': 'Investigação + Documentação + Alerta',
            'tempo_resposta': '4h',
            'notificar': ['Pesquisadores', 'LNA', 'Comunidade científica']
        },
        'RISCO_COLISAO_ISS': {
            'prioridade': 'CRÍTICA',
            'acao': 'Alerta imediato + Cálculo de manobra',
            'tempo_resposta': '15min',
            'notificar': ['NASA', 'Roscosmos', 'AEB', 'LNA']
        }
    }
    
    def executar_protocolo(self, tipo_ameaca, dados):
        """Executa protocolo de resposta para ameaça identificada."""
        protocolo = self.PROTOCOLOS.get(tipo_ameaca, self.PROTOCOLOS['OBJETO_ANOMALO'])
        
        resultado = {
            'tipo': tipo_ameaca,
            'prioridade': protocolo['prioridade'],
            'acao': protocolo['acao'],
            'timestamp': pd.Timestamp.now(),
            'dados': dados,
            'notificacoes': []
        }
        
        # Executar ações
        if 'manobra' in protocolo['acao'].lower():
            orbital = MecanicaOrbital()
            manobra = orbital.manobra_evasiva(dados['estado'], dados['obstaculo'])
            resultado['manobra'] = manobra
        
        # Enviar notificações
        for destinatario in protocolo['notificar']:
            resultado['notificacoes'].append({
                'destinatario': destinatario,
                'status': 'ENVIADO',
                'timestamp': pd.Timestamp.now()
            })
        
        return resultado
```

---

## 📈 FASE 5 — Otimização e Métricas

### 5.1 Função de Otimização

```
MAXIMIZAR: Consciência Situacional (CS)

CS = f(Detectabilidade, Precisão, Cobertura, Latência, Resposta)

Onde:
  Detectabilidade = TP / (TP + FN)
  Precisão = TP / (TP + FP)
  Cobertura = Ω_monitorado / Ω_total
  Latência = 1 / Δt_médio
  Resposta = Σ(protocolos_executados) / Σ(alertas_gerados)

Sujeito a:
  - Recursos computacionais limitados
  - Tempo de processamento < 5s por objeto
  - Taxa de falsos positivos < 5%
  - Cobertura > 80% do céu visível
```

### 5.2 Dashboard de Métricas

```python
class DashboardMetricas:
    """Dashboard em tempo real de métricas de consciência situacional."""
    
    def gerar_dashboard(self, historico):
        """Gera dados para dashboard interativo."""
        return {
            'resumo': {
                'total_objetos': sum(h['objetos_detectados'] for h in historico),
                'alertas_ativos': sum(len(h['alertas']) for h in historico),
                'anomalias_hoje': sum(len(h['anomalias']) for h in historico[-24:]),
                'taxa_deteccao_media': np.mean([h['metricas']['taxa_deteccao'] for h in historico])
            },
            'distribuicao': self._distribuicao_categorias(historico),
            'tendencias': self._calcular_tendencias(historico),
            'alertas_recentes': self._alertas_recentes(historico, n=10),
            'mapa_ceu': self._gerar_mapa_ceu(historico)
        }
    
    def _distribuicao_categorias(self, historico):
        """Distribuição de objetos por categoria."""
        dist = {}
        for h in historico:
            for cat, objs in h['objetos_classificados'].items():
                dist[cat] = dist.get(cat, 0) + len(objs)
        return dist
    
    def _calcular_tendencias(self, historico):
        """Calcula tendências temporais."""
        if len(historico) < 2:
            return {}
        
        deteccoes = [h['objetos_detectados'] for h in historico]
        return {
            'deteccoes_tendencia': 'crescente' if deteccoes[-1] > deteccoes[0] else 'decrescente',
            'media_movel_7d': np.mean(deteccoes[-7:]),
            'variacao_percentual': (deteccoes[-1] - deteccoes[0]) / deteccoes[0] * 100
        }
    
    def _alertas_recentes(self, historico, n=10):
        """Retorna alertas mais recentes."""
        todos_alertas = []
        for h in historico:
            todos_alertas.extend(h['alertas'])
        return sorted(todos_alertas, key=lambda x: x.get('timestamp', ''), reverse=True)[:n]
    
    def _gerar_mapa_ceu(self, historico):
        """Gera dados para mapa celeste."""
        posicoes = []
        for h in historico:
            for cat, objs in h['objetos_classificados'].items():
                for obj in objs:
                    if 'posicao' in obj.get('features', {}):
                        posicoes.append({
                            'posicao': obj['features']['posicao'],
                            'categoria': cat,
                            'confianca': obj['confianca']
                        })
        return posicoes
```

---

## 🗓️ Cronograma de Implementação

| Fase | Descrição | Duração | Dependências |
|------|-----------|---------|--------------|
| **F1** | Análise Exploratória de Dados | 2-3 meses | Dados do OPD |
| **F2** | Física Matemática Aplicada | 3-4 meses | F1 |
| **F3** | Inteligência Artificial | 4-6 meses | F1, F2 |
| **F4** | Defesa e Resposta | 2-3 meses | F3 |
| **F5** | Otimização e Métricas | Contínuo | F1-F4 |

---

## 📚 Referências

1. **NASA Orbital Debris Program Office** — https://orbitaldebris.jsc.nasa.gov/
2. **ESA Space Situational Awareness** — https://www.esa.int/Space_Safety/Space_Situational_Awareness
3. **NORAD Two-Line Element Sets** — https://www.space-track.org/
4. **Astropy** — https://www.astropy.org/
5. **scikit-learn** — https://scikit-learn.org/
6. **TensorFlow** — https://www.tensorflow.org/

---

## 🎯 Próximos Passos

- [ ] Coletar dados reais do OPD
- [ ] Implementar pipeline de AED (Fase 1)
- [ ] Validar modelos de mecânica orbital
- [ ] Treinar classificador com dados reais
- [ ] Integrar com sistema COAFA
- [ ] Apresentar resultados ao LNA

---

*Documento criado por Daniel Palma — UNIFEI — Julho 2026*
