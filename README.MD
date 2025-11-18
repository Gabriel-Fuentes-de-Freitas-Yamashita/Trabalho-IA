# End-to-End Gaze Tracking Fraud Detection

Este projeto implementa um sistema completo de detecção de fraude baseado no **rastreamento do olhar (gaze tracking)** em tempo real, combinando:

- **MediaPipe Face Mesh** – detecção rápida e precisa dos olhos  
- **EyeGestures (versão matemática)** – cálculo da orientação 2D/3D do olhar  
- **Lógica Python** – heurística para detectar comportamento suspeito  
- **Streamlit** – aplicação final consumível via webcam

---

## 📌 Objetivo

Identificar quando um aluno **não está olhando para a tela** durante uma prova online, marcando possíveis situações de consulta externa.

---

## 🧩 Arquitetura do Sistema

### 1. Captura — *Streamlit + Webcam*
Fluxo de vídeo em tempo real é capturado diretamente no navegador.

### 2. Processamento — *MediaPipe Face Mesh*
Extrai landmarks faciais, incluindo:
- Olho esquerdo
- Olho direito
- Pose da cabeça

### 3. Cálculo — *EyeGestures*
A partir dos landmarks:
- Calcula a posição dos olhos
- Gera vetor de direção do olhar (dx, dy)

### 4. Detecção de Fraude — *Lógica Python*
Regra:
> Se o olhar permanece fora da área central por mais de 3 segundos → **SUSPEITO**

### 5. Apresentação — *Streamlit*
A aplicação mostra:
- Vídeo com landmarks
- Indicador de fraude
- Estado “fora da tela: sim/não”


--- 
### Rodar a aplicação Streamlit
streamlit run app.py

### Instalar dependências
```bash
pip install -r requirements.txt