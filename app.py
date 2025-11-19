# app.py

import streamlit as st
import cv2
import mediapipe as mp
import time
# Importa o módulo de análise do olhar atualizado
from utils.gaze_analyzer import GazeAnalyzer

# --- Configurações Iniciais ---
st.set_page_config(page_title="Gaze Tracking - Testes Online", layout="wide")
st.title("👁️ Sistema de Monitoramento de Olhar (Streamlit)")
st.markdown("---")

# Inicializa o MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
# Especificação de desenho para o mesh facial
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Inicializa o Gaze Analyzer (cálculo do olhar)
# O tamanho do frame (640x480) é uma suposição para inicialização
analyzer = GazeAnalyzer(screen_w=640, screen_h=480) 

# Variáveis de Estado para Detecção de Fraude Contínua
if 'cheating_start_time' not in st.session_state:
    st.session_state.cheating_start_time = None
if 'cheating_history' not in st.session_state:
    st.session_state.cheating_history = []
    
# Parâmetros de Fraude (Ajustável)
# Reduzido para 2 segundos para maior sensibilidade no rastreamento ocular
CHEAT_DURATION_THRESHOLD = 2 # Segundos contínuos de desvio (status "FRAUDE") para registrar

# --- Streamlit UI ---
col1, col2 = st.columns([3, 2]) # Ajustado o layout para dar mais espaço ao vídeo

with col2:
    st.header("Status de Foco")
    status_placeholder = st.empty()
    st.markdown("---")
    st.header("Histórico de Desvios (Fraude)")
    history_placeholder = st.empty()


# Inicia a captura de vídeo
col1.header("Webcam em Tempo Real")
frame_placeholder = col1.empty()

# Cria o objeto FaceMesh
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    # Abrir a webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Não foi possível acessar a webcam. Verifique as permissões.")
        
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Erro ao ler o frame da webcam.")
            break

        # Inverte o frame para espelhamento natural
        frame = cv2.flip(frame, 1)
        
        # Ajusta o analyzer para o tamanho real do frame
        H, W, _ = frame.shape
        if analyzer.screen_w != W or analyzer.screen_h != H:
            # Re-inicializa o analyzer se o tamanho do frame mudar
            analyzer = GazeAnalyzer(screen_w=W, screen_h=H)

        # Converter a cor BGR para RGB antes do processamento
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processamento com MediaPipe
        results = face_mesh.process(rgb_frame)

        # Variáveis de rastreamento do estado atual
        status_message = "Aguardando detecção..."
        current_status = "Aguardando" 

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                
                # 1. Desenhar a malha facial para debug/UX
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_spec)
                
                # 2. Analisar o olhar com o módulo customizado (Recebe o novo 'status')
                frame, is_cheating_now, message, status = analyzer.analyze(frame, face_landmarks)
                
                status_message = message
                current_status = status # Atualizando o status

        # --- Lógica de Estado Contínuo (Detecção de Fraude) ---
        current_time = time.time()
        
        # A contagem de fraude só deve ocorrer se o status for estritamente "FRAUDE"
        if current_status == "FRAUDE":
            if st.session_state.cheating_start_time is None:
                # Inicia a contagem
                st.session_state.cheating_start_time = current_time
            
            elapsed = current_time - st.session_state.cheating_start_time
            
            if elapsed >= CHEAT_DURATION_THRESHOLD:
                # Fraude detectada e persistente
                
                # Alerta visual mais forte (simulando a modal do main.py)
                status_placeholder.error(f"🚨 FRAUDE CONTINUA (Duração: {elapsed:.1f}s)") 
                
                if not st.session_state.cheating_history or (current_time - st.session_state.cheating_history[-1]['time']) > 5:
                    # Registra apenas se a última fraude foi há mais de 5s (para evitar spam)
                    
                    # Exibe um Toast (alerta pop-up rápido) na primeira ocorrência de fraude contínua
                    st.toast("🚨 ALERTA: Mantenha seus olhos na tela!", icon='🚨')
                    
                    st.session_state.cheating_history.append({
                        "time": current_time,
                        "type": status_message.split(':')[0], # Tipo de Fraude: "FRAUDE"
                        "duration": f"{elapsed:.1f}s"
                    })
                    
            else:
                # Desvio de olhar, mas ainda não atingiu o limite de tempo contínuo
                status_placeholder.warning(f"{status_message.split(':')[0]}: Contagem para fraude: {elapsed:.1f}s")
                
        elif current_status == "INSTRUÇÃO":
            # Exibe a instrução, mas não inicia nem mantém a contagem de fraude
            st.session_state.cheating_start_time = None 
            status_placeholder.info(status_message)

        else: # Status é "OK" ou "Aguardando"
            # Não está trapaceando
            st.session_state.cheating_start_time = None # Reseta a contagem
            status_placeholder.success(status_message) # Usa a mensagem "OK"
            
        # --- Atualiza o UI ---
        
        # Histórico (coluna 2)
        history_table_data = [
            {"Horário": time.strftime("%H:%M:%S", time.localtime(h['time'])), 
             "Tipo": h['type'], 
             "Duração": h['duration']} 
            for h in st.session_state.cheating_history
        ]
        
        # DataFrame sem o argumento 'width' (Linha corrigida)
        history_placeholder.dataframe(history_table_data, hide_index=True)


        # Converte de volta para BGR e exibe o frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Image sem o argumento 'width' (Linha corrigida)
        frame_placeholder.image(frame, channels="RGB")

    cap.release()