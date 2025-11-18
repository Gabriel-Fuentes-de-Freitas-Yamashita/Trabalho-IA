import cv2
import streamlit as st
import mediapipe as mp
import numpy as np
import time

st.set_page_config(layout="wide")

# --- CONFIGURAÇÃO DO MEDIAPIPE ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -------------------------------------------------------------------
# Função para calcular yaw e pitch de forma simples e estável
# -------------------------------------------------------------------
def calcular_orientacao(landmarks):
    # Pega pontos importantes: olhos e nariz
    left_eye = np.array([landmarks[33].x, landmarks[33].y])      # canto do olho
    right_eye = np.array([landmarks[263].x, landmarks[263].y])   # canto do outro olho
    nose = np.array([landmarks[1].x, landmarks[1].y])

    # Vetores básicos
    eye_vector = right_eye - left_eye
    face_vector = nose - (left_eye + right_eye) / 2

    yaw = np.degrees(np.arctan2(eye_vector[1], eye_vector[0]))     # esquerda/direita
    pitch = np.degrees(np.arctan2(face_vector[1], face_vector[0])) # cima/baixo

    # Ajuste para evitar valores extremos bugados
    if abs(yaw) > 90: yaw = 0
    if abs(pitch) > 90: pitch = 0

    return yaw, pitch

# -------------------------------------------------------------------
# Lógica mais estável (SEM FALSO POSITIVO)
# -------------------------------------------------------------------
contador_fora = 0
LIMITE_YAW = 40
LIMITE_PITCH = 35
FRAMES_NECESSARIOS = 30  # ~1 segundo a 30 FPS

st.title("Gaze Tracking – Detecção de Fraude em Tempo Real")

st.write("""
Sistema de detecção de desvio de olhar:
- MediaPipe Face Mesh
- Estimativa de orientação (yaw/pitch)
- Lógica Python simples e robusta
- Webcam em tempo real via Streamlit
""")

ativar = st.checkbox("Iniciar Webcam")

frame_slot = st.empty()

# -------------------------------------------------------------------
# LOOP PRINCIPAL
# -------------------------------------------------------------------
if ativar:
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Erro ao acessar webcam.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = face_mesh.process(frame_rgb)

        fora_da_tela = True
        suspeito = True
        yaw, pitch = 0, 0

        if resultado.multi_face_landmarks:
            face_landmarks = resultado.multi_face_landmarks[0].landmark

            # calcular yaw/pitch
            yaw, pitch = calcular_orientacao(face_landmarks)

            # -------- lógica sem falso positivo --------
            if abs(yaw) > LIMITE_YAW or abs(pitch) > LIMITE_PITCH:
                contador_fora += 1
            else:
                contador_fora = 0

            fora_da_tela = contador_fora > FRAMES_NECESSARIOS
            suspeito = fora_da_tela

            # Desenhar texto
            texto = f"Suspeito: {suspeito} | Fora da tela: {fora_da_tela}"
            cv2.putText(frame, texto, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, f"Yaw: {yaw:.2f}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, f"Pitch: {pitch:.2f}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # Mostrar imagem
        frame_slot.image(frame, channels="BGR")

        # Modo real-time
        time.sleep(0.01)
