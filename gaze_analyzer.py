# utils/gaze_analyzer.py

import numpy as np
import cv2
import math

class GazeAnalyzer:
    """
    Classe responsável por processar os landmarks do MediaPipe
    e calcular a direção da cabeça (Pose Estimation) para detecção robusta de fraude.
    A lógica inclui sanitização de ângulos para evitar estouro numérico (Gimbal Lock).
    """
    
    # --- CALIBRAÇÃO DE POSE (Ajuste Final de Estabilidade) ---
    # Se o Pitch RAW estiver sempre próximo de 90.0 (indicando instabilidade do PnP),
    # definimos o OFFSET como 90.0 para garantir que o PITCH CAL seja 0.0 na pose neutra.
    PITCH_OFFSET = 90.0 

    # Pontos de referência para PnP (Cabeça - 6 pontos principais)
    HEAD_MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),             # Nariz (1)
        (-225.0, 170.0, -135.0),     # Canto Olho Esq (33)
        (225.0, 170.0, -135.0),      # Canto Olho Dir (263)
        (-150.0, -150.0, -125.0),    # Boca Esq (61)
        (150.0, -150.0, -125.0),     # Boca Dir (291)
        (0.0, 0.0, -300.0)           # Testa (199)
    ]) / 500

    def __init__(self, screen_w=640, screen_h=480):
        # Dimensões da tela/frame
        self.screen_w = screen_w
        self.screen_h = screen_h
        
        # Matriz de calibração intrínseca (estimada)
        self.focal_length = screen_w 
        self.cam_matrix = np.array([
            [self.focal_length, 0, screen_w / 2],
            [0, self.focal_length, screen_h / 2],
            [0, 0, 1]
        ], dtype="double")
        
        self.dist_coeffs = np.zeros((4, 1))

    def _get_2d_3d_head_points(self, landmarks):
        """Prepara os pontos 2D/3D para a estimativa da pose da cabeça."""
        head_image_points = np.array([
            (landmarks.landmark[1].x * self.screen_w, landmarks.landmark[1].y * self.screen_h),
            (landmarks.landmark[33].x * self.screen_w, landmarks.landmark[33].y * self.screen_h),
            (landmarks.landmark[263].x * self.screen_w, landmarks.landmark[263].y * self.screen_h),
            (landmarks.landmark[61].x * self.screen_w, landmarks.landmark[61].y * self.screen_h),
            (landmarks.landmark[291].x * self.screen_w, landmarks.landmark[291].y * self.screen_h),
            (landmarks.landmark[199].x * self.screen_w, landmarks.landmark[199].y * self.screen_h)
        ], dtype="double")
        return head_image_points, self.HEAD_MODEL_POINTS

    # Função simples para verificar se os olhos estão abertos
    def _get_eye_openness(self, landmarks):
        """Estima o quão abertos os olhos estão, baseado na posição vertical dos landmarks oculares."""
        
        # Olho esquerdo (pontos verticais)
        y_top_l = landmarks.landmark[159].y * self.screen_h
        y_bottom_l = landmarks.landmark[145].y * self.screen_h
        
        # Olho direito (pontos verticais)
        y_top_r = landmarks.landmark[386].y * self.screen_h
        y_bottom_r = landmarks.landmark[374].y * self.screen_h
        
        # Distância vertical média (uma estimativa simples)
        openness = ((y_bottom_l - y_top_l) + (y_bottom_r - y_top_r)) / 2.0
        
        return openness

    def analyze(self, frame, landmarks):
        """Calcula a direção da cabeça e detecta o desvio para fraude."""
        
        # 1. Estimar a Pose da Cabeça
        head_image_points, head_model_points = self._get_2d_3d_head_points(landmarks)
        (success, rot_vec_head, trans_vec_head) = cv2.solvePnP(
            head_model_points, head_image_points, self.cam_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        # --- Inicialização Padrão ---
        is_cheating_now = False
        status = "INSTRUÇÃO" 
        message = "⚠️ INSTRUÇÃO: Posicione o rosto na tela e mantenha os olhos visíveis."
        
        yaw_debug = 0.0
        pitch_debug = 0.0
        
        if not success:
             return frame, is_cheating_now, message, status

        # 2. Calcular Ângulos (Yaw e Pitch) da Cabeça
        try:
            (rot_mat, jac) = cv2.Rodrigues(rot_vec_head)
            proj_matrix = np.hstack((rot_mat, trans_vec_head))
            eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
            
            # Ângulos brutos (para debug)
            yaw_debug = eulerAngles[1, 0] 
            pitch_debug = eulerAngles[0, 0]
            
            # --- SANITIZAÇÃO DE ÂNGULOS ---
            # Previne estouro numérico/Gimbal lock. Limita o Pitch bruto a um intervalo razoável.
            pitch_debug = max(min(pitch_debug, 90.0), -90.0)
            # --- FIM SANITIZAÇÃO ---
            
            # Ângulos para LÓGICA DE FRAUDE (com offset aplicado)
            yaw = yaw_debug 
            pitch = pitch_debug - self.PITCH_OFFSET 
            
        except Exception:
            yaw = 0.0
            pitch = 0.0
            yaw_debug = 0.0
            pitch_debug = 0.0
        
        # 3. Lógica de Detecção de Fraude (Baseada na Pose da Cabeça)
        
        # Limites de Pose (agora relativos à calibração)
        CHEAT_LIMIT_YAW = 30     # Virar a cabeça mais que 30 graus (muito lateral)
        # Reduzido para ser mais sensível, já que a pose neutra é 0.0
        CHEAT_LIMIT_PITCH = 15    # Inclinar a cabeça mais que 15 graus (olhar para baixo/cima)
        
        openness = self._get_eye_openness(landmarks)
        EYE_OPENNESS_THRESHOLD = 5.0 

        status = "OK"
        # Usamos o Pitch CALIBRADO na mensagem de foco.
        message = f"✅ FOCO: Yaw: {yaw:.1f}°, Pitch: {pitch:.1f}°"
        
        # Checagem de Desvio de Pose
        if abs(yaw) > CHEAT_LIMIT_YAW:
            is_cheating_now = True
            status = "FRAUDE"
            message = f"🚨 FRAUDE: Cabeça virada ({abs(yaw):.1f}°). Mantenha o foco à frente."
        elif abs(pitch) > CHEAT_LIMIT_PITCH:
            is_cheating_now = True
            status = "FRAUDE"
            # Usamos o Pitch CALIBRADO na mensagem de fraude.
            message = f"🚨 FRAUDE: Cabeça inclinada ({abs(pitch):.1f}°). Olhe para frente."
        elif openness < EYE_OPENNESS_THRESHOLD:
            # Mantemos esta checagem simples como um placeholder
            pass
            
        # 4. Desenhar o Eixo da Cabeça (Visualização)
        if success:
            
            # Desenha o Eixo da Cabeça (usando ângulos brutos para o desenho ser correto)
            (nose_end_point2D, _) = cv2.projectPoints(
                np.array([(0.0, 0.0, 1000.0)]), rot_vec_head, trans_vec_head, self.cam_matrix, self.dist_coeffs
            )
            p1 = (int(head_image_points[0][0]), int(head_image_points[0][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            # Cor é baseada no status LÓGICO (após offset)
            if status == "OK":
                color = (0, 255, 0) # Verde
            else:
                color = (0, 0, 255) # Vermelho
            
            # Adiciona os ângulos na imagem para debug
            cv2.putText(frame, f"Yaw: {yaw_debug:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Pitch (RAW): {pitch_debug:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Pitch (CAL): {pitch:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Desenha o Eixo Principal
            cv2.line(frame, p1, p2, color, 3) 

            
        return frame, is_cheating_now, message, status