import cv2
import mediapipe as mp
import numpy as np
import time

class GazeAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.suspect_time = 0
        self.last_timestamp = time.time()

    # -------------------------
    # HEAD POSE ESTIMATION
    # -------------------------
    def estimate_head_pose(self, landmarks, img_w, img_h):
        # Pontos 3D (modelo da cabeça)
        model_points = np.array([
            (0.0, 0.0, 0.0),               # Nose tip
            (0.0, -330.0, -65.0),          # Chin
            (-225.0, 170.0, -135.0),       # Left eye left corner
            (225.0, 170.0, -135.0),        # Right eye right corner
            (-150.0, -150.0, -125.0),      # Mouth left corner
            (150.0, -150.0, -125.0)        # Mouth right corner
        ], dtype=np.float64)

        # Índices dos landmarks mediapipe
        lm_ids = [1, 152, 33, 263, 61, 291]
        image_points = []

        for idx in lm_ids:
            lm = landmarks[idx]
            x, y = int(lm.x * img_w), int(lm.y * img_h)
            image_points.append((x, y))

        image_points = np.array(image_points, dtype=np.float64)

        # Parâmetros da câmera (aproximação)
        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))

        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

        yaw = euler_angles[1][0]     # esquerda/direita
        pitch = euler_angles[0][0]   # cima/baixo

        return yaw, pitch

    # ---------------------------------------------------
    # PROCESSAMENTO PRINCIPAL DO FRAME
    # ---------------------------------------------------
    def analyze_frame(self, frame):
        img_h, img_w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return frame, False, False, 0, 0

        face_landmarks = results.multi_face_landmarks[0].landmark

        # Calcular yaw/pitch reais
        yaw, pitch = self.estimate_head_pose(face_landmarks, img_w, img_h)

        # -------------------------
        # LÓGICA DE FRAUDE
        # -------------------------
        THRESH_YAW = 20       # >= 20° vira suspeito
        THRESH_PITCH = 25

        outside = abs(yaw) > THRESH_YAW or abs(pitch) > THRESH_PITCH

        # Controle temporal (5 segundos)
        now = time.time()
        dt = now - self.last_timestamp
        self.last_timestamp = now

        if outside:
            self.suspect_time += dt
        else:
            self.suspect_time = 0

        SUSPECT_THRESHOLD = 5
        suspect = self.suspect_time > SUSPECT_THRESHOLD

        # -------------------------------------
        # DESENHAR INFORMAÇÕES NA TELA
        # -------------------------------------
        cv2.putText(frame, f"Yaw: {yaw:.1f}°", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Pitch: {pitch:.1f}°", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame, suspect, outside, yaw, pitch
