import cv2
import os
import numpy as np

# --- Parámetros de Calibración ---
# Aumenta el número de características a detectar para mayor robustez.
N_FEATURES = 2000
# Ratio para el test de Lowe. Un valor más bajo es más estricto.
RATIO_LANCZOS = 0.75
# Número mínimo de coincidencias buenas para considerar una detección válida.
MIN_MATCH_COUNT = 20


# --- Inicialización ---

# Crear carpeta si no existe
escena_dir = "escenas_guardadas"
os.makedirs(escena_dir, exist_ok=True)

# ORB y matcher
# Usamos más features y BFMatcher para knnMatch (k-nearest neighbors)
orb = cv2.ORB_create(N_FEATURES)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Inicializar captura de cámara
ip_cam_url = "http://172.32.15.105:8080/video"
cap = cv2.VideoCapture(ip_cam_url)
if not cap.isOpened():
    print("⚠️ Cámara IP no disponible. Probando webcam local...")
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ No se pudo acceder a ninguna cámara.")
    exit()

# Variables para la imagen de referencia
ref_img = None
ref_kp = None
ref_des = None
ref_name = None

print("\n[INFO] Cámara iniciada. Presiona 's' para capturar una imagen de referencia.")
print("[INFO] Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Falló la captura.")
        break

    # Redimensionar para un rendimiento consistente
    frame = cv2.resize(frame, (800, 600))
    gray_live = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_live, des_live = orb.detectAndCompute(gray_live, None)

    # Si no hay imagen de referencia, mostramos el video en vivo
    if ref_img is None:
        display_frame = frame.copy()
        cv2.putText(display_frame, "Presiona 's' para capturar referencia",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.imshow("Matching ORB", display_frame)

    # Si ya tenemos una referencia, procedemos a hacer el matching
    else:
        h_ref, w_ref = ref_img.shape[:2]
        
        # Combinar el frame de video y la referencia para visualización
        display_frame = np.zeros((max(h_ref, 600), w_ref + 800, 3), dtype=np.uint8)
        display_frame[0:h_ref, 0:w_ref] = ref_img
        display_frame[0:600, w_ref:] = frame

        if des_live is not None and len(des_live) > 0 and ref_des is not None:
            # 1. Matching con k-Nearest Neighbors (k=2)
            matches = bf.knnMatch(ref_des, des_live, k=2)

            # 2. Aplicar Ratio Test de Lowe para filtrar buenas coincidencias
            good_matches = []
            if matches and len(matches[0]) == 2:
                for m, n in matches:
                    if m.distance < RATIO_LANCZOS * n.distance:
                        good_matches.append(m)

            # 3. Si hay suficientes coincidencias buenas, buscar homografía
            if len(good_matches) >= MIN_MATCH_COUNT:
                # Extraer coordenadas de los puntos emparejados
                src_pts = np.float32([ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_live[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Calcular la homografía con RANSAC
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                
                # Si se encuentra una homografía válida
                if M is not None:
                    # Dibujar el contorno del objeto detectado
                    h, w = ref_img.shape[:2]
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)
                    
                    # Dibujar el polígono en la imagen de video (frame, no en 'display_frame')
                    frame_with_box = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                    
                    # Dibujar las coincidencias (inliers)
                    matchesMask = mask.ravel().tolist()
                    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None,
                                       matchesMask=matchesMask, flags=2)
                    display_frame = cv2.drawMatches(ref_img, ref_kp, frame_with_box, kp_live, good_matches, None, **draw_params)

                    cv2.putText(display_frame, f"MATCH ENCONTRADO: {ref_name}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    # Homografía no encontrada
                    cv2.putText(display_frame, "BUSCANDO OBJETO...", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                # No hay suficientes coincidencias buenas
                cv2.putText(display_frame, f"SIN COINCIDENCIA (Matches: {len(good_matches)}/{MIN_MATCH_COUNT})",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "ERROR: No se detectaron descriptores",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Matching ORB", display_frame)


    # --- Manejo de Teclas ---
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        ref_img = frame.copy()
        gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        ref_kp, ref_des = orb.detectAndCompute(gray_ref, None)
        
        if ref_des is None:
            print("❌ No se pudieron detectar características en la imagen de referencia. Intenta de nuevo.")
            ref_img = None # Resetear
        else:
            escena_id = len(os.listdir(escena_dir)) // 2 + 1
            ref_name = f"escena_{escena_id}"
            cv2.imwrite(os.path.join(escena_dir, f"{ref_name}.jpg"), ref_img)
            print(f"[✓] Imagen de referencia '{ref_name}' capturada y guardada.")
            print(f"[i] {len(ref_kp)} características detectadas en la referencia.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()