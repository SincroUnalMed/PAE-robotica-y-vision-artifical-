import cv2
import os
import numpy as np
import sys

# --- Parámetros de Calibración ---
N_FEATURES = 2000
RATIO_LANCZOS = 0.75
MIN_MATCH_COUNT = 20

# --- Ruta de la Imagen de Referencia ---
# ❗️ MODIFICA ESTA LÍNEA con la ruta a tu imagen de referencia (logo, objeto, etc.)
RUTA_IMAGEN_REFERENCIA = "imagenes_prueba/ferrari.jpg"


# --- Inicialización ---

# 1. Cargar la imagen de referencia desde el archivo
if not os.path.exists(RUTA_IMAGEN_REFERENCIA):
    print(f"❌ ERROR: No se encontró el archivo en la ruta: {RUTA_IMAGEN_REFERENCIA}")
    sys.exit() # Termina el script si el archivo no existe

ref_img_color = cv2.imread(RUTA_IMAGEN_REFERENCIA)
if ref_img_color is None:
    print(f"❌ ERROR: OpenCV no pudo leer la imagen. Revisa el archivo: {RUTA_IMAGEN_REFERENCIA}")
    sys.exit()

ref_img = cv2.cvtColor(ref_img_color, cv2.COLOR_BGR2GRAY)
ref_name = os.path.splitext(os.path.basename(RUTA_IMAGEN_REFERENCIA))[0]
print(f"✅ Imagen de referencia '{ref_name}' cargada correctamente.")

# 2. Inicializar ORB y calcular descriptores de la referencia
orb = cv2.ORB_create(N_FEATURES)
ref_kp, ref_des = orb.detectAndCompute(ref_img, None)

if ref_des is None:
    print("❌ No se pudieron detectar características en la imagen de referencia. Prueba con otra imagen.")
    sys.exit()

print(f"[i] {len(ref_kp)} características detectadas en la referencia.")

# 3. Inicializar Matcher y Cámara
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

ip_cam_url = "http://172.32.15.105:8080/video"
cap = cv2.VideoCapture(ip_cam_url)
if not cap.isOpened():
    print("⚠️ Cámara IP no disponible. Probando webcam local...")
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ No se pudo acceder a ninguna cámara.")
    sys.exit()

print("\n[INFO] Cámara iniciada. Buscando la imagen de referencia en el video...")
print("[INFO] Presiona 'q' para salir.")


# --- Bucle Principal de Detección ---
h_ref, w_ref = ref_img.shape

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Falló la captura.")
        break

    frame = cv2.resize(frame, (800, 600))
    h_live, w_live = frame.shape[:2]
    
    gray_live = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_live, des_live = orb.detectAndCompute(gray_live, None)

    # Combinar el frame de video y la referencia para visualización
    display_frame = np.zeros((max(h_ref, h_live), w_ref + w_live, 3), dtype=np.uint8)
    display_frame[0:h_ref, 0:w_ref] = ref_img_color
    display_frame[0:h_live, w_ref:] = frame

    if des_live is not None and len(des_live) > 0:
        # Matching con k-Nearest Neighbors
        matches = bf.knnMatch(ref_des, des_live, k=2)

        # Aplicar Ratio Test de Lowe
        good_matches = []
        if matches and len(matches[0]) == 2:
            for m, n in matches:
                if m.distance < RATIO_LANCZOS * n.distance:
                    good_matches.append(m)

        # Si hay suficientes coincidencias, buscar homografía
        if len(good_matches) >= MIN_MATCH_COUNT:
            src_pts = np.float32([ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_live[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if M is not None:
                # Dibujar el contorno del objeto detectado
                pts = np.float32([[0, 0], [0, h_ref - 1], [w_ref - 1, h_ref - 1], [w_ref - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                
                # Ajustar las coordenadas del polígono al frame de la derecha
                dst_on_display = dst + np.float32([w_ref, 0])

                # Dibujar el polígono en la vista combinada
                display_frame = cv2.polylines(display_frame, [np.int32(dst_on_display)], True, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.putText(display_frame, f"MATCH: {ref_name}", (w_ref + 20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "BUSCANDO OBJETO...", (w_ref + 20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            cv2.putText(display_frame, f"SIN COINCIDENCIA (Matches: {len(good_matches)}/{MIN_MATCH_COUNT})",
                        (w_ref + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(display_frame, "BUSCANDO...", (w_ref + 20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Deteccion de Objetos con ORB", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()