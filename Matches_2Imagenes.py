import cv2
import numpy as np
import math

# Cargar y redimensionar imágenes
img1 = cv2.imread("imagenes_prueba/reloj1.jpg")
img2 = cv2.imread("imagenes_prueba/reloj2.jpg")
img1 = cv2.resize(img1, (600, 400))
img2 = cv2.resize(img2, (600, 400))

# Convertir a grises
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ORB + Matching
orb = cv2.ORB_create(1500)
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)

# Análisis de los mejores 30 matches
dxs, dys, distancias, angulos = [], [], [], []
for m in matches[:30]:
    pt1 = kp1[m.queryIdx].pt
    pt2 = kp2[m.trainIdx].pt
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    dist = np.sqrt(dx**2 + dy**2)
    ang = math.degrees(math.atan2(dy, dx))
    dxs.append(dx)
    dys.append(dy)
    distancias.append(dist)
    angulos.append(ang)

# Métricas clave
num_matches = len(matches)
dx_prom = np.mean(dxs)
dy_prom = np.mean(dys)
dist_prom = np.mean(distancias)
mod_prom = np.sqrt(dx_prom**2 + dy_prom**2)
ang_prom = np.mean(angulos)
ang_var = np.var(angulos)

# -----------------------------------
# 1. Imagen con matches ORB
img_match = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None, flags=2)

# -----------------------------------
# 2. Crear imagen solo para la tabla

# Datos para tabla
rows = [
    ["Métrica",             "Valor"],
    ["Matches",             f"{num_matches}"],
    ["Dist. promedio",      f"{dist_prom:.2f} px"],
    ["dx promedio",         f"{dx_prom:.2f} px"],
    ["dy promedio",         f"{dy_prom:.2f} px"],
    ["Magnitud desplaz.",   f"{mod_prom:.2f} px"],
    ["Ángulo promedio",     f"{ang_prom:.2f}°"],
    ["Varianza de ángulo",  f"{ang_var:.2f}"]
]

# Tamaño imagen tabla
row_h = 40
col_w = [240, 200]
tabla_w = sum(col_w)
tabla_h = row_h * len(rows)
tabla_img = np.zeros((tabla_h, tabla_w, 3), dtype=np.uint8)
tabla_img[:] = (30, 30, 30)  # fondo gris oscuro

# Dibujar bordes y líneas
for i in range(len(rows) + 1):
    y = i * row_h
    cv2.line(tabla_img, (0, y), (tabla_w, y), (255, 255, 255), 1)
for j in range(1, len(col_w)):
    x = col_w[0]
    cv2.line(tabla_img, (x, 0), (x, tabla_h), (255, 255, 255), 1)
cv2.rectangle(tabla_img, (0, 0), (tabla_w - 1, tabla_h - 1), (255, 255, 255), 2)

# Insertar texto
font = cv2.FONT_HERSHEY_DUPLEX
for i, row in enumerate(rows):
    y = (i + 1) * row_h - 12
    cv2.putText(tabla_img, row[0], (10, y), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(tabla_img, row[1], (col_w[0] + 10, y), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

# -----------------------------------
# Mostrar ambas ventanas
cv2.imshow("ORB - Matches entre imágenes", img_match)
cv2.imshow("Tabla de análisis geométrico", tabla_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
