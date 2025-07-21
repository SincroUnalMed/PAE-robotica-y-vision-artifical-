Este repositorio contiene una colección de scripts en Python que utilizan el algoritmo ORB (Oriented FAST and Rotated BRIEF) de OpenCV para la detección y correspondencia de características. Los ejemplos van desde la comparación básica entre dos imágenes estáticas hasta la detección de objetos en tiempo real y la integración con el ecosistema de robótica ROS2.

El objetivo principal es demostrar cómo se pueden usar los descriptores de características para tareas como:

-Establecer correspondencias entre escenas.

-Detectar un objeto conocido en un flujo de video.

-Capturar una "escena" de referencia y volver a reconocerla.

-Integrar estas técnicas en un nodo de ROS2, sentando las bases para aplicaciones en robótica y SLAM.

A continuación se detalla el propósito y funcionamiento de cada script.

1. Matches_2Imagenes.py
Este script es una introducción básica a la correspondencia de características con ORB.

Propósito: Comparar dos imágenes estáticas, encontrar puntos de características coincidentes y mostrar un análisis geométrico del desplazamiento entre ellas.

Funcionamiento:

-Carga dos imágenes (reloj1.jpg y reloj2.jpg) desde la carpeta imagenes_prueba/.

-Calcula los puntos clave (keypoints) y descriptores ORB para ambas.

-Utiliza un BFMatcher (Brute-Force Matcher) para encontrar las mejores coincidencias.

-Visualiza las 30 mejores coincidencias en una sola imagen.

-Calcula y muestra una tabla con métricas como la distancia promedio, el desplazamiento (dx, dy) y el ángulo promedio entre los puntos coincidentes.

2. buscando_objetos.py
Este script demuestra cómo encontrar un objeto específico (definido por una imagen) dentro de un flujo de video en tiempo real.

Propósito: Detección de un objeto de referencia en tiempo real usando una cámara.

Funcionamiento:

-Carga una imagen de referencia (ej. imagenes_prueba/ferrari.jpg). Debes modificar la ruta en el script si usas otra imagen.

-Calcula los descriptores ORB de la imagen de referencia una sola vez.

-Inicia la captura de video (desde una cámara IP o una webcam local).

-En cada fotograma, busca correspondencias con la imagen de referencia usando el Ratio Test de Lowe para filtrar coincidencias de baja calidad.

-Si se encuentran suficientes coincidencias, calcula la homografía usando RANSAC para encontrar la posición y perspectiva del objeto y dibuja un cuadro verde a su alrededor.

3. reconocimiento_objetos.py
Similar al anterior, pero en lugar de cargar un objeto desde un archivo, permite al usuario capturar una "escena de referencia" directamente desde la cámara.

Propósito: Reconocer una escena u objeto capturado dinámicamente desde el video en vivo.

Funcionamiento:

-Inicia la cámara y muestra el video en vivo.

-El usuario presiona la tecla 's' para capturar el fotograma actual como la nueva imagen de referencia.

-La imagen de referencia se guarda en la carpeta escenas_guardadas/.

-A partir de ese momento, el script busca continuamente la escena capturada en el video, dibujando un cuadro verde cuando la encuentra, aplicando los mismos filtros de Ratio Test y RANSAC.

4. simulacion_orb_ros2.py
Este script adapta la lógica de reconocimiento de objetos para funcionar dentro del framework ROS2 (Robot Operating System 2).

Propósito: Crear un nodo de ROS2 que se suscribe a un tópico de imagen, realiza la detección de características y publica los resultados en otro tópico. Es una simulación de un componente de percepción para un robot.

Funcionamiento:

-Inicia un nodo de ROS2 llamado orb_feature_matcher.

-Se suscribe al tópico /camera/image_raw para recibir imágenes.

-Al igual que reconocimiento_objetos.py, el usuario presiona 's' para capturar una imagen de referencia.

-Compara la referencia con las imágenes entrantes usando ORB, Ratio Test y RANSAC.

-Publica una imagen con las coincidencias visualizadas en el tópico /orb_matches.
