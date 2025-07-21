import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class OrbFeatureMatcher(Node):
    """
    Este nodo se suscribe a un topic de imagen RAW, permite al usuario capturar
    una imagen de referencia presionando 's', y luego muestra continuamente las
    coincidencias de características (usando ORB) entre la referencia y la imagen en vivo.
    Utiliza filtros avanzados (Ratio Test y RANSAC) para mejorar la fiabilidad.
    """
    
    def __init__(self):
        super().__init__('orb_feature_matcher')
        
        # --- Parámetros y Constantes ---
        self.MIN_MATCH_COUNT = 10  # Mínimo de coincidencias para considerar que es la misma escena
        self.LOWE_RATIO = 0.75     # Umbral para el Ratio Test
        self.reference_img = None
        self.reference_kp = None
        self.reference_des = None
        self.bridge = CvBridge()

        # --- Inicialización de OpenCV ---
        self.orb = cv2.ORB_create(nfeatures=2000) # Aumentamos un poco las features
        # Quitamos crossCheck para usar knnMatch, que es necesario para el Ratio Test
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        # --- Comunicaciones ROS2 ---
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw', 
            self.image_callback,
            10)
        
        self.publisher_ = self.create_publisher(Image, '/orb_matches', 10)
        
        self.get_logger().info('✅ Nodo de comparación ORB (Avanzado) iniciado.')
        self.get_logger().info("Presiona 's' en la ventana de la cámara para capturar la referencia.")

    def image_callback(self, msg):
        try:
            current_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Fallo al convertir la imagen: {e}')
            return

        # Si no hay referencia, solo mostrar la cámara
        if self.reference_img is None:
            cv2.imshow("Live Feed - Presiona 's' para capturar", current_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                self.capture_reference(current_frame)
            return

        # --- Proceso de Detección de Coincidencias ---
        kp2, des2 = self.orb.detectAndCompute(current_frame, None)

        # Asegurarse de que tenemos descriptores para comparar
        if self.reference_des is None or des2 is None:
            cv2.imshow("ORB Matches", current_frame)
            cv2.waitKey(1)
            return

        # 1. RATIO TEST para filtrar coincidencias ambiguas
        matches = self.bf.knnMatch(self.reference_des, des2, k=2)
        good_matches = []
        # Asegurarse de que knnMatch devolvió pares de puntos
        if matches and len(matches[0]) == 2:
            for m, n in matches:
                if m.distance < self.LOWE_RATIO * n.distance:
                    good_matches.append(m)
        
        # 2. FILTRO RANSAC para asegurar consistencia geométrica
        if len(good_matches) > self.MIN_MATCH_COUNT:
            # Extraer las coordenadas de los puntos coincidentes
            src_pts = np.float32([self.reference_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Encontrar la homografía con RANSAC
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()
            
            # Contar los inliers (coincidencias que se ajustan al modelo geométrico)
            num_inliers = np.sum(matches_mask)
            
            # Dibujar el contorno de la referencia en la escena actual si se encuentra
            h, w, _ = self.reference_img.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            if M is not None:
                dst = cv2.perspectiveTransform(pts, M)
                current_frame_with_box = cv2.polylines(current_frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
            else:
                current_frame_with_box = current_frame
            
            # Preparar texto para mostrar el número de coincidencias buenas
            display_text = f"Coincidencias: {int(num_inliers)}"
            text_color = (0, 255, 0) # Verde si hay suficientes
            
        else:
            # No hay suficientes coincidencias buenas
            matches_mask = None
            current_frame_with_box = current_frame
            display_text = f"Coincidencias: {len(good_matches)} (Muy pocas)"
            text_color = (0, 0, 255) # Rojo si no hay suficientes

        # Visualizar el resultado
        draw_params = dict(matchColor=(0, 255, 0), # Dibujar inliers en verde
                           singlePointColor=None,
                           matchesMask=matches_mask, # Solo dibujar inliers
                           flags=2)

        match_img = cv2.drawMatches(self.reference_img, self.reference_kp, current_frame_with_box, kp2, good_matches, None, **draw_params)

        # Añadir el texto con el conteo
        cv2.putText(match_img, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

        cv2.imshow("ORB Matches", match_img)
        
        # Publicar la imagen con las coincidencias
        try:
            match_msg = self.bridge.cv2_to_imgmsg(match_img, 'bgr8')
            match_msg.header = msg.header
            self.publisher_.publish(match_msg)
        except Exception as e:
            self.get_logger().error(f'Fallo al publicar la imagen: {e}')
        
        cv2.waitKey(1)


    def capture_reference(self, frame):
        """Captura y procesa la imagen de referencia."""
        self.reference_img = frame.copy()
        self.reference_kp, self.reference_des = self.orb.detectAndCompute(self.reference_img, None)
        
        if self.reference_des is not None:
            self.get_logger().info(f'📸 ¡Referencia capturada! Se encontraron {len(self.reference_kp)} características.')
            cv2.destroyWindow("Live Feed - Presiona 's' para capturar")
        else:
            self.get_logger().warn('No se encontraron características en la referencia. Inténtalo de nuevo.')
            self.reference_img = None

def main(args=None):
    rclpy.init(args=args)
    orb_matcher_node = OrbFeatureMatcher()
    
    try:
        rclpy.spin(orb_matcher_node)
    except KeyboardInterrupt:
        pass
    finally:
        orb_matcher_node.destroy_node()
        cv2.destroyAllWindows()
        if not rclpy.is_shutdown():
            rclpy.shutdown()

if __name__ == '__main__':
    main()