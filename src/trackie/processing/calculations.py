# src/trackie/processing/calculations.py

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# TODO: Estes mapeamentos virão do arquivo de configuração
# Por enquanto, vamos mantê-los aqui para referência.
YOLO_CLASS_MAP = { "celular": ["cell phone"], "mesa": ["table", "desk"], ... }
SURFACE_CLASSES = ["table", "desk", "dining table", "bench", "bed"]
METERS_PER_STEP = 0.7

def find_best_yolo_match(
    object_query: str,
    yolo_results: List[Any],
    yolo_model_names: List[str]
) -> Optional[Tuple[Dict[str, int], float, str]]:
    """
    Encontra a melhor correspondência YOLO para um objeto nos resultados.

    Args:
        object_query: O nome do objeto a ser procurado (ex: "celular").
        yolo_results: Os resultados brutos da predição YOLO.
        yolo_model_names: A lista de nomes de classes do modelo YOLO.

    Returns:
        Uma tupla contendo (bounding_box, confiança, nome_da_classe) ou None.
    """
    best_match = None
    highest_confidence = -1.0
    
    target_classes = YOLO_CLASS_MAP.get(object_query.lower(), [object_query.lower()])

    for result in yolo_results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            if class_id < len(yolo_model_names):
                detected_class = yolo_model_names[class_id]
                if detected_class in target_classes and confidence > highest_confidence:
                    highest_confidence = confidence
                    coords = list(map(int, box.xyxy[0]))
                    bbox = {'x1': coords[0], 'y1': coords[1], 'x2': coords[2], 'y2': coords[3]}
                    best_match = (bbox, confidence, detected_class)
    
    return best_match

def estimate_direction_from_bbox(bbox: Dict[str, int], frame_width: int) -> str:
    """
    Estima a direção de um objeto com base em sua bounding box.

    Args:
        bbox: A bounding box do objeto.
        frame_width: A largura do frame da imagem.

    Returns:
        Uma string descrevendo a direção (ex: "à sua esquerda").
    """
    if frame_width == 0: return "em uma direção indeterminada"
    
    box_center_x = (bbox['x1'] + bbox['x2']) / 2.0
    one_third = frame_width / 3.0
    
    if box_center_x < one_third:
        return "à sua esquerda"
    elif box_center_x > (frame_width - one_third):
        return "à sua direita"
    else:
        return "à sua frente"

def check_if_on_surface(
    target_bbox: Dict[str, int],
    yolo_results: List[Any],
    yolo_model_names: List[str]
) -> bool:
    """
    Verifica se um objeto parece estar sobre uma superfície detectada.

    Args:
        target_bbox: A bounding box do objeto de interesse.
        yolo_results: Os resultados brutos da predição YOLO.
        yolo_model_names: A lista de nomes de classes do modelo YOLO.

    Returns:
        True se o objeto parece estar sobre uma superfície, False caso contrário.
    """
    target_bottom_y = target_bbox['y2']
    target_center_x = (target_bbox['x1'] + target_bbox['x2']) / 2.0

    for result in yolo_results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id < len(yolo_model_names):
                detected_class = yolo_model_names[class_id]
                if detected_class in SURFACE_CLASSES:
                    s_coords = list(map(int, box.xyxy[0]))
                    s_x1, s_y1, s_x2, _ = s_coords
                    
                    is_horizontally_aligned = s_x1 < target_center_x < s_x2
                    y_tolerance = 30  # pixels
                    is_vertically_aligned = (s_y1 - y_tolerance) < target_bottom_y < (s_y1 + y_tolerance * 1.5)
                    
                    if is_horizontally_aligned and is_vertically_aligned:
                        return True
    return False

def estimate_distance_in_steps(depth_map: np.ndarray, bbox: Dict[str, int]) -> Optional[int]:
    """
    Estima a distância até um objeto em passos, usando o mapa de profundidade.

    Args:
        depth_map: O mapa de profundidade gerado pelo MiDaS.
        bbox: A bounding box do objeto.

    Returns:
        O número estimado de passos, ou None se não for possível estimar.
    """
    try:
        obj_center_x = int((bbox['x1'] + bbox['x2']) / 2)
        obj_center_y = int((bbox['y1'] + bbox['y2']) / 2)

        # Garante que as coordenadas estão dentro dos limites
        obj_center_y = max(0, min(obj_center_y, depth_map.shape[0] - 1))
        obj_center_x = max(0, min(obj_center_x, depth_map.shape[1] - 1))
        
        depth_value = depth_map[obj_center_y, obj_center_x]

        # A heurística para converter o valor de profundidade em metros
        # MiDaS produz profundidade inversa (valores maiores = mais perto)
        # Esta lógica é empírica e precisa de calibração.
        estimated_meters = -1.0
        if depth_value > 1e-6:
            if depth_value > 250: estimated_meters = np.random.uniform(0.3, 1.0)
            elif depth_value > 150: estimated_meters = np.random.uniform(1.0, 2.5)
            elif depth_value > 75: estimated_meters = np.random.uniform(2.5, 5.0)
            elif depth_value > 25: estimated_meters = np.random.uniform(5.0, 10.0)
            else: estimated_meters = np.random.uniform(10.0, 15.0)
        
        if estimated_meters > 0:
            num_steps = max(1, round(estimated_meters / METERS_PER_STEP))
            return num_steps
            
    except (IndexError, TypeError) as e:
        # logger.error(f"Erro ao processar mapa de profundidade: {e}") # O logger será importado depois
        pass
        
    return None