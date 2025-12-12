import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
import pygame


# Audio file paths
WELCOME = 'welcome.wav'
SUITCASE_LEFT_SOUND = 'suitecase.wav'
ACCESS_DENIED_SOUND = 'accessDenied.wav'
ALARM_SOUND = 'alarm.wav'

def set_background_image(image_path):
    """Set background image for the app"""
    import base64
    
    # Check if image exists
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        
        # CSS to set background with blur and better contrast
        st.markdown(
            f"""
               <style>
            /* Background with blur effect */
            .stApp::before {{
                content: "";
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-image: url("data:image/png;base64,{encoded_string}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                filter: blur(3px) brightness(0.4);
                z-index: -1;
            }}
            
            .stApp {{
                background: transparent;
            }}
            
            /* Make header more visible */
            .stApp > header {{
                background-color: rgba(0, 0, 0, 0.9) !important;
                backdrop-filter: blur(10px);
            }}
            
            /* Sidebar with dark background */
            section[data-testid="stSidebar"] {{
                background-color: rgba(15, 23, 42, 0.98) !important;
                backdrop-filter: blur(10px);
            }}
            
            section[data-testid="stSidebar"] > div {{
                background-color: transparent !important;
            }}
            
            /* Remove default card backgrounds */
            div[data-testid="stVerticalBlock"] > div {{
                background-color: transparent !important;
            }}
            
            /* Style main content container */
            div[data-testid="stVerticalBlock"] {{
                background-color: transparent !important;
            }}
            
            /* Column containers - make them look like cards */
            div[data-testid="column"] > div {{
                background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.90)) !important;
                backdrop-filter: blur(12px);
                padding: 25px !important;
                border-radius: 15px !important;
                border: 1px solid rgba(59, 130, 246, 0.3) !important;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
                transition: all 0.3s ease !important;
            }}
            
            div[data-testid="column"] > div:hover {{
                border-color: rgba(59, 130, 246, 0.6) !important;
                box-shadow: 0 12px 48px rgba(59, 130, 246, 0.2) !important;
                transform: translateY(-2px);
            }}
            
            /* Title styling */
            h1 {{
                color: #ffffff !important;
                text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.9);
                font-weight: 800 !important;
                padding: 20px;
                background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.90)) !important;
                border-radius: 15px;
                border: 1px solid rgba(59, 130, 246, 0.4);
                margin-bottom: 10px !important;
            }}
            
            h2 {{
                color: #ffffff !important;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.9);
                font-weight: 700 !important;
                padding: 15px 10px;
                background: linear-gradient(90deg, rgba(59, 130, 246, 0.2), transparent);
                border-left: 4px solid rgba(59, 130, 246, 0.8);
                border-radius: 8px;
                margin-bottom: 15px !important;
            }}
            
            h3 {{
                color: #93c5fd !important;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
                font-weight: 600 !important;
            }}
            
            /* All text elements */
            .stMarkdown, .stText, label, p, span {{
                color: #ffffff !important;
                text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8);
            }}
            
            /* Subtitle/Description */
            .stMarkdown p {{
                background-color: rgba(15, 23, 42, 0.8);
                padding: 10px 15px;
                border-radius: 8px;
                border-left: 3px solid rgba(59, 130, 246, 0.6);
            }}
            
            /* Input fields */
            input, textarea, select {{
                background-color: rgba(15, 23, 42, 0.95) !important;
                color: #ffffff !important;
                border: 2px solid rgba(59, 130, 246, 0.4) !important;
                border-radius: 8px !important;
            }}
            
            input:focus, textarea:focus, select:focus {{
                border-color: rgba(59, 130, 246, 0.8) !important;
                box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
            }}
            
            /* Buttons - better contrast */
            .stButton > button {{
                background: linear-gradient(135deg, rgba(30, 41, 59, 0.98), rgba(51, 65, 85, 0.95)) !important;
                color: #ffffff !important;
                border: 2px solid rgba(59, 130, 246, 0.6) !important;
                font-weight: 700 !important;
                text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.9);
                border-radius: 10px !important;
                padding: 12px 24px !important;
                transition: all 0.3s ease !important;
            }}
            
            .stButton > button:hover {{
                background: linear-gradient(135deg, rgba(59, 130, 246, 0.9), rgba(37, 99, 235, 0.9)) !important;
                border-color: rgba(59, 130, 246, 1) !important;
                transform: scale(1.05);
                box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important;
            }}
            
            /* Primary button (Démarrer) */
            .stButton > button[kind="primary"] {{
                background: linear-gradient(135deg, rgba(220, 38, 38, 0.98), rgba(185, 28, 28, 0.95)) !important;
                border: 2px solid rgba(239, 68, 68, 0.8) !important;
            }}
            
            .stButton > button[kind="primary"]:hover {{
                background: linear-gradient(135deg, rgba(239, 68, 68, 1), rgba(220, 38, 38, 1)) !important;
                border-color: rgba(248, 113, 113, 1) !important;
                box-shadow: 0 6px 20px rgba(239, 68, 68, 0.5) !important;
            }}
            
            /* Radio buttons */
            div[data-testid="stRadio"] {{
                background-color: rgba(15, 23, 42, 0.9) !important;
                padding: 15px;
                border-radius: 10px;
                border: 1px solid rgba(59, 130, 246, 0.3);
            }}
            
            /* Metrics - card style */
            div[data-testid="stMetric"] {{
                background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.90)) !important;
                padding: 20px !important;
                border-radius: 12px !important;
                border: 2px solid rgba(59, 130, 246, 0.4) !important;
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3) !important;
            }}
            
            div[data-testid="stMetric"] label {{
                color: #93c5fd !important;
                font-weight: 700 !important;
                font-size: 1.1rem !important;
            }}
            
            div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
                color: #ffffff !important;
                font-size: 2.2rem !important;
                font-weight: 800 !important;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
            }}
            
            /* Dataframe */
            div[data-testid="stDataFrame"] {{
                background-color: rgba(15, 23, 42, 0.98) !important;
                border-radius: 12px;
                border: 1px solid rgba(59, 130, 246, 0.3);
                padding: 10px;
            }}
            
            /* Expander */
            div[data-testid="stExpander"] {{
                background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.90)) !important;
                border: 1px solid rgba(59, 130, 246, 0.4) !important;
                border-radius: 10px;
                backdrop-filter: blur(10px);
            }}
            
            /* Number input */
            div[data-baseweb="input"] {{
                background-color: rgba(15, 23, 42, 0.95) !important;
                border-radius: 8px;
            }}
            
            /* Slider */
            div[data-testid="stSlider"] {{
                background-color: rgba(15, 23, 42, 0.9) !important;
                padding: 15px;
                border-radius: 10px;
                border: 1px solid rgba(59, 130, 246, 0.3);
            }}
            
            /* Checkbox */
            div[data-testid="stCheckbox"] {{
                background-color: rgba(15, 23, 42, 0.9) !important;
                padding: 10px;
                border-radius: 8px;
                border: 1px solid rgba(59, 130, 246, 0.3);
            }}
            
            /* Select box */
            div[data-baseweb="select"] {{
                background-color: rgba(15, 23, 42, 0.98) !important;
                border-radius: 8px;
            }}
            
            /* Text input */
            div[data-baseweb="base-input"] {{
                background-color: rgba(15, 23, 42, 0.98) !important;
            }}
            
            /* File uploader */
            section[data-testid="stFileUploader"] {{
                background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.90)) !important;
                border: 2px dashed rgba(59, 130, 246, 0.6) !important;
                border-radius: 12px;
                padding: 25px;
                backdrop-filter: blur(10px);
            }}
            
            /* Info, warning, success boxes */
            div[data-testid="stAlert"] {{
                background-color: rgba(15, 23, 42, 0.98) !important;
                backdrop-filter: blur(10px);
                border-radius: 10px;
            }}
            
            /* Markdown horizontal rule */
            hr {{
                border: none !important;
                height: 2px !important;
                background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.5), transparent) !important;
                margin: 20px 0 !important;
            }}
            
            /* Improve video placeholder visibility */
            div[data-testid="stImage"] {{
                border: 2px solid rgba(59, 130, 246, 0.6);
                border-radius: 12px;
                background-color: rgba(0, 0, 0, 0.7);
                padding: 5px;
                box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
            }}
            
            /* Empty placeholder styling */
            .element-container {{
                background-color: transparent !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning(f"⚠️ Image de fond introuvable: {image_path}")


# Configuration
st.set_page_config(
    page_title="Système de Surveillance de Sécurité Aéroportuaire",
    page_icon=" Md.Md",
    layout="wide"
)
# Set background image
set_background_image('port.jpg')  # Change to your image filename
# Initialize session state
def init_session_state():
    defaults = {
        'authorized_ids': set(),
        'restricted_zones': [],
        'object_tracking': {},
        'person_tracking': {},
        'alarm_log': [],
        'running': False,
        'audio_enabled': True,
        'alarm_cooldown': 0,
        'weapon_detection_count': 0
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Constants
WEAPONS = ['knife', 'gun', 'rifle', 'pistol', 'weapon', 'handgun', 'shotgun']
SUSPICIOUS_OBJECTS = ['backpack', 'suitcase', 'handbag', 'bag', 'umbrella']
ABANDON_THRESHOLD = 60

# Initialize pygame for audio
try:
    pygame.mixer.init()
    AUDIO_AVAILABLE = True
except Exception as e:
    AUDIO_AVAILABLE = False
    st.warning("⚠️ Audio non disponible sur ce système")


@st.cache_resource
def load_yolo_model(model_path='yolov8n.pt'):
    """Load YOLOv8 model with proper error handling"""
    try:
        from torch.nn.modules.container import Sequential, ModuleList
        from torch.nn.modules.activation import SiLU
        from collections import OrderedDict
        from ultralytics.nn.modules import Conv, C2f, SPPF, Bottleneck
        from ultralytics.nn.tasks import DetectionModel

        safe_classes = [Sequential, ModuleList, SiLU, OrderedDict,
                        Conv, C2f, SPPF, Bottleneck, DetectionModel]
        torch.serialization.add_safe_globals(safe_classes)

        try:
            with torch.serialization.safe_globals(safe_classes):
                if os.path.exists(model_path) and model_path != 'yolov8n.pt':
                    model = YOLO(model_path)
                    st.success(f"Modèle personnalisé chargé: {model_path}")
                else:
                    model = YOLO('yolov8n.pt')
                    st.info("Utilisation du modèle YOLOv8 par défaut")
            return model

        except Exception as e:
            st.warning(f"⚠️ Échec du chargement sécurisé: {e}")
            original_load = torch.load
            
            def patched_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            
            torch.load = patched_load
            model = YOLO(model_path)
            torch.load = original_load
            st.success("✓ Modèle chargé avec workaround")
            return model

    except Exception as e:
        st.error(f"Impossible de charger le modèle: {e}")
        return None


def create_alarm_sound():
    """Create alarm sound file if it doesn't exist"""
    if not os.path.exists(ALARM_SOUND):
        try:
            from scipy.io.wavfile import write
            
            sample_rate = 44100
            duration = 0.5
            frequency = 1000
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.sin(2 * np.pi * frequency * t) * 0.5
            audio += np.sin(2 * np.pi * frequency * 1.5 * t) * 0.3
            audio = (audio * 32767).astype(np.int16)
            
            write(ALARM_SOUND, sample_rate, audio)
        except Exception as e:
            st.warning(f"Impossible de créer le son d'alarme: {e}")


def play_alarm(audio_file):
    """Play alarm sound"""
    if AUDIO_AVAILABLE and st.session_state.audio_enabled:
        try:
            if os.path.exists(audio_file):
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
        except Exception:
            pass


def save_config():
    """Save configuration to JSON"""
    config = {
        'authorized_ids': list(st.session_state.authorized_ids),
        'restricted_zones': st.session_state.restricted_zones,
        'audio_enabled': st.session_state.audio_enabled
    }
    try:
        with open('security_config.json', 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        st.error(f"Erreur de sauvegarde: {e}")


def load_config():
    """Load configuration from JSON"""
    if os.path.exists('security_config.json'):
        try:
            with open('security_config.json', 'r') as f:
                config = json.load(f)
                st.session_state.authorized_ids = set(config.get('authorized_ids', []))
                st.session_state.restricted_zones = config.get('restricted_zones', [])
                st.session_state.audio_enabled = config.get('audio_enabled', True)
        except Exception as e:
            st.error(f"Erreur de chargement: {e}")


def log_alarm(alarm_type, message, severity="AVERTISSEMENT"):
    """Log security alarm"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.alarm_log.insert(0, {
        'timestamp': timestamp,
        'type': alarm_type,
        'message': message,
        'severity': severity
    })
    st.session_state.alarm_log = st.session_state.alarm_log[:100]
    
    if severity == "CRITIQUE" and st.session_state.alarm_cooldown == 0:
        # play_alarm(ALARM_SOUND)
        st.session_state.alarm_cooldown = 30


def point_in_polygon(point, polygon):
    """Check if point is inside polygon using ray casting algorithm"""
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(n):
        p2x, p2y = polygon[(i + 1) % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def is_in_restricted_zone(bbox, zones):
    """Check if bbox center is in any restricted zone"""
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    center = (center_x, center_y)
    
    for zone in zones:
        if point_in_polygon(center, zone['coords']):
            return True, zone['name']
    
    return False, None


def process_frame(frame, model):
    """Process frame for object detection and tracking"""
    results = model.track(frame, persist=True, verbose=False)
    
    detections = {
        'persons': [],
        'weapons': [],
        'objects': []
    }
    
    if results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            track_id = int(box.id[0]) if box.id is not None else None
            
            class_name = model.names[cls].lower()
            
            detection = {
                'bbox': xyxy,
                'confidence': conf,
                'class': class_name,
                'id': track_id
            }
            
            if class_name == 'person':
                detections['persons'].append(detection)
            elif any(weapon in class_name for weapon in WEAPONS):
                detections['weapons'].append(detection)
            elif class_name in SUSPICIOUS_OBJECTS:
                detections['objects'].append(detection)
    
    return results, detections


def draw_annotations(frame, results, detections, zones, abandon_threshold):
    """Draw all annotations on frame"""
    annotated_frame = results[0].plot()
    current_time = time.time()
    height, width = frame.shape[:2]
    
    # Draw restricted zones
    for zone in zones:
        pts = np.array(zone['coords'], np.int32)
        pts = pts.reshape((-1, 1, 2))
        
        # Draw zone border
        cv2.polylines(annotated_frame, [pts], True, (0, 0, 255), 3)
        
        # Draw semi-transparent overlay
        overlay = annotated_frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 255))
        cv2.addWeighted(overlay, 0.2, annotated_frame, 0.8, 0, annotated_frame)
        
        # Draw zone label
        label_pos = tuple(map(int, zone['coords'][0]))
        cv2.putText(annotated_frame, f"ZONE RESTREINTE: {zone['name']}", 
                   label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Track persons and check restricted zones
    for person in detections['persons']:
        if person['id'] is not None:
            track_id = person['id']
            
            # Initialize tracking for new person
            if track_id not in st.session_state.person_tracking:
                st.session_state.person_tracking[track_id] = {
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'authorized': track_id in st.session_state.authorized_ids,
                    'in_zone_logged': False
                }
            else:
                st.session_state.person_tracking[track_id]['last_seen'] = current_time
            
            # Check if person is in restricted zone
            in_zone, zone_name = is_in_restricted_zone(person['bbox'], zones)
            
            if in_zone:
                is_authorized = st.session_state.person_tracking[track_id]['authorized']
                x1, y1, x2, y2 = map(int, person['bbox'])
                
                if not is_authorized:
                    # Draw red box for unauthorized access
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                    cv2.putText(annotated_frame, "⚠️ NON AUTORISE!", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # Log alarm once per zone entry
                    if not st.session_state.person_tracking[track_id].get('in_zone_logged', False):
                        log_alarm("ZONE_RESTREINTE", 
                                 f"Personne non autorisée (ID:{track_id}) dans {zone_name}", 
                                 "CRITIQUE")
                        play_alarm(ACCESS_DENIED_SOUND)
                        st.session_state.person_tracking[track_id]['in_zone_logged'] = True
                else:
                    # Draw green box for authorized access
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(annotated_frame, "✓ AUTORISE", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # Reset zone flag when person leaves zone
                if track_id in st.session_state.person_tracking:
                    st.session_state.person_tracking[track_id]['in_zone_logged'] = False
    
    # Weapon detection with alarm
    if detections['weapons']:
        st.session_state.weapon_detection_count += 1
        
        for weapon in detections['weapons']:
            x1, y1, x2, y2 = map(int, weapon['bbox'])
            
            # Pulsing red box
            pulse = int(abs(np.sin(current_time * 4) * 255))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, pulse), 5)
            
            # Warning text
            cv2.putText(annotated_frame, f"🚨 ARME: {weapon['class'].upper()}", 
                       (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.putText(annotated_frame, f"Confiance: {weapon['confidence']:.2f}", 
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Screen-wide alert
        cv2.rectangle(annotated_frame, (0, 0), (width, 80), (0, 0, 255), -1)
        cv2.putText(annotated_frame, "🚨 ALERTE ARME DETECTEE 🚨", 
                   (width//2 - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
        
        # Log alarm periodically
        if st.session_state.weapon_detection_count % 10 == 0:
            log_alarm("ARME", 
                     f"Arme détectée: {detections['weapons'][0]['class']} "
                     f"(Confiance: {detections['weapons'][0]['confidence']:.2f})", 
                     "CRITIQUE")
    
    # Track abandoned objects
    for obj in detections['objects']:
        if obj['id'] is not None:
            obj_id = obj['id']
            
            if obj_id not in st.session_state.object_tracking:
                st.session_state.object_tracking[obj_id] = {
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'class': obj['class'],
                    'bbox': obj['bbox'],
                    'logged': False
                }
            else:
                st.session_state.object_tracking[obj_id]['last_seen'] = current_time
                st.session_state.object_tracking[obj_id]['bbox'] = obj['bbox']
                
                duration = current_time - st.session_state.object_tracking[obj_id]['first_seen']
                
                if duration > abandon_threshold:
                    x1, y1, x2, y2 = map(int, obj['bbox'])
                    pulse = int(abs(np.sin(current_time * 2) * 255))
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, pulse), 4)
                    
                    cv2.putText(annotated_frame, 
                               f"⚠️ OBJET ABANDONNE {obj['class'].upper()} ({int(duration)}s)", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    if not st.session_state.object_tracking[obj_id].get('logged', False):
                        log_alarm("OBJET_ABANDONNE", 
                                 f"{obj['class']} abandonné pendant {int(duration)}s",
                                 "AVERTISSEMENT")
                        play_alarm(SUITCASE_LEFT_SOUND)
                        st.session_state.object_tracking[obj_id]['logged'] = True
    
    # Cleanup old tracking data
    persons_to_remove = [tid for tid, data in st.session_state.person_tracking.items() 
                        if current_time - data['last_seen'] > 5]
    for tid in persons_to_remove:
        del st.session_state.person_tracking[tid]
    
    objects_to_remove = [oid for oid, data in st.session_state.object_tracking.items() 
                        if current_time - data['last_seen'] > 5]
    for oid in objects_to_remove:
        del st.session_state.object_tracking[oid]
    
    # Status overlay
    cv2.putText(annotated_frame, f"Personnes: {len(detections['persons'])}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Objets: {len(st.session_state.object_tracking)}", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Decrease alarm cooldown
    if st.session_state.alarm_cooldown > 0:
        st.session_state.alarm_cooldown -= 1
    
    return annotated_frame


# ============================================================================
# UI
# ============================================================================

st.title("Système de Surveillance de Sécurité Aéroportuaire MaydayMayday")
st.markdown("**Surveillance Avancée Alimentée par IA avec Détection d'Armes**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration du Système")
    
    # Model selection
    st.markdown("###  Sélection du Modèle")
    model_options = ['yolov8n.pt (Défaut)']
    
    custom_models = list(Path('.').glob('*.pt')) + list(Path('runs/detect').glob('*/weights/best.pt'))
    for model_path in custom_models:
        if model_path.name != 'yolov8n.pt':
            model_options.append(str(model_path))
    
    selected_model = st.selectbox("Modèle YOLO", model_options)
    
    if selected_model != 'yolov8n.pt (Défaut)':
        st.info(f"Utilisation du modèle: {selected_model}")
    
    # Audio settings
    st.markdown("###  Paramètres Audio")
    audio_toggle = st.checkbox("Activer Alarme Sonore", value=st.session_state.audio_enabled)
    if audio_toggle != st.session_state.audio_enabled:
        st.session_state.audio_enabled = audio_toggle
        save_config()
    
    if st.button("🔊 Tester Alarme"):
        play_alarm(ACCESS_DENIED_SOUND)
    
    # Config management
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        if st.button("📂 Charger", use_container_width=True):
            load_config()
            st.success("Config chargée")
    with col_cfg2:
        if st.button("💾 Sauvegarder", use_container_width=True):
            save_config()
            st.success("Config sauvegardée")
    
    # Authorized personnel
    st.markdown("###  Personnel Autorisé")
    new_auth_id = st.number_input("ID de Suivi", min_value=1, step=1, key="auth_id_input")
    
    if st.button(" Autoriser", use_container_width=True):
        st.session_state.authorized_ids.add(new_auth_id)
        save_config()
        st.success(f" ID autorisé: {new_auth_id}")
    
    if st.session_state.authorized_ids:
        st.text("IDs autorisés: " + ", ".join(map(str, sorted(st.session_state.authorized_ids))))
        if st.button(" Effacer Autorisations"):
            st.session_state.authorized_ids.clear()
            save_config()
            st.success("Autorisations effacées")
    
    # Restricted zones
    st.markdown("---")
    st.markdown("###  Zones Restreintes")
    zone_name = st.text_input("Nom Zone", value="Zone Restreinte 1", key="zone_name")
    
    st.text("Définir rectangle (X1,Y1) → (X2,Y2)")
    col1, col2 = st.columns(2)
    with col1:
        x1 = st.number_input("X1", value=100, step=10, key="zone_x1")
        y1 = st.number_input("Y1", value=100, step=10, key="zone_y1")
    with col2:
        x2 = st.number_input("X2", value=400, step=10, key="zone_x2")
        y2 = st.number_input("Y2", value=400, step=10, key="zone_y2")
    
    if st.button("➕ Ajouter Zone", use_container_width=True):
        # Ensure proper rectangle coordinates
        min_x, max_x = min(x1, x2), max(x1, x2)
        min_y, max_y = min(y1, y2), max(y1, y2)
        
        zone_coords = [
            (min_x, min_y),  # Top-left
            (max_x, min_y),  # Top-right
            (max_x, max_y),  # Bottom-right
            (min_x, max_y)   # Bottom-left
        ]
        
        st.session_state.restricted_zones.append({
            'name': zone_name,
            'coords': zone_coords
        })
        save_config()
        st.success(f" Zone ajoutée: {zone_name}")
    
    if st.session_state.restricted_zones:
        st.text(f"📍 Zones actives: {len(st.session_state.restricted_zones)}")
        for i, zone in enumerate(st.session_state.restricted_zones):
            st.text(f"  {i+1}. {zone['name']}")
        
        if st.button(" Effacer Toutes les Zones"):
            st.session_state.restricted_zones = []
            save_config()
            st.success(" Zones effacées")
    
    # Settings
    st.markdown("---")
    abandon_time = st.slider("⏱️ Seuil Objet Abandonné (s)", 10, 300, 60)
    
    # Metrics
    st.markdown("---")
    st.metric("🔫 Détections d'Armes", st.session_state.weapon_detection_count)
    critical_count = len([a for a in st.session_state.alarm_log if a['severity'] == 'CRITIQUE'])
    st.metric("🚨 Alertes Critiques", critical_count)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Flux Caméra")
    camera_option = st.radio("Source", ["Webcam", "Télécharger Vidéo"], horizontal=True)
    video_placeholder = st.empty()
    
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        start_button = st.button("Démarrer", type="primary", use_container_width=True)
    with col_btn2:
        stop_button = st.button(" Arrêter", use_container_width=True)
    with col_btn3:
        clear_logs = st.button(" Effacer Logs", use_container_width=True)

with col2:
    st.subheader("🚨 Alertes en Direct")
    alerts_placeholder = st.empty()
    
    st.subheader("📊 Statistiques")
    stats_placeholder = st.empty()

if clear_logs:
    st.session_state.alarm_log = []
    st.session_state.weapon_detection_count = 0
    st.success(" Logs effacés")

if start_button:
    st.session_state.running = True

if stop_button:
    st.session_state.running = False

# Video processing
# Video processing
if st.session_state.running:
    model_path = selected_model.replace(' (Défaut)', '')
    model = load_yolo_model(model_path)
    
    if model is not None:
        cap = None
        
        if camera_option == "Webcam":
            cap = cv2.VideoCapture(0)
        else:
            uploaded_video = st.file_uploader("📁 Fichier Vidéo", type=['mp4', 'avi', 'mov'])
            if uploaded_video:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_video.read())
                cap = cv2.VideoCapture(tfile.name)
            else:
                st.warning("⚠️ Veuillez télécharger une vidéo")
                st.session_state.running = False
        
        if cap and cap.isOpened():
            # Play welcome sound when camera starts
            if 'welcome_played' not in st.session_state:
                st.session_state.welcome_played = False
            
            if not st.session_state.welcome_played:
                play_alarm(WELCOME)
                st.session_state.welcome_played = True
            
            frame_count = 0
            
            while st.session_state.running:
                ret, frame = cap.read()
                
                if not ret:
                    st.info("Fin de vidéo")
                    st.session_state.running = False
                    break
                
                frame_count += 1
                
                # Process frame
                results, detections = process_frame(frame, model)
                annotated_frame = draw_annotations(
                    frame, results, detections,
                    st.session_state.restricted_zones,
                    abandon_time
                )
                
                # Display video
                video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
                
                # Update alerts
                if st.session_state.alarm_log:
                    alert_html = ""
                    for log in st.session_state.alarm_log[:8]:
                        emoji = "🔴" if log['severity'] == "CRITIQUE" else "🟡"
                        color = "#ff4444" if log['severity'] == "CRITIQUE" else "#ffaa00"
                        
                        alert_html += f"""
                        <div style='background-color: {color}22; padding: 10px; margin: 5px 0; 
                                    border-left: 4px solid {color}; border-radius: 5px;'>
                            <small><b>{emoji} {log['timestamp']}</b></small><br>
                            <b>{log['type']}</b>: {log['message']}
                        </div>
                        """
                    alerts_placeholder.markdown(alert_html, unsafe_allow_html=True)
                
                # Update stats
                stats_html = f"""
                <div style='background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333;'>
                    <p><b style="color: #00ff00; background-color: #003300; padding: 2px 6px; border-radius: 4px;">👥 Personnes:</b> {len(detections['persons'])}</p>
                    <p><b style="color: #ff0000; background-color: #330000; padding: 2px 6px; border-radius: 4px;">🔫 Armes Détectées:</b> {st.session_state.weapon_detection_count}</p>
                    <p><b style="color: #00ff00; background-color: #003300; padding: 2px 6px; border-radius: 4px;">📦 Objets Suivis:</b> {len(st.session_state.object_tracking)}</p>
                    <p><b style="color: #ff8800; background-color: #332200; padding: 2px 6px; border-radius: 4px;">🚨 Alertes Critiques:</b> {critical_count}</p>
                    <p><b style="color: #00ff00; background-color: #003300; padding: 2px 6px; border-radius: 4px;">📋 Total Alertes:</b> {len(st.session_state.alarm_log)}</p>
                    <p><b style="color: #00ff00; background-color: #003300; padding: 2px 6px; border-radius: 4px;">🎞️ Frame:</b> {frame_count}</p>
                </div>
                """
                stats_placeholder.markdown(stats_html, unsafe_allow_html=True)
                
                time.sleep(0.03)
            
            cap.release()
            # Reset welcome flag when camera stops
            st.session_state.welcome_played = False
# Security log
st.markdown("---")
st.subheader("📋 Journal de Sécurité")

if st.session_state.alarm_log:
    import pandas as pd
    
    log_df_data = [{
        'Horodatage': log['timestamp'],
        'Gravité': log['severity'],
        'Type': log['type'],
        'Message': log['message']
    } for log in st.session_state.alarm_log]
    
    df = pd.DataFrame(log_df_data)
    st.dataframe(df, use_container_width=True, height=300)
    
    if st.button("Exporter Journal"):
        csv = df.to_csv(index=False)
        st.download_button(
            "Télécharger CSV",
            csv,
            f"journal_securite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )
else:
    st.info("Aucune alerte pour le moment")

# Instructions
with st.expander("Guide d'Utilisation"):
    st.markdown("""
    ### Configuration:
    
    1. **Modèle de détection:**
       - Par défaut: YOLOv8n (détection générale)
       - Recommandé: Entraînez un modèle personnalisé pour une meilleure précision
       - Sélectionnez votre modèle personnalisé dans la barre latérale
    
    2. **Alarme sonore:**
       - Activez/désactivez l'alarme dans les paramètres
       - L'alarme se déclenche automatiquement lors de détections critiques
       - Testez l'alarme avant utilisation
    
    3. **Zones restreintes:**
       - Définissez les coordonnées X1,Y1 (coin supérieur gauche) et X2,Y2 (coin inférieur droit)
       - Ajoutez plusieurs zones si nécessaire
       - Les personnes autorisées (IDs ajoutés) peuvent entrer sans alerte
    
    4. **Personnel autorisé:**
       - Ajoutez les IDs de suivi des personnes autorisées
       - Ces IDs apparaissent lors du suivi en temps réel
       - Les personnes non autorisées déclenchent des alarmes dans les zones restreintes
    
    ### Fonctionnalités:
    - Détection d'armes en temps réel
    - Alarme sonore automatique
    - Contrôle d'accès aux zones restreintes
    - Suivi des objets abandonnés
    - Journal de sécurité complet
    - Support modèles personnalisés
    - Export des logs en CSV
    """)