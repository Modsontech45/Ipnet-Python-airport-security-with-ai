import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
from datetime import datetime
import json
import os
from collections import defaultdict
import tempfile
import pygame  # For audio alarms
from pathlib import Path

# Configuration
st.set_page_config(
    page_title="Système de Surveillance de Sécurité Aéroportuaire",
    page_icon="MD.MD",
    layout="wide"
)

# Initialize session state
if 'authorized_ids' not in st.session_state:
    st.session_state.authorized_ids = set()
if 'restricted_zones' not in st.session_state:
    st.session_state.restricted_zones = []
if 'object_tracking' not in st.session_state:
    st.session_state.object_tracking = {}
if 'person_tracking' not in st.session_state:
    st.session_state.person_tracking = {}
if 'alarm_log' not in st.session_state:
    st.session_state.alarm_log = []
if 'running' not in st.session_state:
    st.session_state.running = False
if 'audio_enabled' not in st.session_state:
    st.session_state.audio_enabled = True
if 'alarm_cooldown' not in st.session_state:
    st.session_state.alarm_cooldown = 0
if 'weapon_detection_count' not in st.session_state:
    st.session_state.weapon_detection_count = 0

# Constants
WEAPONS = ['knife', 'gun', 'rifle', 'pistol', 'weapon', 'handgun', 'shotgun']
SUSPICIOUS_OBJECTS = ['backpack', 'suitcase', 'handbag', 'bag', 'umbrella']
ABANDON_THRESHOLD = 60

# Initialize pygame for audio
try:
    pygame.mixer.init()
    AUDIO_AVAILABLE = True
except:
    AUDIO_AVAILABLE = False
    st.warning("⚠️ Audio non disponible sur ce système")

@st.cache_resource
def load_yolo_model(model_path='yolov8n.pt'):
    """
    Load YOLOv8 model safely with all required classes registered as safe globals.
    Works with PyTorch 2.6+ and avoids weights-only loading errors.
    """
    try:
        import torch
        from torch.nn.modules.container import Sequential, ModuleList
        from torch.nn.modules.activation import SiLU
        from collections import OrderedDict
        from ultralytics.nn.modules import Conv, C2f, SPPF, Bottleneck
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics import YOLO

        # Register safe globals for PyTorch checkpoint loading and try safe-context load
        safe_classes = [Sequential, ModuleList, SiLU, OrderedDict,
                        Conv, C2f, SPPF, Bottleneck, DetectionModel]
        torch.serialization.add_safe_globals(safe_classes)

        try:
            # Load model inside a safe_globals context to allowlist ultralytics classes
            with torch.serialization.safe_globals(safe_classes):
                if os.path.exists(model_path) and model_path != 'yolov8n.pt':
                    model = YOLO(model_path)
                    st.success(f" Modèle personnalisé chargé: {model_path}")
                else:
                    model = YOLO('yolov8n.pt')
                    st.info("Utilisation du modèle YOLOv8 par défaut. Entraînez un modèle personnalisé pour une meilleure détection d'armes!")

            return model

        except Exception as e:
            # Fallback: attempt weights_only=False (trusted checkpoint only!)
            st.warning(f"⚠️ Échec du chargement sécurisé du modèle: {e}. Tentative de fallback...")
            try:
                original_load = torch.load
                def patched_load(*args, **kwargs):
                    kwargs['weights_only'] = False
                    return original_load(*args, **kwargs)
                torch.load = patched_load

                model = YOLO(model_path)
                torch.load = original_load
                st.success("✓ Modèle chargé avec workaround (weights_only=False)")
                return model
            except Exception as e2:
                st.error(f"❌ Impossible de charger le modèle même avec workaround: {e2}")
                return None

    except Exception as e:
        # Fallback: attempt weights_only=False (trusted checkpoint only!)
        st.warning(f"⚠️ Échec du chargement normal du modèle: {e}")
        try:
            original_load = torch.load
            def patched_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            torch.load = patched_load

            model = YOLO(model_path)
            torch.load = original_load
            st.success("✓ Modèle chargé avec workaround (weights_only=False)")
            return model
        except Exception as e2:
            st.error(f"❌ Impossible de charger le modèle même avec workaround: {e2}")
            return None

def create_alarm_sound():
    """Create an alarm sound file if it doesn't exist"""
    if not os.path.exists('alarm.wav'):
        try:
            import numpy as np
            from scipy.io.wavfile import write
            
            sample_rate = 44100
            duration = 0.5
            frequency = 1000  # Hz
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            # Create siren-like sound
            audio = np.sin(2 * np.pi * frequency * t) * 0.5
            audio += np.sin(2 * np.pi * frequency * 1.5 * t) * 0.3
            audio = (audio * 32767).astype(np.int16)
            
            write('alarm.wav', sample_rate, audio)
            st.success("✅ Fichier d'alarme créé")
        except Exception as e:
            st.warning(f"⚠️ Impossible de créer le son d'alarme: {e}")

def play_alarm():
    """Play alarm sound when weapon detected"""
    if AUDIO_AVAILABLE and st.session_state.audio_enabled:
        try:
            if os.path.exists('alarm.wav'):
                pygame.mixer.music.load('alarm.wav')
                pygame.mixer.music.play()
            else:
                # System beep as fallback
                import winsound
                winsound.Beep(1000, 300)
        except Exception as e:
            pass  # Silent fail if audio not available

def save_config():
    """Save configuration"""
    config = {
        'authorized_ids': list(st.session_state.authorized_ids),
        'restricted_zones': st.session_state.restricted_zones,
        'audio_enabled': st.session_state.audio_enabled
    }
    with open('security_config.json', 'w') as f:
        json.dump(config, f)

def load_config():
    """Load configuration"""
    if os.path.exists('security_config.json'):
        with open('security_config.json', 'r') as f:
            config = json.load(f)
            st.session_state.authorized_ids = set(config.get('authorized_ids', []))
            st.session_state.restricted_zones = config.get('restricted_zones', [])
            st.session_state.audio_enabled = config.get('audio_enabled', True)

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
    
    # Play alarm for critical events
    if severity == "CRITIQUE" and st.session_state.alarm_cooldown == 0:
        play_alarm()
        st.session_state.alarm_cooldown = 30  # 1 second cooldown at 30fps

def point_in_polygon(point, polygon):
    """Check if point is inside polygon"""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
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
    """Check if bbox center is in restricted zone"""
    x1, y1, x2, y2 = bbox
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    
    for zone in zones:
        if point_in_polygon(center, zone['coords']):
            return True, zone['name']
    return False, None

def process_frame(frame, model):
    """Process frame for detection and tracking"""
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
        cv2.polylines(annotated_frame, [pts], True, (0, 0, 255), 3)
        
        overlay = annotated_frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 255))
        cv2.addWeighted(overlay, 0.2, annotated_frame, 0.8, 0, annotated_frame)
        
        cv2.putText(annotated_frame, f"ZONE RESTREINTE: {zone['name']}", 
                   tuple(zone['coords'][0]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Track persons
    for person in detections['persons']:
        if person['id'] is not None:
            track_id = person['id']
            
            if track_id not in st.session_state.person_tracking:
                st.session_state.person_tracking[track_id] = {
                    'first_seen': current_time,
                    'last_seen': current_time,
                    'authorized': track_id in st.session_state.authorized_ids,
                    'in_zone_logged': False
                }
            else:
                st.session_state.person_tracking[track_id]['last_seen'] = current_time
            
            # Check restricted zones
            in_zone, zone_name = is_in_restricted_zone(person['bbox'], zones)
            if in_zone:
                is_authorized = st.session_state.person_tracking[track_id]['authorized']
                
                if not is_authorized:
                    x1, y1, x2, y2 = person['bbox']
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                                (0, 0, 255), 4)
                    cv2.putText(annotated_frame, "⚠️ NON AUTORISE!", 
                               (int(x1), int(y1) - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    if not st.session_state.person_tracking[track_id].get('in_zone_logged', False):
                        log_alarm("ZONE_RESTREINTE", 
                                 f"Personne non autorisée (ID:{track_id}) dans {zone_name}", 
                                 "CRITIQUE")
                        st.session_state.person_tracking[track_id]['in_zone_logged'] = True
                else:
                    x1, y1, x2, y2 = person['bbox']
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                                (0, 255, 0), 3)
                    cv2.putText(annotated_frame, "✓ AUTORISE", 
                               (int(x1), int(y1) - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                st.session_state.person_tracking[track_id]['in_zone_logged'] = False
    
    # **WEAPON DETECTION WITH ALARM**
    if detections['weapons']:
        st.session_state.weapon_detection_count += 1
        
        for weapon in detections['weapons']:
            x1, y1, x2, y2 = weapon['bbox']
            
            # Pulsing red box
            pulse = int(abs(np.sin(current_time * 4) * 255))
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                        (0, 0, pulse), 5)
            
            # Large warning text
            cv2.putText(annotated_frame, f"🚨 ARME: {weapon['class'].upper()}", 
                       (int(x1), int(y1) - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.putText(annotated_frame, f"Confiance: {weapon['confidence']:.2f}", 
                       (int(x1), int(y1) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Screen-wide alert
        cv2.rectangle(annotated_frame, (0, 0), (width, 80), (0, 0, 255), -1)
        cv2.putText(annotated_frame, "🚨 ALERTE ARME DETECTEE 🚨", 
                   (width//2 - 300, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
        
        # Log alarm
        if st.session_state.weapon_detection_count % 10 == 0:  # Log every 10 detections
            log_alarm("ARME", 
                     f"Arme détectée: {detections['weapons'][0]['class']} (Confiance: {detections['weapons'][0]['confidence']:.2f})", 
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
                    x1, y1, x2, y2 = obj['bbox']
                    pulse = int(abs(np.sin(current_time * 2) * 255))
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                                (0, 0, pulse), 4)
                    
                    cv2.putText(annotated_frame, 
                               f"⚠️ OBJET ABANDONNE {obj['class'].upper()} ({int(duration)}s)", 
                               (int(x1), int(y1) - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    if not st.session_state.object_tracking[obj_id].get('logged', False):
                        play_alarm()
                        log_alarm("OBJET_ABANDONNE", 
                                 f"{obj['class']} abandonné pendant {int(duration)}s",
                                 "AVERTISSEMENT")
                            
                        st.session_state.object_tracking[obj_id]['logged'] = True
    
    # Cleanup old tracking
    current_ids = set([p['id'] for p in detections['persons'] if p['id'] is not None])
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

st.title(" Système de Surveillance de Sécurité Aéroportuaire MaydayMayday")
st.markdown("**Surveillance Avancée Alimentée par IA avec Détection d'Armes Personnalisée**")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration du Système")
    
    # Model selection
    st.markdown("###  Sélection du Modèle")
    model_options = ['yolov8n.pt (Défaut)']
    
    # Check for custom models
    custom_models = list(Path('.').glob('*.pt')) + list(Path('runs/detect').glob('*/weights/best.pt'))
    for model_path in custom_models:
        if model_path.name != 'yolov8n.pt':
            model_options.append(str(model_path))
    
    selected_model = st.selectbox("Modèle YOLO", model_options)
    
    if selected_model != 'yolov8n.pt (Défaut)':
        st.info(f" Utilisation du modèle: {selected_model}")
    
    # Audio settings
    st.markdown("### 🔊 Paramètres Audio")
    audio_toggle = st.checkbox("Activer Alarme Sonore", value=st.session_state.audio_enabled)
    if audio_toggle != st.session_state.audio_enabled:
        st.session_state.audio_enabled = audio_toggle
        save_config()
    
    if st.button("🔊 Tester Alarme"):
        play_alarm()
    
    # Load/Save config
    if st.button("📂 Charger Configuration"):
        load_config()
        st.success("Configuration chargée")
    
    st.markdown("### Personnel Autorisé")
    new_auth_id = st.number_input("ID de Suivi", min_value=1, step=1)
    #style the button
    if st.button("Autoriser"):
        st.session_state.authorized_ids.add(new_auth_id)
        save_config()
        st.success(f"ID autorisé: {new_auth_id}")
    
    if st.session_state.authorized_ids:
        st.text(", ".join(map(str, sorted(st.session_state.authorized_ids))))
        if st.button("Effacer Autorisations"):
            st.session_state.authorized_ids.clear()
            save_config()
    
    st.markdown("---")
    st.markdown("### 🚫 Zones Restreintes")
    zone_name = st.text_input("Nom Zone", value="Zone Restreinte 1")
    col1, col2 = st.columns(2)
    with col1:
        x1 = st.number_input("X1", value=100)
        y1 = st.number_input("Y1", value=100)
    with col2:
        x2 = st.number_input("X2", value=400)
        y2 = st.number_input("Y2", value=400)
    
    if st.button("Ajouter Zone"):
        zone_coords = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        st.session_state.restricted_zones.append({'name': zone_name, 'coords': zone_coords})
        save_config()
        st.success(f"Zone ajoutée: {zone_name}")
    
    if st.session_state.restricted_zones:
        st.text(f"Zones: {len(st.session_state.restricted_zones)}")
        if st.button("Effacer Zones"):
            st.session_state.restricted_zones = []
            save_config()
    
    st.markdown("---")
    abandon_time = st.slider("Seuil Objet Abandonné (s)", 10, 300, 60)
    
    st.markdown("---")
    st.metric("Détections d'Armes", st.session_state.weapon_detection_count)
    st.metric("Alertes Critiques", len([a for a in st.session_state.alarm_log if a['severity'] == 'CRITIQUE'][:10]))

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
        stop_button = st.button("Arrêter", use_container_width=True)
    with col_btn3:
        clear_logs = st.button("Effacer Logs", use_container_width=True)

with col2:
    st.subheader("🚨 Alertes en Direct")
    alerts_placeholder = st.empty()
    
    st.subheader("Statistiques")
    stats_placeholder = st.empty()

if clear_logs:
    st.session_state.alarm_log = []
    st.session_state.weapon_detection_count = 0

if start_button:
    st.session_state.running = True

if stop_button:
    st.session_state.running = False

# Video processing
if st.session_state.running:
    # Create alarm sound
    create_alarm_sound()
    
    # Load model
    model_path = selected_model.replace(' (Défaut)', '')
    model = load_yolo_model(model_path)
    
    if model is not None:
        cap = None
        
        if camera_option == "Webcam":
            cap = cv2.VideoCapture(0)
        else:
            uploaded_video = st.file_uploader("Fichier Vidéo", type=['mp4', 'avi', 'mov'])
            if uploaded_video:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_video.read())
                cap = cv2.VideoCapture(tfile.name)
            else:
                st.warning("Téléchargez une vidéo")
                st.session_state.running = False
        
        if cap and cap.isOpened():
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
                
                # Display
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
                critical_alerts = len([a for a in st.session_state.alarm_log if a['severity'] == 'CRITIQUE'])
                
                stats_html = f"""
                <div style='background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333;'>
                    <p><b style="color: #00ff00; background-color: ermelad; padding: 2px 6px; border-radius: 4px;">Personnes:</b> {len(detections['persons'])}</p>
                    <p><b style="color: #00ff00; background-color: #003300; padding: 2px 6px; border-radius: 4px;">Armes Détectées:</b> {st.session_state.weapon_detection_count}</p>
                    <p><b style="color: #00ff00; background-color: #003300; padding: 2px 6px; border-radius: 4px;">Objets Suivis:</b> {len(st.session_state.object_tracking)}</p>
                    <p><b style="color: #00ff00; background-color: #003300; padding: 2px 6px; border-radius: 4px;">Alertes Critiques:</b> {critical_alerts}</p>
                    <p><b style="color: #00ff00; background-color: #003300; padding: 2px 6px; border-radius: 4px;">Total Alertes:</b> {len(st.session_state.alarm_log)}</p>
                    <p><b style="color: #00ff00; background-color: #003300; padding: 2px 6px; border-radius: 4px;">Image:</b> {frame_count}</p>
                </div>
                """
                stats_placeholder.markdown(stats_html, unsafe_allow_html=True)
                
                time.sleep(0.03)
            
            cap.release()

# Security log
st.markdown("---")
st.subheader("Journal de Sécurité")

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
    
    if st.button(" Exporter Journal"):
        csv = df.to_csv(index=False)
        st.download_button(
            "Télécharger CSV",
            csv,
            f"journal_securite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )

# Instructions
with st.expander("📖 Guide d'Utilisation"):
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
    
    3. **Entraîner votre modèle:**
       - Collectez 500-1000+ images d'armes
       - Annotez avec LabelImg ou Roboflow
       - Entraînez avec le script fourni
       - Placez best.pt dans le dossier du projet
    
    ### Fonctionnalités:
    - ✅ Détection d'armes en temps réel
    - ✅ Alarme sonore automatique
    - ✅ Contrôle d'accès aux zones restreintes
    - ✅ Suivi des objets abandonnés
    - ✅ Journal de sécurité complet
    - ✅ Support modèles personnalisés
    """)