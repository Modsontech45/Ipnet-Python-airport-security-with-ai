import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

class AdvancedAnomalyDetector:
    """
    Advanced human anomaly detector with:
    - Full body detection
    - Weapon detection (knife, gun)
    - Crowd density monitoring
    - Restricted zone alerts
    - Aggressive behavior detection
    - Fast movement detection
    """
    
    def __init__(self, video_source=0):
        # MediaPipe Pose for full body tracking
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Video capture
        self.cap = cv2.VideoCapture(video_source)
        ret, frame = self.cap.read()
        if ret:
            self.frame_height, self.frame_width = frame.shape[:2]
        else:
            self.frame_height, self.frame_width = 480, 640
        
        # HOG Person Detector (better than background subtraction)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # Background subtractor (backup method)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
        
        # Tracking data
        self.person_trackers = {}
        self.next_person_id = 0
        self.history_length = 30
        
        # IMPROVED THRESHOLDS
        self.running_speed_threshold = 30  # Reduced for better sensitivity
        self.aggression_hand_speed_threshold = 0.12  # Hand movement threshold
        self.loitering_time = 120  # frames (~4 seconds)
        self.min_body_visibility = 0.6  # Minimum body parts visible
        
        # Crowd density settings
        self.crowd_density_threshold = 5  # Max people in zone
        self.crowd_zone_size = 200  # pixels
        
        # Restricted zones (you can define areas)
        self.restricted_zones = []
        self.zone_mode = False  # Set to True to draw zones
        
        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        print("\n" + "="*70)
        print("ADVANCED HUMAN ANOMALY DETECTION SYSTEM")
        print("="*70)
        print("Features:")
        print("  ✓ Full body detection and tracking")
        print("  ✓ Fast movement/running detection")
        print("  ✓ Aggressive behavior detection")
        print("  ✓ Weapon detection (knife, gun, stick)")
        print("  ✓ Crowd density monitoring")
        print("  ✓ Restricted zone alerts")
        print("\nControls:")
        print("  Q - Quit")
        print("  Z - Toggle zone drawing mode")
        print("  C - Clear all zones")
        print("="*70 + "\n")
    
    def detect_persons_hog(self, frame):
        """Detect persons using HOG descriptor (more accurate)"""
        # Resize for faster processing
        scale = 0.5
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        # Detect people
        boxes, weights = self.hog.detectMultiScale(
            small_frame, 
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.05,
            hitThreshold=0.5
        )
        
        # Scale back to original size
        person_boxes = []
        for (x, y, w, h) in boxes:
            x = int(x / scale)
            y = int(y / scale)
            w = int(w / scale)
            h = int(h / scale)
            person_boxes.append([x, y, x+w, y+h])
        
        return person_boxes
    
    def detect_persons_background(self, frame):
        """Backup person detection using background subtraction"""
        fg_mask = self.bg_subtractor.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        person_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 3000:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / (w + 1e-6)
                if 1.0 < aspect_ratio < 4 and h > 100:
                    person_boxes.append([x, y, x+w, y+h])
        
        return person_boxes
    
    def check_full_body_visible(self, landmarks):
        """Check if full body is visible (head to feet)"""
        if not landmarks:
            return False, 0
        
        # Key landmarks to check
        required_landmarks = [
            self.mp_pose.PoseLandmark.NOSE,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE,
        ]
        
        visible_count = 0
        for landmark_type in required_landmarks:
            landmark = landmarks[landmark_type.value]
            if landmark.visibility > 0.5:
                visible_count += 1
        
        visibility_ratio = visible_count / len(required_landmarks)
        is_full_body = visibility_ratio >= self.min_body_visibility
        
        return is_full_body, visibility_ratio
    
    def detect_weapon(self, landmarks, frame, bbox):
        """Detect if person is holding a weapon-like object"""
        if not landmarks:
            return False, "none"
        
        # Get hand and elbow positions
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        
        weapon_detected = False
        weapon_type = "none"
        
        # Check for raised arm (threatening pose)
        left_arm_raised = left_wrist.y < left_shoulder.y - 0.1
        right_arm_raised = right_wrist.y < right_shoulder.y - 0.1
        
        # Check for extended arm (pointing/holding)
        left_arm_extended = abs(left_wrist.x - left_shoulder.x) > 0.3
        right_arm_extended = abs(right_wrist.x - right_shoulder.x) > 0.3
        
        # Weapon-like pose detection
        if (left_arm_raised or right_arm_raised) and (left_arm_extended or right_arm_extended):
            weapon_detected = True
            weapon_type = "weapon-like pose"
        
        # Check for stick-like object (vertical line near hand)
        x1_roi = max(0, int(x1 + left_wrist.x * w - 30))
        x2_roi = min(frame.shape[1], int(x1 + left_wrist.x * w + 30))
        y1_roi = max(0, int(y1 + left_wrist.y * h - 50))
        y2_roi = min(frame.shape[0], int(y1 + left_wrist.y * h + 50))
        
        if x2_roi > x1_roi and y2_roi > y1_roi:
            hand_roi = frame[y1_roi:y2_roi, x1_roi:x2_roi]
            if hand_roi.size > 0:
                gray_roi = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray_roi, 50, 150)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, minLineLength=40, maxLineGap=10)
                
                if lines is not None and len(lines) > 3:
                    weapon_detected = True
                    weapon_type = "stick/knife-like object"
        
        return weapon_detected, weapon_type
    
    def detect_fast_movement(self, person_id):
        """Improved fast movement detection"""
        if person_id not in self.person_trackers:
            return False, 0, "none"
        
        history = self.person_trackers[person_id]['position_history']
        
        if len(history) < 6:
            return False, 0, "none"
        
        # Calculate speeds
        speeds = []
        for i in range(len(history) - 1):
            dist = np.linalg.norm(np.array(history[i+1]) - np.array(history[i]))
            speeds.append(dist)
        
        # Recent speed analysis
        recent_speeds = speeds[-5:]
        avg_speed = np.mean(recent_speeds)
        max_speed = max(recent_speeds)
        
        movement_type = "none"
        is_fast = False
        
        # Running detection
        if avg_speed > self.running_speed_threshold:
            is_fast = True
            movement_type = "RUNNING"
        
        # Sudden movement/darting
        elif max_speed > self.running_speed_threshold * 1.5:
            is_fast = True
            movement_type = "SUDDEN MOVEMENT"
        
        return is_fast, avg_speed, movement_type
    
    def detect_aggressive_behavior(self, person_id, landmarks):
        """Enhanced aggressive behavior detection"""
        if person_id not in self.person_trackers or not landmarks:
            return False, []
        
        aggressive_indicators = []
        
        # Get body landmarks
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE.value]
        
        # Track wrist movements
        if 'wrist_history' not in self.person_trackers[person_id]:
            self.person_trackers[person_id]['wrist_history'] = deque(maxlen=15)
        
        avg_wrist_x = (left_wrist.x + right_wrist.x) / 2
        avg_wrist_y = (left_wrist.y + right_wrist.y) / 2
        
        self.person_trackers[person_id]['wrist_history'].append([avg_wrist_x, avg_wrist_y])
        wrist_history = self.person_trackers[person_id]['wrist_history']
        
        if len(wrist_history) >= 5:
            # Calculate wrist speed
            wrist_speeds = []
            for i in range(len(wrist_history) - 1):
                dist = np.linalg.norm(np.array(wrist_history[i+1]) - np.array(wrist_history[i]))
                wrist_speeds.append(dist)
            
            max_wrist_speed = max(wrist_speeds[-5:])
            
            # Fast hand movements (punching, swinging)
            if max_wrist_speed > self.aggression_hand_speed_threshold:
                aggressive_indicators.append("RAPID HAND MOVEMENT")
        
        # Raised arms (threatening posture)
        left_raised = left_wrist.y < left_shoulder.y
        right_raised = right_wrist.y < right_shoulder.y
        
        if left_raised and right_raised:
            aggressive_indicators.append("ARMS RAISED")
        
        # Hands near head (fighting stance or aggressive gesture)
        left_near_head = abs(left_wrist.y - nose.y) < 0.15
        right_near_head = abs(right_wrist.y - nose.y) < 0.15
        
        if left_near_head or right_near_head:
            aggressive_indicators.append("HANDS NEAR HEAD")
        
        return len(aggressive_indicators) > 0, aggressive_indicators
    
    def detect_crowd_density(self, person_boxes):
        """Detect overcrowding in specific areas"""
        if len(person_boxes) < 3:
            return []
        
        crowded_zones = []
        
        # Check density in grid zones
        for person_box in person_boxes:
            x1, y1, x2, y2 = person_box
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Count nearby people
            nearby_count = 0
            for other_box in person_boxes:
                ox1, oy1, ox2, oy2 = other_box
                other_center_x = (ox1 + ox2) / 2
                other_center_y = (oy1 + oy2) / 2
                
                distance = np.sqrt((center_x - other_center_x)**2 + (center_y - other_center_y)**2)
                
                if distance < self.crowd_zone_size:
                    nearby_count += 1
            
            if nearby_count >= self.crowd_density_threshold:
                crowded_zones.append({
                    'center': (int(center_x), int(center_y)),
                    'count': nearby_count
                })
        
        return crowded_zones
    
    def check_restricted_zones(self, person_box):
        """Check if person is in restricted zone"""
        if not self.restricted_zones:
            return False, None
        
        x1, y1, x2, y2 = person_box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        for i, zone in enumerate(self.restricted_zones):
            zx1, zy1, zx2, zy2 = zone
            
            if zx1 <= center_x <= zx2 and zy1 <= center_y <= zy2:
                return True, i
        
        return False, None
    
    def update_trackers(self, detections):
        """Update person trackers"""
        current_centers = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            current_centers.append(center)
        
        matched_ids = []
        
        for i, center in enumerate(current_centers):
            best_match = None
            min_dist = float('inf')
            
            for pid, tracker in self.person_trackers.items():
                if tracker['position_history']:
                    last_pos = tracker['position_history'][-1]
                    dist = np.linalg.norm(np.array(center) - np.array(last_pos))
                    
                    if dist < min_dist and dist < 150:
                        min_dist = dist
                        best_match = pid
            
            if best_match is not None:
                matched_ids.append(best_match)
                self.person_trackers[best_match]['position_history'].append(center)
                self.person_trackers[best_match]['bbox'] = detections[i]
                self.person_trackers[best_match]['frame_count'] += 1
            else:
                new_id = self.next_person_id
                self.next_person_id += 1
                matched_ids.append(new_id)
                
                self.person_trackers[new_id] = {
                    'position_history': deque([center], maxlen=self.history_length),
                    'bbox': detections[i],
                    'frame_count': 0
                }
        
        to_remove = [pid for pid in self.person_trackers if pid not in matched_ids]
        for pid in to_remove:
            del self.person_trackers[pid]
        
        return matched_ids
    
    def process_frame(self, frame):
        """Process frame and detect all anomalies"""
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect persons using HOG (primary method)
        person_boxes = self.detect_persons_hog(frame)
        
        # If HOG finds nothing, use background subtraction
        if len(person_boxes) == 0:
            person_boxes = self.detect_persons_background(frame)
        
        # Update trackers
        person_ids = self.update_trackers(person_boxes)
        
        # Check crowd density
        crowded_zones = self.detect_crowd_density(person_boxes)
        
        anomalies = []
        
        for person_id, bbox in zip(person_ids, person_boxes):
            x1, y1, x2, y2 = bbox
            
            person_roi = frame_rgb[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            
            if person_roi.size == 0:
                continue
            
            pose_results = self.pose.process(person_roi)
            
            anomaly_list = []
            box_color = (0, 255, 0)  # Green default
            
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                
                # Check full body visibility
                is_full_body, visibility = self.check_full_body_visible(landmarks)
                
                # Weapon detection
                has_weapon, weapon_type = self.detect_weapon(landmarks, frame, bbox)
                if has_weapon:
                    anomaly_list.append(f"⚠ WEAPON: {weapon_type}")
                    box_color = (0, 0, 255)  # Red
                
                # Aggressive behavior
                is_aggressive, aggression_types = self.detect_aggressive_behavior(person_id, landmarks)
                if is_aggressive:
                    for agg_type in aggression_types:
                        anomaly_list.append(f"⚠ AGGRESSIVE: {agg_type}")
                    box_color = (0, 100, 255)  # Orange-red
                
                # Draw skeleton
                for landmark in landmarks:
                    px = int(x1 + landmark.x * (x2 - x1))
                    py = int(y1 + landmark.y * (y2 - y1))
                    if 0 <= px < w and 0 <= py < h and landmark.visibility > 0.5:
                        cv2.circle(frame, (px, py), 2, (0, 255, 255), -1)
            
            # Fast movement detection
            is_fast, speed, movement_type = self.detect_fast_movement(person_id)
            if is_fast:
                anomaly_list.append(f"🏃 {movement_type} (speed: {speed:.1f})")
                if box_color == (0, 255, 0):
                    box_color = (255, 165, 0)  # Orange
            
            # Restricted zone check
            in_restricted, zone_id = self.check_restricted_zones(bbox)
            if in_restricted:
                anomaly_list.append(f" IN RESTRICTED ZONE {zone_id + 1}")
                box_color = (255, 0, 255)  # Magenta
            
            # Draw bounding box
            thickness = 3 if anomaly_list else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
            
            # Draw person ID
            cv2.putText(frame, f"ID:{person_id}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if anomaly_list:
                anomalies.append({
                    'person_id': person_id,
                    'bbox': bbox,
                    'anomalies': anomaly_list
                })
        
        # Draw crowd density warnings
        for crowd_zone in crowded_zones:
            center = crowd_zone['center']
            count = crowd_zone['count']
            cv2.circle(frame, center, self.crowd_zone_size, (0, 0, 255), 2)
            cv2.putText(frame, f"CROWD: {count} people", (center[0] - 60, center[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            anomalies.append({
                'person_id': 'CROWD',
                'bbox': None,
                'anomalies': [f"⚠ OVERCROWDING: {count} people in area"]
            })
        
        # Draw restricted zones
        for i, zone in enumerate(self.restricted_zones):
            zx1, zy1, zx2, zy2 = zone
            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 0, 255), 2)
            cv2.putText(frame, f"RESTRICTED ZONE {i+1}", (zx1 + 5, zy1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame, anomalies
    
    def run(self):
        """Main detection loop"""
        drawing_zone = False
        zone_start = None
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing_zone, zone_start
            
            if not self.zone_mode:
                return
            
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing_zone = True
                zone_start = (x, y)
            
            elif event == cv2.EVENT_LBUTTONUP and drawing_zone:
                drawing_zone = False
                zone_end = (x, y)
                
                x1 = min(zone_start[0], zone_end[0])
                y1 = min(zone_start[1], zone_end[1])
                x2 = max(zone_start[0], zone_end[0])
                y2 = max(zone_start[1], zone_end[1])
                
                if x2 - x1 > 20 and y2 - y1 > 20:
                    self.restricted_zones.append([x1, y1, x2, y2])
                    print(f"✓ Restricted zone {len(self.restricted_zones)} created: ({x1}, {y1}) to ({x2}, {y2})")
        
        cv2.namedWindow('Advanced Anomaly Detection')
        cv2.setMouseCallback('Advanced Anomaly Detection', mouse_callback)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("End of video or camera error")
                break
            
            # Process frame
            processed_frame, anomalies = self.process_frame(frame)
            
            # Calculate FPS
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                elapsed = time.time() - self.start_time
                self.fps = self.frame_count / elapsed
            
            # Display alerts
            y_offset = 30
            for anomaly in anomalies:
                person_id = anomaly['person_id']
                for anom_text in anomaly['anomalies']:
                    if person_id == 'CROWD':
                        text = anom_text
                    else:
                        text = f"Person {person_id}: {anom_text}"
                    
                    cv2.putText(processed_frame, text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    y_offset += 25
                    
                    print(f"[ALERT] {text}")
            
            # Display info
            info_text = f"People: {len(self.person_trackers)} | FPS: {self.fps:.1f} | Zones: {len(self.restricted_zones)}"
            cv2.putText(processed_frame, info_text, (10, processed_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if self.zone_mode:
                cv2.putText(processed_frame, "ZONE DRAWING MODE: Click and drag to create restricted zones",
                           (10, processed_frame.shape[0] - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('Advanced Anomaly Detection', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('z'):
                self.zone_mode = not self.zone_mode
                print(f"Zone drawing mode: {'ON' if self.zone_mode else 'OFF'}")
            elif key == ord('c'):
                self.restricted_zones = []
                print("All restricted zones cleared")
        
        self.cleanup()
    
    def cleanup(self):
        """Release resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        self.pose.close()
        print("\n" + "="*70)
        print("Detector stopped. Summary:")
        print(f"  Total frames processed: {self.frame_count}")
        print(f"  Average FPS: {self.fps:.2f}")
        print(f"  Restricted zones defined: {len(self.restricted_zones)}")
        print("="*70)

if __name__ == "__main__":
    # Use 0 for webcam, or provide video file path
    detector = AdvancedAnomalyDetector(video_source=0)
    detector.run()