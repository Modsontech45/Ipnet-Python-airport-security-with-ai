import cv2
import torch

# FIX for PyTorch 2.6+: Import the classes first, then add as safe globals
try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except Exception as e:
    print(f"Note: Could not pre-register safe globals: {e}")
    print("Trying alternative method...\n")

from ultralytics import YOLO

# Load YOLOv8 model
print("Loading YOLOv8 model...")
try:
    model = YOLO('yolov8n.pt')
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"\n❌ Error loading model: {e}")
    print("\nTrying workaround...")
    
    # Workaround: Temporarily disable weights_only check
    import ultralytics.nn.tasks as tasks_module
    original_load = torch.load
    
    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    
    torch.load = patched_load
    
    try:
        model = YOLO('yolov8n.pt')
        print("✓ Model loaded with workaround!")
    finally:
        torch.load = original_load

# Load video
video_path = './test.mp4'
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print(f"\n❌ Error: Could not open video '{video_path}'")
    print("\nTrying webcam instead...")
    video_path = 0
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Could not open webcam either")
        print("\nPlease:")
        print("  1. Make sure 'test.mp4' exists in the current directory")
        print("  2. Or change video_path to 0 to use webcam")
        print("  3. Or provide the full path to your video file")
        exit()
    else:
        print("✓ Using webcam")

# Get video properties (compatible with all OpenCV versions)
try:
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30  # Default for webcam
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
except AttributeError:
    # Fallback for older OpenCV versions
    fps = int(cap.get(cv2.cv.CV_CAP_PROP_FPS)) if hasattr(cv2, 'cv') else 30
    width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)) if hasattr(cv2, 'cv') else 640
    height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)) if hasattr(cv2, 'cv') else 480
    total_frames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)) if hasattr(cv2, 'cv') else 0

print(f"\nVideo Info:")
print(f"  Resolution: {width}x{height}")
print(f"  FPS: {fps}")
if total_frames > 0:
    print(f"  Total Frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.1f}s")
else:
    print(f"  Source: Live camera/webcam")
print("\nControls:")
print("  Press 'Q' to quit")
print("  Press 'P' to pause/resume\n")

frame_count = 0
paused = False

# Read frames
while True:
    if not paused:
        ret, frame = cap.read()
        
        if not ret:
            print("\n✓ End of video reached")
            break
        
        frame_count += 1
        
        # Detect and track objects
        results = model.track(frame, persist=True, verbose=False)
        
        # Plot results with bounding boxes and labels
        frame_annotated = results[0].plot()
        
        # Add frame counter and info
        info_text = f"Frame: {frame_count}"
        if total_frames > 0:
            info_text += f"/{total_frames}"
        
        cv2.putText(frame_annotated, info_text, 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Count detected objects
        boxes = results[0].boxes
        if boxes is not None:
            num_objects = len(boxes)
            cv2.putText(frame_annotated, f"Objects: {num_objects}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        # If paused, keep showing the same frame
        cv2.putText(frame_annotated, "PAUSED - Press 'P' to resume", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display frame
    cv2.imshow('YOLOv8 Object Tracking', frame_annotated)
    
    # Handle key presses
    key = cv2.waitKey(25) & 0xFF
    if key == ord('q'):
        print("\nStopped by user")
        break
    elif key == ord('p'):
        paused = not paused
        print("Paused" if paused else "Resumed")

# Release resources
cap.release()
cv2.destroyAllWindows()
print(f"\n✓ Processed {frame_count} frames")
print("Done!")