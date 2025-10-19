import os
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import hashlib
import requests
from streamlit_lottie import st_lottie
import time
from pathlib import Path
import sqlite3
from contextlib import contextmanager

# --- Custom Modules (from your original script) ---
try:
    from image_ops import loader, preprocess, label
    from models import predict_age, predict_feature
except ImportError:
    st.error("Could not import custom modules (image_ops, models). Please ensure they are in the correct path.")
    st.stop()


# --- Project Structure and Configuration ---
ROOT_DIR = Path(__file__).parent.absolute()
MODELS_DIR = ROOT_DIR / "models"
CASCADE_DIR = ROOT_DIR / "image_ops"
DATA_DIR = ROOT_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "dermalscan.db"

# Ensure all necessary directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Correct, portable paths to model and cascade files
FEATURES_MODEL_PATH = MODELS_DIR / "efficientnet_b0_face_classifier_finetuned.h5"
AGE_MODEL_PATH = MODELS_DIR / "age_mobilenet_regression_stratified_old.h5"
CASCADE_FILENAME = CASCADE_DIR / "haarcascade_frontalface_default.xml"

# --- Database Management ---

@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def initialize_database():
    """Initialize SQLite database with required tables."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Analysis logs table - stores images as BLOBs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                image_filename TEXT NOT NULL,
                patient_name TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                face_array BLOB NOT NULL,
                annotated_image BLOB NOT NULL,
                clear_skin_percent REAL,
                dark_spots_percent REAL,
                puffy_eyes_percent REAL,
                wrinkles_percent REAL,
                skin_age REAL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Create indexes for better query performance
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_analysis_user_id 
            ON analysis_logs(user_id)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_analysis_timestamp 
            ON analysis_logs(timestamp DESC)
        ''')
        
        conn.commit()

def hash_password(password):
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def validate_email(email):
    """Basic email validation."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_username(username):
    """Validate username (alphanumeric and underscore only, 3-20 chars)."""
    import re
    pattern = r'^[a-zA-Z0-9_]{3,20}$'
    return re.match(pattern, username) is not None

def validate_password(password):
    """Validate password strength (min 8 chars, at least one number)."""
    return len(password) >= 8 and any(c.isdigit() for c in password)

def create_user(username, email, password):
    """Create a new user account with validation."""
    if not validate_username(username):
        return False, "Username must be 3-20 characters (letters, numbers, underscore only)."
    
    if not validate_email(email):
        return False, "Please enter a valid email address."
    
    if not validate_password(password):
        return False, "Password must be at least 8 characters with at least one number."
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            password_hash = hash_password(password)
            
            cursor.execute(
                'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                (username, email, password_hash)
            )
            conn.commit()
            return True, "Account created successfully! Please sign in."
            
    except sqlite3.IntegrityError as e:
        if 'username' in str(e):
            return False, "Username already exists."
        elif 'email' in str(e):
            return False, "Email already registered."
        else:
            return False, "Registration failed. Please try again."

def verify_user(username, password):
    """Verify user credentials and return user_id if valid."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT id, password_hash FROM users WHERE username = ?',
                (username,)
            )
            user = cursor.fetchone()
            
            if user and hash_password(password) == user['password_hash']:
                return user['id']
    except Exception as e:
        st.error(f"Authentication error: {e}")
    
    return None

def numpy_to_blob(numpy_array):
    """Convert numpy array to binary blob for storage."""
    return numpy_array.tobytes()

def blob_to_numpy(blob_data, shape, dtype=np.float32):
    """Convert binary blob back to numpy array."""
    return np.frombuffer(blob_data, dtype=dtype).reshape(shape)

def image_to_blob(image):
    """Convert OpenCV image to binary blob via PNG encoding."""
    success, encoded = cv2.imencode('.png', image)
    if success:
        return encoded.tobytes()
    return None

def blob_to_image(blob_data):
    """Convert binary blob back to OpenCV image."""
    nparr = np.frombuffer(blob_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def append_log(user_id, log_data):
    """Append analysis results to database with images stored as BLOBs."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Convert images to BLOBs
            face_blob = numpy_to_blob(log_data['face_array'])
            annotated_blob = image_to_blob(log_data['annotated_image'])
            
            if annotated_blob is None:
                st.error("Failed to encode annotated image")
                return False
            
            cursor.execute('''
                INSERT INTO analysis_logs (
                    user_id, image_filename, patient_name, timestamp,
                    face_array, annotated_image,
                    clear_skin_percent, dark_spots_percent,
                    puffy_eyes_percent, wrinkles_percent, skin_age
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                log_data['image_filename'],
                log_data.get('patient_name', ''),
                log_data['timestamp'],
                face_blob,
                annotated_blob,
                log_data['clear_skin_%'],
                log_data['dark_spots_%'],
                log_data['puffy_eyes_%'],
                log_data['wrinkles_%'],
                log_data['skin_age']
            ))
            conn.commit()
            return True
    except Exception as e:
        st.error(f"Failed to save log: {e}")
        return False

def get_user_logs(user_id, limit=None):
    """Retrieve analysis logs for a user with images from BLOBs."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            query = '''
                SELECT id, user_id, image_filename, patient_name, timestamp,
                       face_array, annotated_image,
                       clear_skin_percent, dark_spots_percent,
                       puffy_eyes_percent, wrinkles_percent, skin_age
                FROM analysis_logs 
                WHERE user_id = ? 
                ORDER BY timestamp DESC
            '''
            if limit:
                query += f' LIMIT {limit}'
            
            cursor.execute(query, (user_id,))
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            logs = []
            for row in rows:
                log_dict = dict(row)
                # Keep BLOBs as-is, we'll decode them when displaying
                logs.append(log_dict)
            
            return logs
    except Exception as e:
        st.error(f"Failed to retrieve logs: {e}")
        return []

def get_analysis_by_id(analysis_id, user_id):
    """Retrieve a specific analysis record with verification."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM analysis_logs 
                WHERE id = ? AND user_id = ?
            ''', (analysis_id, user_id))
            row = cursor.fetchone()
            return dict(row) if row else None
    except Exception as e:
        st.error(f"Failed to retrieve analysis: {e}")
        return None

# --- UI Enhancements & Animations ---

@st.cache_data(ttl=3600)
def load_lottieurl(url: str):
    """Fetches a Lottie animation from a URL with caching."""
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        pass
    return None

def apply_custom_css():
    """Applies custom CSS for styling and animations."""
    try:
        with open('styles.css','r') as file:
            css=file.read()
        st.markdown(r'<style>'+css+r'</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        pass  # CSS file is optional

# --- Model Loading ---
@st.cache_resource
def load_models_and_cascade():
    """Load ML models and cascade classifier with caching."""
    try:
        face_cascade = loader.load_cascade(str(CASCADE_FILENAME))
        age_model = predict_age.load_model(str(AGE_MODEL_PATH))
        feature_model = predict_feature.load_model(str(FEATURES_MODEL_PATH))
        return face_cascade, age_model, feature_model
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

# --- UI Pages ---

def show_auth_page():
    """Display the authentication page (login/register)."""
    lottie_url = "https://assets9.lottiefiles.com/packages/lf20_x6s26o.json"
    lottie_json = load_lottieurl(lottie_url)

    if lottie_json:
        st_lottie(lottie_json, speed=1, height=200, key="auth_lottie")
    
    st.title("Welcome to DermalScan")
    st.write("Unlock AI-powered insights into your skin's health and age.")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Sign In")
        
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submit = st.form_submit_button("Sign In", use_container_width=True)
            
            if submit:
                if not username or not password:
                    st.error("Please fill in all fields.")
                else:
                    user_id = verify_user(username, password)
                    if user_id:
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = username
                        st.session_state['user_id'] = user_id
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
    
    with col2:
        st.header("Create Account")
        
        with st.form("register_form"):
            new_username = st.text_input("Choose a Username", key="reg_username")
            new_email = st.text_input("Email Address", key="reg_email")
            new_password = st.text_input("Create a Password", type="password", key="reg_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")
            submit = st.form_submit_button("Create Account", use_container_width=True)
            
            if submit:
                if not all([new_username, new_email, new_password, confirm_password]):
                    st.error("Please fill in all fields.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    success, message = create_user(new_username, new_email, new_password)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)

def show_dashboard():
    """Display the main dashboard for logged-in users."""
    username = st.session_state['username']
    user_id = st.session_state['user_id']

    # Header with logout
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("DermalScan Dashboard")
        st.markdown(f"Hello, **{username}**! Ready for your analysis?")
    with col2:
        if st.button("Logout", use_container_width=True):
            st.session_state['logged_in'] = False
            st.session_state['username'] = None
            st.session_state['user_id'] = None
            st.rerun()

    st.markdown("---")

    # Load models
    face_cascade, age_model, feature_model = load_models_and_cascade()

    # File uploader and Patient Name Input
    st.subheader("Upload Your Image")
    st.info("**Tips**: Use a well-lit, front-facing photo. Remove glasses and accessories for best results.")

    patient_name = st.text_input(
        "Name of person",
        help="If provided, this name will be used in the filenames of saved and downloaded files."
    )

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )

    if uploaded_file:
        # Read and decode image
        img_bytes = uploaded_file.read()
        full_image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        if full_image is None:
            st.error("Could not decode image. Please try a different file.")
            return

        # Show loading animation
        analysis_placeholder = st.empty()
        lottie_analysis_url = "https://assets1.lottiefiles.com/packages/lf20_gijbedz7.json"
        lottie_analysis = load_lottieurl(lottie_analysis_url)

        with analysis_placeholder.container():
            if lottie_analysis:
                st_lottie(lottie_analysis, speed=1, height=200, key="analysis_lottie")

            progress_bar = st.progress(0)
            status_text = st.empty()

            # Simulate progress
            for i in range(100):
                progress_bar.progress(i + 1)
                if i < 30:
                    status_text.text("Detecting face...")
                elif i < 60:
                    status_text.text("Analyzing features...")
                else:
                    status_text.text("Calculating results...")
                time.sleep(0.01)

        # Perform analysis
        try:
            face_image = preprocess.bytes_to_image(img_bytes)
            age = predict_age.predict_age(age_model, face_image)
            features = predict_feature.predict_features(feature_model, face_image)
            annotated_image = label.draw_labels_on_image(
                full_image.copy(), age, features, face_cascade
            )
        except Exception as e:
            analysis_placeholder.empty()
            st.error(f"No face detected or prediction failed: {e}")
            st.info("Try a well-lit image with a clearly visible face.")
            return

        # Clear loading animation
        analysis_placeholder.empty()

        # Display results
        st.header("Analysis Results")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Processed Image")
            st.image(
                cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
                caption="Your analyzed image with annotations",
                use_container_width=True
            )

        with col2:
            st.subheader("Metrics")
            st.metric(
                "Estimated Skin Age",
                f"{age:.1f} years",
                help="Estimated biological age of your skin"
            )

            st.markdown("---")
            st.subheader("Feature Analysis")

            # Display features with progress bars
            for feat, prob in features.items():
                feature_name = feat.replace('_', ' ').title()
                st.write(f"**{feature_name}**")
                progress_value = float(prob.item() if hasattr(prob, 'item') else prob)
                if progress_value > 1.0:
                    progress_value = progress_value / 100.0
                progress_value = max(0.0, min(1.0, progress_value))
                st.progress(progress_value)
                st.caption(f"{progress_value*100:.1f}%")

        with col1:
            st.markdown("---")
            st.subheader("Download")

            timestamp_str_short = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if patient_name and not patient_name.isspace():
                sanitized_name = "".join(c for c in patient_name if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')
                download_base_name = f"{sanitized_name}_{timestamp_str_short}"
            else:
                download_base_name = f"analysis_{timestamp_str_short}"

            success, img_encoded = cv2.imencode(".png", annotated_image)
            if success:
                st.download_button(
                    "Download Annotated Image",
                    img_encoded.tobytes(),
                    f"{download_base_name}_annotated.png",
                    "image/png",
                    use_container_width=True
                )

            # Download predictions CSV
            csv_header = "timestamp,age," + ",".join(features.keys())
            csv_values = [datetime.now().isoformat(), f"{age:.1f}"] + [f"{float(p.item() if hasattr(p, 'item') else p)*100:.1f}%" for p in features.values()]
            csv_row = ",".join(csv_values)
            csv_text = csv_header + "\n" + csv_row

            st.download_button(
                "Download Predictions CSV",
                csv_text,
                f"{download_base_name}_predictions.csv",
                "text/csv",
                use_container_width=True
            )

        # Save files - NO LONGER SAVING TO DISK
        # Images are stored only in database as BLOBs
        
        # Prepare log data with numpy arrays and images
        log_data = {
            'image_filename': uploaded_file.name,
            'patient_name': patient_name if patient_name and not patient_name.isspace() else '',
            'timestamp': datetime.now().isoformat(),
            'face_array': face_image,  # Store numpy array directly
            'annotated_image': annotated_image,  # Store image directly
            'clear_skin_%': float(features.get('clear_face', 0).item() if hasattr(features.get('clear_face', 0), 'item') else features.get('clear_face', 0)) * 100,
            'dark_spots_%': float(features.get('darkspots', 0).item() if hasattr(features.get('darkspots', 0), 'item') else features.get('darkspots', 0)) * 100,
            'puffy_eyes_%': float(features.get('puffy_eyes', 0).item() if hasattr(features.get('puffy_eyes', 0), 'item') else features.get('puffy_eyes', 0)) * 100,
            'wrinkles_%': float(features.get('wrinkles', 0).item() if hasattr(features.get('wrinkles', 0), 'item') else features.get('wrinkles', 0)) * 100,
            'skin_age': float(age.item() if hasattr(age, 'item') else age)
        }

        # Append to database
        if append_log(user_id, log_data):
            st.success("Analysis complete and results saved!")
            st.balloons()

    # Analysis History
    st.markdown("---")
    st.header("Your Analysis History")

    logs = get_user_logs(user_id, limit=10)
    
    if logs:
        st.info(f"Showing your last {len(logs)} analyses")
        for log in logs:
            timestamp = pd.to_datetime(log['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            display_name = f"{log['patient_name']} - " if log['patient_name'] else ""
            
            with st.expander(
                f"{display_name}{timestamp} - `{log['image_filename']}`",
                expanded=False
            ):
                col1, col2 = st.columns([1, 2])
                with col1:
                    # Decode image from BLOB
                    try:
                        annotated_img = blob_to_image(log['annotated_image'])
                        if annotated_img is not None:
                            st.image(
                                cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),
                                caption="Analyzed Image"
                            )
                        else:
                            st.warning("Could not decode image")
                    except Exception as e:
                        st.warning(f"Error loading image: {e}")
                
                with col2:
                    if log['patient_name']:
                        st.write(f"**Patient:** {log['patient_name']}")
                    st.metric("Skin Age", f"{log['skin_age']:.1f} years")
                    st.write(f"**Clear Skin:** {log['clear_skin_percent']:.1f}%")
                    st.write(f"**Dark Spots:** {log['dark_spots_percent']:.1f}%")
                    st.write(f"**Puffy Eyes:** {log['puffy_eyes_percent']:.1f}%")
                    st.write(f"**Wrinkles:** {log['wrinkles_percent']:.1f}%")
                    
                    # Download button for this specific analysis
                    st.markdown("---")
                    success, img_encoded = cv2.imencode(".png", annotated_img)
                    if success:
                        st.download_button(
                            "Download This Image",
                            img_encoded.tobytes(),
                            f"analysis_{log['id']}_{timestamp.replace(':', '-').replace(' ', '_')}.png",
                            "image/png",
                            key=f"download_{log['id']}"
                        )
    else:
        st.info("No analysis history yet. Upload an image to get started!")

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="DermalScan - AI Skin Analysis",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    apply_custom_css()
    initialize_database()

    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.user_id = None
    
    # Route to appropriate page
    if st.session_state.logged_in:
        show_dashboard()
    else:
        show_auth_page()

if __name__ == "__main__":
    main()