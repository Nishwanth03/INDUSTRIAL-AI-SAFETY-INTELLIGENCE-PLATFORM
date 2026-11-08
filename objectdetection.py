import streamlit as st
from streamlit_option_menu import option_menu
import cv2
import tempfile
import os
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import json
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from datetime import datetime, timezone
from ppe_detection import (
    ISSUE_LABELS,
    MODEL_REPOSITORY,
    MODEL_SHA256,
    MODEL_FILENAME,
    PPEAlertValidator,
    PPEDetector,
    download_ppe_model,
)
from safety_guidance import get_worker_guidance

# Streamlit page config
st.set_page_config(page_title="Industrial Safety Project Guide", layout="wide")

# Sidebar Option Menu
with st.sidebar:
    selected = option_menu(
        menu_title="Industrial Safety",
        options=["Project Guide", "PPE Setup & Alerts", "Worker Guide", "About", "Video Detection", "Live Detection", "Objects Detected", "Graphs"],
        icons=["list-check", "shield-check", "shield-check", "info-circle", "camera-video", "camera-reels", "clipboard-data", "bar-chart"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "5px"},
            "icon": {"color": "white", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px",
                "--hover-color": "crimson",
            },
            "nav-link-selected": {"background-color": "#02ab21", "color": "white"},
        },
    )

PPE_MODEL_PATH = Path(__file__).resolve().parent / "models" / MODEL_FILENAME
st.session_state.setdefault("ppe_incidents", [])

def get_aws_settings():
    settings = {}
    secrets_paths = [
        Path(".streamlit") / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    ]
    if any(path.is_file() for path in secrets_paths):
        settings = dict(st.secrets.get("aws", {}))

    environment_settings = {
        "region_name": os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"),
        "sns_topic_arn": os.getenv("SNS_TOPIC_ARN"),
        "bucket_name": os.getenv("S3_BUCKET_NAME"),
    }
    return {key: value or settings.get(key) for key, value in environment_settings.items()}

@st.cache_resource
def load_ppe_detector(weights_path):
    return PPEDetector(Path(weights_path))

def send_sns_alert(incident):
    settings = get_aws_settings()
    topic_arn = settings["sns_topic_arn"]
    if not topic_arn:
        return False
    guidance = get_worker_guidance(incident["event_type"])
    message = (
        "PPE safety alert - human review required\n"
        f"Incident: {incident['incident_id']}\n"
        f"Source: {incident['source_id']}\n"
        f"Event: {incident['label']}\n"
        f"Confidence: {incident['confidence']:.0%}\n"
        f"Detected at: {incident['detected_at']}\n"
        f"Suggested action: {guidance}\n"
        "This AI alert is not a safety certification or automatic control."
    )
    try:
        client = boto3.client("sns", region_name=settings["region_name"])
        client.publish(
            TopicArn=topic_arn,
            Message=message,
            Subject=f"Industrial PPE alert: {incident['label']}",
        )
        return True
    except (BotoCoreError, ClientError) as exc:
        st.error(f"SNS alert failed: {exc}")
        return False

def record_ppe_incident(incident):
    record = incident.to_dict()
    st.session_state["ppe_incidents"].insert(0, record)
    del st.session_state["ppe_incidents"][200:]
    st.warning(f"{record['label']} — {record['confidence']:.0%} confidence. Human review required.")
    st.write(f"Suggested action: {get_worker_guidance(record['event_type'])}")
    if get_aws_settings()["sns_topic_arn"]:
        if send_sns_alert(record):
            st.success("SNS notification sent.")

def show_recent_ppe_incidents():
    incidents = st.session_state["ppe_incidents"]
    st.subheader("Recent PPE alerts")
    if incidents:
        st.dataframe(
            pd.DataFrame(incidents)[
                ["detected_at", "source_id", "label", "confidence", "incident_id"]
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No validated PPE alerts have been generated in this session.")

def analyze_ppe_capture(cap, detector, source_id, confidence, consecutive_frames, cooldown_seconds, duration_seconds=None):
    if not cap.isOpened():
        cap.release()
        st.error("Could not open the selected video or camera.")
        return

    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = st.progress(0.0)
    frame_view = st.empty()
    validators = st.session_state.setdefault("ppe_validators", {})
    validator_key = (source_id, consecutive_frames, cooldown_seconds)
    if validator_key not in validators:
        validators[validator_key] = PPEAlertValidator(
            consecutive_frames=consecutive_frames,
            cooldown_seconds=cooldown_seconds,
        )
    validator = validators[validator_key]
    processed = 0
    detected_objects_dict = {}
    started_at = time.monotonic()
    try:
        while True:
            if duration_seconds is not None and time.monotonic() - started_at >= duration_seconds:
                break
            ok, frame = cap.read()
            if not ok:
                break
            processed += 1
            detections = detector.detect(frame, confidence)
            for detection in detections:
                detected_objects_dict.setdefault(detection.label, []).append(
                    detection.confidence * 100
                )
                x1, y1, x2, y2 = detection.box
                color = (0, 0, 255) if detection.issue_type else (0, 180, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{detection.label} {detection.confidence:.0%}",
                    (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
            frame_view.image(frame, channels="BGR", use_container_width=True)
            incidents = validator.update(
                detections,
                source_id,
                now=datetime.now(timezone.utc),
            )
            for incident in incidents:
                record_ppe_incident(incident)
            if duration_seconds is not None:
                progress.progress(min((time.monotonic() - started_at) / duration_seconds, 1.0))
            elif frame_total > 0:
                progress.progress(min(processed / frame_total, 1.0))
    finally:
        cap.release()
    st.session_state["detected_data"] = detected_objects_dict
    st.success(f"Processed {processed} frames from {source_id}.")
    if get_aws_settings()["bucket_name"]:
        upload_to_s3(detected_objects_dict)

# Upload to S3
def upload_to_s3(detected_objects_dict):
    settings = get_aws_settings()
    bucket_name = settings["bucket_name"]
    if not bucket_name:
        st.info("S3 uploads are disabled. Configure S3_BUCKET_NAME to enable them.")
        return

    output_data_path = os.path.join(tempfile.gettempdir(), "detected_objects.json")
    with open(output_data_path, 'w') as f:
        json.dump(detected_objects_dict, f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    s3_file_name = f"detected_objects_{timestamp}.json"

    try:
        s3 = boto3.client("s3", region_name=settings["region_name"])
        with open(output_data_path, "rb") as f:
            s3.upload_fileobj(f, bucket_name, s3_file_name)
        st.success(f"✅ Detected objects data uploaded to S3 as: `{s3_file_name}`")
    except (BotoCoreError, ClientError, OSError) as exc:
        st.error(f"❌ Failed to upload detected objects data to S3: {exc}")



if selected == "Project Guide":
    st.title("Industrial Safety Project Guide")
    st.write(
        "Use this checklist to set up the existing project and build toward an "
        "industrial PPE monitoring prototype. Complete the steps in order."
    )

    with st.expander("1. Prepare your computer", expanded=True):
        st.markdown(
            "- Install Python 3.11 and Git.\n"
            "- Open PowerShell in the cloned project folder.\n"
            "- Create and activate a virtual environment using the README commands."
        )
    with st.expander("2. Install and start the current prototype"):
        st.markdown(
            "- Install the packages listed in `requirements.txt`.\n"
            "- Run `streamlit run objectdetection.py`.\n"
            "- Open **PPE Setup & Alerts** and click **Download PPE model** once.\n"
            "- Try **Video Detection** with `uk.mp4`; webcam access is optional."
        )
    with st.expander("3. Know what works and what does not"):
        st.warning(
            "The downloaded construction PPE model can flag helmet, vest, and boots-related "
            "classes. Results are suggestions for human review, not safety determinations."
        )
        st.markdown(
            "- The publisher reports mAP50 around 0.76 and recall around 0.72; these results "
            "are not independently verified and are not guarantees for your cameras.\n"
            "- AWS SNS is optional. Safety-shoe/boot visibility, worker tracking, restricted "
            "zones, and incident database features require further validation/development.\n"
            "- Do not use this prototype as a certified safety system or as a replacement "
            "for site procedures or trained supervision."
        )
    with st.expander("4. Build the industrial features in phases"):
        st.markdown(
            "1. Download the model and test an uploaded video.\n"
            "2. Review detections against your actual camera views and PPE policy.\n"
            "3. Configure SNS and test an alert if you have AWS access.\n"
            "4. Collect site-specific evaluation data before relying on alerts.\n"
            "5. Add worker association, restricted zones, persistent incident management, "
            "and future read-only OPC UA integration."
        )
    st.info(
        "Next recommended action: open PPE Setup & Alerts, download the model, then run "
        "Video Detection against the included sample video."
    )

elif selected == "PPE Setup & Alerts":
    st.title("PPE Model Setup & Alerts")
    st.write(
        "This prototype uses an externally published construction PPE checkpoint; "
        "you do not need to train a model or provide a dataset to try it."
    )
    st.warning(
        "AI detections can be wrong or miss hazards. Require a trained person to review "
        "every alert. Do not use this prototype for discipline, automatic machinery "
        "control, emergency shutdown, or as proof of PPE compliance."
    )
    st.markdown(
        "**Model:** [SafeSight Construction PPE YOLOv8](https://huggingface.co/killuminati1/construction-ppe-yolov8)  \n"
        "**Checkpoint license declared by publisher:** Apache-2.0. Review the model card "
        "and underlying dataset terms before redistribution or commercial deployment.  \n"
        "**Publisher-reported metrics:** approximately 0.76 mAP50, 0.43 mAP50-95, "
        "0.81 precision, and 0.72 recall. These are not independently verified and "
        "performance has not been validated on your cameras."
    )
    st.markdown(
        "The model includes explicit `No-Helmet`, `No-Vest`, and `No-Boots` classes. "
        "Its boot classes are not a certification that the footwear is industrial "
        "safety-rated. The app does not identify individual workers."
    )

    if PPE_MODEL_PATH.is_file():
        st.success(f"PPE model file is available: `{PPE_MODEL_PATH.relative_to(Path(__file__).resolve().parent)}`")
        if st.button("Load and verify PPE model"):
            detector = load_ppe_detector(str(PPE_MODEL_PATH))
            st.success(
                "Loaded PPE model. Supported missing-PPE alerts: "
                + ", ".join(sorted(ISSUE_LABELS[issue] for issue in detector.supported_issues))
            )
    else:
        st.info("PPE model is not downloaded yet. The checkpoint is about 22 MB.")
        if st.button("Download PPE model"):
            with st.spinner("Downloading and verifying the PPE checkpoint..."):
                try:
                    downloaded_path = download_ppe_model(PPE_MODEL_PATH)
                    st.success(f"Model downloaded and SHA-256 verified: `{downloaded_path}`")
                except RuntimeError as exc:
                    st.error(str(exc))
        st.caption(
            f"Model repository: {MODEL_REPOSITORY}. Expected SHA-256: `{MODEL_SHA256}`. "
            "Weights are stored locally under `models/` and are excluded from Git."
        )

    settings = get_aws_settings()
    st.subheader("Optional SNS notification setup")
    if settings["sns_topic_arn"]:
        st.success(f"SNS topic configured: `{settings['sns_topic_arn']}`")
        if st.button("Send test SNS notification"):
            try:
                boto3.client("sns", region_name=settings["region_name"]).publish(
                    TopicArn=settings["sns_topic_arn"],
                    Subject="Industrial PPE alert test",
                    Message="Test notification from the Industrial Safety Intelligence Platform.",
                )
                st.success("Test notification published. Check the subscribed email or phone.")
            except (BotoCoreError, ClientError) as exc:
                st.error(f"Could not publish the SNS test notification: {exc}")
    else:
        st.info("SNS is not configured. PPE detections still display locally; no cloud alert will be sent.")
    with st.expander("Configure SNS step by step"):
        st.markdown(
            "1. In AWS Console, create an SNS **Standard** topic and copy its ARN.\n"
            "2. Subscribe an email address or an SMS phone number in international E.164 format; "
            "confirm the email subscription if prompted.\n"
            "3. Configure AWS credentials locally with `aws configure` (or an IAM role/profile "
            "that can call `sns:Publish`). For a named profile use "
            "`aws configure --profile industrial-safety`, then set "
            "`$env:AWS_PROFILE = \"industrial-safety\"` in PowerShell before starting "
            "Streamlit. Never put access keys in source code.\n"
            "4. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and replace "
            "the example region and topic ARN. Alternatively, set `AWS_REGION` and "
            "`SNS_TOPIC_ARN` environment variables.\n"
            "5. Restart Streamlit, return here, and click **Send test SNS notification**. "
            "SNS SMS may incur charges and must follow AWS regional rules."
        )
    st.caption(
        "Automatic SNS notifications are sent only after the configured number of "
        "consecutive frames confirms a missing-PPE class. Continuous violations are "
        "deduplicated; a cooldown applies after the violation clears."
    )

elif selected == "Worker Guide":
    st.title("Worker Safety Guide")
    st.warning(
        "Guidance is shown for AI-generated prototype alerts, not verified compliance. "
        "Follow site procedures and ask a trained supervisor to confirm."
    )
    guidance = {
        "Helmet not detected": "helmet_missing",
        "Safety vest not detected": "vest_missing",
        "Safety shoes not detected": "shoes_missing",
        "Person in a restricted area": "restricted_entry",
        "PPE status uncertain": "uncertain",
    }
    for label, issue in guidance.items():
        with st.expander(label):
            st.write(get_worker_guidance(issue))
    show_recent_ppe_incidents()
    st.caption(
        "Always follow your site's safety procedures and supervisor instructions. "
        "Do not rely on this prototype for emergency response or safety-critical decisions."
    )

elif selected == "About":
    st.markdown("""
        <style>
            body {
                background-color: black;
                color: white;
            }
            .animated-heading {
                font-size: 32px;
                font-weight: bold;
                color: #02ab21;
                text-align: center;
                margin-top: 20px;
                margin-bottom: 30px;
                animation: fadeSlideIn 2s ease-in-out;
            }

            @keyframes fadeSlideIn {
                0% { opacity: 0; transform: translateY(-30px); }
                100% { opacity: 1; transform: translateY(0); }
            }

            .about-container {
                background-color: #f4f9f9;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                text-align: center;
                animation: fadeIn 1.5s ease-in-out;
                margin-top: 20px;
                margin-bottom: 20px;
            }

            @keyframes fadeIn {
                from { opacity: 0; transform: scale(0.95); }
                to { opacity: 1; transform: scale(1); }
            }

            .about-container p {
                font-size: 18px;
                color: #333;
                line-height: 1.6;
                margin-top: 20px;
                margin-bottom: 30px; /* Ensure space below text */
            }

            .tech-button {
                display: inline-block;
                background-color: crimson;
                color: white;
                padding: 4px 12px;
                border-radius: 15px;
                font-size: 15px;
                font-weight: 500;
                margin: 8px; /* Spacing between buttons */
                box-shadow: 0 3px 6px rgba(0,0,0,0.1);
            }

            .tech-buttons-container {
                display: flex;
                justify-content: center;
                flex-wrap: wrap;
                gap: 10px;  /* Spacing between buttons */
                margin-top: 20px;
            }
        </style>

        <div class="animated-heading">Industrial PPE Detection Prototype</div>

        <div class="about-container">
            <p>
                Review helmet, vest, and boots-related model detections in video<br>
            </p>
            <div class="tech-buttons-container">
                <span class="tech-button">YOLOv8</span>
                <span class="tech-button">OpenCV</span><br>
            </div>
            <p> AI findings are prototype alerts for human review, not confirmed PPE compliance.
                SNS alerts are optional and must be configured by the operator. </p>
        </div>
    """, unsafe_allow_html=True)


elif selected == "Video Detection":
    st.title("PPE Detection in Uploaded Video")
    confidence = st.sidebar.slider(
        "PPE detection confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
        help="Higher thresholds reduce weak detections but may miss PPE violations.",
    )
    consecutive_frames = st.sidebar.slider(
        "Frames required before alert",
        min_value=1,
        max_value=20,
        value=5,
    )
    cooldown_seconds = st.sidebar.number_input(
        "Alert cooldown (seconds)",
        min_value=0,
        max_value=3600,
        value=300,
        step=30,
    )
    uploaded_file = st.sidebar.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="ppe_video_upload",
    )
    if not PPE_MODEL_PATH.is_file():
        st.warning("Open PPE Setup & Alerts and download the PPE model before analyzing video.")
    elif uploaded_file and st.button("Analyze uploaded video"):
        detector = load_ppe_detector(str(PPE_MODEL_PATH))
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=Path(uploaded_file.name).suffix,
                delete=False,
            ) as temporary_file:
                temporary_file.write(uploaded_file.getvalue())
                temporary_path = Path(temporary_file.name)
            capture = cv2.VideoCapture(str(temporary_path))
            analyze_ppe_capture(
                capture,
                detector,
                source_id=uploaded_file.name,
                confidence=confidence,
                consecutive_frames=consecutive_frames,
                cooldown_seconds=int(cooldown_seconds),
            )
        finally:
            if temporary_path and temporary_path.exists():
                temporary_path.unlink()
    st.caption(
        "Red boxes indicate model classes that explicitly predict missing PPE. "
        "Each alert still requires human review."
    )
    show_recent_ppe_incidents()

elif selected == "Live Detection":
    st.title("PPE Detection from Local Webcam")
    st.info("The webcam must be connected to the computer running this Streamlit app.")
    capture_seconds = st.number_input(
        "Capture duration (seconds)",
        min_value=5,
        max_value=60,
        value=10,
        step=5,
    )
    confidence = st.sidebar.slider(
        "PPE detection confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
        key="live_ppe_confidence",
    )
    consecutive_frames = st.sidebar.slider(
        "Frames required before alert",
        min_value=1,
        max_value=20,
        value=5,
        key="live_ppe_consecutive",
    )
    cooldown_seconds = st.sidebar.number_input(
        "Alert cooldown (seconds)",
        min_value=0,
        max_value=3600,
        value=300,
        step=30,
        key="live_ppe_cooldown",
    )
    if not PPE_MODEL_PATH.is_file():
        st.warning("Open PPE Setup & Alerts and download the PPE model before using the webcam.")
    elif st.button("Analyze webcam for selected duration"):
        detector = load_ppe_detector(str(PPE_MODEL_PATH))
        cap = cv2.VideoCapture(0)
        analyze_ppe_capture(
            cap,
            detector,
            source_id="local-webcam",
            confidence=confidence,
            consecutive_frames=consecutive_frames,
            cooldown_seconds=int(cooldown_seconds),
            duration_seconds=int(capture_seconds),
        )
    show_recent_ppe_incidents()

# Objects Detected Section
elif selected == "Objects Detected":
    st.markdown("<h1 style='color:#02ab21;'>Detected Objects</h1>", unsafe_allow_html=True)
    detected_objects_dict = st.session_state.get("detected_data", {})

    if detected_objects_dict:
        sorted_objects = sorted(detected_objects_dict.items(), key=lambda x: -len(x[1]))
        for obj, acc_list in sorted_objects:
            avg_acc = round(sum(acc_list) / len(acc_list), 2)
            st.write(f"🔸 **{obj}** — Average Accuracy: {avg_acc}% (Detected {len(acc_list)} times)")
    else:
        st.info("Run the Video Detection section first to see detected objects.")

# Graphs Section
elif selected == "Graphs":
    st.markdown("<h1 style='color:#02ab21;'>Visualization of Objects</h1>", unsafe_allow_html=True)
    detected_objects_dict = st.session_state.get("detected_data", {})

    if detected_objects_dict:
        data = {
            "Object": list(detected_objects_dict.keys()),
            "Frequency": [len(acc_list) for acc_list in detected_objects_dict.values()],
            "Avg Accuracy": [round(sum(acc_list) / len(acc_list), 2) for acc_list in detected_objects_dict.values()]
        }
        df = pd.DataFrame(data)

        # Plotly
        import plotly.express as px
        st.subheader("📉 Plotly: Interactive Bar Chart")
        fig3 = px.bar(df, x='Object', y='Frequency', color='Avg Accuracy',
                      title="Plotly - Object Frequency and Accuracy", text='Frequency')
        fig3.update_layout(xaxis_title="Object", yaxis_title="Frequency", title_x=0.5)
        st.plotly_chart(fig3)

        # Altair
        import altair as alt
        st.subheader("📊 Altair: Accuracy Line Chart")
        chart = alt.Chart(df).mark_line(point=True).encode(
            x='Object',
            y='Avg Accuracy',
            tooltip=['Object', 'Avg Accuracy']
        ).properties(title="Altair - Average Accuracy Trend")
        st.altair_chart(chart, use_container_width=True)

        # Matplotlib
        st.subheader("📊 Matplotlib: Pie Chart")
        fig1, ax1 = plt.subplots()
        ax1.pie(df["Frequency"], labels=df["Object"], autopct='%1.1f%%', startangle=90)
        ax1.axis("equal")
        st.pyplot(fig1)

        # Seaborn
        import seaborn as sns
        st.subheader("📈 Seaborn: Barplot - Object Frequency")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Frequency", y="Object", data=df, ax=ax2, palette="viridis")
        ax2.set_title("Object Frequency by Detection Count")
        st.pyplot(fig2)

        st.subheader("🔢 Seaborn: Count Plot - Object Frequency")
        expanded_data = []
        for obj, confs in detected_objects_dict.items():
            for conf in confs:
                expanded_data.append({"Object": obj, "Confidence": conf})
        expanded_df = pd.DataFrame(expanded_data)

        fig_count, ax_count = plt.subplots(figsize=(10, 6))
        sns.countplot(data=expanded_df, y="Object", order=df.sort_values("Frequency", ascending=False)["Object"])
        ax_count.set_title("Detected Object Frequency (Raw Count)")
        st.pyplot(fig_count)

        st.subheader("📦 Seaborn: Box Plot - Confidence Score Distribution")
        fig_box, ax_box = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=expanded_df, x="Object", y="Confidence", palette="Set2")
        ax_box.set_title("Confidence Score Distribution by Object")
        ax_box.set_xticklabels(ax_box.get_xticklabels(), rotation=45)
        st.pyplot(fig_box)

    else:
        st.warning("No detection data available. Please upload and process a video first.")
