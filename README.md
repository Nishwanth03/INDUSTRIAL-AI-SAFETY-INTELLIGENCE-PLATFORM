# Industrial Safety Intelligence Platform (Prototype)

This application now supports PPE object detection for uploaded video and the
local webcam using a public construction-site checkpoint. You do not need to
train a model or provide a dataset to try it. **All results are suggestions for
human review, not proof of compliance or a safety-rated system.**

## Run on Windows

From PowerShell in this project folder:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run objectdetection.py
```

Open the local URL printed by Streamlit (normally <http://localhost:8501>).
Python 3.11 or newer is recommended. If the virtual environment already exists,
reuse it rather than creating it again.

## Download and test PPE detection

1. In the app, open **PPE Setup & Alerts**.
2. Click **Download PPE model**. The app downloads a pinned checkpoint from
   Hugging Face and verifies its SHA-256 before saving it under `models/`.
   Internet access is needed for this one-time download. Model weights are
   ignored by Git and are not committed.
3. Open **Video Detection**, upload the included `uk.mp4`, and click
   **Analyze uploaded video**. Review the class labels and red violation boxes.
4. Open **Objects Detected** and **Worker Guide** to review this session's
   detections and suggested actions.
5. To test a local webcam, open **Live Detection**, set the capture duration,
   and click **Analyze webcam for selected duration**. The webcam must be
   available to the computer running Streamlit.

The app starts with a 0.35 confidence threshold, five consecutive frames before
an alert, and a 300-second alert cooldown. Adjust the controls in the sidebar
when testing. These are prototype settings, not validated operating thresholds.

## PPE model and limitations

The default model is
[SafeSight Construction PPE YOLOv8](https://huggingface.co/killuminati1/construction-ppe-yolov8).
Its model card declares Apache-2.0 for the checkpoint and reports approximately
0.76 mAP50, 0.43 mAP50-95, 0.81 precision, and 0.72 recall. Those numbers are
publisher-reported, not independently verified; performance is not established
for this project's sample video, cameras, sites, lighting, or PPE rules.

The checkpoint has classes for `No-Helmet`, `No-Vest`, and `No-Boots`. The boot
class does not identify a safety certification or prove that footwear meets a
site's required standard. A missed detection is not evidence that a worker is
compliant. The model also detects objects independently; this prototype does
not associate every PPE item with a tracked individual worker.

Review the model card and the underlying training-data terms before commercial
use or redistribution. Ultralytics and the model checkpoint have their own
license obligations. The app is for evaluation and human-reviewed prototyping
only; never use it to automatically discipline workers, control machinery, or
replace safety procedures or trained inspectors.

## Configure AWS SNS alerts

SNS is optional. The app works locally and shows validated alerts without AWS.
To send email or SMS alerts:

1. In the AWS Console, create an SNS **Standard** topic and copy its topic ARN.
2. Add an email or SMS subscription. Confirm the email subscription if AWS
   sends a confirmation message. SMS recipients should use an international
   E.164 phone number; SMS availability and pricing vary by region.
3. Configure local AWS credentials with an AWS profile, IAM role, or the
   standard AWS credential chain. For a named local profile, run
   `aws configure --profile industrial-safety`, then set
   `$env:AWS_PROFILE = "industrial-safety"` in the same PowerShell session
   before starting Streamlit. The identity needs permission to call
   `sns:Publish` for the topic. Do not put access keys in Python files or commit
   credentials.
4. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and set
   your actual region and topic ARN:

   ```toml
   [aws]
   region_name = "ap-south-1"
   sns_topic_arn = "arn:aws:sns:ap-south-1:123456789012:industrial-safety-alerts"
   ```

   Alternatively, set `AWS_REGION` and `SNS_TOPIC_ARN` in the environment.
5. Restart the app, open **PPE Setup & Alerts**, and click **Send test SNS
   notification**. Check the subscribed inbox or phone.

Validated detection alerts include the source, event, confidence, timestamp,
incident ID, and suggested action. A test notification is a real SNS publish
and may incur AWS SMS charges. No notifications are sent if SNS is not
configured. Session incident history is currently in memory; it is not a
persistent incident database and PPE evidence is not uploaded to S3.

## Tests

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests
```

Tests cover PPE class mapping, consecutive-frame validation, cooldown, alert
deduplication, and worker guidance. They do not prove model accuracy or safety
performance on site footage.
