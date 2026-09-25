# Face Recognition Attendance System

A Streamlit web app that registers people by face and marks attendance from a live webcam feed. Face embeddings come from InsightFace and are stored and matched in Redis.

## What it does

- **Register**: enter a name and role (Student or Teacher), look at the camera for a few seconds, and the app collects face embeddings, averages them, and saves the result to Redis.
- **Real-time attendance**: the live camera feed is compared against every registered face. Matches are logged with name, role, and timestamp. Logs are flushed to Redis every 60 seconds.
- **Report**: view registered people, raw logs, and an attendance report. Records can be deleted from here.

## How matching works

1. InsightFace (`buffalo_sc` model, ONNX runtime) detects faces and returns a 512-dimension embedding per face.
2. Embeddings of registered people are loaded from a Redis hash into a DataFrame.
3. Each live embedding is compared with cosine similarity. The best match above a 0.5 threshold is accepted, anything below is shown as Unknown.

## Stack

| Layer | Tech |
| --- | --- |
| UI | Streamlit, streamlit-webrtc (browser camera stream) |
| Face recognition | InsightFace `buffalo_sc`, onnxruntime, OpenCV |
| Matching | scikit-learn cosine similarity, pandas, numpy |
| Storage | Redis (hash for registrations, list for logs) |
| Deployment | AWS EC2, Apache reverse proxy with SSL (`configure.sh`) |

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` in the project root:

```
REDIS_HOST=
REDIS_PORT=
REDIS_PASSWORD=
```

Start the app:

```bash
streamlit run Home.py --server.fileWatcherType none
```

The `fileWatcherType none` flag avoids a known conflict between Streamlit's file watcher and torch.

## Deploy on a server

`main.sh` writes an Apache virtual host that redirects HTTP to HTTPS and proxies both the page and the WebSocket endpoint (`/_stcore`) to Streamlit on port 8501, then starts the app. Replace `<domain or ip address>` in `configure.sh` and point the SSL lines at your certificate before running it.

## Redis keys

| Key | Type | Content |
| --- | --- | --- |
| `academy:register` | hash | `name@role` → averaged face embedding (bytes) |
| `attendence:logs` | list | `name@role@timestamp` per detection |

## Project layout

```
Home.py                       entry page, loads the model and connects to Redis
face_rec.py                   embedding, matching, registration and logging logic
pages/1_Real_Time_Prediction.py
pages/2_Registration_Form.py
pages/3_Report.py
insightFace_models/           buffalo_sc model files
configure.sh, main.sh         server setup and start scripts
requirements.txt
```

## Limitations

- Single Redis hash means registrations are loaded fully into memory on each page load. Fine for a class or office, not for thousands of people.
- Threshold 0.5 was chosen by testing on a small group. Tune it for your lighting and camera.
