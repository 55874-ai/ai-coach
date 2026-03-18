import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import mediapipe as mp
import numpy as np
import av

# --- ส่วน Logic เดิมของคุณ (ยกมาวาง) ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def neck_lateral_angle(lm):
    # (ฟังก์ชันเดิมของคุณ...)
    ls, rs = lm[mp_pose.PoseLandmark.LEFT_SHOULDER], lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    le, re = lm[mp_pose.PoseLandmark.LEFT_EAR], lm[mp_pose.PoseLandmark.RIGHT_EAR]
    lh, rh = lm[mp_pose.PoseLandmark.LEFT_HIP], lm[mp_pose.PoseLandmark.RIGHT_HIP]
    sh_mid = np.array([(ls.x + rs.x)/2, (ls.y + rs.y)/2]); ear_mid = np.array([(le.x + re.x)/2, (le.y + re.y)/2])
    hip_mid = np.array([(lh.x + rh.x)/2, (lh.y + rh.y)/2]); trunk_vec = hip_mid - sh_mid; head_vec = ear_mid - sh_mid
    cosang = np.dot(trunk_vec, head_vec) / (np.linalg.norm(trunk_vec) * np.linalg.norm(head_vec) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

# --- ฟังก์ชันจัดการวิดีโอ ---
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    if res.pose_landmarks:
        # วาดเส้นจุด
        mp.solutions.drawing_utils.draw_landmarks(img, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # คำนวณมุม (ตัวอย่าง)
        angle = neck_lateral_angle(res.pose_landmarks.landmark)
        cv2.putText(img, f"Neck Angle: {int(angle)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI ของ Streamlit ---
st.title("AI Upper Body Coach 🏋️‍♂️")
st.write("ระบบตรวจจับท่าทางแบบ Real-time")

webrtc_streamer(key="example", video_frame_callback=video_frame_callback)
