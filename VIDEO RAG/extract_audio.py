from moviepy import VideoFileClip
import whisper
import json
import os

# ── Directories ──────────────────────────────────────────────
VIDEO_DIR = r"VIDEO RAG\data\video"
AUDIO_DIR = r"VIDEO RAG\data\audio"
TRANSCRIPT_DIR = r"VIDEO RAG\data\transcripts"

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

# ── Load Whisper model once ───────────────────────────────────
print("Loading Whisper model...")
model = whisper.load_model("base")


def extract_audio(video_path, audio_path):
    """Extract audio from a video file and save as .wav"""
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)
    video.close()
    print(f"Audio extracted: {audio_path}")


def transcribe_and_chunk(audio_path, chunk_duration=60):
    """Transcribe audio and split into chunks of `chunk_duration` seconds"""
    print(f"Transcribing: {audio_path}")
    result = model.transcribe(audio_path, verbose=False)

    chunks = []
    current_chunk = {"start": 0, "end": 0, "text": ""}

    for segment in result["segments"]:
        seg_start = segment["start"]
        seg_end = segment["end"]
        seg_text = segment["text"].strip()

        if seg_end - current_chunk["start"] <= chunk_duration:
            current_chunk["end"] = seg_end
            current_chunk["text"] += " " + seg_text
        else:
            chunks.append(current_chunk)
            current_chunk = {"start": seg_start, "end": seg_end, "text": seg_text}

    chunks.append(current_chunk)  # Add the last chunk
    return chunks


# ── Main Pipeline ─────────────────────────────────────────────
for file in os.listdir(VIDEO_DIR):
    if file.endswith((".mp4", ".avi", ".mkv")):
        base_name = os.path.splitext(file)[0]

        video_path      = os.path.join(VIDEO_DIR, file)
        audio_path      = os.path.join(AUDIO_DIR, base_name + ".wav")
        transcript_path = os.path.join(TRANSCRIPT_DIR, base_name + ".json")

        # Step 1: Extract audio
        extract_audio(video_path, audio_path)

        # Step 2: Transcribe and chunk
        chunks = transcribe_and_chunk(audio_path)

        # Step 3: Save transcript
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=4, ensure_ascii=False)

        print(f"Transcript saved: {transcript_path} ({len(chunks)} chunks)\n")

print("All videos processed!")