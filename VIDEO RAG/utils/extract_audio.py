from moviepy import VideoFileClip
from faster_whisper import WhisperModel
from langchain_core.documents import Document
import os

from utils.constants import AUDIO_PATH,VIDEO_PATH

os.makedirs(AUDIO_PATH, exist_ok=True)

model = WhisperModel("base", device="cpu", compute_type="int8")

def extract_audio(video_path, audio_path):
    """Extract audio from a video file and save as .wav"""
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)
    video.close()
    print(f"Audio extracted: {audio_path}")


def transcribe_and_chunk(chunk_duration=60):
    chunks = []
    for file in os.listdir(VIDEO_PATH):
        base_name = os.path.splitext(file)[0]
        video_path = os.path.join(VIDEO_PATH, file)
        audio_path = os.path.join(AUDIO_PATH, base_name + ".wav")
        extract_audio(video_path,audio_path)
        print(f"Transcribing: {audio_path}")

        segments, info = model.transcribe(
            audio_path,
            vad_filter=True
        )

        current_chunk = None

        for segment in segments:

            seg_start = segment.start
            seg_end = segment.end
            seg_text = segment.text.strip()

            if current_chunk is None:
                current_chunk = {
                    "video_content": seg_text,
                    "metadata": {
                        "start": seg_start,
                        "end": seg_end,
                        "audio_path": audio_path,
                        "chunk_id": 1
                    }
                }
                continue

            # Same 60-second window
            if seg_end - current_chunk["metadata"]["start"] <= chunk_duration:
                current_chunk["metadata"]["end"] = seg_end
                current_chunk["video_content"] += " " + seg_text
            # Create a new chunk
            else:
                chunks.append(current_chunk)
                current_chunk = {
                    "video_content": seg_text,
                    "metadata": {
                        "start": seg_start,
                        "end": seg_end,
                        "audio_path": audio_path,
                        "chunk_id": len(chunks) + 1
                    }
                }
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
    return chunks
