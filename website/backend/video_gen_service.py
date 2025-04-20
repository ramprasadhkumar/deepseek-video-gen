# Service to manage model loading, video generation, and video retrieval
import threading
from pathlib import Path
from typing import Optional

class VideoGenService:
    def __init__(self, model_loader, video_dir: str = "generated_videos"):
        self.model = model_loader()
        self.video_dir = Path(video_dir)
        self.video_dir.mkdir(exist_ok=True)
        self.latest_video: Optional[Path] = None
        self.lock = threading.Lock()

    def generate_video(self, input_data: str) -> str:
        # Placeholder: Generate video using the model
        # Save video to self.video_dir
        with self.lock:
            video_path = self.video_dir / f"video_{len(list(self.video_dir.iterdir())) + 1}.mp4"
            with open(video_path, "wb") as f:
                f.write(b"FAKE_VIDEO_DATA")  # Replace with real video bytes
            self.latest_video = video_path
        return str(video_path)

    def get_latest_video(self):
        if self.latest_video and self.latest_video.exists():
            return self.latest_video, self.latest_video.name
        else:
            videos = sorted(self.video_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
            if videos:
                self.latest_video = videos[0]
                return videos[0], videos[0].name
        return None, None
