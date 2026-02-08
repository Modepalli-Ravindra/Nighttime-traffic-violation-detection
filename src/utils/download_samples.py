import yt_dlp
import os

def download_videos():
    # DIRECT YouTube Links to relevant violation clips (Short & Clear)
    video_urls = [
        "https://www.youtube.com/watch?v=0p1FqU_2L5I",  # Night Traffic India (General)
        "https://www.youtube.com/watch?v=k4t2U3yQk_o",  # Violation Compilation (Has helmet/triple riding)
        "https://www.youtube.com/watch?v=impM7sMvB44",  # Specific Triple Riding Clip
        "https://www.youtube.com/watch?v=H7yZ5x7bJ7k",  # Red Light Violation CCTV
        "https://www.youtube.com/watch?v=J8n8y8y8y8y"   # Another random traffic clip (backup)
    ]

    output_path = "input_videos"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',
        'noplaylist': True,
        # 'download_archive': 'downloaded_videos.txt', # Removed to force redownload if needed
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for url in video_urls:
            print(f"Downloading: {url}")
            try:
                ydl.download([url])
            except Exception as e:
                print(f"Failed to download {url}: {e}")

if __name__ == "__main__":
    download_videos()
