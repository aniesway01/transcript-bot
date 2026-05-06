"""
What: Video transcript pipeline for GitHub Actions
Why: Download bilibili/youtube audio, transcribe, summarize, email result
When: Triggered by GitHub Actions workflow via repository_dispatch
How: yt-dlp -> Groq Whisper ASR -> NVIDIA DeepSeek summary -> QQ SMTP email
"""
import json
import logging
import os
import smtplib
import subprocess
import tempfile
import time
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from urllib.parse import urlparse

import requests

log = logging.getLogger("pipeline")

SMTP_HOST = "smtp.qq.com"
SMTP_PORT = 465
SENDER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
RECIPIENT = os.environ.get("RECIPIENT_EMAIL", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")


def download_and_get_metadata(url: str, output_dir: Path) -> tuple:
    """Download audio and extract metadata in a single yt-dlp call.
    Returns (audio_path, metadata_dict).
    """
    log.info("[DL] Downloading audio + metadata...")
    output_path = output_dir / "audio.mp3"
    cmd = [
        "yt-dlp", "-x", "--audio-format", "mp3",
        "--audio-quality", "5", "--print-json",
        "-o", str(output_path.with_suffix(".%(ext)s")),
    ]
    cookie_str = os.environ.get("BILIBILI_COOKIES", "").lstrip("﻿").strip()
    if cookie_str and "bilibili" in url:
        cookie_file = output_dir / "cookies.txt"
        cookie_file.write_text(cookie_str, encoding="utf-8")
        cmd.extend(["--cookies", str(cookie_file)])
    cmd.append(url)
    result = subprocess.run(
        cmd,
        capture_output=True, timeout=1800, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr[:500]}")
    data = json.loads(result.stdout)
    metadata = {
        "title": data.get("title", "unknown"),
        "upload_date": data.get("upload_date", "00000000"),
        "duration": data.get("duration", 0),
        "uploader": data.get("uploader", "unknown"),
        "webpage_url": data.get("webpage_url", url),
    }
    for ext in [".mp3", ".m4a", ".wav", ".opus"]:
        candidate = output_path.with_suffix(ext)
        if candidate.exists():
            return candidate, metadata
    raise FileNotFoundError(f"No audio file found in {output_dir}")


def transcribe_audio(audio_path: Path) -> dict:
    file_size = audio_path.stat().st_size
    log.info(f"[ASR] Audio: {file_size / 1024 / 1024:.1f} MB")
    if file_size > 25 * 1024 * 1024:
        return _transcribe_chunked(audio_path)
    return _call_groq_whisper(audio_path)


def _call_groq_whisper(audio_path: Path, language: str = "zh", max_retries: int = 3) -> dict:
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    for attempt in range(max_retries):
        with open(audio_path, "rb") as f:
            files = {"file": (audio_path.name, f, "audio/mpeg")}
            data = {
                "model": "whisper-large-v3-turbo",
                "language": language,
                "response_format": "verbose_json",
                "temperature": 0,
                "prompt": "This is a Chinese speech.",
            }
            log.info(f"[ASR] Sending to Groq Whisper (attempt {attempt + 1})...")
            start = time.time()
            resp = requests.post(url, headers=headers, files=files, data=data, timeout=300)
        elapsed = time.time() - start
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 30 * (attempt + 1)))
            log.warning(f"[ASR] Rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue
        if resp.status_code != 200:
            raise RuntimeError(f"Groq API error {resp.status_code}: {resp.text[:300]}")
        result = resp.json()
        segments = [
            {"start": s.get("start", 0), "end": s.get("end", 0), "text": s.get("text", "")}
            for s in result.get("segments", [])
        ]
        log.info(f"[ASR] Done in {elapsed:.1f}s, {len(result.get('text', ''))} chars")
        return {"text": result.get("text", ""), "segments": segments}
    raise RuntimeError("Groq API rate limit exceeded after retries")


def _transcribe_chunked(audio_path: Path) -> dict:
    chunk_dir = audio_path.parent / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    ffmpeg_result = subprocess.run(
        [
            "ffmpeg", "-i", str(audio_path), "-f", "segment",
            "-segment_time", "900", "-c:a", "libmp3lame", "-q:a", "5",
            str(chunk_dir / "chunk_%03d.mp3"),
        ],
        capture_output=True, timeout=300
    )
    if ffmpeg_result.returncode != 0:
        raise RuntimeError(f"ffmpeg chunking failed: {ffmpeg_result.stderr[:300]}")
    chunks = sorted(chunk_dir.glob("chunk_*.mp3"))
    if not chunks:
        raise RuntimeError("ffmpeg chunking produced no files")
    all_segments, all_text, time_offset = [], [], 0.0
    for chunk in chunks:
        result = _call_groq_whisper(chunk)
        all_text.append(result["text"])
        for seg in result["segments"]:
            seg["start"] += time_offset
            seg["end"] += time_offset
            all_segments.append(seg)
        if result["segments"]:
            time_offset = result["segments"][-1]["end"]
    return {"text": " ".join(all_text), "segments": all_segments}


def generate_summary(transcript_text: str, title: str) -> str:
    log.info("[LLM] Generating summary with NVIDIA DeepSeek V4 Flash...")
    prompt = f"""You are a content analyst. Analyze the following transcript of a video titled "{title}".

Please provide your response entirely in Chinese:
1. A structured summary (3-5 paragraphs covering the main points)
2. A timeline with key topics and timestamps

Format your response EXACTLY like this:

## Summary

(3-5 paragraphs in Chinese summarizing the key points)

## Timeline

- 00:00 - Topic description
- MM:SS - Topic description

Here is the transcript:

{transcript_text[:80000]}
"""
    if len(transcript_text) > 80000:
        log.warning(f"[LLM] Transcript truncated from {len(transcript_text)} to 80000 chars")
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-ai/deepseek-v4-flash",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 4000,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=180)
    if resp.status_code == 200:
        text = resp.json()["choices"][0]["message"]["content"]
        log.info(f"[LLM] Summary generated, {len(text)} chars")
        return text

    log.warning(f"[LLM] DeepSeek failed ({resp.status_code}), trying llama-3.3-70b fallback...")
    payload["model"] = "meta/llama-3.3-70b-instruct"
    resp = requests.post(url, headers=headers, json=payload, timeout=180)
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]

    log.error(f"[LLM] All providers failed: {resp.status_code}")
    return "## Summary\n\n(Summary generation failed)\n\n## Timeline\n\n(Timeline generation failed)"


def fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def assemble_markdown(metadata: dict, summary: str, segments: list) -> str:
    title = metadata["title"]
    date = metadata["upload_date"]
    date_fmt = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    dur_m, dur_s = divmod(int(metadata["duration"]), 60)
    lines = [
        f"# {title}",
        "",
        "## Metadata",
        f"- **URL**: {metadata['webpage_url']}",
        f"- **UP**: {metadata['uploader']}",
        f"- **Date**: {date_fmt}",
        f"- **Duration**: {dur_m}:{dur_s:02d}",
        "",
        summary,
        "",
        "## Transcript",
        "",
    ]
    for seg in segments:
        text = seg["text"].strip()
        if text:
            lines.append(f"[{fmt_ts(seg['start'])}] {text}")
    lines.append("")
    return "\n".join(lines)


def send_email(subject: str, body_text: str, attachment_name: str, attachment_bytes: bytes):
    log.info(f"[EMAIL] Sending to {RECIPIENT}...")
    msg = MIMEMultipart()
    msg["From"] = SENDER
    msg["To"] = RECIPIENT
    msg["Subject"] = subject
    msg.attach(MIMEText(body_text, "plain", "utf-8"))
    att = MIMEApplication(attachment_bytes, Name=attachment_name)
    att["Content-Disposition"] = f'attachment; filename="{attachment_name}"'
    msg.attach(att)
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
        server.login(SENDER, SMTP_PASSWORD)
        server.send_message(msg)
    log.info("[EMAIL] Sent successfully")


def sanitize_filename(name: str) -> str:
    for ch in '<>:"/\\|?*':
        name = name.replace(ch, "_")
    return name.strip()[:80]


def validate_url(url: str):
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"URL must use https, got: {parsed.scheme}")
    allowed = {"www.bilibili.com", "bilibili.com", "www.youtube.com", "youtube.com", "youtu.be", "m.bilibili.com"}
    if parsed.hostname not in allowed:
        raise ValueError(f"Unsupported host: {parsed.hostname}")


def run(url: str):
    validate_url(url)
    for var in ("SMTP_USER", "SMTP_PASSWORD", "RECIPIENT_EMAIL", "GROQ_API_KEY", "NVIDIA_API_KEY"):
        if not os.environ.get(var):
            raise RuntimeError(f"Missing required env var: {var}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        audio_path, metadata = download_and_get_metadata(url, Path(tmp_dir))
        asr_result = transcribe_audio(audio_path)

    title = metadata["title"]
    uploader = metadata["uploader"]
    dur_m, dur_s = divmod(int(metadata["duration"]), 60)
    log.info(f"[META] {title} ({uploader}, {dur_m}:{dur_s:02d})")

    summary = generate_summary(asr_result["text"], title)
    md_content = assemble_markdown(metadata, summary, asr_result["segments"])

    date = metadata["upload_date"]
    date_fmt = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
    safe_title = sanitize_filename(title)
    filename = f"{date_fmt}-{safe_title}.md"

    email_subject = f"[Transcript] {title} - {uploader} {date_fmt}"
    email_body = (
        f"Video: {title}\n"
        f"URL: {metadata['webpage_url']}\n"
        f"Duration: {dur_m}:{dur_s:02d} | UP: {uploader}\n\n"
        f"Full transcript (with summary and timeline) is attached."
    )
    send_email(email_subject, email_body, filename, md_content.encode("utf-8"))
    log.info(f"[OK] Done: {filename}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    url = os.environ.get("VIDEO_URL", "")
    if not url:
        raise RuntimeError("VIDEO_URL env var is required")
    run(url)


if __name__ == "__main__":
    main()
