from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/workspace")
HINDI_DIR = ROOT / "episode2-assets" / "hindi"
ASSET_DIR = ROOT / "episode2-assets" / "english"
OUT_DIR = ROOT / "episode2-final-renders"
THUMB_DIR = ROOT / "episode2-final-thumbnails"
MEDIA_DIR = ROOT / "media"
FPS = 30

LANDSCAPE_IMAGES = [
    "1793555866425.png",
    "1793555876969.png",
    "1793555883757.png",
    "1793555884089.png",
    "1793555888822.png",
    "1793555889404.png",
    "1793555894274.png",
    "1793555895498.png",
    "1793555897526.png",
    "1793555900893.png",
    "1793555901701.png",
    "1793555905915.png",
    "1793555906598.png",
    "1793555910686.png",
    "1793555911002.png",
    "1793555914234.png",
    "1793555918561.png",
    "1793555920631.png",
    "1793555924680.png",
]

LANDSCAPE_WEIGHTS = [
    0.9,
    1.0,
    0.95,
    0.9,
    1.0,
    0.95,
    0.9,
    0.8,
    0.9,
    1.0,
    1.0,
    1.0,
    0.95,
    0.95,
    0.95,
    1.0,
    1.0,
    1.05,
    1.1,
]

REEL_IMAGES = [
    "1793555888822.png",
    "1793555897526.png",
    "1793555900893.png",
    "1793555911002.png",
    "1793555914234.png",
    "1793555920631.png",
    "1793555924680.png",
]

REEL_WEIGHTS = [1.0, 1.0, 1.05, 0.9, 1.0, 1.0, 1.05]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def probe_duration(path: Path) -> float:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    ).strip()
    return float(out)


def detect_trim_points(path: Path, threshold_db: str = "-38dB", min_silence: float = 0.35) -> tuple[float, float]:
    duration = probe_duration(path)
    proc = subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(path),
            "-af",
            f"silencedetect=n={threshold_db}:d={min_silence}",
            "-f",
            "null",
            "-",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    stderr = proc.stderr
    start_match = re.search(r"silence_end: ([0-9.]+)", stderr)
    last_silence_start = None
    for match in re.finditer(r"silence_start: ([0-9.]+)", stderr):
        last_silence_start = float(match.group(1))

    start = float(start_match.group(1)) if start_match else 0.0
    end = last_silence_start if last_silence_start and last_silence_start > duration * 0.75 else duration

    start = max(0.0, start - 0.05)
    end = min(duration, end + 0.05)
    if end <= start:
        return 0.0, duration
    return start, end


def trim_audio(src: Path, dest: Path) -> None:
    start, end = detect_trim_points(src)
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-c:a",
            "libmp3lame",
            "-q:a",
            "2",
            str(dest),
        ]
    )


def make_audio_concat(inputs: list[Path], output: Path) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        for path in inputs:
            fh.write(f"file '{path.as_posix()}'\n")
        concat_file = fh.name
    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_file,
            "-c",
            "copy",
            str(output),
        ]
    )


def build_clip(image: Path, output: Path, width: int, height: int, duration: float, vertical: bool) -> None:
    frames = max(1, round(duration * FPS))
    scale_w = width * 2
    scale_h = height * 2
    if vertical:
        vf = (
            f"scale={scale_w}:{scale_h},"
            f"zoompan=z='min(zoom+0.0008,1.12)':"
            f"x='iw/2-(iw/zoom/2)':"
            f"y='ih/2-(ih/zoom/2)':"
            f"d={frames}:s={width}x{height}:fps={FPS},format=yuv420p"
        )
    else:
        vf = (
            f"scale={scale_w}:{scale_h},"
            f"zoompan=z='min(zoom+0.0007,1.10)':"
            f"x='iw/2-(iw/zoom/2)':"
            f"y='ih/2-(ih/zoom/2)':"
            f"d={frames}:s={width}x{height}:fps={FPS},format=yuv420p"
        )
    run(
        [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            str(image),
            "-vf",
            vf,
            "-frames:v",
            str(frames),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]
    )


def scaled_durations(total_duration: float, weights: list[float]) -> list[float]:
    weight_sum = sum(weights)
    durations = [total_duration * w / weight_sum for w in weights]
    durations[-1] += max(0.0, total_duration - sum(durations))
    return durations


def concat_video_clips(clips: list[Path], output: Path) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        for path in clips:
            fh.write(f"file '{path.as_posix()}'\n")
        concat_file = fh.name
    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_file,
            "-c",
            "copy",
            str(output),
        ]
    )


def mux_audio(video: Path, audio: Path, output: Path) -> None:
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-i",
            str(audio),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(output),
        ]
    )


def render_video(audio: Path, images: list[Path], weights: list[float], output: Path, width: int, height: int, vertical: bool) -> None:
    duration = probe_duration(audio)
    clips_dir = MEDIA_DIR / f"{output.stem}_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    durations = scaled_durations(duration, weights)
    clips: list[Path] = []
    for idx, (image, clip_duration) in enumerate(zip(images, durations), start=1):
        clip = clips_dir / f"{idx:02d}.mp4"
        build_clip(image, clip, width, height, clip_duration, vertical)
        clips.append(clip)
    silent = output.with_name(f"{output.stem}_silent.mp4")
    concat_video_clips(clips, silent)
    mux_audio(silent, audio, output)


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as img:
        img = img.convert("RGB")
        src_ratio = img.width / img.height
        dst_ratio = size[0] / size[1]
        if src_ratio > dst_ratio:
            new_height = size[1]
            new_width = int(new_height * src_ratio)
        else:
            new_width = size[0]
            new_height = int(new_width / src_ratio)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        left = (new_width - size[0]) // 2
        top = (new_height - size[1]) // 2
        return img.crop((left, top, left + size[0], top + size[1]))


def draw_gradient(im: Image.Image, start_y: int, opacity: int) -> None:
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for y in range(start_y, im.height):
        t = (y - start_y) / max(1, im.height - start_y)
        alpha = int(opacity * t)
        draw.line((0, y, im.width, y), fill=(0, 0, 0, alpha))
    im.alpha_composite(overlay)


def text_block_height(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, spacing: int) -> int:
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing)
    return bbox[3] - bbox[1]


def create_thumbnail(background: Path, output: Path, title: str, subtitle: str, vertical: bool = False, hindi: bool = False) -> None:
    size = (1080, 1920) if vertical else (1280, 720)
    im = fit_image(background, size).convert("RGBA")
    draw_gradient(im, int(size[1] * 0.35), 220)
    draw = ImageDraw.Draw(im)

    if hindi:
        font_path = "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf"
    else:
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    title_font = ImageFont.truetype(font_path, 108 if vertical else 80)
    subtitle_font = ImageFont.truetype(font_path, 72 if vertical else 44)

    if vertical:
        box = [30, size[1] - 520, size[0] - 30, size[1] - 30]
        title_y = size[1] - 430
    else:
        box = [36, size[1] - 250, int(size[0] * 0.76), size[1] - 24]
        title_y = size[1] - 205
    draw.rounded_rectangle(box, radius=28, fill=(0, 0, 0, 155))
    draw.rounded_rectangle([box[0] + 16, box[1] + 20, box[0] + 30, box[3] - 20], radius=10, fill=(70, 160, 255, 255))

    padding_x = box[0] + 40
    subtitle_y = title_y + text_block_height(draw, title, title_font, 8) + 18
    draw.multiline_text((padding_x, title_y), title, font=title_font, fill="white", spacing=8, stroke_width=2, stroke_fill="black")
    draw.multiline_text((padding_x, subtitle_y), subtitle, font=subtitle_font, fill=(255, 215, 120), spacing=6, stroke_width=2, stroke_fill="black")
    im.convert("RGB").save(output, quality=95)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    trimmed = OUT_DIR / "trimmed-audio"
    trimmed.mkdir(parents=True, exist_ok=True)

    hindi_part1 = trimmed / "hindi-part1.mp3"
    hindi_part2 = trimmed / "hindi-part2.mp3"
    hindi_reel = trimmed / "hindi-reel.mp3"
    english_part1 = trimmed / "english-part1.mp3"
    english_part2 = trimmed / "english-part2.mp3"
    english_reel = trimmed / "english-reel.mp3"

    trim_audio(HINDI_DIR / "ash_and_rudraksha.mp3", hindi_part1)
    trim_audio(HINDI_DIR / "the_awakening_at_gokarna.mp3", hindi_part2)
    trim_audio(HINDI_DIR / "the_brahmana_of_bashkala.mp3", hindi_reel)
    trim_audio(ASSET_DIR / "gemini_music (10).mp3", english_part1)
    trim_audio(ASSET_DIR / "gemini_music (9).mp3", english_part2)
    trim_audio(ASSET_DIR / "gemini_music (11).mp3", english_reel)

    hindi_full = OUT_DIR / "episode2-hindi-full-audio.mp3"
    english_full = OUT_DIR / "episode2-english-full-audio.mp3"
    make_audio_concat([hindi_part1, hindi_part2], hindi_full)
    make_audio_concat([english_part1, english_part2], english_full)

    landscape_paths = [ASSET_DIR / name for name in LANDSCAPE_IMAGES]
    reel_paths = [ASSET_DIR / name for name in REEL_IMAGES]

    render_video(hindi_full, landscape_paths, LANDSCAPE_WEIGHTS, OUT_DIR / "shiv-puran-episode2-hindi-youtube-final.mp4", 1920, 1080, False)
    render_video(english_full, landscape_paths, LANDSCAPE_WEIGHTS, OUT_DIR / "shiv-puran-episode2-english-youtube-final.mp4", 1920, 1080, False)
    render_video(hindi_reel, reel_paths, REEL_WEIGHTS, OUT_DIR / "shiv-puran-episode2-hindi-reel-final.mp4", 1080, 1920, True)
    render_video(english_reel, reel_paths, REEL_WEIGHTS, OUT_DIR / "shiv-puran-episode2-english-reel-final.mp4", 1080, 1920, True)

    create_thumbnail(
        ASSET_DIR / "1793555920631.png",
        THUMB_DIR / "shiv-puran-episode2-hindi-youtube-thumbnail.png",
        "शिव पुराण EP 2",
        "पश्चाताप से मुक्ति",
        vertical=False,
        hindi=True,
    )
    create_thumbnail(
        ASSET_DIR / "1793555920631.png",
        THUMB_DIR / "shiv-puran-episode2-english-youtube-thumbnail.png",
        "SHIVA PURANA EP 2",
        "REPENTANCE & REDEMPTION",
        vertical=False,
        hindi=False,
    )
    create_thumbnail(
        ASSET_DIR / "1793555924680.png",
        THUMB_DIR / "shiv-puran-episode2-hindi-reel-cover.png",
        "पाप से",
        "वापसी?",
        vertical=True,
        hindi=True,
    )
    create_thumbnail(
        ASSET_DIR / "1793555924680.png",
        THUMB_DIR / "shiv-puran-episode2-english-reel-cover.png",
        "A WAY BACK",
        "FROM SIN?",
        vertical=True,
        hindi=False,
    )


if __name__ == "__main__":
    main()
