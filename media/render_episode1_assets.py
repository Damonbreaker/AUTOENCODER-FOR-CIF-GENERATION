from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/workspace")
HINDI_DIR = ROOT / "final-assets" / "from-drive"
ENGLISH_DIR = ROOT / "final-assets" / "english-drive"
OUT_DIR = ROOT / "final-renders"
THUMB_DIR = ROOT / "final-thumbnails"
MEDIA_DIR = ROOT / "media"

FPS = 30

LANDSCAPE_IMAGES = [
    "1793544693095.png",
    "1793544702321.png",
    "1793544731979.png",
    "1793544760270.png",
    "1793544795765.png",
    "1793544818475.png",
    "1793544840098.png",
    "1793544862515.png",
    "1793544885175.png",
    "1793544905796.png",
    "1793544929416.png",
    "1793545957448.png",
    "1793545981783.png",
    "1793546038321.png",
    "1793546046597.png",
    "1793546057706.png",
    "1793546065171.png",
    "1793546117207.png",
]

VERTICAL_IMAGES = [
    "1793546070889.png",
    "1793546075353.png",
    "1793546077623.png",
    "1793546083631.png",
    "1793546087955.png",
    "1793546108694.png",
    "1793546112583.png",
]

LANDSCAPE_WEIGHTS = [
    1.25,
    1.0,
    0.95,
    0.95,
    0.9,
    1.05,
    0.9,
    1.0,
    1.0,
    0.9,
    0.95,
    0.75,
    0.95,
    1.0,
    1.0,
    0.95,
    1.05,
    1.05,
]

VERTICAL_WEIGHTS = [
    1.25,
    1.0,
    1.0,
    1.0,
    0.9,
    0.9,
    1.1,
]


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


def build_clip(image: Path, output: Path, width: int, height: int, duration: float, mode: str) -> None:
    frames = max(1, round(duration * FPS))
    if mode == "landscape":
        vf = (
            f"scale={width*2}:{height*2},"
            f"zoompan=z='min(zoom+0.0007,1.10)':"
            f"x='iw/2-(iw/zoom/2)':"
            f"y='ih/2-(ih/zoom/2)':"
            f"d={frames}:s={width}x{height}:fps={FPS},"
            "format=yuv420p"
        )
    else:
        vf = (
            f"scale={width*2}:{height*2},"
            f"zoompan=z='min(zoom+0.0008,1.11)':"
            f"x='iw/2-(iw/zoom/2)':"
            f"y='ih/2-(ih/zoom/2)':"
            f"d={frames}:s={width}x{height}:fps={FPS},"
            "format=yuv420p"
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


def scaled_durations(total_duration: float, weights: list[float]) -> list[float]:
    weight_sum = sum(weights)
    durations = [total_duration * w / weight_sum for w in weights]
    durations[-1] += max(0.0, total_duration - sum(durations))
    return durations


def render_video(
    audio: Path,
    images: list[Path],
    weights: list[float],
    output: Path,
    width: int,
    height: int,
    mode: str,
) -> None:
    duration = probe_duration(audio)
    clips_dir = MEDIA_DIR / f"{output.stem}_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clips: list[Path] = []
    durations = scaled_durations(duration, weights)

    for idx, (image, clip_duration) in enumerate(zip(images, durations), start=1):
        clip = clips_dir / f"{idx:02d}.mp4"
        build_clip(image, clip, width, height, clip_duration, mode)
        clips.append(clip)

    silent_video = output.with_name(f"{output.stem}_silent.mp4")
    concat_video_clips(clips, silent_video)
    mux_audio(silent_video, audio, output)


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
        t = (y - start_y) / max(1, (im.height - start_y))
        alpha = int(opacity * t)
        draw.line((0, y, im.width, y), fill=(0, 0, 0, alpha))
    im.alpha_composite(overlay)


def text_block_height(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, spacing: int) -> int:
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing)
    return bbox[3] - bbox[1]


def create_thumbnail(
    background: Path,
    output: Path,
    title: str,
    subtitle: str,
    vertical: bool = False,
    hindi: bool = False,
) -> None:
    size = (1080, 1920) if vertical else (1280, 720)
    im = fit_image(background, size).convert("RGBA")
    draw_gradient(im, int(size[1] * 0.35), 220)
    draw = ImageDraw.Draw(im)

    if hindi:
        font_path = "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf"
    else:
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    title_font_size = 108 if (vertical and not hindi) else (124 if vertical else 84)
    title_font = ImageFont.truetype(font_path, title_font_size)
    subtitle_font = ImageFont.truetype(font_path, 76 if vertical else 48)

    padding_x = 64
    title_y = size[1] - (420 if vertical else 220)
    subtitle_y = title_y + text_block_height(draw, title, title_font, 8) + 24

    if not vertical:
        box = [36, size[1] - 250, int(size[0] * 0.72), size[1] - 24]
    else:
        box = [30, size[1] - 520, size[0] - 30, size[1] - 30]
    draw.rounded_rectangle(box, radius=28, fill=(0, 0, 0, 150))

    accent_x = box[0] + 20
    draw.rounded_rectangle(
        [accent_x, box[1] + 20, accent_x + 14, box[3] - 20],
        radius=10,
        fill=(70, 160, 255, 255),
    )

    draw.multiline_text(
        (padding_x + 18, title_y),
        title,
        font=title_font,
        fill="white",
        spacing=8,
        stroke_width=2,
        stroke_fill="black",
    )
    draw.multiline_text(
        (padding_x + 18, subtitle_y),
        subtitle,
        font=subtitle_font,
        fill=(255, 215, 120),
        spacing=6,
        stroke_width=2,
        stroke_fill="black",
    )

    im.convert("RGB").save(output, quality=95)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    THUMB_DIR.mkdir(parents=True, exist_ok=True)

    hindi_long_parts = [
        HINDI_DIR / "gemini_music (5).mp3",
        HINDI_DIR / "gemini_music (4).mp3",
    ]
    hindi_reel = min([p for p in HINDI_DIR.glob("*.mp3")], key=probe_duration)

    english_part1 = ENGLISH_DIR / "the_question_of_shaunaka.mp3"
    english_part2 = ENGLISH_DIR / "the_turning_at_pratishthana.mp3"
    english_reel = ENGLISH_DIR / "the_path_to_kailasa.mp3"

    hindi_full_audio = OUT_DIR / "episode1-hindi-full-audio.mp3"
    english_full_audio = OUT_DIR / "episode1-english-full-audio.mp3"
    make_audio_concat(hindi_long_parts, hindi_full_audio)
    make_audio_concat([english_part1, english_part2], english_full_audio)

    landscape_paths = [HINDI_DIR / name for name in LANDSCAPE_IMAGES]
    vertical_paths = [HINDI_DIR / name for name in VERTICAL_IMAGES]

    render_video(
        hindi_full_audio,
        landscape_paths,
        LANDSCAPE_WEIGHTS,
        OUT_DIR / "shiv-puran-episode1-hindi-youtube-final.mp4",
        1920,
        1080,
        "landscape",
    )
    render_video(
        english_full_audio,
        landscape_paths,
        LANDSCAPE_WEIGHTS,
        OUT_DIR / "shiv-puran-episode1-english-youtube-final.mp4",
        1920,
        1080,
        "landscape",
    )
    render_video(
        hindi_reel,
        vertical_paths,
        VERTICAL_WEIGHTS,
        OUT_DIR / "shiv-puran-episode1-hindi-reel-final.mp4",
        1080,
        1920,
        "vertical",
    )
    render_video(
        english_reel,
        vertical_paths,
        VERTICAL_WEIGHTS,
        OUT_DIR / "shiv-puran-episode1-english-reel-final.mp4",
        1080,
        1920,
        "vertical",
    )

    create_thumbnail(
        HINDI_DIR / "1793546117207.png",
        THUMB_DIR / "shiv-puran-episode1-hindi-youtube-thumbnail.png",
        "शिव पुराण",
        "देवराज की मुक्ति",
        vertical=False,
        hindi=True,
    )
    create_thumbnail(
        HINDI_DIR / "1793546117207.png",
        THUMB_DIR / "shiv-puran-episode1-english-youtube-thumbnail.png",
        "SHIVA PURANA",
        "DEVARAJA'S REDEMPTION",
        vertical=False,
        hindi=False,
    )
    create_thumbnail(
        HINDI_DIR / "1793546108694.png",
        THUMB_DIR / "shiv-puran-episode1-hindi-reel-cover.png",
        "महापापी को",
        "मुक्ति?",
        vertical=True,
        hindi=True,
    )
    create_thumbnail(
        HINDI_DIR / "1793546108694.png",
        THUMB_DIR / "shiv-puran-episode1-english-reel-cover.png",
        "FALLEN MAN",
        "SAVED?",
        vertical=True,
        hindi=False,
    )


if __name__ == "__main__":
    main()
