import json
import logging
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple, Union

from pydantic import HttpUrl
from dotenv import load_dotenv

from app.downloaders.base import Downloader
from app.db.video_task_dao import delete_task_by_video, insert_video_task
from app.enmus.exception import NoteErrorEnum, ProviderErrorEnum
from app.enmus.task_status_enums import TaskStatus
from app.enmus.note_enums import DownloadQuality
from app.exceptions.note import NoteError
from app.exceptions.provider import ProviderError
from app.gpt.base import GPT
from app.gpt.gpt_factory import GPTFactory
from app.models.audio_model import AudioDownloadResult
from app.models.gpt_model import GPTSource
from app.models.model_config import ModelConfig
from app.models.notes_model import NoteResult
from app.models.transcriber_model import TranscriptResult, TranscriptSegment
from app.services.constant import SUPPORT_PLATFORM_MAP
from app.services.provider import ProviderService
from app.transcriber.base import Transcriber
from app.transcriber.transcriber_provider import get_transcriber, _transcribers
from app.utils.note_helper import replace_content_markers
from app.utils.video_helper import generate_screenshot
from app.utils.video_reader import VideoReader

# ------------------ 环境变量 ------------------

load_dotenv()
IS_CLI_MODE = os.getenv("CLI_MODE") == "true"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------ 路径配置 ------------------

if IS_CLI_MODE:
    NOTE_OUTPUT_DIR = Path("/tmp/billnote_runtime")
    IMAGE_OUTPUT_DIR = "/tmp/billnote_images"
    IMAGE_BASE_URL = ""
else:
    NOTE_OUTPUT_DIR = Path(os.getenv("NOTE_OUTPUT_DIR", "note_results"))
    IMAGE_OUTPUT_DIR = os.getenv("OUT_DIR", "./static/screenshots")
    IMAGE_BASE_URL = os.getenv("IMAGE_BASE_URL", "/static/screenshots")

NOTE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)


class NoteGenerator:
    """
    Web / CLI 双模式 NoteGenerator
    CLI 模式：纯下载 + 转写，零副作用
    """

    def __init__(self):
        self.transcriber_type = os.getenv("TRANSCRIBER_TYPE", "fast-whisper")
        self.transcriber: Transcriber = self._init_transcriber()
        self.video_path: Optional[Path] = None
        self.video_img_urls: List[str] = []
        logger.info(f"NoteGenerator 初始化完成 (CLI_MODE={IS_CLI_MODE})")

    # ---------------- 公有方法 ----------------

    def generate(
            self,
            video_url: Union[str, HttpUrl],
            platform: str,
            quality: DownloadQuality = DownloadQuality.medium,
            task_id: Optional[str] = None,
            model_name: Optional[str] = None,
            provider_id: Optional[str] = None,
            link: bool = False,
            screenshot: bool = False,
            _format: Optional[List[str]] = None,
            style: Optional[str] = None,
            extras: Optional[str] = None,
            output_path: Optional[str] = None,
            video_understanding: bool = False,
            video_interval: int = 0,
            grid_size: Optional[List[int]] = None,
            summary: bool = True,
            transcribe: bool = True,
    ) -> Optional[NoteResult]:

        grid_size = grid_size or []

        # CLI 强制阉割视觉 & GPT
        if IS_CLI_MODE:
            screenshot = False
            video_understanding = False
            _format = []
            link = False
            summary = False

        try:
            logger.info(f"开始任务 (task_id={task_id})")

            downloader = self._get_downloader(platform)

            gpt = None
            if summary and model_name and provider_id:
                gpt = self._get_gpt(model_name, provider_id)

            audio_cache_file = NOTE_OUTPUT_DIR / f"{task_id}_audio.json"
            transcript_cache_file = NOTE_OUTPUT_DIR / f"{task_id}_transcript.json"
            markdown_cache_file = NOTE_OUTPUT_DIR / f"{task_id}_markdown.md"

            audio_meta = self._download_media(
                downloader,
                video_url,
                quality,
                audio_cache_file,
                platform,
                output_path,
                screenshot,
                video_understanding,
                video_interval,
                grid_size,
            )

            transcript = None
            if transcribe:
                transcript = self._transcribe_audio(
                    audio_meta.file_path,
                    transcript_cache_file,
                )

            markdown = ""
            if gpt and transcript:
                markdown = self._summarize_text(
                    audio_meta,
                    transcript,
                    gpt,
                    markdown_cache_file,
                    link,
                    screenshot,
                    _format or [],
                    style,
                    extras,
                    self.video_img_urls,
                    )

            if not IS_CLI_MODE and _format:
                markdown = self._post_process_markdown(
                    markdown, self.video_path, _format, audio_meta, platform
                )

            if not IS_CLI_MODE:
                self._save_metadata(audio_meta.video_id, platform, task_id)

            return NoteResult(markdown=markdown, transcript=transcript, audio_meta=audio_meta)

        except Exception as e:
            logger.exception("NoteGenerator 失败")
            return None

    # ---------------- 私有方法 ----------------

    def _update_status(self, *args, **kwargs):
        """
        🔥 C3 修复：CLI 模式下完全不写状态文件
        """
        if IS_CLI_MODE:
            return
        # Web 模式原逻辑已移除（由 router 处理）

    def _save_metadata(self, video_id: str, platform: str, task_id: str):
        """
        🔥 C3 修复：CLI 模式不入库
        """
        if IS_CLI_MODE:
            return
        insert_video_task(video_id=video_id, platform=platform, task_id=task_id)

    def _init_transcriber(self) -> Transcriber:
        if self.transcriber_type not in _transcribers:
            raise Exception(f"不支持的转写器：{self.transcriber_type}")
        return get_transcriber(transcriber_type=self.transcriber_type)

    def _get_gpt(self, model_name: str, provider_id: str) -> GPT:
        provider = ProviderService.get_provider_by_id(provider_id)
        if not provider:
            raise ProviderError(
                code=ProviderErrorEnum.NOT_FOUND,
                message=ProviderErrorEnum.NOT_FOUND.message,
            )
        config = ModelConfig(
            api_key=provider["api_key"],
            base_url=provider["base_url"],
            model_name=model_name,
            provider=provider["type"],
            name=provider["name"],
        )
        return GPTFactory().from_config(config)

    def _get_downloader(self, platform: str) -> Downloader:
        downloader_cls = SUPPORT_PLATFORM_MAP.get(platform)
        if not downloader_cls:
            raise NoteError(
                code=NoteErrorEnum.PLATFORM_NOT_SUPPORTED.code,
                message=NoteErrorEnum.PLATFORM_NOT_SUPPORTED.message,
            )
        return downloader_cls

    def _download_media(
            self,
            downloader: Downloader,
            video_url: Union[str, HttpUrl],
            quality: DownloadQuality,
            audio_cache_file: Path,
            platform: str,
            output_path: Optional[str],
            screenshot: bool,
            video_understanding: bool,
            video_interval: int,
            grid_size: List[int],
    ) -> AudioDownloadResult:

        if audio_cache_file.exists():
            data = json.loads(audio_cache_file.read_text(encoding="utf-8"))
            return AudioDownloadResult(**data)

        need_video = screenshot or video_understanding

        if need_video:
            self.video_path = Path(downloader.download_video(video_url))
            if grid_size:
                self.video_img_urls = VideoReader(
                    video_path=str(self.video_path),
                    grid_size=tuple(grid_size),
                    frame_interval=video_interval,
                ).run()

        audio = downloader.download(
            video_url=video_url,
            quality=quality,
            output_dir=output_path,
            need_video=need_video,
        )
        audio_cache_file.write_text(
            json.dumps(asdict(audio), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return audio

    def _transcribe_audio(
            self,
            audio_file: str,
            transcript_cache_file: Path,
    ) -> TranscriptResult:

        if transcript_cache_file.exists():
            data = json.loads(transcript_cache_file.read_text(encoding="utf-8"))
            segments = [TranscriptSegment(**seg) for seg in data.get("segments", [])]
            return TranscriptResult(
                language=data["language"],
                full_text=data["full_text"],
                segments=segments,
            )

        transcript = self.transcriber.transcript(file_path=audio_file)
        transcript_cache_file.write_text(
            json.dumps(asdict(transcript), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return transcript

    def _summarize_text(
            self,
            audio_meta: AudioDownloadResult,
            transcript: TranscriptResult,
            gpt: GPT,
            markdown_cache_file: Path,
            link: bool,
            screenshot: bool,
            formats: List[str],
            style: Optional[str],
            extras: Optional[str],
            video_img_urls: List[str],
    ) -> str:

        if markdown_cache_file.exists():
            return markdown_cache_file.read_text(encoding="utf-8")

        source = GPTSource(
            title=audio_meta.title,
            segment=transcript.segments,
            tags=audio_meta.raw_info.get("tags", []),
            screenshot=screenshot,
            video_img_urls=video_img_urls,
            link=link,
            _format=formats,
            style=style,
            extras=extras,
        )
        markdown = gpt.summarize(source)
        markdown_cache_file.write_text(markdown, encoding="utf-8")
        return markdown

    def _post_process_markdown(
            self, markdown: str, video_path: Path, formats: List[str], audio_meta, platform
    ) -> str:
        if IS_CLI_MODE:
            return markdown

        if "screenshot" in formats and video_path:
            markdown = self._insert_screenshots(markdown, video_path)

        if "link" in formats:
            markdown = replace_content_markers(
                markdown, video_id=audio_meta.video_id, platform=platform
            )
        return markdown

    def _insert_screenshots(self, markdown: str, video_path: Path) -> str:
        matches = self._extract_screenshot_timestamps(markdown)
        for idx, (marker, ts) in enumerate(matches):
            img_path = generate_screenshot(
                str(video_path), str(IMAGE_OUTPUT_DIR), ts, idx
            )
            filename = Path(img_path).name
            img_url = f"{IMAGE_BASE_URL.rstrip('/')}/{filename}"
            markdown = markdown.replace(marker, f"![]({img_url})", 1)
        return markdown

    @staticmethod
    def _extract_screenshot_timestamps(markdown: str) -> List[Tuple[str, int]]:
        pattern = r"(?:\*Screenshot-(\d{2}):(\d{2})|Screenshot-\[(\d{2}):(\d{2})\])"
        results = []
        for match in re.finditer(pattern, markdown):
            mm = match.group(1) or match.group(3)
            ss = match.group(2) or match.group(4)
            results.append((match.group(0), int(mm) * 60 + int(ss)))
        return results
