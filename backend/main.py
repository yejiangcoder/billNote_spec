import sys
import os
import json
import logging
import re
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from dotenv import load_dotenv

# =============================
# CLI / Web 模式判定
# =============================
IS_CLI_MODE = len(sys.argv) > 1

# ⚠️ CLI 模式下不要引入 DB / 事件
if not IS_CLI_MODE:
    from app.db.init_db import init_db
    from events import register_handler

from app.utils.logger import get_logger
from app import create_app
from app.transcriber.transcriber_provider import get_transcriber
from ffmpeg_helper import ensure_ffmpeg_or_raise

from app.services.note import NoteGenerator
from app.enmus.note_enums import DownloadQuality
from app.exceptions.exception_handlers import register_exception_handlers

logger = get_logger(__name__)
load_dotenv()

# =============================
# CLI 模式环境变量注入
# =============================
if IS_CLI_MODE:
    os.environ["CLI_MODE"] = "true"
    os.environ["NOTE_OUTPUT_DIR"] = "/tmp/billnote_runtime"
    os.environ["DISABLE_SCREENSHOT"] = "true"

# =============================
# 工具函数
# =============================
def extract_bvid(url: str) -> str:
    """从 URL 提取 BV 号，作为稳定主键"""
    match = re.search(r"(BV[0-9A-Za-z]{10})", url)
    if not match:
        return f"UNKNOWN_{abs(hash(url))}"
    return match.group(1)


def save_cli_result(bvid: str, payload: dict):
    """
    ✅ 原子写入
    /data/content/{bvid}/result.json
    """
    MOUNT_ROOT = "/data/content"
    target_dir = os.path.join(MOUNT_ROOT, bvid)
    os.makedirs(target_dir, exist_ok=True)

    final_path = os.path.join(target_dir, "result.json")
    tmp_path = final_path + ".tmp"

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # ✅ 原子替换，Java 永远读到完整文件
        os.replace(tmp_path, final_path)

        print(f"STATUS=WRITTEN FILE={final_path}")
        logger.info(f"[CLI] 写入成功: {final_path}")

    except Exception as e:
        logger.error(f"[CLI] 写入失败: {e}")
        print(f"STATUS=ERROR_WRITE {e}")

# =============================
# Web 模式配置（保持原样）
# =============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not IS_CLI_MODE:
        register_handler()
        init_db()
        get_transcriber()
    yield


if not IS_CLI_MODE:
    static_path = os.getenv("STATIC", "/static")
    static_dir = "static"
    uploads_dir = "uploads"

    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(uploads_dir, exist_ok=True)

    app = create_app(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_exception_handlers(app)
    app.mount(static_path, StaticFiles(directory=static_dir), name="static")
    app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")

# =============================
# 🔥 主入口：CLI / Web 分流
# =============================
if __name__ == "__main__":

    if IS_CLI_MODE:
        ensure_ffmpeg_or_raise()

        video_url = sys.argv[1]
        bvid = extract_bvid(video_url)
        task_id = f"task_{bvid}"

        logger.info(f"[CLI] 启动采集 | BVID={bvid}")
        print(f"STATUS=START BVID={bvid}")

        try:
            generator = NoteGenerator()

            note_result = generator.generate(
                video_url=video_url,
                platform="bilibili",
                quality=DownloadQuality.medium,
                task_id=task_id,
                model_name=None,      # 🔒 关闭 GPT
                provider_id=None,
                summary=False,
                transcribe=True,
            )

            if not note_result or not note_result.transcript:
                raise RuntimeError("转写失败或结果为空")

            # 🔥 这里的改动：增加了 fullText 字段，双重保险
            transcript_obj = {
                "full_text": note_result.transcript.full_text,
                "fullText": note_result.transcript.full_text,  # 👈 给 Java 的 Jackson/FastJSON 兜底
                "segments": [
                    {"start": s.start, "end": s.end, "text": s.text}
                    for s in note_result.transcript.segments
                ],
            }

            result_payload = {
                "bvid": bvid,
                "platform": "bilibili",
                "url": video_url,
                "engine": "billnote-whisper-pure",
                "stage": "transcript",
                "status": "SUCCESS",
                "transcript": transcript_obj,
                "summary": "",  # Java 侧安全占位
                "meta": note_result.audio_meta.raw_info if note_result.audio_meta else {},
            }

            save_cli_result(bvid, result_payload)
            print("STATUS=SUCCESS")
            sys.exit(0)

        except Exception as e:
            logger.exception("[CLI] 执行失败")

            error_payload = {
                "bvid": bvid,
                "platform": "bilibili",
                "url": video_url,
                "engine": "billnote-whisper-pure",
                "status": "FAILED",
                "error": str(e),
                "summary": "",
            }

            save_cli_result(bvid, error_payload)
            print(f"STATUS=FAILED ERROR={e}")
            sys.exit(1)

    else:
        host = os.getenv("BACKEND_HOST", "0.0.0.0")
        port = int(os.getenv("BACKEND_PORT", 8483))
        uvicorn.run(app, host=host, port=port, reload=False)