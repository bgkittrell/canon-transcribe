from math import log
import os
import requests
import redis
from celery import shared_task
from celery.utils.log import get_task_logger

import whisperx

logger = get_task_logger(__name__)
r = redis.Redis(host='localhost', port=6379)

@shared_task
def checker():
    logger.info("Checking for new episodes...")
    response = requests.get('http://localhost:8000/api/contents/needs_transcription')
    content_list = response.json()
    logger.info(content_list)
    for content in content_list:
        logger.info(content['id'])
        logger.info(content['content_url'])
        if not r.get(content['id']):
            r.set(content['id'], 1)
            logger.info("Queuing!")
            transcribe.delay(content['id'], content['content_url'])
        else:
            logger.info("Already queued!")

@shared_task
def transcribe(content_id, content_url):
    logger.info("Transcribing...")

    output_directory = f"output_docs/{content_id}"
    os.makedirs(output_directory, exist_ok=True)
    save_path = f"{output_directory}/audio.mp3"
    _download_mp3(content_url, save_path)
    # transcript = _transcribe(save_path)
    transcript = "test"
    _write_transcript(output_directory, transcript)
    _save_transcript(content_id, transcript)
    r.delete(content_id)
    logger.info("Transcribed!")

def _write_transcript(output_directory, transcript):
    logger.info("Writing transcript...")
    with open(f"{output_directory}/transcript.txt", "w") as f:
        f.write(transcript)
    logger.info("Written!")

def _save_transcript(content_id, transcript):
    logger.info("Saving transcript...")
    requests.patch(f'http://localhost:8000/api/contents/{content_id}', data={'transcript': transcript})    
    logger.info("Saved!")

def _download_mp3(url, save_path):
    logger.info(f"Downloading mp3 from {url} to {save_path}")
    response = requests.get(url, stream=True)
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def _transcribe(save_path):
    device = "cuda" 
    audio_file = save_path
    batch_size = 16 # reduce if low on GPU mem
    compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    logger.info(result["segments"]) # before alignment

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    logger.info(result["segments"]) # after alignment

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

    return result["segments"]
