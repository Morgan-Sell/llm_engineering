import json
import os
import pickle
from functools import lru_cache
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from tqdm.notebook import tqdm

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 200
BATCHES_FOLDER = "batches"
OUTPUT_FOLDER = "output"
state = Path("batches.pkl")

SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""


@lru_cache(maxsize=1)
def client() -> Anthropic:
    load_dotenv(override=True)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Set ANTHROPIC_API_KEY before running batch preprocessing.")
    return Anthropic(api_key=api_key)


class Batch:
    BATCH_SIZE = 1_000

    batches = []

    def __init__(self, items, start, end, lite):
        self.items = items
        self.start = start
        self.end = end
        self.filename = f"{start}_{end}.jsonl"
        self.file_id = None
        self.batch_id = None
        self.output_file_id = None
        self.requests_payload = None
        self.done = False
        folder = Path("lite") if lite else Path("full")
        self.batches = folder / BATCHES_FOLDER
        self.output = folder / OUTPUT_FOLDER
        self.batches.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=True)

    def make_request(self, item):
        params = {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "system": SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": item.full},
            ],
        }
        return {
            "custom_id": str(item.id),
            "params": params,
        }

    def make_jsonl(self, item):
        return json.dumps(self.make_request(item))

    def make_file(self):
        batch_file = self.batches / self.filename
        with batch_file.open("w") as f:
            for item in self.items[self.start : self.end]:
                f.write(self.make_jsonl(item))
                f.write("\n")

    def send_file(self):
        batch_file = self.batches / self.filename
        with batch_file.open("r") as f:
            self.requests_payload = [json.loads(line) for line in f if line.strip()]
        self.file_id = str(batch_file)

    def submit_batch(self):
        response = client().messages.batches.create(requests=self.requests_payload)
        self.batch_id = response.id

    def is_ready(self):
        response = client().messages.batches.retrieve(self.batch_id)
        status = response.processing_status
        if status == "ended":
            self.output_file_id = response.results_url
        return status == "ended"

    def fetch_output(self):
        output_file = self.output / self.filename
        with output_file.open("w") as f:
            for result in client().messages.batches.results(self.batch_id):
                f.write(json.dumps(self.normalize_result(result)))
                f.write("\n")

    @staticmethod
    def _text_from_blocks(blocks):
        texts = []
        for block in blocks:
            block_type = getattr(block, "type", None)
            if block_type is None and isinstance(block, dict):
                block_type = block.get("type")
            if block_type != "text":
                continue
            text = getattr(block, "text", None)
            if text is None and isinstance(block, dict):
                text = block.get("text", "")
            texts.append(text or "")
        return "\n".join(texts).strip()

    @classmethod
    def normalize_result(cls, result):
        payload = {"custom_id": result.custom_id, "type": result.result.type}
        if result.result.type == "succeeded":
            payload["summary"] = cls._text_from_blocks(result.result.message.content)
        elif result.result.type == "errored":
            payload["error_type"] = result.result.error.error.type
            payload["message"] = result.result.error.error.message
        return payload

    def item_lookup(self):
        return {str(item.id): item for item in self.items[self.start : self.end]}

    def apply_output(self):
        output_file = self.output / self.filename
        item_lookup = self.item_lookup()
        failures = []
        with output_file.open("r") as f:
            for line in f:
                json_line = json.loads(line)
                custom_id = json_line["custom_id"]
                if json_line["type"] != "succeeded":
                    failures.append(custom_id)
                    continue
                item_lookup[custom_id].summary = json_line["summary"]
        if failures:
            raise RuntimeError(
                f"{len(failures)} requests in batch {self.batch_id} failed or expired. "
                f"Inspect {output_file} for details."
            )
        self.done = True

    @classmethod
    def create(cls, items, lite):
        for start in range(0, len(items), cls.BATCH_SIZE):
            end = min(start + cls.BATCH_SIZE, len(items))
            batch = Batch(items, start, end, lite)
            cls.batches.append(batch)
        print(f"Created {len(cls.batches)} batches")

    @classmethod
    def run(cls):
        for batch in tqdm(cls.batches):
            batch.make_file()
            batch.send_file()
            batch.submit_batch()
        print(f"Submitted {len(cls.batches)} batches")

    @classmethod
    def fetch(cls):
        for batch in tqdm(cls.batches):
            if not batch.done:
                if batch.is_ready():
                    batch.fetch_output()
                    batch.apply_output()
        finished = [batch for batch in cls.batches if batch.done]
        print(f"Finished {len(finished)} of {len(cls.batches)} batches")

    @classmethod
    def save(cls):
        items = cls.batches[0].items
        for batch in cls.batches:
            batch.items = None
        with state.open("wb") as f:
            pickle.dump(cls.batches, f)
        for batch in cls.batches:
            batch.items = items
        print(f"Saved {len(cls.batches)} batches")

    @classmethod
    def load(cls, items):
        with state.open("rb") as f:
            cls.batches = pickle.load(f)
        for batch in cls.batches:
            batch.items = items
        print(f"Loaded {len(cls.batches)} batches")
