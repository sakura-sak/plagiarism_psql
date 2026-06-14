import base64
import io
import math
import os
import time
import uuid
import warnings
import zlib
from typing import Dict

import numpy as np
import psycopg2
import psycopg2.extras
import urllib3

urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _parse_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


# BLAS/OpenMP: выставляем до import torch, иначе MKL не подхватит.
# На CPU раньше было жёстко 1 поток — из‑за этого LM forward на rugpt3small очень долгий.
_default_cpu = min(8, max(2, os.cpu_count() or 4))
_cpu_threads = _parse_positive_int("TORCH_CPU_THREADS", _default_cpu)
os.environ["OMP_NUM_THREADS"] = str(_cpu_threads)
os.environ["MKL_NUM_THREADS"] = str(_cpu_threads)

import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, QueryRequest, VectorParams
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


class AntiPlagiarismWorker:
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "university_docs",
    ):
        self.pg_dsn = "dbname=db user=user password=password host=host port=port"  # строка подключения к бд
        requested_device = os.getenv("DEVICE", "auto").strip().lower()
        if requested_device not in {"auto", "cpu", "cuda"}:
            requested_device = "auto"

        if requested_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "DEVICE=cuda requested but CUDA is not available in this runtime. "
                    "Install CUDA-enabled PyTorch and run the container with GPU access."
                )
            self.device = "cuda"
        elif requested_device == "cpu":
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.device == "cpu":
            torch.set_num_threads(_cpu_threads)

        self.ai_rank_good = _parse_positive_int("AI_RANK_GOOD", 50)
        self.ai_rank_bad = _parse_positive_int("AI_RANK_BAD", 1000)
        self.ai_p_ratio_threshold = float(os.getenv("AI_P_RATIO_THRESHOLD", "0.85"))
        self.ai_m_ratio_threshold = float(os.getenv("AI_M_RATIO_THRESHOLD", "0.01"))
        self.ai_c_ratio_threshold = float(os.getenv("AI_C_RATIO_THRESHOLD", "0.45"))
        self.ai_weight_p = float(os.getenv("AI_WEIGHT_P", "0.60"))
        self.ai_weight_m = float(os.getenv("AI_WEIGHT_M", "0.30"))
        self.ai_weight_c = float(os.getenv("AI_WEIGHT_C", "0.10"))
        self.qdrant_score_threshold = float(os.getenv("QDRANT_SCORE_THRESHOLD", "0.92"))

        self.encoder = SentenceTransformer(
            "cointegrated/rubert-tiny2", device=self.device
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )

        model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
        self.ai_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.ai_model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.ai_model.eval()

        self.qdrant = QdrantClient(
            url=f"http://{qdrant_host}:{qdrant_port}", prefer_grpc=False, timeout=30.0
        )
        self.collection_name = collection_name
        self._ensure_qdrant_collection()

    def _ensure_qdrant_collection(self) -> None:
        """
        The API relies on Qdrant for similarity search. If the expected collection
        does not exist, Qdrant returns 404 and the API replies 500.
        Create the collection at startup to make the service self-contained.
        """
        # Qdrant may still be booting when the analysis container starts.
        last_err: Exception | None = None
        for _ in range(30):
            try:
                existing = {c.name for c in self.qdrant.get_collections().collections}
                if self.collection_name in existing:
                    return

                vector_size = int(self.encoder.get_sentence_embedding_dimension())
                self.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
                return
            except Exception as e:  # includes connection errors / Qdrant not ready yet
                last_err = e
                time.sleep(1)

        if last_err:
            raise RuntimeError(f"Qdrant is not ready: {last_err}") from last_err

    def extract_real_text(self, text_input: str) -> str:
        if not isinstance(text_input, str) or not text_input.startswith("FILE_BASE64|"):
            return text_input
        try:
            parts = text_input.split("|", 2)
            if len(parts) != 3:
                return text_input
            _, ext, b64_data = parts
            file_bytes = base64.b64decode(b64_data)
            file_stream = io.BytesIO(file_bytes)
            if ext.lower() == "pdf":
                from pypdf import PdfReader

                reader = PdfReader(file_stream)
                extracted = "\n".join(
                    [page.extract_text() or "" for page in reader.pages]
                )
                return extracted.strip()
            elif ext.lower() in ["docx", "doc"]:
                import docx

                doc = docx.Document(file_stream)
                extracted = "\n".join([p.text for p in doc.paragraphs])
                return extracted.strip()
        except Exception as e:
            print(f"Error parsing base64 file: {e}")
            return "Ошибка парсинга"
        return text_input

    def _analyze_ai_chunk(self, text: str) -> float:
        token_ids = self.ai_tokenizer.encode(
            text, return_tensors="pt", max_length=1024, truncation=True
        ).to(self.device)
        if token_ids.shape[1] < 2:
            return 0.0
        with torch.no_grad():
            logits = self.ai_model(token_ids).logits[0, :-1, :]
            actual_next_tokens = token_ids[0, 1:].unsqueeze(-1)
            sorted_indices = torch.argsort(logits, dim=-1, descending=True)
            ranks = (sorted_indices == actual_next_tokens).nonzero(as_tuple=True)[
                1
            ].cpu().numpy() + 1
        total_words = len(ranks)
        if total_words == 0:
            return 0.0
        p_ratio = np.sum(ranks <= self.ai_rank_good) / total_words
        p_score = 1.0 / (1.0 + math.exp(-20.0 * (p_ratio - self.ai_p_ratio_threshold)))
        m_ratio = np.sum(ranks > self.ai_rank_bad) / total_words
        m_score = 1.0 / (1.0 + math.exp(300.0 * (m_ratio - self.ai_m_ratio_threshold)))
        text_bytes = text.encode("utf-8")
        c_ratio = len(zlib.compress(text_bytes)) / len(text_bytes)
        c_score = (
            1.0
            / (1.0 + math.exp(15.0 * (c_ratio - self.ai_c_ratio_threshold)))
            * min(1.0, len(text_bytes) / 1000.0)
        )
        return math.sqrt(self.ai_weight_p * p_score + self.ai_weight_m * m_score + self.ai_weight_c * c_score * 100) / 100

    def process_text(self, text: str, filename: str, verbose: bool = True) -> Dict:
        actual_text = self.extract_real_text(text)
        chunks = self.text_splitter.split_text(actual_text)
        if not chunks:
            return {
                "plagiarism_percent": 0.0,
                "ai_percent": 0.0,
                "parsed_text": actual_text,
            }
        if verbose:
            print(f"Analysis {filename[:20]}... (batches: {len(chunks)})")
        ai_scores = []
        for chunk in chunks:
            if len(chunk.strip()) > 100:
                ai_scores.append(self._analyze_ai_chunk(chunk))
        ai_percent = (
            round((sum(ai_scores) / len(ai_scores)) * 100, 2) if ai_scores else 0.0
        )
        vectors = self.encoder.encode(
            chunks, batch_size=16, show_progress_bar=False
        ).tolist()
        requests = [
            QueryRequest(query=vec, limit=1, score_threshold=self.qdrant_score_threshold) for vec in vectors
        ]
        try:
            batch_search_results = self.qdrant.query_batch_points(
                collection_name=self.collection_name, requests=requests
            )
        except UnexpectedResponse as e:
            # Most common production issue: collection was deleted or never created.
            if getattr(e, "status_code", None) == 404:
                self._ensure_qdrant_collection()
                batch_search_results = self.qdrant.query_batch_points(
                    collection_name=self.collection_name, requests=requests
                )
            else:
                raise
        plagiarized_chunks = 0
        points_to_insert = []
        for chunk_text, vector, qdrant_res in zip(
            chunks, vectors, batch_search_results
        ):
            if len(qdrant_res.points) > 0:
                plagiarized_chunks += 1
            points_to_insert.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={"filename": filename, "text": chunk_text},
                )
            )
        if points_to_insert:
            self.qdrant.upsert(
                collection_name=self.collection_name, points=points_to_insert
            )
        plagiarism_percent = round((plagiarized_chunks / len(chunks)) * 100, 2)
        return {
            "plagiarism_percent": plagiarism_percent,
            "ai_percent": ai_percent,
            "parsed_text": actual_text,
        }

    def _fetch_next_task(self):
        try:
            with psycopg2.connect(self.pg_dsn) as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                    cursor.execute(
                        "UPDATE tasks SET status = 'PROCESSING' WHERE id = (SELECT id FROM tasks WHERE status = 'PENDING' LIMIT 1 FOR UPDATE SKIP LOCKED) RETURNING id, filename, content"
                    )
                    row = cursor.fetchone()
                    conn.commit()
                    return dict(row) if row else None
        except Exception as e:
            print(f"Postgres error: {e}")
            return None

    def _save_task_result(self, task_id: int, result: dict):
        try:
            with psycopg2.connect(self.pg_dsn) as conn:
                with conn.cursor() as cursor:
                    if "parsed_text" in result:
                        cursor.execute(
                            "UPDATE tasks SET status = 'DONE', plagiarism_percent = %s, ai_percent = %s, content = %s WHERE id = %s",
                            (
                                result["plagiarism_percent"],
                                result["ai_percent"],
                                result["parsed_text"],
                                task_id,
                            ),
                        )
                    else:
                        cursor.execute(
                            "UPDATE tasks SET status = 'DONE', plagiarism_percent = %s, ai_percent = %s WHERE id = %s",
                            (
                                result["plagiarism_percent"],
                                result["ai_percent"],
                                task_id,
                            ),
                        )
                conn.commit()
        except Exception as e:
            print(f"Postgres error: {e}")

    def run_worker_loop(self, worker_id: int):
        print(f"worker #{worker_id} launched...")
        while True:
            task = self._fetch_next_task()
            if task:
                result = self.process_text(task["content"], task["filename"])
                self._save_task_result(task["id"], result)
                print(f"Processed: {task['filename'][:20]}")
            else:
                time.sleep(2)
