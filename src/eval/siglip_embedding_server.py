import argparse
import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

from src.eval.topo_memory import DEFAULT_SIGLIP_PATH, SiglipMultimodalEncoder


class SiglipEmbeddingHandler(BaseHTTPRequestHandler):
    encoder = None
    encode_lock = threading.Lock()

    def log_message(self, fmt, *args):
        return

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/encode", "/encode_text"):
            self._send_json(404, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            with self.encode_lock:
                if self.path == "/encode":
                    image_bytes = base64.b64decode(payload["image"])
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    embedding = self.encoder.encode(image)
                else:
                    embedding = self.encoder.encode_text(payload["text"])
            self._send_json(200, {
                "dim": int(embedding.shape[0]),
                "embedding": embedding.astype(float).tolist(),
            })
        except Exception as exc:
            self._send_json(500, {"error": repr(exc)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_SIGLIP_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8301)
    args = parser.parse_args()

    SiglipEmbeddingHandler.encoder = SiglipMultimodalEncoder(args.model_path, device=args.device)
    SiglipEmbeddingHandler.encoder._load()
    server = ThreadingHTTPServer((args.host, args.port), SiglipEmbeddingHandler)
    print(
        f"SigLIP embedding server ready on http://{args.host}:{args.port} "
        f"model={args.model_path} device={args.device}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
