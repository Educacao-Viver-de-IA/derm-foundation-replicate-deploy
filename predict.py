"""Derm Foundation — Google's dermatology embedding model (TF SavedModel)."""
import os
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

print(f"[module] predict.py loading at t={time.time()}", flush=True)
sys.stdout.flush()
import numpy as np
import tensorflow as tf
print(f"[module] tf {tf.__version__} | GPUs: {len(tf.config.list_physical_devices('GPU'))}", flush=True)
sys.stdout.flush()
from PIL import Image
from cog import BasePredictor, Input, Path
sys.stdout.flush()

MODEL_DIR = "/src/weights/derm-foundation"


class Predictor(BasePredictor):
    def setup(self):
        t0 = time.time()
        print(f"[setup] === START === t={t0}", flush=True)
        sys.stdout.flush()
        self.model = None
        self.setup_error = None
        try:
            print(f"[setup] files: {sorted(os.listdir(MODEL_DIR))}", flush=True)
            print(f"[setup] loading SavedModel...", flush=True)
            sys.stdout.flush()
            # derm-foundation é exportado como TF SavedModel
            self.model = tf.saved_model.load(MODEL_DIR)
            print(f"[setup] DONE (t={time.time()-t0:.1f}s)", flush=True)
            # Lista signatures pra debug
            sigs = list(getattr(self.model, 'signatures', {}).keys())
            print(f"[setup] signatures: {sigs}", flush=True)
            sys.stdout.flush()
        except Exception as e:
            import traceback
            print(f"[setup] FATAL: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            self.setup_error = f"setup failed: {e}"

    def predict(
        self,
        image: Path = Input(description="Foto de lesão dermatológica."),
        image_size: int = Input(default=448, ge=224, le=512,
            description="Resolução de entrada (derm-foundation default 448)."),
    ) -> dict:
        if self.model is None:
            return {"error": f"Modelo não carregou: {getattr(self, 'setup_error', '?')}"}

        t0 = time.time()
        pil = Image.open(str(image)).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)  # [1, H, W, 3]

        # Derm Foundation usa signature default 'serving_default'
        infer = self.model.signatures.get('serving_default')
        if infer is None:
            return {"error": f"No serving_default signature found. Available: {list(self.model.signatures.keys())}"}

        try:
            input_name = list(infer.structured_input_signature[1].keys())[0]
            output = infer(**{input_name: tf.constant(arr)})
        except Exception as e:
            return {"error": f"Inference failed: {e}"}

        # Output é dict de tensors — extrai embedding
        out_dict = {k: v.numpy() for k, v in output.items()}
        # Acha o vetor de embedding (geralmente o que tem shape [1, D] com D grande)
        emb = None
        for k, v in out_dict.items():
            if v.ndim == 2 and v.shape[0] == 1 and v.shape[1] > 32:
                emb = v[0]
                emb_key = k
                break
        if emb is None:
            # fallback: pega o primeiro tensor
            k = list(out_dict.keys())[0]
            emb = out_dict[k].flatten()
            emb_key = k

        return {
            "embedding": emb.tolist(),
            "embedding_dim": int(emb.shape[0]),
            "embedding_key": emb_key,
            "available_outputs": list(out_dict.keys()),
            "predict_time_s": round(time.time() - t0, 3),
        }
