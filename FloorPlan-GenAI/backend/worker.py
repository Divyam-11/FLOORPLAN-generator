"""One offline inference process. Only this process imports the upstream runtime."""
import json
import os
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

from .paths import CHECKPOINT, CLIP, SOURCE, checkpoint_matches, clip_files_present


def prepare_models(send):
    if not SOURCE.joinpath("backend_cli.py").is_file():
        raise FileNotFoundError("The pinned source checkout is missing. Run scripts/setup.py.")
    if not checkpoint_matches():
        send("health", checkpoint="unavailable")
        raise ValueError("The released Flow Matching checkpoint is missing or has an invalid SHA-256. Run scripts/setup.py.")
    send("health", checkpoint="verified", message="Loading the Flow Matching checkpoint.")

    os.environ.update(
        TEXT2REVIT_BACKEND="1",
        TEXT2REVIT_CHECKPOINT=str(CHECKPOINT),
        TEXT2REVIT_CLIP_DIR=str(CLIP),
        HF_HUB_OFFLINE="1",
        MPLBACKEND="Agg",
    )
    if os.environ.get("FLOORPLAN_DEVICE", "cpu").lower() != "auto":
        os.environ["TEXT2REVIT_DEVICE"] = "cpu"
    else:
        os.environ.pop("TEXT2REVIT_DEVICE", None)
    sys.path.insert(0, str(SOURCE))
    import backend_cli
    import torch
    from transformers import CLIPTextModel, CLIPTokenizer

    torch.set_num_threads(int(os.environ.get("FLOORPLAN_THREADS", "4")))
    pipeline = backend_cli.pipeline
    try:
        flow_model = pipeline.load_flow_model(CHECKPOINT)
        if pipeline.train.DEVICE.type == "cuda":
            # Exercise the GPU build before reporting readiness.
            torch.zeros(1, device=pipeline.train.DEVICE).add_(1)
            torch.cuda.synchronize()
    except RuntimeError:
        if pipeline.train.DEVICE.type != "cuda":
            raise
        pipeline.train.DEVICE = torch.device("cpu")
        flow_model = pipeline.load_flow_model(CHECKPOINT)

    send("health", checkpoint="loaded", message="Loading the local CLIP text encoder.")
    if not clip_files_present():
        send("health", clip="unavailable")
        raise FileNotFoundError("The local CLIP encoder is missing. Run scripts/setup.py.")
    tokenizer = CLIPTokenizer.from_pretrained(CLIP, local_files_only=True)
    text_model, loading = CLIPTextModel.from_pretrained(
        CLIP, local_files_only=True, output_loading_info=True,
    )
    if any(loading.get(field) for field in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("The local CLIP encoder has missing or incompatible weights. Run model setup again.")
    text_model.eval()

    def cached_flow_model(_path):
        return flow_model.to(pipeline.train.DEVICE)

    @torch.inference_mode()
    def encode_texts(texts, device):
        # Same tokenization, pooling and output dtype as pipeline_generate.encode_texts.
        encoded = tokenizer(texts, padding="max_length", truncation=True,
                            max_length=77, return_tensors="pt")
        output = text_model.to(device)(
            input_ids=encoded["input_ids"].to(device),
            attention_mask=encoded["attention_mask"].to(device),
        )
        pooled = output.pooler_output
        if pooled is None:
            pooled = output.last_hidden_state[:, 0]
        return pooled.float().cpu().numpy()

    probe = encode_texts(["A 2-bedroom apartment with 1 bathroom and 1 balcony."],
                         pipeline.train.DEVICE)
    if probe.shape != (1, 768) or not pipeline.np.isfinite(probe).all():
        raise ValueError("CLIP returned an invalid 768-dimensional embedding during startup.")
    # Keep the upstream CLI, solver, validation, retries and exports intact.
    # Only the two per-attempt model loading call sites use cached instances.
    pipeline.load_flow_model = cached_flow_model
    pipeline.encode_texts = encode_texts
    assert "train" not in sys.modules
    send("health", state="ready", checkpoint="loaded", clip="loaded",
         device=pipeline.train.DEVICE.type, threads=torch.get_num_threads(),
         steps=pipeline.INFERENCE_STEPS, message="Both pretrained models are loaded.")
    return backend_cli


def generate(cli, command, send):
    response = Path(command["directory"]) / "plan.json"
    request = response.with_name("request.json")
    request.write_text(json.dumps(command["request"]), encoding="utf-8")
    prior_argv = sys.argv

    def publish_progress(message, *, flush=False):
        if cli.pipeline.train.DEVICE.type == "cpu":
            send("health", device="cpu")
        send("progress", message=message)

    try:
        # The original Revit/browser entry point takes arguments from sys.argv.
        # This isolated process handles exactly one request at a time.
        sys.argv = [cli.__file__, "--request", str(request), "--response", str(response)]
        cli.print = publish_progress
        cli.main()
        assert "train" not in sys.modules
    finally:
        sys.argv = prior_argv
        del cli.print
        request.unlink(missing_ok=True)


def main():
    protocol = sys.stdout

    def send(event, **payload):
        print(json.dumps({"event": event, **payload}), file=protocol, flush=True)

    with redirect_stdout(sys.stderr):
        try:
            cli = prepare_models(send)
        except Exception as error:
            traceback.print_exc()
            send("health", state="error", message=str(error))
            return 1
        for line in sys.stdin:
            try:
                generate(cli, json.loads(line), send)
                send("completed")
            except Exception as error:
                traceback.print_exc()
                send("failed", message=str(error) or type(error).__name__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
