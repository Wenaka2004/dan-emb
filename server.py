"""FastAPI server for Danbooru tag expansion with optional Gradio WebUI."""

import argparse
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import (
    SERVER_HOST, SERVER_PORT, WEBUI_PORT,
    SILICONFLOW_API_KEY, LLM_MODEL, EMBEDDING_MODEL,
    DEFAULT_TOP_K, DEFAULT_TEMPERATURE, validate,
)
from expander import DanbooruRAG, expand_tags, expand_tags_streaming

# ── Shared state ────────────────────────────────────────────────────────────

rag: DanbooruRAG | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    rag = DanbooruRAG()
    yield
    rag = None


# ── FastAPI app ─────────────────────────────────────────────────────────────

app = FastAPI(title="Danbooru Tag Expander", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ExpandRequest(BaseModel):
    prompt: str
    top_k: int = DEFAULT_TOP_K
    category_filter: list[str] | None = None
    temperature: float = DEFAULT_TEMPERATURE
    two_stage: bool = True


class ExpandResponse(BaseModel):
    input: str
    english_input: str
    tags: list[str]
    tag_count: int
    latency_ms: float
    steps: dict | None = None
    rag_top5: list[dict] | None = None


@app.post("/expand", response_model=ExpandResponse)
async def api_expand(req: ExpandRequest):
    result = expand_tags(
        user_input=req.prompt,
        rag=rag,
        top_k=req.top_k,
        category_filter=req.category_filter,
        temperature=req.temperature,
        two_stage=req.two_stage,
    )
    return ExpandResponse(
        input=result["input"],
        english_input=result.get("english_input", req.prompt),
        tags=result["tags"],
        tag_count=result["tag_count"],
        latency_ms=result["latency_ms"],
        steps=result.get("steps"),
        rag_top5=[
            {"tag": r["tag"], "category": r["category"], "score": round(r["score"], 4)}
            for r in result.get("rag_results", [])[:5]
        ],
    )


@app.get("/health")
async def health():
    return {"status": "ok", "vectors": len(rag.embeddings) if rag else 0}


# ── Gradio WebUI ────────────────────────────────────────────────────────────

def create_webui():
    import gradio as gr

    def expand_ui(prompt: str, temperature: float, two_stage: bool):
        if not prompt.strip():
            yield "", "", ""
            return

        status = ""
        tags = ""
        detail = ""

        for event in expand_tags_streaming(
            user_input=prompt,
            rag=rag,
            temperature=temperature,
            two_stage=two_stage,
        ):
            etype = event["type"]

            if etype == "status":
                icons = {"translate": "🌐", "rag1": "🔍", "llm1": "✨", "rag2": "🔍", "llm2": "✨"}
                icon = icons.get(event["step"], "⏳")
                status = f"{icon} {event['message']}"
                yield tags, status, detail

            elif etype == "rag":
                lines = []
                for r in event["results"]:
                    cp = f"  [{', '.join(r['copyrights'][:2])}]" if r.get("copyrights") else ""
                    lines.append(f"  {r['score']:.3f} [{r['category']:10s}] {r['tag']}{cp}")
                detail = detail + "\n".join(lines) + "\n" if detail else "\n".join(lines) + "\n"
                yield tags, status, detail

            elif etype == "stream":
                stage_label = "Stage 1" if event["stage"] == 1 else "Stage 2"
                tags = f"[{stage_label}] {event['text']}"
                yield tags, status, detail

            elif etype == "done":
                tags = ", ".join(event["tags"])
                status = f"✅ Done — {event['tag_count']} tags, {event['latency_ms']:.0f}ms"
                detail += f"\n---\nTotal: {event['tag_count']} tags, {event['latency_ms']:.0f}ms"
                yield tags, status, detail

    with gr.Blocks(title="Danbooru Tag Expander") as demo:
        gr.Markdown("# Danbooru Tag Expander\nEnter a description in any language, get Danbooru tags.")

        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(
                    label="Description",
                    placeholder="e.g. 银发红眼穿校服的少女 / a girl with dragon horns and tail",
                    lines=3,
                    autofocus=True,
                )
                with gr.Row():
                    temperature = gr.Slider(0.1, 0.8, value=DEFAULT_TEMPERATURE, step=0.1, label="Temperature")
                    two_stage = gr.Checkbox(value=True, label="Two-stage")
                btn = gr.Button("Expand", variant="primary")

            with gr.Column(scale=4):
                tags_output = gr.Textbox(label="Tags", lines=4)
                status_output = gr.Textbox(label="Status", lines=1, interactive=False)
                detail_output = gr.Textbox(label="RAG References", lines=12, interactive=False)

        btn.click(fn=expand_ui, inputs=[prompt, temperature, two_stage], outputs=[tags_output, status_output, detail_output])
        prompt.submit(fn=expand_ui, inputs=[prompt, temperature, two_stage], outputs=[tags_output, status_output, detail_output])

    return demo


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Danbooru Tag Expander Server")
    parser.add_argument("--host", default=SERVER_HOST)
    parser.add_argument("--port", type=int, default=SERVER_PORT)
    parser.add_argument("--webui", action="store_true", help="Launch Gradio WebUI")
    parser.add_argument("--webui-port", type=int, default=WEBUI_PORT)
    args = parser.parse_args()

    validate()

    rag = DanbooruRAG()

    if args.webui:
        import threading

        demo = create_webui()

        def run_api():
            uvicorn.run(app, host=args.host, port=args.port, log_level="warning")

        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()
        print(f"\nFastAPI: http://{args.host}:{args.port}")
        print(f"API docs: http://{args.host}:{args.port}/docs")
        print(f"WebUI: http://localhost:{args.webui_port}\n")
        demo.launch(server_name=args.host, server_port=args.webui_port, share=False)
    else:
        uvicorn.run(app, host=args.host, port=args.port)
