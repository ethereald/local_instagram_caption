# Local Instagram Caption

Rewrites a factual image description into an Instagram caption with a local
Hugging Face causal language model. It does not use Ollama, an API, or network
downloads.

## Workflow

1. Load `Florence-2-base` or `Florence-2-pixelprose` with
   `Florence2ModelLoader`.
2. Connect an image and the model to `Florence2Run`.
3. Set the Florence task to `more_detailed_caption`.
4. Connect `Florence2Run.caption` to
   `Local Instagram Caption.image_description`.
5. Queue the workflow. The caption appears on the node and is also returned as
   a `STRING` output.

Models are discovered below `ComfyUI/models/LLM`. Only directories whose
`config.json` declares a `*ForCausalLM` architecture are listed.

The caption node's seed supports ComfyUI's `fixed`, `increment`, `decrement`,
and `randomize` controls.
