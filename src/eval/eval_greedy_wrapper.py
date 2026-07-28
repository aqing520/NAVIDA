import runpy

import transformers


RealGenerationConfig = transformers.GenerationConfig


def GreedyGenerationConfig(*args, **kwargs):
    kwargs["do_sample"] = False
    kwargs.pop("temperature", None)
    kwargs.pop("top_p", None)
    return RealGenerationConfig(*args, **kwargs)


transformers.GenerationConfig = GreedyGenerationConfig

runpy.run_path("src/eval/eval.py", run_name="__main__")
