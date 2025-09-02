from d3pm_absorbing import D3PMAbsorbing
from md4 import MD4

run_length_args = {
    "short": {
        "warmup_iters": 25,
        "max_iters": 300,
        "eval_interval": 100,
    },
    "long": {
        "warmup_iters": 2_500,
        "max_iters": 500_000,
        "eval_interval": 25_000,
    },
}


def text8(model_type="d3pm", run_length="long"):
    if run_length not in run_length_args:
        run_length = "long"
        print(f"Run length {run_length} not found, defaulting to long")
    
    model_args = dict(
        vocab_size=27,
        n_embed=768,
        n_heads=768 // 64,
        n_blocks=12,
        n_cond=128,
        dropout=0.025,
        T=1000,
    )
    if model_type == "d3pm":
        model_cls = D3PMAbsorbing
        model_args["lambda_ce"] = 0.05
    elif model_type == "md4":
        model_cls = MD4
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    training_args = dict(
        batch_size=256,
        learning_rate=1e-3,
        min_lr=1e-5,
        gradient_accumulation_steps=4,
        warmup_iters=run_length_args[run_length]["warmup_iters"],
        max_iters=run_length_args[run_length]["max_iters"],
        eval_interval=run_length_args[run_length]["eval_interval"],
        eval_iters=1000,
        weight_decay=0.1,
        training_seed=1,
    )
    return model_cls, model_args, training_args

def text8_2gpu(model_type="d3pm", run_length="long"):
    model, model_args, training_args = text8(model_type, run_length)
    training_args["gradient_accumulation_steps"] = 2
    training_args["eval_iters"] = 500
    return model, model_args, training_args

def text8_4gpu(model_type="d3pm", run_length="long"):
    model, model_args, training_args = text8(model_type, run_length)
    training_args["gradient_accumulation_steps"] = 1
    training_args["eval_iters"] = 250
    return model, model_args, training_args


def openwebtext_8gpu(model_type="d3pm", run_length="long"):
    if run_length not in run_length_args:
        run_length = "long"
        print(f"Run length {run_length} not found, defaulting to long")

    model_args = dict(
        vocab_size=50257,
        n_embed=768,
        n_heads=768 // 64,
        n_blocks=12,
        n_cond=128,
        dropout=0.0,
        T=1000,
    )
    if model_type == "d3pm":
        model_cls = D3PMAbsorbing
        model_args["lambda_ce"] = 0.05
    elif model_type == "md4":
        model_cls = MD4
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    training_args = dict(
        dataset="openwebtext",
        batch_size=16,
        seq_len=1024,
        learning_rate=6e-4,
        min_lr=1e-5,
        gradient_accumulation_steps=8,
        warmup_iters=run_length_args[run_length]["warmup_iters"],
        max_iters=run_length_args[run_length]["max_iters"],
        eval_interval=run_length_args[run_length]["eval_interval"],
        eval_iters=400,
        weight_decay=0.1,
        training_seed=9,
    )
    return model_cls, model_args, training_args


def openwebtext_32gpu(model_type="d3pm", run_length="long"):
    model, model_args, training_args = openwebtext_8gpu(model_type, run_length)
    training_args["gradient_accumulation_steps"] = 2
    training_args["eval_iters"] = 50
    return model, model_args, training_args

