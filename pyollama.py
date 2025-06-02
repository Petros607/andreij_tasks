import ollama # pip install ollama

def convert_nanosec_to_sec(nanosec):
    return nanosec / 1_000_000_000

def run_model(model, prompt, stream=False):
    try:
        res = ollama.generate(
            model=model,
            prompt=prompt,
            stream=stream
        )
        return res
    except Exception:
        return 'Нет ответа от модели'

def list_models_names():
    models = [model for model in ollama.list()['models']]
    return [model['model'] for model in models]