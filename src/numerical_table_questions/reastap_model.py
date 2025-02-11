from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def reastap_model(hf_version_path: str = "Yale-LILY/reastap-large"):
    return AutoModelForSeq2SeqLM.from_pretrained(hf_version_path)
