from .clip import ClipTextEncoder, ClipTokenizer
from .siglip import SigLIP2TextEncoder, SigLIP2Tokenizer


def fetch_text_encoders(model_name):
    """Return encoder class and latent dimension."""
    if model_name == 'clip':
        return ClipTextEncoder(), 512
    if model_name == 'siglip2':
        enc = SigLIP2TextEncoder()
        return enc, enc.hidden_size
    return None, None


def fetch_tokenizers(model_name):
    """Return tokenizer class."""
    if model_name == 'clip':
        return ClipTokenizer()
    if model_name == 'siglip2':
        return SigLIP2Tokenizer()
    return None
