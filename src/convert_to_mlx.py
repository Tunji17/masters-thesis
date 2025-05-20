import os
import mlx.core as mx
from transformers import (AutoModelForTokenClassification,
                          AutoTokenizer, AutoConfig)


def convert_hf_bert_to_mlx(hf_model_name: str, mlx_model_path: str):
    """
    Converts a Hugging Face BERT-based model to MLX format
    and saves the weights, tokenizer, and configuration.

    Args:
      hf_model_name (str): The name of the Hugging Face model
      (e.g., "medicalai/ClinicalBERT" ou "blaze999/Medical-NER").
      mlx_model_path (str): The path to save the MLX model files.
    """
    print(f"Creating directory for MLX model: {mlx_model_path}")
    os.makedirs(mlx_model_path, exist_ok=True)

    # Load the Hugging Face model, tokenizer, and config
    # Using AutoModelForTokenClassification as ClinicalBERT
    # is often used for NER/token classification tasks
    # and this will include the classification head weights.
    print(f"Loading Hugging Face model '{hf_model_name}'...")
    try:
        model = AutoModelForTokenClassification.from_pretrained(hf_model_name)
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        config = AutoConfig.from_pretrained(hf_model_name)
    except Exception as e:
        print(f"Error loading model from Hugging Face: {e}")
        return

    # Save tokenizer and config to the MLX model directory
    print(f"Saving tokenizer and configuration to {mlx_model_path}...")
    try:
        tokenizer.save_pretrained(mlx_model_path)
        config.save_pretrained(mlx_model_path)  # Saves config.json
    except Exception as e:
        print(f"Error saving tokenizer/config: {e}")
        return

    # Convert PyTorch state dictionary to MLX weights
    print("Converting model weights to MLX format...")
    hf_state_dict = model.state_dict()
    mlx_weights = {}
    for name, tensor in hf_state_dict.items():
        # Convert PyTorch tensor to NumPy array, then to MLX array
        # For standard BERT models,
        # weight names and structures are generally compatible.
        mlx_weights[name] = mx.array(tensor.numpy())
        print(f"  Converted: {name} with shape {tensor.shape}")

    # Save MLX weights
    weights_path = os.path.join(mlx_model_path, "weights.safetensors")
    print(f"Saving MLX weights to {weights_path}...")
    try:
        mx.save_safetensors(weights_path, mlx_weights)
    except Exception as e:
        print(f"Error saving MLX weights: {e}")
        return

    print("\nConversion complete.")
    print(f"""MLX model weights, tokenizer,
          and config saved in: {mlx_model_path}""")
    print(f"- Weights: {weights_path}")
    print(f"- Config: {os.path.join(mlx_model_path, 'config.json')}")


if __name__ == "__main__":

    # hugging_face_model_name = "dslim/bert-large-NER"
    # hugging_face_model_name = "Simonlee711/Clinical_ModernBERT"
    # hugging_face_model_name = "blaze999/Medical-NER"
    hugging_face_model_name = "d4data/biomedical-ner-all"
    mlx_output_path = "./models/mlx_biomedical-ner"

    convert_hf_bert_to_mlx(hugging_face_model_name, mlx_output_path)
