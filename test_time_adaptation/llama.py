import torch
import transformers
from transformers import AutoTokenizer

class Llama():
    def __init__(self, model_name, num_sequences=5):
        super(Llama, self).__init__()

        self._model_name = model_name
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)

        self._pipeline = transformers.pipeline(
            "text-generation",
            model = self._model_name,
            torch_dtype = torch.float16,
        )

        self._num_sequences = num_sequences

    def generate_sentence(self, input_words):

        sequences = self._pipeline(
            input_words,
            do_sample=True,
            top_k=10,
            num_return_sequences=self._num_sequences,
            eos_token_id = self._tokenizer.eos_token_id,
            truncation=True,
            max_length=100
        )

        return sequences

# just for testing
# remove this when integrating with the main code

# model_name = "meta-llama/Llama-Guard-3-8B-INT8"
# llama = Llama(model_name, 5)
# sequences = llama.generate_sentence("dog", 5)

# for seq in sequences:
#     print(f"Results: {seq['generated_text']}")