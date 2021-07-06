import numpy as np
import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from .utils import get_available_devices


class GPTLM():
    def __init__(self, model_name_or_path='gpt2'):
        self.start_token = "<|endoftext|>"

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path, bos_token=self.start_token)

        self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        self.device, gpu_ids = get_available_devices()
        self.model.to(self.device)
        self.model.eval()

    def get_probabilities(self, raw_text, top_k=1000):

        inputs = self.tokenizer(raw_text, return_tensors="pt", truncation= True, max_length = 1024).to(self.device
                                                                  )
        # add input_seq as labels to predict probabilities for each timestep
        logits = self.model(**inputs, labels=inputs['input_ids'])  # (1, input_size, vocab_size)

        # do softmax to get probabilities
        pred_probs = torch.softmax(logits.logits[0, :-1], dim=-1)
        y = inputs['input_ids'][0, 1:]

        # Sort the probabiltiies for each timestep
        sorted_preds = np.argsort(-pred_probs.data.cpu().numpy())

        # find where the true token is positioned in the predicted sorted probabilities list
        true_topk_pos = [int(np.where(sorted_preds[i] == y[i].item())[0][0]) for i in range(y.shape[0])]

        # Probabilities for our original input sequences
        true_topk_probs = pred_probs[np.arange(0, y.shape[0], 1), y].data.cpu().numpy().tolist()
        true_topk_probs = list(map(lambda x: round(x, 5), true_topk_probs))

        true_topk = list(zip(true_topk_pos, true_topk_probs))

        bpe_strings = [self.tokenizer.decoder[s.item()] for s in inputs['input_ids'][0]]
        bpe_strings = [self.postprocess(s) for s in bpe_strings]

        # Get the k predicted probabilties for each timestep
        pred_topk = [
            list(zip([self.tokenizer.decoder[p] for p in sorted_preds[i][:top_k]],
                     list(map(lambda x: round(x, 5),
                              pred_probs[i][sorted_preds[i][
                                            :top_k]].data.cpu().numpy().tolist()))))
            for i in range(y.shape[0])]
        pred_topk = [[(self.postprocess(t[0]), t[1]) for t in pred] for pred in pred_topk]

        response = {
            'bpe_strings': bpe_strings,
            'true_topk': true_topk,
            'pred_topk': pred_topk
        }
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return response

    @staticmethod
    def postprocess(token):
        with_space = False
        with_break = False
        if token.startswith('Ġ'):
            with_space = True
            token = token[1:]
        elif token.startswith('â'):
            token = ' '
        elif token.startswith('Ċ'):
            token = ' '
            with_break = True

        token = '-' if token.startswith('â') else token
        token = '“' if token.startswith('ľ') else token
        token = '”' if token.startswith('Ŀ') else token
        token = "'" if token.startswith('Ļ') else token

        return token