import torch
import sentence_transformers
from transformers import AutoModel, AutoConfig
from functools import partial
from operator import itemgetter
import re


def mean_pooling(emb, mask):
    mask = mask.unsqueeze(-1).expand(emb.size()).float()
    sum_emb = torch.sum(emb * mask, 1)
    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
    return sum_emb / sum_mask


class BaseTransformer(torch.nn.Module):
    """
    Base class for Transformers

    Arguments:
    ----------
    transformer: str or a transformer
        from Sentence Transformer or HF (not tested with others)
    pooler: str
        method to reduce the output of transformer layers
    normalize: boolean
        return normalized outputs or as it is 
    """
    def __init__(self, transformer, pooler, normalize, **kwargs):
        super(BaseTransformer, self).__init__()
        self.transform, self.pooler, self.normalize = self.construct(
            transformer, pooler, normalize, **kwargs)
        self.__normalize = normalize
        self.__pooler = pooler

    def construct(self, transformer, pooler, normalize, **kwargs):
        """
        Construct the transformer and the pooler
        """
        return self.construct_transformer(transformer, **kwargs), \
            self.construct_pooler(pooler, **kwargs), \
                self.construct_normalizer(normalize)

    def encode(self, x):
        ids, mask = x
        return self.normalize(self.pooler(self.transform(ids, mask), mask))

    def forward(self, x):
        return self.encode(x)

    def construct_normalizer(self, normalize):
        if normalize:
            return torch.nn.functional.normalize
        else:
            return lambda x : x

    def construct_transformer(self, *args, **kwargs):
        """
        Construct the transformer
        """
        raise NotImplementedError("")

    def construct_pooler(self, *args, **kwargs):
        """
        Construct pooler to reduce output of Transformer layers
        """
        return lambda x: x

    @property
    def repr_dims(self):
        """
        The dimensionality of output/embedding space
        """
        raise NotImplementedError("")

    @property
    def _pooler(self):
        return self.__pooler

    @property
    def _vocabulary(self):
        raise NotImplementedError("")

    @property
    def _normalize(self):
        return self.__normalize

    @property
    def config(self):
        """
        The dimensionality of output/embedding space
        """
        return f"V: {self._vocabulary}; D: {self.repr_dims}; Normalize: {self._normalize}; Pooler: {self._pooler}"


class STransformer(BaseTransformer):
    """
    Create Transformers using Sentence Bert Library
    * Use default pooler of trained model (yields better results)
    * Use HTransformer if you want to customize pooler
    * mean pooler is equivalent to using mean_pooling on 
      HTransformer's last_hidden_state followed by an optional normalize layer


    Arguments:
    ----------
    transformer: str or a transformer
        from Sentence Transformer 
    normalize: boolean
        return normalized outputs or as it is 
    """
    def __init__(self, transformer='roberta-base', normalize=False, **kwargs):
        super(STransformer, self).__init__(transformer, None, normalize)

    def construct_transformer(self, transformer, **kwargs):
        if isinstance(transformer, str):
            return sentence_transformers.SentenceTransformer(transformer)
        else:
            return transformer

    def encode(self, x):
        ids, mask = x
        out = self.transform({'input_ids': ids, 'attention_mask': mask})
        return self.normalize(out['sentence_embedding'])

    @property
    def repr_dims(self):
        return self.transform[1].word_embedding_dimension

    @property
    def _pooler(self):
        keys = [x for x in self.transform[1].__dict__['config_keys']\
             if re.match("pooling*", x)]        
        return dict(zip(keys, itemgetter(*keys)(self.transform[1].__dict__)))

    @property
    def _vocabulary(self):
        return self.transform[0].vocab_size


class HTransformer(BaseTransformer):
    """
    Create Transformers using Huggingface library

    Arguments:
    ----------
    transformer: str or a transformer
        from Huggingface 
    pooler: str
        method to reduce the output of transformer layers
        * Support for mean, None (identity), concat and cls
    normalize: boolean
        return normalized outputs or as it is 
    c_layers: list
        concatenate these layers when pooler is concat (ignored otherwise)
    """
    def __init__(self, transformer='roberta-base', pooler=None, normalize=False, c_layers=[-1, -4]):
        if pooler != "concat":
            c_layers = None
        super(HTransformer, self).__init__(transformer, pooler, normalize, c_layers=c_layers)
        self._c_layers = c_layers

    def construct_transformer(self, transformer, c_layers):
        output_hidden_states = True if isinstance(c_layers, list) else True
        if isinstance(transformer, str):
            config = AutoConfig.from_pretrained(
                transformer, 
                output_hidden_states=output_hidden_states)
            return AutoModel.from_pretrained(transformer, config=config)
        else:
            return transformer

    def encode(self, ids, mask):
        out = self.transform(input_ids=ids, attention_mask= mask)
        return self.normalize(self.pooler(out, mask))

    def construct_pooler(self, pooler: str, c_layers: list or None):
        if pooler is None:
                return lambda x: x
        elif pooler == 'concat':
            assert isinstance(c_layers, list), "list is expected for concat"
            def f(x, m, c_l):
                r = []
                for l in c_l:
                    r.append(
                        mean_pooling(x['last_hidden_state'][l], m))
                return torch.hstack(r)
            return partial(f)(c_layers=c_layers)
        elif pooler == 'mean':
            def f(x, m):
                return mean_pooling(x['last_hidden_state'], m)
            return f
        elif pooler == 'cls':
            def f(x, *args):
                return x['last_hidden_state'][:, 0]
            return f
        else:
            print(f'Unknown pooler type encountered: {pooler}')

    @property
    def repr_dims(self):
        d = self.transform.embeddings.word_embeddings.embedding_dim
        if self._pooler == "concat":
            return d * len(self._c_layers) 
        else:
            return d

    @property
    def config(self):
        """
        The dimensionality of output/embedding space
        """
        return self.transform.config
