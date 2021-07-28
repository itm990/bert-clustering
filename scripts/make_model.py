from sentence_transformers import SentenceTransformer, models


def set_pooling(pooling_type):
    pooling_dict = { "cls": False, "max" : False, "mean": False }
    pooling_dict[pooling_type] = True
    return pooling_dict


def make_model(model_name_or_path, pooling_strategy):
    transformer = models.Transformer(model_name_or_path=model_name_or_path)
    pooling_setting = set_pooling(pooling_strategy)
    pooling = models.Pooling(
        word_embedding_dimension=transformer.get_word_embedding_dimension(),
        pooling_mode_cls_token=pooling_setting["cls"],
        pooling_mode_max_tokens=pooling_setting["max"],
        pooling_mode_mean_tokens=pooling_setting["mean"],
    )
    model = SentenceTransformer(modules=[transformer, pooling])
    model.max_seq_length = 128
    return model
