# Language-driven Visual Saliency Prediction

This code implements the framework that simultaneously enhances the interpretability and generalizability of saliency prediction models with language representation. It contains three principal components:
- A prompting-based method for extracting image descriptions and the spatial mappings of semantics.
- A principled framework that derives human attention based on language, and augments visual saliency models with improved generalizability across datasets.
- An analytic framework for studying the impacts of semantics on attention deployment without requiring manual annotations.

### Requirements
1. Requirements for Pytorch. We use Pytorch 2.1.2 in our experiments.
2. Requirements for Tensorflow. We only use the tensorboard for visualization.
3. Requirements for LLaVA-Next with Llama-8B. 

### Data Preparation
To obtain language data for saliency prediction, you need to first modify a few lines in the llava library to enable the generation of attention and language features. Specifically,
1. Disabling the memory-efficient attention (e.g., FlashAttention2), which does not output an attention map, and use the standard attention. In particular, in line 700 of modeling_llama.py (llava library), switch attention to "inefficient_scaled_dot_product_attention".
2. By default, LLaVA only outputs the predicted logit for faster inference. To get the language features and attention, in line 1360 of utils.py (llava library). set "generation_config.output_attentions=True", "generation_config.output_hidden_states", and "generation_config.return_dict_in_generate = True".
3. Adjust the data directory in our "prompting_scripy.py", reshape all images to be 640x480, and run the script.

### Model training
To train the language-only model in our paper:
```
python main.py --batch_size 10 --epochs 20 --lr 1e-4 --anno_path SALICON_PATH --language_path PATH_TO_LANGUAGE_FEATURE --log_dir LOG
```

To train the reweighting model with sequential training:
```
python main.py --batch_size 10 --epochs 20 --lr 1e-4 --anno_path SALICON_PATH --language_path PATH_TO_LANGUAGE_FEATURE --log_dir LOG --is_reweight --weights WEIGHT_OF_LANGUGE_ONLY_MODEL
```

To train the reweighting model with joint training:
```
python main.py --batch_size 10 --epochs 20 --lr 1e-4 --anno_path SALICON_PATH --language_path PATH_TO_LANGUAGE_FEATURE --log_dir LOG --is_reweight --joint_training
```

### Semantic Weights
To obtain the semantic weights for attention modeling:
```
python main.py --batch_size 10 --epochs 20 --lr 1e-4 --anno_path SALICON_PATH --language_path PATH_TO_LANGUAGE_FEATURE --weights WEIGHT_OF_LANGUGE_ONLY_MODEL --mode language_parsing --save_dir SAVE_JSON
```
```
python main.py --batch_size 10 --epochs 20 --lr 1e-4 --anno_path SALICON_PATH --language_path PATH_TO_LANGUAGE_FEATURE --weights WEIGHT_OF_LANGUGE_ONLY_MODEL --mode language_stat --save_dir SAVE_JSON
```


