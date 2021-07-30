# BERTを使用した文書クラスタリング

## 説明

Transformers, SentenceTransformersを使用して実装した，BERTによる文書クラスタリングの実装です．
文書データとその文書のラベルデータを使用して，ファインチューニング，クラスタリングを行います．
ファインチューニングには，文書分類，文類似度のいずれかを使用します．

## 要件

- Python 3.7.3
- PyToch 1.6.0
- NumPy 1.19.4
- scikit-learn 0.23.2
- Transformers 4.2.0dev0
- SentenceTransformers 0.4.1
- tqdm 3.4.3

## 使用方法

- 文書分類を使用

  - ファインチューニング
  
  ```
  $ python finetuning_text_classification.py \
      --max_epoch 1 \
      --class_num 2 \
      --logging_steps 10 \
      --eval_steps 100 \
      --train_sent [train document data] \
      --train_cls [train label data] \
      --valid_sent [validation document data] \
      --valid_cls [validation label data] \
      --model_name_or_path [pretrained model data] \
      --output_dir [save model name] \
      --seed 42 \
      --multi_gpu
  ```

- 文書類似度を使用

  - データの作成
  
  ```
  $ python make_examples.py \
      --model_name_or_path [pretrained model data] \
      --pooling_strategy [mean or cls or max] \
      --load_sent [document data] \
      --load_cls [label data] \
      --save_file [save examples name] \
      --seed 42 \
      --data_type [single, double or triple] \
      --sampling_strategy [random, hard or semi-hard] \
      --triplet_margin 5.0
  ```
  
  - ファインチューニング
  
  ```
  $ python finetuning_text_similarity.py
      --model_name_or_path [pretrained model data] \
      --train_examples [train examples data] \
      --valid_examples [validation examples data] \
      --output_path [save model name] \
      --pooling_strategy [mean, cls or max] \
      --loss_type [batch_all_triplet, batch_hard_soft_margin_triplet, batch_hard_triplet, batch_semi_hard_triplet, cosine_similarity or triplet] \
      --check_steps 3 \
      --epochs 1 \
      --batch_size 16 \
      --evaluation_steps 1000 \
      --warmup_steps 10000 \
      --limit_steps 3 \
      --seed 42 \
      --use_amp
  ```

- クラスタリング
  
  ```
  $ python clustering.py
      --cls_type [kmeans, euclidean, cosine or manhattan] \
      --use_classifier \
      --model_path [model data] \
      --num_cls 2 \
      --input_file [document data] \
      --output_file [predicted label data]
  ```

- クラスタの評価
  
  ```
  $ python evaluate_cluster.py
      --gold_path [gold label data] \
      --pred_path [predicted label data] \
      --output_path [evaluated data]
  ```