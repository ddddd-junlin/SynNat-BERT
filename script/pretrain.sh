python train_roberta.py \
--dataset_path=dataset \
--tokenizer_path=ChemBERTa-100M-MLM \
--model_type=mlm \
--output_dir=pubchem_coconut \
--run_name=pubchem_coconut \
--vocab_size=7924 \
--num_train_epochs=3 \
--save_steps=100
