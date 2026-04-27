# seq2seqYorubaADR
Finetuning Byt5 Model for Sequence-To-Sequence Yoruba Diacritic Restoration

Directions:
Pull local copy and run files within 'training'.
Trained models may then be evaluated using files within 'eval'.
Returns evaluation results along metrics BLEU (Bilingual Evaluation Understudy) and Character Access.

Directory:

Training
Files for executing model finetuning on Byt5 Small and Base models at varied training parameters, outputs trained parameters to local root directory.
Assumes user access to Yankari 2024 Huggingsace dataset.

Eval:
Contains model evalutation on Yoruba 2024 Benchmark Dataset included in root level file yad_test.json
