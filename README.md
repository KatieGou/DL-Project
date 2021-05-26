# DL-Porject
There are three files of code in this project, including train.py, test.py and setting.py.

- The train.py includes the implement of transformer model and training peorcesss. 
- The test.py includes the BLEU score of test data, based on the trained model.
- The setting.py includes all hyper-parameters which can be modified for the model.

## To train and test a model:
- **Step 1** Modify hyper-parameters in setting.py.
- **Step 2** Run train.py by the command "python train.py". Dataset will be downloaded automatically. Subword vocabulary will be stored for test.
- **Step 3** Run test.py by the command "python test.py". The final BLEU score of test data will be computed and printed.
