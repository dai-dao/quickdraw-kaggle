mkdir input
mkdir input/train_simplified

chmod 600 /root/.kaggle/kaggle.json

kaggle competitions download quickdraw-doodle-recognition -f train_simplified.zip -p input
unzip input/train_simplified.zip -d input/train_simplified

kaggle competitions download quickdraw-doodle-recognition -f test_simplified.csv -p input
kaggle competitions download quickdraw-doodle-recognition -f sample_submission.csv -p input
