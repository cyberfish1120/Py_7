#python data_precess.py --maxlen 256
#python main.py --maxlen 256 --bert_layers 24 --bert_lr 1e-05 --batch_size 16
_seed=(1997 11 20 12 21)

for((i=0;i<5;++i))
do
  python main.py --seed ${_seed[i]}
done
