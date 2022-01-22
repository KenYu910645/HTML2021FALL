TRAIN='hw4_train_trans.dat'
TEST='hw4_train_trans.dat'

./train -q -s 0 -c 5000 -e 0.000001 ../$TRAIN; ./predict ../$TEST $TRAIN.model 123.csv
./train -q -s 0 -c 50 -e 0.000001 ../$TRAIN; ./predict ../$TEST $TRAIN.model 123.csv
./train -q -s 0 -c 0.5 -e 0.000001 ../$TRAIN; ./predict ../$TEST $TRAIN.model 123.csv
./train -q -s 0 -c 0.005 -e 0.000001 ../$TRAIN; ./predict ../$TEST $TRAIN.model 123.csv
./train -q -s 0 -c 0.00005 -e 0.000001 ../$TRAIN; ./predict ../$TEST $TRAIN.model 123.csv