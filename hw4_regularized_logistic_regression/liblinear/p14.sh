TRAIN='hw4_p14_train.dat'
TEST='hw4_p14_val.dat'

./train -q -s 0 -c 5000 -e 0.000001 ../$TRAIN; ./predict ../$TEST $TRAIN.model 123.csv
./train -q -s 0 -c 50 -e 0.000001 ../$TRAIN; ./predict ../$TEST $TRAIN.model 123.csv
./train -q -s 0 -c 0.5 -e 0.000001 ../$TRAIN; ./predict ../$TEST $TRAIN.model 123.csv
./train -q -s 0 -c 0.005 -e 0.000001 ../$TRAIN; ./predict ../$TEST $TRAIN.model 123.csv
./train -q -s 0 -c 0.00005 -e 0.000001 ../$TRAIN; ./predict ../$TEST $TRAIN.model 123.csv

# Test on real testing set
./train -q -s 0 -c 0.005 -e 0.000001 ../$TRAIN; ./predict ../$TEST $TRAIN.model 123.csv
./predict ../hw4_test_trans.dat $TRAIN.model 123.csv
