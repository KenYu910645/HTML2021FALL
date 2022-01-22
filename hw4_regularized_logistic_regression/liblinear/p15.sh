TRAIN='hw4_train_trans.dat'
TEST='hw4_test_trans.dat'

# Test on real testing set
./train -q -s 0 -c 0.005 -e 0.000001 ../$TRAIN; ./predict ../$TEST $TRAIN.model 123.csv