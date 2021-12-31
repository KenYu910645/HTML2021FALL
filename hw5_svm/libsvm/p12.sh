echo "============ 2 vs. not 2 =============="
./svm-train -s 0 -t 1 -d 3 -c 10 -r 1 -g 1 satimage_2vo.scale
./svm-predict satimage_2vo.scale satimage_2vo.scale.model output.txt
echo "============ 3 vs. not 3 =============="
./svm-train -s 0 -t 1 -d 3 -c 10 -r 1 -g 1 satimage_3vo.scale
./svm-predict satimage_3vo.scale satimage_3vo.scale.model output.txt
echo "============ 4 vs. not 4 =============="
./svm-train -s 0 -t 1 -d 3 -c 10 -r 1 -g 1 satimage_4vo.scale
./svm-predict satimage_4vo.scale satimage_4vo.scale.model output.txt
echo "============ 5 vs. not 5 =============="
./svm-train -s 0 -t 1 -d 3 -c 10 -r 1 -g 1 satimage_5vo.scale
./svm-predict satimage_5vo.scale satimage_5vo.scale.model output.txt
echo "============ 6 vs. not 6 =============="
./svm-train -s 0 -t 1 -d 3 -c 10 -r 1 -g 1 satimage_6vo.scale
./svm-predict satimage_6vo.scale satimage_6vo.scale.model output.txt
