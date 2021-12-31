echo "============ r = 0.1 =============="
./svm-train -s 0 -t 2 -g 0.1 -c 0.1 satimage_1vo.scale
./svm-predict satimage_1vo.scale.t satimage_1vo.scale.model output.txt
echo "============ r = 1 =============="
./svm-train -s 0 -t 2 -g 1 -c 0.1 satimage_1vo.scale
./svm-predict satimage_1vo.scale.t satimage_1vo.scale.model output.txt
echo "============ r = 10 =============="
./svm-train -s 0 -t 2 -g 10 -c 0.1 satimage_1vo.scale
./svm-predict satimage_1vo.scale.t satimage_1vo.scale.model output.txt
echo "============ r = 100 =============="
./svm-train -s 0 -t 2 -g 100 -c 0.1 satimage_1vo.scale
./svm-predict satimage_1vo.scale.t satimage_1vo.scale.model output.txt
echo "============ r = 1000 =============="
./svm-train -s 0 -t 2 -g 1000 -c 0.1 satimage_1vo.scale
./svm-predict satimage_1vo.scale.t satimage_1vo.scale.model output.txt