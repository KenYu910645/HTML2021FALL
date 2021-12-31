echo "============ C = 0.01 =============="
./svm-train -s 0 -t 2 -g 10 -c 0.01 satimage_1vo.scale
./svm-predict satimage_1vo.scale.t satimage_1vo.scale.model output.txt
echo "============ C = 0.1 =============="
./svm-train -s 0 -t 2 -g 10 -c 0.1 satimage_1vo.scale
./svm-predict satimage_1vo.scale.t satimage_1vo.scale.model output.txt
echo "============ C = 1 =============="
./svm-train -s 0 -t 2 -g 10 -c 1 satimage_1vo.scale
./svm-predict satimage_1vo.scale.t satimage_1vo.scale.model output.txt
echo "============ C = 10 =============="
./svm-train -s 0 -t 2 -g 10 -c 10 satimage_1vo.scale
./svm-predict satimage_1vo.scale.t satimage_1vo.scale.model output.txt
echo "============ C = 100 =============="
./svm-train -s 0 -t 2 -g 10 -c 100 satimage_1vo.scale
./svm-predict satimage_1vo.scale.t satimage_1vo.scale.model output.txt