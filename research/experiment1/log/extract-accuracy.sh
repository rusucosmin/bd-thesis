echo "teacher accuracy"
awk 'FNR==113 {print FILENAME, $0}' $1
echo "student accuracy"
awk 'FNR==418 {print FILENAME, $0}' $1
echo "student2 accuracy"
awk 'FNR==519 {print FILENAME, $0}' $1
echo "student3 accuracy"
awk 'FNR==620 {print FILENAME, $0}' $1
echo "student4 accuracy"
awk 'FNR==216 {print FILENAME, $0}' $1
echo "student5 accuracy"
awk 'FNR==317 {print FILENAME, $0}' $1
