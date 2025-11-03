for prefix in ran rbf; do
  for num in 050 100 150 200 250 300; do
    mkdir "${prefix}-${num}"
    sed -e "s/XXX/$num/g" -e "s/YYY/$prefix/g" model_training_tutorial.py > $prefix-$num/model.py
  done
done

ls -d */ > dir-list.txt
