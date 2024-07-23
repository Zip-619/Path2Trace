move_types=('move_in' 'move_out')

# 对所有的数据进行训练集和测试集的划分

for move_type in ${move_types[*]}
do
    echo "正在处理"${move_type}"数据..."
    path='../data/city_inmigration/data/edgelist/'${move_type}

    for edgelistfile in $(ls ${path})
    do 
        echo $edgelistfile; 
        t=${edgelistfile%-*};
        # echo ${t};
        inputfile=${path}'/'${edgelistfile}
        # outputfile="../pengpai/city_graph/data/embedding/"${move_type}'/'${t}"-embedding.txt" 
        python3 LP_construct_dataset.py --input ${inputfile} 
    done
done
