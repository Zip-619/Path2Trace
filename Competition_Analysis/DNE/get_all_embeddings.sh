move_types=('move_in' 'move_out')

for move_type in ${move_types[*]}
do
    echo "正在处理"${move_type}"类型数据..."
    path='../../data/city_inmigration/data/LP_edgelist_70/'${move_type}
    
    for time in $(ls ${path})
    do
        # shellcheck disable=SC2086
        base=${path}'/'${time}
        # model = 
        # output='DNE/emb/LP_edgelist/'${move_type}'/'${time}
        # mkdir ${output}
        python3 DNE-model.py --type 1 --model weighted --base ${base} 
    done

    cp -a ../../data/city_inmigration/data/emb/. ../../data/city_inmigration/data/${move_type}'/'
    rm -r ../../data/city_inmigration/data/emb/*

done


# python3 DNE-model.py --base ../data/LP_edgelist