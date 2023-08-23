source ~/anaconda/etc/profile.d/conda.sh
conda activate artifact

nvcc -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75  -lcusparse -o cusparse_convert cusparse_convert.cu
nvcc -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -o pit_csr pit_csr.cu
mkdir -p log
H=4096
W=4096
rm convert.csv
for sparsity in 0.4 0.5 0.9 0.95 0.99
do
    echo Sparsity:$sparsity
    ./cusparse_convert $sparsity $H $W > log/1_${sparsity}.log
done
python triton_convert.py --block 16
python triton_convert.py --block 32
python pit_convert.py
for sparsity in 0.4 0.5 0.9 0.95 0.99
do
    echo Sparsity:$sparsity
    ./pit_csr $sparsity $H $W >> convert.csv
done

python plot.py
