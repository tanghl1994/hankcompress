python maincifar_scatter.py -a resnet18 --lr 0.0001 --batch-size 1024 --epochs 200 --freeze-epoch -1 --dist-url 'tcp://127.0.0.1:7008' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 ../img