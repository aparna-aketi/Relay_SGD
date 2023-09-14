python trainer.py --lr=0.1 --skew=0.1 --epochs=200 --arch=resnet --momentum=0.9 --seed=1234 --batch-size=512 --world_size=16
cd ./outputs
python dict_to_csv.py --lr=0.1 --seed=1234 --arch=resnet --world_size=16  --skew=0.1
cd ..

