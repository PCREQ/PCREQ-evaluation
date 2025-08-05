import subprocess
import sys
import os
import time
import json

#转化为你的项目路径
proj_path_dict = {
  "SimpleHTR": "/home/lei/compatibility_analysis/tensorflow/2.4/SimpleHTR/src",
  "StyleGAN-Tensorflow": "/home/lei/compatibility_analysis/tensorflow/1.13/StyleGAN-Tensorflow",
  "cnn_captcha": "/home/lei/compatibility_analysis/tensorflow/1.7/cnn_captcha",
  "deep-belief-network": "/home/lei/compatibility_analysis/tensorflow/1.5/deep-belief-network",
  "crnn.pytorch": "/home/lei/compatibility_analysis/pytorch/1.2/crnn.pytorch",
  "siren": "/home/lei/compatibility_analysis/pytorch/1.3/siren",
  "Deep-SAD-PyTorch": "/home/lei/compatibility_analysis/pytorch/1.1/Deep-SAD-PyTorch/src",
  "PyTorch-ENet": "/home/lei/compatibility_analysis/pytorch/1.1/PyTorch-ENet",
  "ConSinGAN": "/home/lei/compatibility_analysis/pytorch/1.1/ConSinGAN",
  "spert": "/home/lei/compatibility_analysis/pytorch/1.4/spert",
  "hifi-gan": "/home/lei/compatibility_analysis/pytorch/1.4/hifi-gan",
  "DexiNed": "/home/lei/compatibility_analysis/pytorch/1.5/DexiNed",
  "KiU-Net-pytorch": "/home/lei/compatibility_analysis/pytorch/1.4/KiU-Net-pytorch",
  "Sentence-VAE": "/home/lei/compatibility_analysis/pytorch/1.5/Sentence-VAE",
  "MASTER-pytorch": "/home/lei/compatibility_analysis/pytorch/1.5/MASTER-pytorch",
  "nlp_classification": "/home/lei/compatibility_analysis/pytorch/1.5/nlp_classification/Character-level_Convolutional_Networks_for_Text_Classification",
  "pytorch-hed": "/home/lei/compatibility_analysis/pytorch/1.7/pytorch-hed",
  "3d-ken-burns": "/home/lei/compatibility_analysis/pytorch/1.7/3d-ken-burns",
  "sepconv-slomo": "/home/lei/compatibility_analysis/pytorch/1.7/sepconv-slomo",
  "svoice": "/home/lei/compatibility_analysis/pytorch/1.6/svoice",
  "LaneATT": "/home/lei/compatibility_analysis/pytorch/1.6/LaneATT",
  "JointBERT": "/home/lei/compatibility_analysis/pytorch/1.6/JointBERT",
  "pytorch-liteflownet": "/home/lei/compatibility_analysis/pytorch/1.6/pytorch-liteflownet",
  "R-BERT": "/home/lei/compatibility_analysis/pytorch/1.6/R-BERT",
  "pytorch-spynet": "/home/lei/compatibility_analysis/pytorch/1.6/pytorch-spynet",
  "GLCIC-PyTorch": "/home/lei/compatibility_analysis/pytorch/1.9/GLCIC-PyTorch",
  "DeepMosaics": "/home/lei/compatibility_analysis/pytorch/1.7/DeepMosaics",
  "PedalNetRT": "/home/lei/compatibility_analysis/pytorch/1.7/PedalNetRT",
  "BERT-NER": "/home/lei/compatibility_analysis/pytorch/1.2/BERT-NER",
  "Federated-Learning-PyTorch": "/home/lei/compatibility_analysis/pytorch/1.2/Federated-Learning-PyTorch/src",
  "RetinaFace_Pytorch": "/home/lei/compatibility_analysis/pytorch/1.1/RetinaFace_Pytorch",
  "Bert-Multi-Label-Text-Classification": "/home/lei/compatibility_analysis/pytorch/1.0/Bert-Multi-Label-Text-Classification",
  "siamese-pytorch": "/home/lei/compatibility_analysis/pytorch/1.0/siamese-pytorch",
  "graphSAGE-pytorch": "/home/lei/compatibility_analysis/pytorch/1.0/graphSAGE-pytorch",
  "pt.darts": "/home/lei/compatibility_analysis/pytorch/1.0/pt.darts"
}
python_command_dict = {
  "SimpleHTR": "python main.py",
  "StyleGAN-Tensorflow": "python main.py --dataset FFHQ --img_size 8 --gpu_num 3 --progressive True --phase train",
  "cnn_captcha": "python3 train_model.py",
  "deep-belief-network": "python example_classification.py",
  "crnn.pytorch": "python demo.py",
  "siren": "python scripts/train_inpainting_siren.py",
  "Deep-SAD-PyTorch": "python main.py mnist mnist_LeNet ../log/DeepSAD/mnist_test ../data --ratio_known_outlier 0.01 --ratio_pollution 0.1 --lr 0.0001 --n_epochs 1 --lr_milestone 50 --batch_size 128 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 1 --ae_batch_size 128 --ae_weight_decay 0.5e-3 --normal_class 0 --known_outlier_class 1 --n_known_outlier_classes 1 --seed 2",
  "PyTorch-ENet": "python main.py -m test --save-dir save/ENet_CamVid",
  "ConSinGAN": "python main_train.py --gpu 0 --train_mode retarget --input_name Images/Generation/colusseum.png --niter 100 --train_stages 3",
  "spert": "python ./spert.py train --config configs/example_train.conf",
  "hifi-gan": "python train.py --config config_v1.json",
  "DexiNed": "python main.py",
  "KiU-Net-pytorch": "python train.py --train_dataset \"dataset/train\" --val_dataset \"dataset/val\" --direc \"result\" --batch_size 1 --epoch 1 --save_freq 1 --modelname \"kiunet\" --learning_rate 0.0001",
  "Sentence-VAE": "python train.py",
  "MASTER-pytorch": "python train.py -c configs/config_lmdb.json -d 1 -dist false",
  "nlp_classification": "python train.py",
  "pytorch-hed": "python run.py --model bsds500 --in ./images/sample.png --out ./out.png",
  "3d-ken-burns": "python depthestim.py --in ./images/doublestrike.jpg --out ./depthestim.npy",
  "sepconv-slomo": "python run.py --model lf --one ./images/one.png --two ./images/two.png --out ./out.png",
  "svoice": "python -m svoice.evaluate outputs/exp_/checkpoint.th egs/debug/tr",
  "LaneATT": "python main.py train --exp_name laneatt_r34_tusimple --cfg cfgs/laneatt_tusimple_resnet34.yml",
  "JointBERT": "python3 predict.py --input_file sample_pred_in.txt --output_file sample_pred_out.txt --model_dir atis_model",
  "pytorch-liteflownet": "python run.py --model default --one ./images/one.png --two ./images/two.png --out ./out.flo",
  "R-BERT": "python3 predict.py --input_file sample_pred_in.txt --output_file out_result.txt --model_dir model",
  "pytorch-spynet": "python run.py --model sintel-final --first ./images/first.png --second ./images/second.png --out ./out.flo",
  "GLCIC-PyTorch": "python train.py datasets/img_align_celeba results/demo/",
  "DeepMosaics": "python deepmosaic.py --media_path ./imgs/ruoruo.jpg --model_path ./pretrained_models/mosaic/add_face.pth --gpu_id 0",
  "PedalNetRT": "python train.py data/ts9_test1_in_FP32.wav data/ts9_test1_out_FP32.wav",
  "BERT-NER": "python run_ner.py --data_dir=data/ --bert_model=bert-base-cased --task_name=ner --output_dir=out_base --max_seq_length=128 --do_eval --warmup_proportion=0.1",
  "Federated-Learning-PyTorch": "python baseline_main.py --model=mlp --dataset=mnist --epochs=1",
  "RetinaFace_Pytorch": "python train.py --data_path ./widerface --batch 16 --save_path ./out",
  "Bert-Multi-Label-Text-Classification": "python run_bert.py --do_train --save_best --do_lower_case",
  "siamese-pytorch": "python3 train.py --train_path omniglot/python/images_background --test_path omniglot/python/images_evaluation --gpu_ids 0 --model_path models",
  "graphSAGE-pytorch": "python -m src.main --epochs 1 --cuda --learn_method unsup",
  "pt.darts": "python augment.py --name cifar10 --dataset cifar10"
}
def run_python_command_in_env(env_name, python_command, proj_path, proj_name, target_library, target_version):
    # 在虚拟环境中运行 Python 命令

    #command = f"bash -i -c 'source /home/lei/anaconda3/bin/activate {env_name} && cd {proj_path} && {python_command} > {target_path}/{target_version}.txt 2>&1'"
    if proj_name == "Bert-Multi-Label-Text-Classification" and target_library == "transformers":
        command = f"bash -c 'source /home/lei/anaconda3/bin/activate {env_name} && cd {proj_path} && PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python {python_command}'"
    else:
        command = f"bash -i -c 'source /home/lei/anaconda3/bin/activate {env_name} && cd {proj_path} && {python_command}'"
    process = subprocess.Popen(command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #print(f"Running command: {command}")
    stdout, stderr = process.communicate()  # 等待命令执行完成并获取输出

if __name__ == "__main__":
    with open(f"./pip.json", "r") as f:
        result_dict = json.load(f)

    for key in result_dict:
        proj_name = key  # 项目名称
        proj_path = proj_path_dict[proj_name]  # 项目路径
        python_command = python_command_dict[proj_name]  # 要执行的 Python 命令
        for library in result_dict[key]:
            target_library = library  # 库名称
            for version in result_dict[key][library]:
                target_version = version  # 库版本
                env_name = f"{proj_name}-{target_library}-{target_version}"  # Conda 虚拟环境名称
                print(f"Running experiment for {proj_name}-{target_library}-{target_version}...")
                run_python_command_in_env(env_name, python_command, proj_path, proj_name, target_library, target_version)
                


        


            
