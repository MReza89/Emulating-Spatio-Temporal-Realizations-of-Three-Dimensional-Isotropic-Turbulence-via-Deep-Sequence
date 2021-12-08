# Emulating Spatio-Temporal Realizations of Three-Dimensional Isotropic Turbulence via Deep Sequence Learning Models
This is an implementation of [Emulating Spatio-Temporal Realizations of Three-Dimensional Isotropic Turbulence via Deep Sequence Learning Models](https://arxiv.org/abs/2112.03469)

## Requirements
 - see requirements.txt

## Instruction
 - Global hyperparameters are configured in config.yml
 - Hyperparameters can be found at process_control() in utils.py 

## Examples
 - Train vqvae, compression scale 1, regularization parameter <img src="https://render.githubusercontent.com/render/math?math=\alpha=0.1,\gamma=0">
    ```ruby
    python train_vqvae.py --data_name Turb --model_name vqvae --control_name 1_chs-1_exact-physcis_0.1_0
    ```
 - Train Conv-LSTM model, compression scale 2, regularization parameter <img src="https://render.githubusercontent.com/render/math?math=\alpha=0.1,\gamma=0.001">, 3 time steps for training and 3 time steps for testing, with teacher-forcing
    ```ruby
    python train_convlstm.py --data_name Turb --model_name convlstm --control_name 2_chs-1_exact-physcis_0.1_0.001_3-3_0
    ```
 - Train Conv-Transformer model, compression scale 3, regularization parameter <img src="https://render.githubusercontent.com/render/math?math=\alpha=0.1,\gamma=0.001">, 3 time steps for training and 3 time steps for testing, with cyclic training
    ```ruby
    python train_transformer_cyclic.py --data_name Turb --model_name transformer --control_name 3_chs-1_exact-physcis_0.1_0.001_3-3_0_1
    ```
## Results
- Predictions of velocity field using Conv-LSTM models.

![Conv-LSTM](/asset/Conv-LSTM.png)

- Predictions of velocity field using Conv-Transformer models.

![Conv-Transformer](/asset/Conv-Transformer.png)

## Acknowledgement
*Mohammadreza Momenifar  
Enmao Diao  
Vahid Tarokh  
Andrew D. Bragg*
