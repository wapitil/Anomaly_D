cd /home/wapiti/Projects/Anomaly_D
python train.py

python export_stfpm_onnx.py --ckpt /home/wapiti/Projects/Anomaly_D/results/Stfpm/MVTecAD/leather/latest/weights/lightning/model.ckpt --output stfpm_leather.onnx
