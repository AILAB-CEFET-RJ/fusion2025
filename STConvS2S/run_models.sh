#!/bin/bash

current_branch=$(git branch --show-current)
echo "Starting experiments with datasets in branch $current_branch."

# 1. ERA5+SIA experiment (Sirenes + INMET + Alerta Rio)
nohup python main.py --cuda 1 -i 1 -v 4 -m stconvs2s-r -e 200 -p 100 --plot --dropout 0.5 -dsp "data/output_dataset_replaced-ERA5+SIA.nc" -r "RunModels_ERA5+SIA" &> RunModels_ERA5+SIA.log &
pid1=$!
echo "Started ERA5+SIA process with PID $pid1"
wait $pid1
echo "ERA5+SIA process finished."

# 2. ERA5+IA experiment (INMET + Alerta Rio)
nohup python main.py --cuda 1 -i 1 -v 4 -m stconvs2s-r -e 200 -p 100 --plot --dropout 0.5 -dsp "data/output_dataset_replaced-ERA5+IA.nc" -r "RunModels_ERA5+IA" &> RunModels_ERA5+IA.log &
pid2=$!
echo "Started ERA5+IA process with PID $pid2"
wait $pid2
echo "ERA5+IA process finished."

# 3. ERA5+SA experiment (Sirenes + Alerta Rio)
nohup python main.py --cuda 1 -i 1 -v 4 -m stconvs2s-r -e 200 -p 100 --plot --dropout 0.5 -dsp "data/output_dataset_replaced-ERA5+SA.nc" -r "RunModels_ERA5+SA" &> RunModels_ERA5+SA.log &
pid3=$!
echo "Started ERA5+SA process with PID $pid3"
wait $pid3
echo "ERA5+SA process finished."

# 4. ERA5+A experiment (Alerta Rio only)
nohup python main.py --cuda 1 -i 1 -v 4 -m stconvs2s-r -e 200 -p 100 --plot --dropout 0.5 -dsp "data/output_dataset_replaced-ERA5+A.nc" -r "RunModels_ERA5+A" &> RunModels_ERA5+A.log &
pid4=$!
echo "Started ERA5+A process with PID $pid4"
wait $pid4
echo "ERA5+A process finished."

# 5. ERA5+SI (Sirenes + INMET)
nohup python main.py --cuda 1 -i 1 -v 4 -m stconvs2s-r -e 200 -p 100 --plot --dropout 0.5 \
  -dsp "data/no_overlapping_output_dataset_websirenes_inmet_2011-01_2024-10.nc" \
  -r "RunModels_ERA5+sirenes+inmet_dr0.5_balanced_MAE" &> RunModels_ERA5+sirenes+inmet_dr0.5_balanced_MAE.log &
pid5=$!
echo "Started ERA5+SI process with PID $pid5"
wait $pid5
echo "ERA5+SI process finished."

# 6. ERA5+I (INMET only)
nohup python main.py --cuda 1 -i 1 -v 4 -m stconvs2s-r -e 200 -p 100 --plot --dropout 0.5 \
  -dsp "data/no_overlapping_output_dataset_inmet_only_2011-01_2024-10.nc" \
  -r "RunModels_ERA5+inmet_only_dr0.5_balanced_MAE" &> RunModels_ERA5+inmet_only_dr0.5_balanced_MAE.log &
pid6=$!
echo "Started ERA5+I process with PID $pid6"
wait $pid6
echo "ERA5+I process finished."

# 7. ERA5 (ERA5 only)
nohup python main.py --cuda 1 -i 1 -v 4 -m stconvs2s-r -e 200 -p 100 --plot --dropout 0.5 \
  -dsp "data/no_overlapping_output_dataset_era5_only_2011-01_2024-10.nc" \
  -r "RunModels_ERA5_only_dr0.5_balanced_MAE" &> RunModels_ERA5_only_dr0.5_balanced_MAE.log &
pid7=$!
echo "Started ERA5 process with PID $pid7"
wait $pid7
echo "ERA5 process finished."

# 8. ERA5+S (Sirenes only)
nohup python main.py --cuda 1 -i 1 -v 4 -m stconvs2s-r -e 200 -p 100 --plot --dropout 0.5 \
  -dsp "data/no_overlapping_output_dataset_sirenes_only_2011-01_2024-10.nc" \
  -r "RunModels_ERA5+sirenes_only_dr0.5_balanced_MAE" &> RunModels_ERA5+sirenes_only_dr0.5_balanced_MAE.log &
pid8=$!
echo "Started ERA5+S process with PID $pid8"
wait $pid8
echo "ERA5+S process finished."

echo "All processes have completed"
