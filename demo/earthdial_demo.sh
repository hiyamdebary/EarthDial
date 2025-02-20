#!/bin/bash

# Run the Streamlit app
nohup streamlit run app.py --server.port 10003 -- --controller_url http://0.0.0.0:40000 --sd_worker_url http://0.0.0.0:39999 > logs/streamlit_app.log 2>&1 &

echo "Streamlit app started on port 10003. Logs: streamlit_app.log"

# Wait for 5 seconds
sleep 10
# Run the controller
nohup python controller.py --host 0.0.0.0 --port 40000 > logs/controller.log 2>&1 &

echo "Controller started on port 40000. Logs: controller.log"
# Wait for 5 seconds
#sleep 30

echo "Now loading model into UI"
# Run the model worker
nohup CUDA_VISIBLE_DEVICES=0 python model_worker.py \
    --host 0.0.0.0 \
    --controller http://0.0.0.0:40000 \
    --port 40001 \
    --worker http://0.0.0.0:40001 \
    --model-path /cos/Model_Files/Model_Weights/4B_Full_9Nov_pretrain_VIT_MLP_LLM_1_RGBFinetune_Change \
    > bash_logs/model_worker.log 2>&1 &

# echo "Model worker started on port 40001. Logs: model_worker.log"
echo " Model running on this http://localhost:10003/ "
echo "All processes have been started."
