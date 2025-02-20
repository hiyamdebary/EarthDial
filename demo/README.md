# <img src="../images/EarthDial_logo.png" height="30"> EarthDial Demo

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx EarthDial-ChatGPT">
</p>

## EarthDial Demo

This repository provides a demo setup for EarthDial, a system that integrates multiple components for AI-driven tasks. Follow the steps below to set up and run the demo.

---

## **Setup Guide**

### **Step 1: Set Environment Variables**
Before running the application, set the necessary environment variables in your terminal:

```sh
export SD_SERVER_PORT=39999
export WEB_SERVER_PORT=10003
export CONTROLLER_PORT=40000
export CONTROLLER_URL=http://0.0.0.0:$CONTROLLER_PORT
export SD_WORKER_URL=http://0.0.0.0:$SD_SERVER_PORT
```

---

### **Step 2: Start the Streamlit Web Server**
Run the following command to start the Streamlit web server on the specified port:

```sh
streamlit run app.py --server.port $WEB_SERVER_PORT -- --controller_url $CONTROLLER_URL --sd_worker_url $SD_WORKER_URL
```

---

### **Step 3: Start the Controller**
Launch the controller process using the command below:

```sh
python controller.py --host 0.0.0.0 --port $CONTROLLER_PORT
```

---

### **Step 4: Start the Model Workers**
You can start different EarthDial model workers with varying model sizes. Below is an example command for starting the **EarthDialRGB Worker** on port `40001`:

```sh
CUDA_VISIBLE_DEVICES=0 python model_worker.py --host 0.0.0.0 --controller $CONTROLLER_URL --port 40001 --worker http://0.0.0.0:40001 --model-path /cos/Model_Files/Model_Weights/4B_Full_9Nov_pretrain_VIT_MLP_LLM_1_RGBFinetune_Change

```

---

### **Notes:**
- Ensure that all dependencies are installed before running the setup.
- Modify the `CUDA_VISIBLE_DEVICES` value as needed based on your GPU availability.
- The ports used can be changed based on your requirements.


**Happy Coding! 🚀**

