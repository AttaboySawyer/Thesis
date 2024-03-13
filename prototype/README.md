# Prototype Usage Instructions

To run the prototype, ensure both the server and frontend are operational concurrently. Below are step-by-step instructions to set up and launch both components.

### Server Setup

1. Download the server folder. The essential file required is `app.py`.
2. Install any missing dependencies using the following command:
```
pip install requiredDependency
```
3. Initiate the server by executing:
```
flask run
```

### Frontend Setup

1. Download the frontend folder.
2. As the frontend is developed using React and JavaScript, download all necessary packages by navigating to the frontend directory and execute:
```
npm install
```
3. Launch the frontend by executing:
```
npm start
```

### Utilizing the Frontend
Once both components are operational, you'll encounter a homepage resembling the following:
![image](https://user-images.githubusercontent.com/83662258/206776840-939711ec-0d08-4c79-b075-7c953550bdf4.png)

Click on the "SELECT FILE" button and utilize the provided example temp mp4 file located in this folder. With the server running, the mp4 file will undergo processing, eventually leading you to a screen similar to this:
![image](https://user-images.githubusercontent.com/83662258/206777188-8aaffb13-f3af-4724-afd4-d289fda829df.png)

Here, you can select images and annotate them. Please note that the download buttons are currently placeholders, as the primary focus was on integrating the network with the frontend.
